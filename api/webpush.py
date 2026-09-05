"""Server-side Web Push for the community WebUI (RFC 8030 / RFC 8291).

Implements VAPID-signed Web Push (est. #3196) so phones can receive
notifications even when the WebUI page/tab is dead or backgrounded.

- VAPID keypair persists at ``<STATE_DIR>/webpush_keys.pem``.
- Push subscriptions persist at ``<STATE_DIR>/webpush_subscriptions.json``.
- Delivery uses ``pywebpush``; externally imported in a way that can never
  break the request/streaming path (all sending happens on daemon threads and
  every failure is swallowed + logged).

No endpoint code lives here; the HTTP layer (api/routes.py) calls into this
module. Keeping the plumbing isolated means a pywebpush import failure or a
network blip can never take down stream handling.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from urllib.parse import quote

from api.config import STATE_DIR

logger = logging.getLogger("webpush")

_SUBS_FILE = Path(STATE_DIR) / "webpush_subscriptions.json"
_KEYS_FILE = Path(STATE_DIR) / "webpush_keys.pem"

# Default contact -- set HERMES_WEBUI_VAPID_SUBJECT=mailto:you@example.com to
# override. Push services reject VAPID JWTs without a valid "sub".
VAPID_SUBJECT = os.getenv("HERMES_WEBUI_VAPID_SUBJECT", "mailto:admin@localhost")

# Hard cap on notification preview text.
_PREVIEW_MAX_CHARS = 500

_state_lock = threading.Lock()

try:  # optional import -- web push is best-effort
    from py_vapid import Vapid, b64urlencode
    from pywebpush import WebPushException, webpush

    _ENABLED = True
except Exception as exc:  # pragma: no cover - env-dependent
    Vapid = None
    b64urlencode = None
    webpush = None
    WebPushException = Exception
    _ENABLED = False
    logger.warning("pywebpush unavailable; web push disabled: %s", exc)


def available() -> bool:
    return _ENABLED


# ── VAPID keys ────────────────────────────────────────────────────────────────
def _vapid() -> "Vapid":
    if not _ENABLED:
        raise RuntimeError("pywebpush not available")
    if _KEYS_FILE.exists():
        return Vapid().from_file(_KEYS_FILE)
    with _state_lock:
        if not _KEYS_FILE.exists():
            v = Vapid()
            v.generate_keys()
            v.save_key(_KEYS_FILE)
    return Vapid().from_file(_KEYS_FILE)


def get_vapid_public_key() -> str:
    """URL-safe base64 of the raw (uncompressed X962) public key, for the
    frontend's ``applicationServerKey``. py_vapid's .public_key is a
    cryptography ECPublicKey object, so serialize explicitly."""
    from cryptography.hazmat.primitives import serialization

    v = _vapid()
    raw = v.public_key.public_bytes(
        serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint
    )
    return b64urlencode(raw)


# ── Subscription store ────────────────────────────────────────────────────────
def _load_subs() -> list[dict]:
    try:
        data = json.loads(_SUBS_FILE.read_text())
        if isinstance(data, list):
            return _sanitize_subs(data)
    except FileNotFoundError:
        pass
    except Exception as exc:
        logger.debug("webpush: failed to read subscriptions: %s", exc)
    return []


def _sanitize_subs(raw: list) -> list[dict]:
    out = []
    for s in raw:
        if isinstance(s, dict) and isinstance(s.get("endpoint"), str) and isinstance(
            s.get("keys"), dict
        ):
            out.append(s)
    return out


def _save_subs(subs: list[dict]) -> None:
    try:
        _SUBS_FILE.parent.mkdir(parents=True, exist_ok=True)
        _SUBS_FILE.write_text(json.dumps(subs, indent=2))
    except Exception as exc:
        logger.warning("webpush: failed to save subscriptions: %s", exc)


def add_subscription(raw: dict) -> int:
    """Register (or update) a subscription keyed by endpoint. Returns count."""
    endpoint = (raw or {}).get("endpoint") or ""
    keys = (raw or {}).get("keys") or {}
    if not endpoint or not isinstance(keys, dict) or not keys.get("p256dh"):
        return len(_load_subs())
    with _state_lock:
        subs = [s for s in _load_subs() if s.get("endpoint") != endpoint]
        subs.append(
            {
                "endpoint": endpoint,
                "keys": {
                    "p256dh": str(keys.get("p256dh", "")),
                    "auth": str(keys.get("auth", "")),
                },
                "created_at": time.time(),
            }
        )
        _save_subs(subs)
        return len(subs)


def remove_subscription(endpoint: str) -> int:
    with _state_lock:
        subs = [s for s in _load_subs() if s.get("endpoint") != endpoint]
        _save_subs(subs)
        return len(subs)


def subscription_count() -> int:
    return len(_load_subs())


# ── Sending ───────────────────────────────────────────────────────────────────
def _send_one(sub: dict, payload: str, claims: dict) -> None:
    """Deliver one push. Expires a subscription on 404/410 (gone)."""
    try:
        webpush(
            subscription_info=sub,
            data=payload,
            vapid_private_key=_vapid(),
            vapid_claims=claims,
            timeout=10,
        )
    except WebPushException as exc:
        resp = getattr(exc, "response", None)
        if resp is not None and getattr(resp, "status_code", None) in (404, 410):
            remove_subscription(sub.get("endpoint", ""))
            logger.info(
                "webpush: expired subscription removed (HTTP %s)",
                resp.status_code,
            )
        else:
            logger.warning("webpush: send failed: %s", exc)
    except Exception as exc:  # never let a push failure propagate
        logger.warning("webpush: send exception: %s", exc)


def push_notify(title: str, body: str, url: str = "/", subs: list[dict] | None = None) -> int:
    """Fire a push to all stored subscriptions on background threads. Returns
    the number of subscriptions dispatched to."""
    if not _ENABLED:
        return 0
    subs = _load_subs() if subs is None else subs
    if not subs:
        return 0
    payload = json.dumps({"title": title, "body": body, "url": url})
    claims = {"sub": VAPID_SUBJECT}
    for s in subs:
        try:
            threading.Thread(
                target=_send_one, args=(s, payload, claims), daemon=True
            ).start()
        except Exception:
            logger.debug("webpush: failed to spawn sender")
    return len(subs)


# ── Completion hook used by api/streaming.py ─────────────────────────────────
def _completion_preview(session) -> str:
    msgs = getattr(session, "messages", None)
    if not isinstance(msgs, list):
        return ""
    for m in reversed(msgs):
        if not isinstance(m, dict) or m.get("role") != "assistant":
            continue
        content = m.get("content", "")
        if isinstance(content, list):
            parts = []
            for p in content:
                if isinstance(p, dict):
                    parts.append(p.get("text") or p.get("content") or "")
            content = "".join(parts)
        text = (str(content or "")).strip()
        if text:
            return text[:_PREVIEW_MAX_CHARS]
    return ""


def notify_completion(session_id: str, session, done_payload: dict | None) -> None:
    """Fire a 'Response complete' push after a successful agent turn. Runs on a
    daemon thread so it can never delay or break stream teardown."""
    def _work() -> None:
        try:
            if not _ENABLED:
                return
            body = _completion_preview(session) or "Task finished"
            sid = session_id or ""
            url = "/session/" + quote(sid, safe="") if sid else "/"
            push_notify("Response complete", body, url)
        except Exception:
            logger.debug("webpush: completion notify failed", exc_info=True)

    try:
        threading.Thread(target=_work, daemon=True).start()
    except Exception:
        logger.debug("webpush: failed to spawn completion thread")