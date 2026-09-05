"""Offline verification of the server-side Web Push crypto path (no network).

Proves that the VAPID signing and RFC 8291 (`aes128gcm`) message encryption the
server sends are correct, by:

  * capturing the outgoing push HTTP request that ``pywebpush`` builds
    (requests.post() is monkeypatched -- nothing hits the network),
  * verifying the ES256 VAPID JWT against the ``k=`` public key and confirming
    the ``aud``/``sub``/``exp`` claims,
  * decrypting the ``aes128gcm`` body with the receiver's private key and
    asserting it round-trips to exactly the payload we sent.

Also smoke-tests the VAPID key format and the subscription store CRUD.

These tests never touch the real STATE_DIR -- the storage paths are isolated to
a tmp dir via monkeypatch.
"""

import base64
import json
import os

import pytest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec, utils as ec_utils
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from api import webpush


def _urlb64(b: bytes) -> str:
    """URL-safe base64 without padding (the Push API wire format)."""
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def _b64url_dec(s: str) -> bytes:
    s = s.replace("-", "+").replace("_", "/")
    return base64.b64decode(s + "=" * ((4 - len(s) % 4) % 4))


@pytest.fixture()
def _isolated_store(tmp_path, monkeypatch):
    """Point the module's key + subscription files at a tmp dir so tests can
    never touch the live ~/.hermes/webui store during a run."""
    monkeypatch.setattr(webpush, "_KEYS_FILE", tmp_path / "k.pem")
    monkeypatch.setattr(webpush, "_SUBS_FILE", tmp_path / "subs.json")
    yield


def test_vapid_public_key_is_raw_p256_point(_isolated_store):
    pk = webpush.get_vapid_public_key()
    raw = _b64url_dec(pk)
    # Uncompressed P-256 X.962 public point = 0x04 || X(32) || Y(32) = 65 bytes.
    assert len(raw) == 65
    assert raw[0] == 0x04
    # 65 raw bytes -> 87 url-safe base64 chars (no padding).
    assert len(pk) == 87


def test_vapid_keys_persist_and_are_stable(_isolated_store, tmp_path):
    first = webpush.get_vapid_public_key()
    second = webpush.get_vapid_public_key()
    assert first == second
    assert (tmp_path / "k.pem").is_file()


def test_subscription_store_crud(_isolated_store):
    fake = {
        "endpoint": "https://push.example.test/edge/1",
        "keys": {"p256dh": "B" * 65, "auth": "A" * 16},
    }
    assert webpush.subscription_count() == 0
    assert webpush.add_subscription(fake) == 1
    # second distinct subscription
    assert (
        webpush.add_subscription(
            {**fake, "endpoint": "https://push.example.test/edge/2"}
        )
        == 2
    )
    # re-adding the same endpoint dedupes (keys updated, count stable)
    assert (
        webpush.add_subscription(
            {**fake, "keys": {"p256dh": "C" * 65, "auth": "D" * 16}}
        )
        == 2
    )
    # malformed payloads are ignored
    assert webpush.add_subscription({"endpoint": "bad"}) == 2
    assert webpush.add_subscription({}) == 2
    assert webpush.subscription_count() == 2
    assert webpush.remove_subscription(fake["endpoint"]) == 1


def test_push_roundtrip_vapid_sign_and_aes128gcm(_isolated_store, monkeypatch):
    captured = {}

    class _FakeResp:
        status_code = 201
        text = ""

        def json(self):
            return {}

    def _fake_post(url, headers=None, data=None, timeout=None, **kw):
        captured["url"] = url
        captured["headers"] = headers or {}
        captured["body"] = (
            data.encode() if isinstance(data, str) else (data or b"")
            if isinstance(data, bytes)
            else b""
        )
        return _FakeResp()

    import requests

    monkeypatch.setattr(requests, "post", _fake_post)

    # A stand-in "device" subscription (real receivers are created by the
    # browser's PushManager; this gives us a private key to decrypt with).
    recv_priv = ec.generate_private_key(ec.SECP256R1())
    recv_pub_raw = recv_priv.public_key().public_bytes(
        Encoding.X962, PublicFormat.UncompressedPoint
    )
    auth_secret = os.urandom(16)
    sub = {
        "endpoint": "https://updates.push.services.mozilla.com/wpush/v2/gAAAAABx?x=1",
        "keys": {"p256dh": _urlb64(recv_pub_raw), "auth": _urlb64(auth_secret)},
    }
    payload = json.dumps(
        {"title": "Response complete", "body": "hello reply", "url": "/session/abc"}
    )

    webpush._send_one(
        sub, payload, {"sub": "mailto:test@example.com"}
    )

    assert "url" in captured, "pywebpush never issued the push request"
    headers = captured["headers"]
    assert headers.get("Content-Encoding") == "aes128gcm"
    assert headers.get("TTL") is not None

    # -- verify the VAPID JWT -------------------------------------------------
    auth = headers.get("Authorization")
    assert auth and auth.startswith("vapid "), f"missing vapid header: {auth!r}"
    params = dict(p.split("=", 1) for p in auth[len("vapid "):].split(","))
    jwt = params["t"].strip('"')
    k_b64 = params["k"].strip('"')

    vapid_pub = ec.EllipticCurvePublicKey.from_encoded_point(
        ec.SECP256R1(), _b64url_dec(k_b64)
    )
    header_b64, payload_b64, sig_b64 = jwt.split(".")
    sig_raw = _b64url_dec(sig_b64)
    r = int.from_bytes(sig_raw[:32], "big")
    s = int.from_bytes(sig_raw[32:], "big")
    vapid_pub.verify(
        ec_utils.encode_dss_signature(r, s),
        (header_b64 + "." + payload_b64).encode(),
        ec.ECDSA(hashes.SHA256()),
    )  # raises on bad signature

    header = json.loads(_b64url_dec(header_b64))
    claims = json.loads(_b64url_dec(payload_b64))
    endpoint_origin = "/".join(captured["url"].split("/")[:3])

    assert header["alg"] == "ES256"
    assert claims["aud"] == endpoint_origin
    assert claims["sub"] == "mailto:test@example.com"
    assert claims["exp"] > 1_000_000_000

    # -- decrypt the payload with the receiver's private key ------------------
    from http_ece import decrypt

    plain = decrypt(
        captured["body"],
        private_key=recv_priv,
        auth_secret=auth_secret,
        version="aes128gcm",
    )
    assert json.loads(plain) == json.loads(payload)