"""Gemini "Nano Banana" AI photo edit (F7 stretch, FR-017).

The one place an image may leave our machine — and only ever a photo of just the
requesting user, opt-in and explicitly consented (Constitution IV; the solo-photo
guard lives in ``services.export.is_solo_editable``). Calls the free-tier Google
AI Studio image model over plain HTTPS (stdlib, no SDK dependency) and returns the
edited bytes; the original is never modified.

Disabled unless ``GEMINI_API_KEY`` is set — ``gemini_available()`` gates the
endpoint, which returns 503 otherwise. Nothing here runs at import time.
"""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request

from app.config import get_settings

_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models"


def gemini_available() -> bool:
    return bool(get_settings().gemini_api_key)


def build_request_payload(image_b64: str, mime_type: str, prompt: str) -> dict:
    """The generateContent body: the source image inline + the edit instruction."""
    return {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": mime_type, "data": image_b64}},
                ],
            }
        ],
        "generationConfig": {"responseModalities": ["IMAGE"]},
    }


def parse_image_response(payload: dict) -> bytes:
    """Extract the first returned image's bytes from a generateContent response."""
    for cand in payload.get("candidates", []):
        for part in cand.get("content", {}).get("parts", []):
            inline = part.get("inline_data") or part.get("inlineData")
            if inline and inline.get("data"):
                return base64.b64decode(inline["data"])
    raise ValueError("Gemini response contained no image")


def edit_image(image_bytes: bytes, prompt: str, mime_type: str = "image/jpeg") -> bytes:
    """Send one photo + a text instruction to Gemini; return the edited image."""
    settings = get_settings()
    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY not set — AI editing is disabled")

    body = build_request_payload(
        base64.b64encode(image_bytes).decode("ascii"), mime_type, prompt
    )
    url = f"{_ENDPOINT}/{settings.gemini_image_model}:generateContent"
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": settings.gemini_api_key,
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 (trusted Google host)
        payload = json.loads(resp.read())
    return parse_image_response(payload)


if __name__ == "__main__":  # runnable self-check (Constitution V)
    # Pure helpers are always testable; the network call is exercised only when a
    # key is configured (we don't spend quota in the self-check).
    body = build_request_payload("QUJD", "image/jpeg", "make the sky pink")
    assert body["contents"][0]["parts"][1]["inline_data"]["data"] == "QUJD"
    assert body["generationConfig"]["responseModalities"] == ["IMAGE"]

    fake = {
        "candidates": [
            {"content": {"parts": [{"inlineData": {"mimeType": "image/png", "data": "QUJD"}}]}}
        ]
    }
    assert parse_image_response(fake) == b"ABC"

    empty = {"candidates": [{"content": {"parts": [{"text": "no image"}]}}]}
    try:
        parse_image_response(empty)
        raise AssertionError("expected ValueError on imageless response")
    except ValueError:
        pass

    state = "configured" if gemini_available() else "disabled"
    print(f"edit_gemini self-check OK (Gemini {state})")
