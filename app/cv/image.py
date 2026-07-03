"""Image loading: HEIC decode, EXIF-orientation correction, and a perceptual hash.

A small dHash (difference hash) gives duplicate grouping (FR-006) without a new
dependency — it reuses Pillow + numpy (Constitution V: prefer what's present).
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageOps

# Register HEIC/HEIF so Pillow can open iPhone photos.
try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except Exception:  # pragma: no cover - optional at import time
    pass


def load_image(data: bytes) -> tuple[np.ndarray, int, int, bool]:
    """Decode bytes to an RGB numpy array with EXIF orientation applied.

    Returns (rgb_array, width, height, orientation_applied).
    """
    with Image.open(io.BytesIO(data)) as img:
        had_exif_orientation = bool(img.getexif().get(0x0112, 1) not in (0, 1))
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        arr = np.asarray(img)
        h, w = arr.shape[:2]
        return arr, w, h, had_exif_orientation


def rgb_to_bgr(rgb: np.ndarray) -> np.ndarray:
    """InsightFace/OpenCV expect BGR channel order."""
    return rgb[:, :, ::-1].copy()


def dhash(data: bytes, hash_size: int = 8) -> str:
    """Perceptual difference-hash as a hex string, for duplicate collapse."""
    with Image.open(io.BytesIO(data)) as img:
        img = ImageOps.exif_transpose(img).convert("L").resize(
            (hash_size + 1, hash_size), Image.LANCZOS
        )
        px = np.asarray(img, dtype=np.int16)
    diff = px[:, 1:] > px[:, :-1]
    bits = np.packbits(diff.flatten())
    return bits.tobytes().hex()
