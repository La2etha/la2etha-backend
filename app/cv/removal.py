"""Privacy-preserving background-person removal (F6, FR-016) via local LaMa.

At export the user may opt to remove unclaimed background people from a photo so
it can be shared comfortably. We build a mask over those faces' regions and let
LaMa inpaint them away. Automatic face-blurring is explicitly NOT used (it
degrades the photo); removal reconstructs plausible background instead.

LaMa (torch + ``simple-lama-inpainting``) is a heavy, optional dependency (the
`export` extra). If it's absent, ``lama_available()`` is False and the export
endpoint serves the original photo / returns 503 for a removal request — the app
never fails to import for lack of it. Nothing heavy is imported at module load.

ponytail: the mask covers each target face's region expanded downward for a bit
of neck/torso — a whole-person cutout would need a segmentation model. This
removes the identifying face region, which is the point for shareability; note
the ceiling and upgrade path (person segmentation) here rather than pretending.
"""

from __future__ import annotations

import threading

import numpy as np

_lock = threading.Lock()
_lama = None


def lama_available() -> bool:
    try:
        import simple_lama_inpainting  # noqa: F401
        import torch  # noqa: F401

        return True
    except Exception:
        return False


def _load():
    global _lama
    if _lama is not None:
        return _lama
    with _lock:
        if _lama is None:
            from simple_lama_inpainting import SimpleLama

            _lama = SimpleLama()
    return _lama


def build_mask(
    image_shape: tuple[int, int],
    bboxes: list[dict],
    pad_ratio: float = 0.4,
    torso_ratio: float = 1.2,
) -> np.ndarray:
    """Binary mask (uint8, 255 = remove) over each bbox, padded around the face
    and extended downward to catch some neck/torso. ``image_shape`` is (H, W)."""
    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for b in bboxes:
        bw, bh = float(b["w"]), float(b["h"])
        x0 = int(b["x"] - pad_ratio * bw)
        y0 = int(b["y"] - pad_ratio * bh)
        x1 = int(b["x"] + bw + pad_ratio * bw)
        y1 = int(b["y"] + bh + torso_ratio * bh)  # extend down for neck/torso
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 255
    return mask


def remove_regions(image_rgb: np.ndarray, bboxes: list[dict]) -> np.ndarray:
    """Inpaint away the given face regions; returns an RGB array.

    If there is nothing to remove, returns the image unchanged (no model needed).
    """
    if not bboxes:
        return image_rgb
    mask = build_mask(image_rgb.shape[:2], bboxes)
    if not mask.any():
        return image_rgb

    from PIL import Image

    lama = _load()
    result = lama(Image.fromarray(image_rgb), Image.fromarray(mask))
    return np.asarray(result.convert("RGB"))


if __name__ == "__main__":  # runnable self-check (Constitution V)
    # Mask building is pure and always testable; inpainting is optional.
    boxes = [{"x": 20, "y": 20, "w": 10, "h": 10}]
    mask = build_mask((100, 100), boxes)
    assert mask[25, 25] == 255  # inside the (expanded) face region
    assert mask[90, 90] == 0    # far away stays untouched
    # Expansion clamps at image bounds (no out-of-range error).
    edge = build_mask((30, 30), [{"x": 25, "y": 25, "w": 20, "h": 20}])
    assert edge.shape == (30, 30)
    # Empty mask short-circuits without needing LaMa.
    assert remove_regions(np.zeros((10, 10, 3), np.uint8), []).shape == (10, 10, 3)

    print(f"removal self-check OK (LaMa {'available' if lama_available() else 'absent'})")
