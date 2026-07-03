"""Proximity / background-subject filtering (F4, FR-013).

Decides whether a detected face is an incidental background bystander rather than
an actual subject, so the pipeline can DEMOTE such photos to the person's
secondary gallery section (never hide them). The cue is how prominent the face is
in the frame: a subject's face is large and in focus; a passing stranger's is
small and usually soft.

Primary signal: ``face_area_ratio`` (bbox area / image area). Secondary
corroboration: face-crop sharpness (VoL) — a mid-sized but clearly out-of-focus
face is background depth-of-field. EXIF/estimated depth is an optional future cue
(feature-flagged Depth-Anything module, T048).
"""

from __future__ import annotations

import cv2
import numpy as np


def face_crop_sharpness(image_bgr: np.ndarray, bbox: dict) -> float:
    """Variance-of-Laplacian of the face region. 0.0 if the crop is empty."""
    h, w = image_bgr.shape[:2]
    x = max(0, int(bbox.get("x", 0)))
    y = max(0, int(bbox.get("y", 0)))
    x2 = min(w, int(bbox.get("x", 0) + bbox.get("w", 0)))
    y2 = min(h, int(bbox.get("y", 0) + bbox.get("h", 0)))
    if x2 <= x or y2 <= y:
        return 0.0
    crop = image_bgr[y:y2, x:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def is_background_face(
    face_area_ratio: float | None,
    face_sharpness: float | None,
    area_min: float,
    sharpness_min: float,
) -> bool:
    """True when a face is an incidental background presence, not a subject.

    - Tiny face (area below ``area_min``) → background (a distant bystander).
    - Small-ish face (below 2×``area_min``) that is also clearly out of focus
      (sharpness below ``sharpness_min``) → background depth-of-field.
    """
    if face_area_ratio is None:
        return False
    if face_area_ratio < area_min:
        return True
    if (
        face_sharpness is not None
        and face_sharpness < sharpness_min
        and face_area_ratio < 2 * area_min
    ):
        return True
    return False


if __name__ == "__main__":  # runnable self-check (Constitution V)
    AREA_MIN, SHARP_MIN = 0.012, 30.0
    # Big, sharp face → subject.
    assert not is_background_face(0.08, 500.0, AREA_MIN, SHARP_MIN)
    # Tiny face → background regardless of sharpness.
    assert is_background_face(0.004, 800.0, AREA_MIN, SHARP_MIN)
    # Small-ish AND soft → background depth-of-field.
    assert is_background_face(0.018, 10.0, AREA_MIN, SHARP_MIN)
    # Small-ish but sharp → kept as subject.
    assert not is_background_face(0.018, 400.0, AREA_MIN, SHARP_MIN)
    # Missing area (older rows) → conservatively a subject.
    assert not is_background_face(None, None, AREA_MIN, SHARP_MIN)

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    assert face_crop_sharpness(img, {"x": 10, "y": 10, "w": 20, "h": 20}) == 0.0
    assert face_crop_sharpness(img, {"x": 0, "y": 0, "w": 0, "h": 0}) == 0.0
    print("proximity self-check OK")
