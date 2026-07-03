"""US2 (F3) — quality culling flags blurry/blank shots and keeps good ones.

VoL is deterministic on synthetic images, so we assert the verdict directly
without needing InsightFace or a GPU (the graded metric must be reproducible).
"""

import cv2
import numpy as np

from app.cv.quality import assess_photo_quality, variance_of_laplacian

BLUR_MIN = 55.0


def _detailed_image(seed: int = 0) -> np.ndarray:
    """High-frequency random image → high VoL (a sharp, usable photo)."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(160, 160, 3), dtype=np.uint8)


def _blurred(image: np.ndarray) -> np.ndarray:
    """Heavy Gaussian blur → collapses high-frequency detail (motion blur)."""
    return cv2.GaussianBlur(image, (0, 0), sigmaX=9)


def test_sharp_photo_kept():
    verdict = assess_photo_quality(_detailed_image(), det_scores=[0.9], blur_min=BLUR_MIN)
    assert verdict.verdict == "ok"
    assert verdict.reason is None


def test_blurred_photo_with_face_flagged_blurry():
    verdict = assess_photo_quality(_blurred(_detailed_image()), det_scores=[0.9], blur_min=BLUR_MIN)
    assert verdict.verdict == "culled"
    assert verdict.reason == "blurry"


def test_blank_floor_shot_flagged():
    flat = np.full((160, 160, 3), 120, dtype=np.uint8)
    verdict = assess_photo_quality(flat, det_scores=None, blur_min=BLUR_MIN)
    assert verdict.verdict == "culled"
    assert verdict.reason == "blank_or_floor"


def test_blur_reduces_sharpness_score():
    sharp = _detailed_image()
    assert variance_of_laplacian(sharp) > variance_of_laplacian(_blurred(sharp))


def test_large_sharp_photo_not_flagged_blurry():
    # A big photo with real detail must survive culling — the resolution bias that
    # made every large photo read as "blurry" is removed by normalization.
    sharp = _detailed_image()
    big = np.repeat(np.repeat(sharp, 8, axis=0), 8, axis=1)  # 1280x1280
    assert big.shape[0] >= 1000
    verdict = assess_photo_quality(big, det_scores=[0.9], blur_min=BLUR_MIN)
    assert verdict.verdict == "ok", verdict
