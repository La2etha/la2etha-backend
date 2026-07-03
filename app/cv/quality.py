"""Quality culling (F3, FR-012): flag unusable shots so the pipeline can DEMOTE
them to the secondary gallery section — never delete them.

Signal: Variance-of-Laplacian (VoL) on the grayscale image. It is a single,
dependency-light measure that catches the two dominant "unusable" cases at a
gathering — motion blur and accidental blank floor/pocket shots — both of which
have low high-frequency detail. Detector confidence is used as a secondary cue.

ponytail: closed-eye ("blink") culling is intentionally NOT wired here. A real
eye-aspect-ratio needs per-eye landmark contours; InsightFace ``buffalo_l`` emits
only 5 keypoints (eye/nose/mouth centers), which cannot measure eye openness.
The upgrade path is a 68-landmark model or a dedicated eye-state classifier — add
it as another signal in ``assess_photo_quality`` when that model is loaded.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Variance-of-Laplacian scales with image resolution: on a large photo a sharp
# edge spans several pixels, diluting the 2nd-derivative variance, so an absolute
# threshold would wrongly flag big, sharp photos as blurry. We compute VoL on a
# size-normalized copy (longest side = this many px) so one threshold is
# comparable across a mixed set of phone cameras. Downscale-only: small images
# (and the synthetic ones in tests) are left untouched.
_BLUR_NORM_LONG_SIDE = 640


@dataclass
class QualityVerdict:
    score: float  # Variance-of-Laplacian on the normalized image (higher = sharper)
    verdict: str  # "ok" | "culled"
    reason: str | None  # "blurry" | "blank_or_floor" | None


def _normalize_for_blur(image: np.ndarray) -> np.ndarray:
    """Downscale so the longest side is ``_BLUR_NORM_LONG_SIDE`` (never upscale)."""
    h, w = image.shape[:2]
    long_side = max(h, w)
    if long_side <= _BLUR_NORM_LONG_SIDE:
        return image
    scale = _BLUR_NORM_LONG_SIDE / long_side
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def variance_of_laplacian(image: np.ndarray, normalize: bool = False) -> float:
    """VoL sharpness. Accepts BGR, RGB, or grayscale; converts to gray.

    ``normalize=True`` resolution-normalizes first so the score is comparable
    across photos of different sizes (used for whole-photo culling).
    """
    if normalize:
        image = _normalize_for_blur(image)
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def assess_photo_quality(
    image_bgr: np.ndarray,
    det_scores: list[float] | None,
    blur_min: float,
) -> QualityVerdict:
    """Judge a whole photo's usability.

    A low-detail image is culled. When it also has no confident face, the reason
    is the more descriptive "blank_or_floor" (accidental shot); otherwise "blurry"
    (a real subject, but too soft to keep in the main gallery).
    """
    score = variance_of_laplacian(image_bgr, normalize=True)
    if score < blur_min:
        has_face = bool(det_scores) and max(det_scores) >= 0.55
        reason = "blurry" if has_face else "blank_or_floor"
        return QualityVerdict(score=score, verdict="culled", reason=reason)
    return QualityVerdict(score=score, verdict="ok", reason=None)


if __name__ == "__main__":  # runnable self-check (Constitution V)
    rng = np.random.default_rng(0)
    # High-frequency noise → high VoL → kept.
    sharp = (rng.integers(0, 255, size=(128, 128, 3))).astype(np.uint8)
    # Flat gray → ~zero VoL → culled as blank/floor.
    flat = np.full((128, 128, 3), 127, dtype=np.uint8)

    kept = assess_photo_quality(sharp, det_scores=[0.9], blur_min=55.0)
    culled = assess_photo_quality(flat, det_scores=None, blur_min=55.0)

    assert kept.verdict == "ok", kept
    assert culled.verdict == "culled" and culled.reason == "blank_or_floor", culled
    # A blurry photo that still has a confident face is reported as "blurry".
    blurry_face = assess_photo_quality(flat, det_scores=[0.9], blur_min=55.0)
    assert blurry_face.reason == "blurry", blurry_face

    # Resolution normalization: a large, detailed image is downscaled before VoL,
    # so it is not spuriously flagged blurry just for being big.
    big = np.repeat(np.repeat(sharp, 8, axis=0), 8, axis=1)  # 1024x1024, same detail
    assert _normalize_for_blur(big).shape[0] == _BLUR_NORM_LONG_SIDE
    assert assess_photo_quality(big, det_scores=[0.9], blur_min=55.0).verdict == "ok"
    print("quality self-check OK")
