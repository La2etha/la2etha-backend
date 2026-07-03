"""Quality-culling precision/recall on a seeded set (SC-008).

Treats "unusable" as the positive class: precision = of the photos we culled, how
many were truly bad; recall = of the truly bad photos, how many we culled. Uses
the production ``assess_photo_quality`` so the numbers match the running pipeline.

``evaluate_culling`` scores from pre-computed (image, is_usable) pairs; feed it
real seeded images in the report, or the synthetic pair below for the self-check.

Run: ``python -m eval.metrics_culling``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.cv.quality import assess_photo_quality


@dataclass
class CullingReport:
    precision: float
    recall: float
    f1: float
    culled: int
    total: int


def evaluate_culling(
    samples: list[tuple[np.ndarray, bool]],
    blur_min: float,
    det_scores_per_image: list[list[float] | None] | None = None,
) -> CullingReport:
    """samples: list of (image_bgr, is_usable). Positive class = unusable."""
    tp = fp = fn = 0
    culled = 0
    for i, (image, is_usable) in enumerate(samples):
        det = det_scores_per_image[i] if det_scores_per_image else None
        verdict = assess_photo_quality(image, det, blur_min)
        is_culled = verdict.verdict == "culled"
        culled += int(is_culled)
        truly_bad = not is_usable
        if is_culled and truly_bad:
            tp += 1
        elif is_culled and not truly_bad:
            fp += 1
        elif not is_culled and truly_bad:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return CullingReport(precision, recall, f1, culled, len(samples))


if __name__ == "__main__":  # runnable self-check (Constitution V)
    import cv2

    rng = np.random.default_rng(0)

    def sharp() -> np.ndarray:
        return rng.integers(0, 255, size=(200, 200, 3), dtype=np.uint8)

    def blurred() -> np.ndarray:
        return cv2.GaussianBlur(sharp(), (0, 0), sigmaX=9)

    def flat() -> np.ndarray:
        return np.full((200, 200, 3), 120, dtype=np.uint8)

    # 4 good (usable=True) + 4 bad (usable=False).
    samples: list[tuple[np.ndarray, bool]] = (
        [(sharp(), True) for _ in range(4)]
        + [(blurred(), False) for _ in range(2)]
        + [(flat(), False) for _ in range(2)]
    )
    det = [[0.9]] * 6 + [None, None]

    report = evaluate_culling(samples, blur_min=55.0, det_scores_per_image=det)
    print(f"culling precision={report.precision:.2f} recall={report.recall:.2f} "
          f"f1={report.f1:.2f} ({report.culled}/{report.total} culled)")
    assert report.precision >= 0.9 and report.recall >= 0.9, report
    print("culling metrics self-check OK")
