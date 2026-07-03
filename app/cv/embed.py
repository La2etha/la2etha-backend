"""Face embedding normalization (ArcFace-R50, 512-d, L2-normalized cosine space)."""

from __future__ import annotations

import numpy as np

from app.cv.detect import FaceDetection
from app.cv.vectors import l2_normalize


def normalized_embedding(detection: FaceDetection) -> np.ndarray:
    """Return the detection's ArcFace embedding as a unit-length 512-vector."""
    return l2_normalize(detection.embedding)


def embed_detections(detections: list[FaceDetection]) -> np.ndarray:
    """Stack L2-normalized embeddings for a list of detections → (N, 512)."""
    if not detections:
        return np.empty((0, 512), dtype=np.float32)
    return np.stack([normalized_embedding(d) for d in detections])
