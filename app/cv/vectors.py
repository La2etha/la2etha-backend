"""Small vector helpers shared across the CV pipeline (numpy only)."""

from __future__ import annotations

import numpy as np


def l2_normalize(vec: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32)
    norm = np.linalg.norm(vec)
    return vec / (norm + eps)


def centroid(vectors: np.ndarray) -> np.ndarray:
    """Mean of row vectors, L2-renormalized (a unit-length prototype)."""
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim == 1:
        return l2_normalize(arr)
    return l2_normalize(arr.mean(axis=0))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity; exact dot product when both inputs are unit-length."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)
