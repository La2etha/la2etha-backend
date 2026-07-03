"""Multi-angle enrollment (F2): aggregate identity centroid + cluster matching.

Enrollment builds one identity centroid from 3–5 angles. Matching compares that
centroid against the event's K cluster centroids (O(M+K)); a per-angle embedding
can also clear the threshold as a recall boost for hard cases.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.cv.vectors import centroid, cosine_similarity


@dataclass
class EnrollmentAggregate:
    centroid: np.ndarray  # unit-length identity prototype
    per_angle: list[np.ndarray]  # unit-length per-sample embeddings
    sample_count: int
    quality_ok: bool


# Minimum mean pairwise cosine among samples for a "consistent" enrollment.
_MIN_CONSISTENCY = 0.30


def aggregate_enrollment(
    per_angle_embeddings: list[np.ndarray], min_samples: int = 3
) -> EnrollmentAggregate:
    """Aggregate per-angle embeddings into a centroid + quality verdict."""
    samples = [np.asarray(e, dtype=np.float32) for e in per_angle_embeddings]
    count = len(samples)
    if count == 0:
        raise ValueError("Enrollment requires at least one face sample")

    stacked = np.stack(samples)
    center = centroid(stacked)

    # Quality: enough samples AND the samples agree with each other.
    consistency = _mean_pairwise_similarity(stacked)
    quality_ok = count >= min_samples and consistency >= _MIN_CONSISTENCY

    return EnrollmentAggregate(
        centroid=center,
        per_angle=samples,
        sample_count=count,
        quality_ok=quality_ok,
    )


def _mean_pairwise_similarity(vectors: np.ndarray) -> float:
    n = vectors.shape[0]
    if n < 2:
        return 1.0
    sims = vectors @ vectors.T
    off_diagonal = sims[~np.eye(n, dtype=bool)]
    return float(off_diagonal.mean())


def match_centroid_to_clusters(
    enroll_centroid: np.ndarray,
    per_angle: list[np.ndarray],
    cluster_centroids: dict,
    threshold: float,
) -> list[tuple]:
    """Return [(cluster_id, similarity)] for clusters this identity matches.

    Compares the identity centroid to each cluster centroid; falls back to the
    best per-angle similarity as a recall boost. This touches only the K cluster
    centroids (plus A angles) — never the M individual faces.
    """
    matches: list[tuple] = []
    for cluster_id, cluster_centroid in cluster_centroids.items():
        sim = cosine_similarity(enroll_centroid, cluster_centroid)
        if sim < threshold and per_angle:
            best_angle = max(cosine_similarity(a, cluster_centroid) for a in per_angle)
            sim = max(sim, best_angle)
        if sim >= threshold:
            matches.append((cluster_id, sim))
    return matches
