"""Anonymous face clustering — the cluster-once engine (HDBSCAN, cosine).

Faces are clustered per event over all their embeddings; each cluster yields a
centroid. Enrollment later matches an identity centroid against these K cluster
centroids (O(M+K)), never against every face (Constitution III).

Embeddings are L2-normalized, so we cluster on cosine distance. At event scale
(single-digit-thousands of vectors) a precomputed cosine distance matrix is cheap
and exact. ponytail: for far larger sets, switch to normalized-Euclidean HDBSCAN
(monotonic with cosine) to avoid the O(N^2) matrix.
"""

from __future__ import annotations

from dataclasses import dataclass

import hdbscan
import numpy as np
from sklearn.metrics.pairwise import cosine_distances

from app.cv.vectors import centroid


@dataclass
class ClusterResult:
    labels: np.ndarray  # per-face cluster label; -1 = noise
    centroids: dict[int, np.ndarray]  # label -> unit-length centroid


def cluster_embeddings(
    embeddings: np.ndarray,
    min_cluster_size: int = 3,
    min_samples: int | None = None,
) -> ClusterResult:
    """Cluster (N, 512) L2-normalized embeddings; return labels + centroids."""
    arr = np.asarray(embeddings, dtype=np.float64)
    if arr.shape[0] == 0:
        return ClusterResult(labels=np.empty((0,), dtype=int), centroids={})
    if arr.shape[0] < min_cluster_size:
        # Too few faces to form a cluster — all treated as noise for now.
        return ClusterResult(labels=np.full(arr.shape[0], -1, dtype=int), centroids={})

    distance = cosine_distances(arr).astype(np.float64)
    clusterer = hdbscan.HDBSCAN(
        metric="precomputed",
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
    )
    labels = clusterer.fit_predict(distance)

    centroids: dict[int, np.ndarray] = {}
    for label in sorted(set(int(x) for x in labels)):
        if label == -1:
            continue
        members = arr[labels == label]
        centroids[label] = centroid(members)
    return ClusterResult(labels=labels, centroids=centroids)
