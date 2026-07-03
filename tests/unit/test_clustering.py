"""US1 — synthetic embeddings form the correct clusters + centroids (HDBSCAN)."""

import numpy as np

from app.cv.cluster import cluster_embeddings
from app.cv.vectors import cosine_similarity, l2_normalize


def _person(seed: int, dim: int = 512) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return l2_normalize(rng.normal(size=dim).astype(np.float32))


def _samples_around(identity: np.ndarray, n: int, jitter: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        noise = rng.normal(size=identity.shape).astype(np.float32)
        out.append(l2_normalize(identity + jitter * noise))
    return np.stack(out)


def test_three_people_form_three_clusters():
    people = [_person(1), _person(2), _person(3)]
    blocks = [_samples_around(p, n=10, jitter=0.05, seed=100 + i) for i, p in enumerate(people)]
    embeddings = np.vstack(blocks)

    result = cluster_embeddings(embeddings, min_cluster_size=3)

    # Exactly three real clusters discovered.
    real_labels = {lbl for lbl in result.labels if lbl != -1}
    assert len(real_labels) == 3
    assert len(result.centroids) == 3

    # Each discovered centroid aligns with exactly one true identity.
    for centroid_vec in result.centroids.values():
        sims = sorted(cosine_similarity(centroid_vec, p) for p in people)
        assert sims[-1] > 0.8  # strong match to its own identity
        assert sims[-2] < 0.5  # clearly separated from the others


def test_empty_input_is_safe():
    result = cluster_embeddings(np.empty((0, 512), dtype=np.float32))
    assert result.labels.shape == (0,)
    assert result.centroids == {}
