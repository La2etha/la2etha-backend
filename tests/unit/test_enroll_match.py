"""US1 — enrollment centroid matches the right cluster; O(M+K) not per-photo."""

import numpy as np

from app.cv.enroll import aggregate_enrollment, match_centroid_to_clusters
from app.cv.vectors import centroid, l2_normalize


def _identity(seed: int, dim: int = 512) -> np.ndarray:
    """A stable unit vector standing in for one person's face embedding."""
    rng = np.random.default_rng(seed)
    return l2_normalize(rng.normal(size=dim).astype(np.float32))


def test_centroid_matches_correct_cluster():
    alice = _identity(1)
    bob = _identity(2)
    carol = _identity(3)

    # Cluster centroids computed once from each person's faces (the M faces).
    clusters = {
        "cluster_alice": centroid(np.stack([alice + 0.02 * _identity(10 + i) for i in range(5)])),
        "cluster_bob": centroid(np.stack([bob + 0.02 * _identity(20 + i) for i in range(5)])),
        "cluster_carol": centroid(np.stack([carol + 0.02 * _identity(30 + i) for i in range(5)])),
    }

    # Alice enrolls with multi-angle samples close to her identity.
    samples = [l2_normalize(alice + 0.03 * _identity(40 + i)) for i in range(4)]
    agg = aggregate_enrollment(samples)

    matches = match_centroid_to_clusters(agg.centroid, agg.per_angle, clusters, threshold=0.35)
    matched_ids = {cid for cid, _ in matches}

    assert "cluster_alice" in matched_ids
    assert "cluster_bob" not in matched_ids
    assert "cluster_carol" not in matched_ids


def test_matching_is_cluster_bound_not_per_photo():
    """Matching compares against K cluster centroids only (O(M+K)), never the
    individual faces. We verify each cluster centroid is inspected exactly once."""
    enroll = _identity(1)
    inspected: list[str] = []

    class Spy(np.ndarray):
        pass

    clusters = {}
    for name in ("a", "b", "c"):
        vec = _identity(hash(name) % 1000).view(Spy)
        clusters[name] = vec

    # Wrap similarity to count comparisons.
    import app.cv.enroll as enroll_mod

    original = enroll_mod.cosine_similarity

    def counting(a, b):
        inspected.append("cmp")
        return original(a, b)

    enroll_mod.cosine_similarity = counting
    try:
        match_centroid_to_clusters(enroll, [], clusters, threshold=0.99)
    finally:
        enroll_mod.cosine_similarity = original

    # One comparison per cluster centroid, no per-angle fallback (empty angles).
    assert len(inspected) == len(clusters)


def test_low_quality_enrollment_flagged():
    """Inconsistent samples (different people) fail the quality gate."""
    samples = [_identity(1), _identity(2), _identity(3)]
    agg = aggregate_enrollment(samples)
    assert agg.quality_ok is False
