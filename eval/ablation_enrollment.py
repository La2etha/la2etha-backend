"""Enrollment ablation (SC-004): single-photo vs multi-angle recall.

The product claim (FR-011) is that enrolling several angles recovers photos that
a single frontal photo misses — profile shots, harsh lighting, accessories. This
measures that gain by running the SAME matcher (``match_centroid_to_clusters``)
with a one-sample enrollment vs a multi-angle enrollment against the same event
clusters, and comparing gallery recall.

Run: ``python -m eval.ablation_enrollment``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.cv.enroll import aggregate_enrollment, match_centroid_to_clusters


@dataclass
class AblationResult:
    single_recall: float
    multi_recall: float

    @property
    def gain(self) -> float:
        return self.multi_recall - self.single_recall


def _recall(matched_labels: set[int], truth_labels: set[int]) -> float:
    if not truth_labels:
        return 1.0
    return len(matched_labels & truth_labels) / len(truth_labels)


def run_ablation(
    cluster_centroids: dict[int, np.ndarray],
    truth_labels: set[int],
    single_sample: np.ndarray,
    multi_samples: list[np.ndarray],
    threshold: float,
) -> AblationResult:
    """Compare recall of the person's true clusters under single vs multi enroll.

    ``truth_labels`` are the cluster labels that genuinely belong to the person.
    """
    single_agg = aggregate_enrollment([single_sample], min_samples=1)
    single = {
        cid
        for cid, _ in match_centroid_to_clusters(
            single_agg.centroid, single_agg.per_angle, cluster_centroids, threshold
        )
    }

    multi_agg = aggregate_enrollment(multi_samples)
    multi = {
        cid
        for cid, _ in match_centroid_to_clusters(
            multi_agg.centroid, multi_agg.per_angle, cluster_centroids, threshold
        )
    }

    return AblationResult(
        single_recall=_recall(single, truth_labels),
        multi_recall=_recall(multi, truth_labels),
    )


if __name__ == "__main__":  # runnable self-check (Constitution V)
    from app.cv.vectors import l2_normalize

    # A person has a frontal look F and a profile look P that are only weakly
    # similar. The event has two clusters of THIS person: frontal shots and
    # profile shots. Single (frontal-only) enrollment should miss the profile
    # cluster; multi-angle enrollment should catch it via the per-angle boost.
    rng = np.random.default_rng(0)
    dim = 128

    frontal = l2_normalize(rng.normal(size=dim).astype(np.float32))
    # Profile: a different direction (low cosine to frontal).
    profile = l2_normalize(rng.normal(size=dim).astype(np.float32))

    clusters = {0: frontal, 1: profile}  # both are the same person's clusters
    truth = {0, 1}

    result = run_ablation(
        cluster_centroids=clusters,
        truth_labels=truth,
        single_sample=frontal,
        multi_samples=[frontal, profile, l2_normalize((frontal + profile) / 2)],
        threshold=0.5,
    )

    print(f"single recall={result.single_recall:.2f}  multi recall={result.multi_recall:.2f}  "
          f"gain=+{result.gain:.2f}")
    assert result.single_recall < 1.0, "single-photo should miss the profile cluster"
    assert result.multi_recall > result.single_recall, "multi-angle must improve recall"
    print("enrollment ablation self-check OK")
