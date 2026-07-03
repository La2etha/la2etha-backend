"""Clustering-quality comparison (the gradeable clustering experiment).

Compares our primary clusterer (HDBSCAN, cosine) against two baselines named in
the research plan — DBSCAN and Chinese Whispers — on labeled face embeddings,
reporting homogeneity, completeness, V-measure, ARI, and NMI (sklearn).

Chinese Whispers is implemented here in ~30 lines of numpy rather than pulled in
via dlib: dlib needs a C++ toolchain, and the algorithm is a simple graph
label-propagation. Keeping it dependency-free means this comparison always runs
for the report.

Run: ``python -m eval.metrics_clustering``.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
    v_measure_score,
)
from sklearn.metrics.pairwise import cosine_similarity as cosine_sim_matrix

from app.cv.cluster import cluster_embeddings


def chinese_whispers(
    embeddings: np.ndarray,
    threshold: float = 0.5,
    iterations: int = 20,
    seed: int = 0,
) -> np.ndarray:
    """Graph label-propagation clustering (Fergus/Biemann "Chinese Whispers").

    Nodes = faces; edges connect faces with cosine similarity >= ``threshold``,
    weighted by that similarity. Each node iteratively adopts the highest
    weighted-sum label among its neighbors. Returns per-face integer labels.
    """
    n = embeddings.shape[0]
    if n == 0:
        return np.empty((0,), dtype=int)
    sim = cosine_sim_matrix(embeddings)
    np.fill_diagonal(sim, 0.0)
    adj = np.where(sim >= threshold, sim, 0.0)

    labels = np.arange(n)
    rng = np.random.default_rng(seed)
    for _ in range(iterations):
        changed = False
        for node in rng.permutation(n):
            weights = adj[node]
            if not weights.any():
                continue
            # Sum edge weights per neighbor label; adopt the strongest.
            scores: dict[int, float] = {}
            for nb in np.nonzero(weights)[0]:
                scores[labels[nb]] = scores.get(labels[nb], 0.0) + weights[nb]
            best = max(scores, key=scores.get)
            if labels[node] != best:
                labels[node] = best
                changed = True
        if not changed:
            break
    return labels


def _score(true_labels: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "homogeneity": float(homogeneity_score(true_labels, pred)),
        "completeness": float(completeness_score(true_labels, pred)),
        "v_measure": float(v_measure_score(true_labels, pred)),
        "ari": float(adjusted_rand_score(true_labels, pred)),
        "nmi": float(normalized_mutual_info_score(true_labels, pred)),
        "n_clusters": int(len({int(x) for x in pred if int(x) != -1})),
    }


def compare_clustering(
    embeddings: np.ndarray,
    true_labels: np.ndarray,
    min_cluster_size: int = 3,
    dbscan_eps: float = 0.35,
    cw_threshold: float = 0.5,
) -> dict[str, dict[str, float]]:
    """Run all three clusterers and score each against the true labels."""
    embeddings = np.asarray(embeddings, dtype=np.float64)

    hdb = cluster_embeddings(embeddings, min_cluster_size=min_cluster_size).labels
    # DBSCAN on cosine distance (embeddings are L2-normalized).
    dbscan = DBSCAN(eps=dbscan_eps, min_samples=min_cluster_size, metric="cosine").fit_predict(
        embeddings
    )
    cw = chinese_whispers(embeddings, threshold=cw_threshold)

    return {
        "HDBSCAN": _score(true_labels, hdb),
        "DBSCAN": _score(true_labels, dbscan),
        "ChineseWhispers": _score(true_labels, cw),
    }


if __name__ == "__main__":  # runnable self-check (Constitution V)
    from app.cv.vectors import l2_normalize

    rng = np.random.default_rng(3)
    blocks, truth = [], []
    for label in range(4):
        base = l2_normalize(np.random.default_rng(label + 1).normal(size=64).astype(np.float32))
        for _ in range(12):
            blocks.append(l2_normalize(base + 0.05 * rng.normal(size=64).astype(np.float32)))
            truth.append(label)
    X = np.stack(blocks)
    y = np.array(truth)

    results = compare_clustering(X, y, cw_threshold=0.6)
    for algo, m in results.items():
        print(f"{algo:16s} V={m['v_measure']:.3f} ARI={m['ari']:.3f} "
              f"NMI={m['nmi']:.3f} k={m['n_clusters']}")
    # Every method should recover the well-separated clusters strongly.
    assert results["HDBSCAN"]["homogeneity"] > 0.9
    assert results["ChineseWhispers"]["v_measure"] > 0.8
    print("clustering comparison self-check OK")
