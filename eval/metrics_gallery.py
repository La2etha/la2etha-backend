"""Primary metric — personal gallery recall/precision (SC-002/003).

End-to-end quality of the whole loop: cluster the event's faces once, match each
person's enrollment centroid against the cluster centroids, materialize their
predicted gallery, and score it against ground truth (who is truly in each photo).

This deliberately calls the SAME production code paths as the app
(``cluster_embeddings`` + ``match_centroid_to_clusters``), so the reported numbers
reflect the shipped system, not a re-implementation.

Run: ``python -m eval.metrics_gallery`` (self-check on synthetic embeddings).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.cv.cluster import cluster_embeddings
from app.cv.enroll import aggregate_enrollment, match_centroid_to_clusters
from eval.harness import GroundTruth, recall_precision


@dataclass
class Face:
    photo_id: str
    person: str  # true identity label (ground truth only; clustering never sees it)
    embedding: np.ndarray  # L2-normalized 512-d


@dataclass
class GalleryReport:
    # person -> (recall, precision)
    per_user: dict[str, tuple[float, float]] = field(default_factory=dict)
    macro_recall: float = 0.0
    macro_precision: float = 0.0

    def as_rows(self) -> list[tuple[str, float, float]]:
        return [(p, r, pr) for p, (r, pr) in sorted(self.per_user.items())]


def ground_truth_from_faces(faces: list[Face]) -> GroundTruth:
    photo_people: dict[str, set[str]] = {}
    for f in faces:
        photo_people.setdefault(f.photo_id, set()).add(f.person)
    return GroundTruth(
        photo_people=photo_people, people={f.person for f in faces}
    )


def build_predicted_galleries(
    faces: list[Face],
    enroll_samples: dict[str, list[np.ndarray]],
    threshold: float,
    min_cluster_size: int = 3,
) -> dict[str, set[str]]:
    """Cluster faces once, then match each person's enrollment to cluster centroids
    and collect the photos of their matched clusters — the production algorithm."""
    embeddings = np.stack([f.embedding for f in faces])
    result = cluster_embeddings(embeddings, min_cluster_size=min_cluster_size)

    predicted: dict[str, set[str]] = {}
    for person, samples in enroll_samples.items():
        agg = aggregate_enrollment(samples)
        matches = match_centroid_to_clusters(
            enroll_centroid=agg.centroid,
            per_angle=agg.per_angle,
            cluster_centroids=result.centroids,
            threshold=threshold,
        )
        matched_labels = {cid for cid, _ in matches}
        gallery = {
            faces[i].photo_id
            for i, label in enumerate(result.labels)
            if int(label) in matched_labels
        }
        predicted[person] = gallery
    return predicted


def evaluate_galleries(predicted: dict[str, set[str]], gt: GroundTruth) -> GalleryReport:
    report = GalleryReport()
    recalls, precisions = [], []
    for person in sorted(gt.people):
        r, p = recall_precision(predicted.get(person, set()), gt.photos_for(person))
        report.per_user[person] = (r, p)
        recalls.append(r)
        precisions.append(p)
    if recalls:
        report.macro_recall = float(np.mean(recalls))
        report.macro_precision = float(np.mean(precisions))
    return report


if __name__ == "__main__":  # runnable self-check (Constitution V)
    from app.cv.vectors import l2_normalize

    rng = np.random.default_rng(7)

    def person_vec(seed: int) -> np.ndarray:
        return l2_normalize(np.random.default_rng(seed).normal(size=512).astype(np.float32))

    def near(v: np.ndarray, jitter: float = 0.05) -> np.ndarray:
        return l2_normalize(v + jitter * rng.normal(size=v.shape).astype(np.float32))

    identities = {p: person_vec(i) for i, p in enumerate(["alice", "bob", "carol"], start=1)}

    # Each person appears in 4 photos (enough for a cluster); build the faces.
    faces: list[Face] = []
    for person, vec in identities.items():
        for k in range(4):
            faces.append(Face(photo_id=f"{person}_{k}.jpg", person=person, embedding=near(vec)))

    enroll = {p: [near(v) for _ in range(3)] for p, v in identities.items()}
    gt = ground_truth_from_faces(faces)

    predicted = build_predicted_galleries(faces, enroll, threshold=0.35)
    report = evaluate_galleries(predicted, gt)

    assert report.macro_recall > 0.9, report
    assert report.macro_precision > 0.9, report
    print(f"gallery metrics self-check OK — recall={report.macro_recall:.3f} "
          f"precision={report.macro_precision:.3f}")
