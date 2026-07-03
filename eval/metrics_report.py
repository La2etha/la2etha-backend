"""Eval report generator — assembles the graded numbers into ``eval/REPORT.md``.

Renders markdown tables from the metric objects produced by the other eval
modules. ``main()`` runs a synthetic end-to-end pass (so the file is always
reproducible with no downloads) and writes REPORT.md; swap in real datasets via
``eval.datasets`` to produce the numbers that go in the write-up.

Run: ``python -m eval.metrics_report`` → writes eval/REPORT.md.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from eval.ablation_enrollment import AblationResult
from eval.metrics_culling import CullingReport
from eval.metrics_gallery import GalleryReport

REPORT_PATH = Path(__file__).resolve().parent / "REPORT.md"


def render_gallery(report: GalleryReport) -> str:
    rows = "\n".join(
        f"| {p} | {r:.3f} | {pr:.3f} |" for p, r, pr in report.as_rows()
    )
    return (
        "## Personal gallery quality (SC-002 / SC-003)\n\n"
        "End-to-end recall/precision of each person's gallery vs ground truth.\n\n"
        "| Person | Recall | Precision |\n|---|---|---|\n"
        f"{rows}\n"
        f"| **Macro avg** | **{report.macro_recall:.3f}** | "
        f"**{report.macro_precision:.3f}** |\n"
    )


def render_clustering(results: dict[str, dict[str, float]]) -> str:
    header = (
        "## Clustering comparison (HDBSCAN vs baselines)\n\n"
        "Homogeneity / completeness / V-measure / ARI / NMI on labeled faces.\n\n"
        "| Algorithm | Homog. | Compl. | V | ARI | NMI | k |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    rows = "\n".join(
        f"| {a} | {m['homogeneity']:.3f} | {m['completeness']:.3f} | {m['v_measure']:.3f} "
        f"| {m['ari']:.3f} | {m['nmi']:.3f} | {m['n_clusters']} |"
        for a, m in results.items()
    )
    return header + rows + "\n"


def render_ablation(result: AblationResult) -> str:
    return (
        "## Enrollment ablation (SC-004)\n\n"
        "Gallery recall on hard cases: single-photo vs multi-angle enrollment.\n\n"
        "| Enrollment | Recall |\n|---|---|\n"
        f"| Single photo | {result.single_recall:.3f} |\n"
        f"| Multi-angle | {result.multi_recall:.3f} |\n"
        f"| **Gain** | **+{result.gain:.3f}** |\n"
    )


def render_culling(report: CullingReport) -> str:
    return (
        "## Quality culling (SC-008)\n\n"
        "Precision/recall of the cull decision (positive class = unusable).\n\n"
        "| Metric | Value |\n|---|---|\n"
        f"| Precision | {report.precision:.3f} |\n"
        f"| Recall | {report.recall:.3f} |\n"
        f"| F1 | {report.f1:.3f} |\n"
        f"| Culled | {report.culled} / {report.total} |\n"
    )


def render_component_placeholder() -> str:
    return (
        "## Component benchmarks (public datasets)\n\n"
        "Run against downloaded releases via `eval/datasets.py`. Compare recognition "
        "verification accuracy to published `buffalo_l` numbers (LFW 99.83%, "
        "CFP-FP 98.74%, AgeDB-30 98.28%) as a sanity check, and detection AP on "
        "WIDER FACE val.\n\n"
        "| Benchmark | Metric | Ours | Published |\n|---|---|---|---|\n"
        "| LFW | Accuracy | _TBD_ | 99.83% |\n"
        "| CFP-FP | Accuracy | _TBD_ | 98.74% |\n"
        "| AgeDB-30 | Accuracy | _TBD_ | 98.28% |\n"
        "| WIDER FACE (val) | AP | _TBD_ | — |\n"
    )


def build_report(
    gallery: GalleryReport,
    clustering: dict[str, dict[str, float]],
    ablation: AblationResult,
    culling: CullingReport,
    synthetic: bool,
) -> str:
    note = (
        "> ⚠️ **Synthetic sanity numbers** — generated with `python -m eval.metrics_report` "
        "on random embeddings to prove the metric pipeline. Replace with real/public "
        "datasets (`eval/datasets.py`) for the final write-up.\n\n"
        if synthetic
        else ""
    )
    return (
        f"# Lahza — CV Evaluation Report\n\n"
        f"_Generated {date.today().isoformat()}._\n\n"
        f"{note}"
        f"{render_gallery(gallery)}\n"
        f"{render_clustering(clustering)}\n"
        f"{render_ablation(ablation)}\n"
        f"{render_culling(culling)}\n"
        f"{render_component_placeholder()}"
    )


def _synthetic_run() -> str:
    """Drive all four experiments on synthetic data and render the report."""
    import cv2
    import numpy as np

    from app.cv.vectors import l2_normalize
    from eval.ablation_enrollment import run_ablation
    from eval.metrics_clustering import compare_clustering
    from eval.metrics_culling import evaluate_culling
    from eval.metrics_gallery import (
        Face,
        build_predicted_galleries,
        evaluate_galleries,
        ground_truth_from_faces,
    )

    rng = np.random.default_rng(7)

    def near(v, jitter=0.05):
        return l2_normalize(v + jitter * rng.normal(size=v.shape).astype(np.float32))

    identities = {
        p: l2_normalize(np.random.default_rng(i).normal(size=512).astype(np.float32))
        for i, p in enumerate(["alice", "bob", "carol"], start=1)
    }
    faces = [
        Face(f"{p}_{k}.jpg", p, near(v)) for p, v in identities.items() for k in range(4)
    ]
    enroll = {p: [near(v) for _ in range(3)] for p, v in identities.items()}
    gallery = evaluate_galleries(
        build_predicted_galleries(faces, enroll, threshold=0.35),
        ground_truth_from_faces(faces),
    )

    blocks, truth = [], []
    for label in range(4):
        base = l2_normalize(np.random.default_rng(label + 1).normal(size=64).astype(np.float32))
        for _ in range(12):
            blocks.append(l2_normalize(base + 0.05 * rng.normal(size=64).astype(np.float32)))
            truth.append(label)
    clustering = compare_clustering(np.stack(blocks), np.array(truth), cw_threshold=0.6)

    frontal = l2_normalize(rng.normal(size=128).astype(np.float32))
    profile = l2_normalize(rng.normal(size=128).astype(np.float32))
    ablation = run_ablation(
        {0: frontal, 1: profile}, {0, 1}, frontal,
        [frontal, profile, l2_normalize((frontal + profile) / 2)], threshold=0.5,
    )

    def sharp():
        return rng.integers(0, 255, size=(200, 200, 3), dtype=np.uint8)

    samples = (
        [(sharp(), True) for _ in range(4)]
        + [(cv2.GaussianBlur(sharp(), (0, 0), 9), False) for _ in range(2)]
        + [(np.full((200, 200, 3), 120, np.uint8), False) for _ in range(2)]
    )
    det = [[0.9]] * 6 + [None, None]
    culling = evaluate_culling(samples, blur_min=55.0, det_scores_per_image=det)

    return build_report(gallery, clustering, ablation, culling, synthetic=True)


def main() -> None:
    body = _synthetic_run()
    REPORT_PATH.write_text(body, encoding="utf-8")
    print(f"wrote {REPORT_PATH}")


if __name__ == "__main__":  # runnable self-check (Constitution V)
    body = _synthetic_run()
    for heading in (
        "# Lahza — CV Evaluation Report",
        "## Personal gallery quality",
        "## Clustering comparison",
        "## Enrollment ablation",
        "## Quality culling",
        "## Component benchmarks",
    ):
        assert heading in body, f"missing section: {heading}"
    main()  # write the artifact
    assert REPORT_PATH.exists()
    print("report generator self-check OK")
