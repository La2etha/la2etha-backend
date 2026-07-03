# Lahza — CV Evaluation Report

_Generated 2026-07-04._

> ⚠️ **Synthetic sanity numbers** — generated with `python -m eval.metrics_report` on random embeddings to prove the metric pipeline. Replace with real/public datasets (`eval/datasets.py`) for the final write-up.

## Personal gallery quality (SC-002 / SC-003)

End-to-end recall/precision of each person's gallery vs ground truth.

| Person | Recall | Precision |
|---|---|---|
| alice | 1.000 | 1.000 |
| bob | 1.000 | 1.000 |
| carol | 1.000 | 1.000 |
| **Macro avg** | **1.000** | **1.000** |

## Clustering comparison (HDBSCAN vs baselines)

Homogeneity / completeness / V-measure / ARI / NMI on labeled faces.

| Algorithm | Homog. | Compl. | V | ARI | NMI | k |
|---|---|---|---|---|---|---|
| HDBSCAN | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 4 |
| DBSCAN | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 4 |
| ChineseWhispers | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 4 |

## Enrollment ablation (SC-004)

Gallery recall on hard cases: single-photo vs multi-angle enrollment.

| Enrollment | Recall |
|---|---|
| Single photo | 0.500 |
| Multi-angle | 1.000 |
| **Gain** | **+0.500** |

## Quality culling (SC-008)

Precision/recall of the cull decision (positive class = unusable).

| Metric | Value |
|---|---|
| Precision | 1.000 |
| Recall | 1.000 |
| F1 | 1.000 |
| Culled | 4 / 8 |

## Component benchmarks (public datasets)

Run against downloaded releases via `eval/datasets.py`. Compare recognition verification accuracy to published `buffalo_l` numbers (LFW 99.83%, CFP-FP 98.74%, AgeDB-30 98.28%) as a sanity check, and detection AP on WIDER FACE val.

| Benchmark | Metric | Ours | Published |
|---|---|---|---|
| LFW | Accuracy | _TBD_ | 99.83% |
| CFP-FP | Accuracy | _TBD_ | 98.74% |
| AgeDB-30 | Accuracy | _TBD_ | 98.28% |
| WIDER FACE (val) | AP | _TBD_ | — |
