# Phase 2: Normal-only Anomaly Baselines

Status: completed locally on 2026-10-19. No Kaggle submission was made, and no hidden test label was accessed.

## Method

Each model is trained only on normal training images, separately per category. Images are resized to 64x64 RGB for this CPU-friendly baseline.

- **`knn_tail`** chooses the closest normal reference using luminance MSE, then scores the mean largest 1% of RGB residuals.
- **`template_tail`** compares to a category median normal template using a robust per-pixel MAD scale, then scores its largest 1% residuals.

Three deterministic folds produced two types of proxy samples: held-out normal images (negative) and three explicit synthetic alterations of those images: `cutout`, `scratch`, and `texture_patch` (positive). The threshold maximizes F1 only on this synthetic proxy task. It is not an estimate of hidden-test F1.

## Reproducible proxy benchmark

| Category | Selected proxy method | Proxy F1 | Proxy threshold |
|---|---|---:|---:|
| bottle | knn_tail | 0.9888 | 0.244253 |
| cable | template_tail | 0.8423 | 4.507181 |
| hazelnut | knn_tail | 0.9743 | 0.219480 |
| screw | knn_tail | 0.9472 | 0.271609 |

The baseline therefore selects its method per category rather than forcing a single visual model across objects. Proxy F1 is weaker for cable, which is a warning that this simple pixel-residual approach is less robust there.

## Test scoring output

All 503 test rows were scored in original `test.csv` order. The local numeric candidate contains 85 rows marked `1` and 418 marked `0`. The label mapping has not been confirmed against Kaggle's inaccessible Evaluation tab, so [the candidate file](../artifacts/phase2/submission_candidate_numeric_0_good_1_defective.csv) is **not authorized for submission**.

Visual diagnostic review of the highest-ranked bottle and hazelnut images found strong localized deviations, consistent with the score design. This is a sanity check only; it did not alter predictions or establish ground truth.

## Outputs

- `artifacts/phase2/proxy_validation_scores.csv` — every held-out-normal and synthetic proxy score.
- `artifacts/phase2/proxy_thresholds.csv` — per-category proxy thresholds and F1 values.
- `artifacts/phase2/test_anomaly_scores.csv` — both method scores, selected method, score, and binary candidate decision for each test ID.
- `artifacts/phase2/plots/` — score distributions and highest-score diagnostic montages.

## Next experiment

The most useful next step is an allowed pretrained feature-space method (e.g. a frozen vision encoder with category-wise nearest-neighbour patch scoring), but the Rules page must first confirm that pretrained weights are permitted. It should be benchmarked against this same proxy protocol, with real-label claims kept separate from synthetic-proxy findings.
