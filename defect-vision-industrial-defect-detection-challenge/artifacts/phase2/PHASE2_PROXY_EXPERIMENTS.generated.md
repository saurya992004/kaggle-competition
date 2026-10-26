# Phase 2: Normal-only Anomaly Baselines

## What was run

Two category-specific normal-only baselines were fit exclusively on the 742 training images:

1. **`knn_tail`**: find the closest normal image by low-resolution luminance MSE and average the largest 1% of pixel residuals.
2. **`template_tail`**: form a category-wise RGB median normal template, scale residuals by robust per-pixel MAD, and average the largest 1%.

The benchmark uses three deterministic folds. Each held-out normal is scored both as normal and after three marked **synthetic proxy corruptions** (`cutout`, `scratch`, and `texture_patch`). The threshold maximizes F1 only for those proxy labels. This is useful for a reproducible ordering experiment; it is not a test-set F1 estimate and must not be reported as one.

## Proxy results and candidate counts

| Category | Method | Proxy F1 | Proxy threshold | Test rows predicted defective |
|---|---|---:|---:|---:|
| bottle | knn_tail | 0.9888 | 0.244253 | 108 |
| bottle | template_tail | 0.9876 | 5.038482 | 229 |
| cable | knn_tail | 0.8304 | 0.441958 | 108 |
| cable | template_tail | 0.8423 | 4.507181 | 229 |
| hazelnut | knn_tail | 0.9743 | 0.219480 | 108 |
| hazelnut | template_tail | 0.9414 | 5.693718 | 229 |
| screw | knn_tail | 0.9472 | 0.271609 | 108 |
| screw | template_tail | 0.6792 | 8.710902 | 229 |

## Interpretation and limits

- The test labels were not read, inferred, or reconstructed. Test-score histograms are descriptive only.
- Synthetic corruptions do not represent the unknown real defect distribution, so choosing a final method or threshold solely on proxy F1 risks overfitting to the corruption generator.
- A normal-only holdout measures false-positive behaviour but not defective recall. Real validation labels, allowed external anomaly data, or leaderboard feedback under the confirmed rules are required to calibrate a defensible F1 threshold.
- `submission_candidate_numeric_0_good_1_defective.csv` selects the best proxy method separately for each category and is a local candidate only. It assumes numeric `0=good`, `1=defective`, which remains unverified by the inaccessible Evaluation tab; it must not be submitted until that contract is confirmed.

## Reproduction

```powershell
python -m src.phase2_baseline --data-root . --output-dir artifacts/phase2
python -m unittest discover -s tests -v
```
