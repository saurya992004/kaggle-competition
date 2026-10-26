# Phase 3 Model Comparison and Candidate Selection

## Completed portfolio

Four local normal-only methods were executed in three leakage-safe category-specific folds: pixel nearest-neighbour residual, global shrinkage-Mahalanobis density, patch-statistic memory, and low-rank PCA reconstruction. Frozen pretrained CNN/ViT and pretrained PatchCore were not executed: no weights were cached locally, no compatible package was installed, and Kaggle's pretrained-weight policy remains unresolved. No external model was downloaded.

## Selection evidence

| Category | Selected method | Synthetic proxy AUROC | Synthetic proxy AP | Synthetic proxy F1 | 95th-percentile OOF normal threshold |
|---|---|---:|---:|---:|---:|
| bottle | pixel_knn | 0.9974 | 0.9993 | 0.9913 | 0.191436 |
| cable | patch_memory | 0.9395 | 0.9782 | 0.9374 | 1.540608 |
| hazelnut | pixel_knn | 0.9938 | 0.9981 | 0.9773 | 0.213427 |
| screw | pixel_knn | 0.9683 | 0.9902 | 0.9487 | 0.321100 |

The selected threshold is the 95th percentile of out-of-fold normal scores, a nominal 5% normal false-positive operating point. It is deliberately not the synthetic-F1-optimal threshold and was not calibrated from the test score distribution. Its sampling uncertainty is stored in `experiment_results.csv`.

## Candidate and caveats

`candidate_test_scores.csv` stores continuous scores plus a local binary decision at the normal-calibrated threshold. It is **not a submission** and does not establish real-defect F1. The final test-label encoding and metric must be confirmed before submission.

The two important weaknesses are (1) synthetic anomalies need not match real defects and (2) handcrafted local features lack the semantic invariance a permitted pretrained patch encoder could provide. Pretrained experiments are the justified next escalation only after the competition Rules page confirms their allowance.

## Compact robustness ablation

This fixed, held-out normal split changes resolution only (64 versus 96). Brightness and Gaussian-noise perturbations are applied only at 64 because the main 5%-normal threshold is defined at that resolution. Values shown are the fraction of benignly perturbed held-out normals exceeding the original 64px threshold; they are robustness diagnostics, not defect labels.

| Category | Method | Resolution | Synthetic proxy AUROC | Synthetic proxy AP | Brightness above 64px threshold | Noise above 64px threshold |
|---|---|---:|---:|---:|---:|---:|
| bottle | pixel_knn | 64 | 1.0000 | 1.0000 | 0.17777777777777778 | 0.1111111111111111 |
| cable | patch_memory | 64 | 0.9360 | 0.9775 | 0.0 | 0.08163265306122448 |
| hazelnut | pixel_knn | 64 | 0.9922 | 0.9977 | 0.011764705882352941 | 0.058823529411764705 |
| screw | pixel_knn | 64 | 0.9703 | 0.9910 | 0.15714285714285714 | 0.014285714285714285 |
| bottle | pixel_knn | 96 | 0.9903 | 0.9970 | n/a | n/a |
| cable | patch_memory | 96 | 0.9392 | 0.9798 | n/a | n/a |
| hazelnut | pixel_knn | 96 | 0.9940 | 0.9981 | n/a | n/a |
| screw | pixel_knn | 96 | 0.9559 | 0.9863 | n/a | n/a |

## Saved-model reload verification

| Category | Method | Max absolute score error after reload |
|---|---|---:|
| bottle | pixel_knn | 0.00e+00 |
| cable | patch_memory | 0.00e+00 |
| hazelnut | pixel_knn | 0.00e+00 |
| screw | pixel_knn | 0.00e+00 |
