# Model Comparison and Candidate Selection

Phase 3 completed on 2026-10-20 using only the supplied normal training data and no hidden test labels. The full run took 323.4 seconds on the local CPU; the subsequent robustness-screen refinement reused its completed cross-validation scores and refit only selected final models.

## Methods actually run

| Method | Description | Outcome |
|---|---|---|
| Pixel KNN | Closest category-normal reference by luminance; top 1% RGB residual. | Selected for bottle, hazelnut, screw. |
| Patch memory | 8x8 local color/texture descriptors; bounded 512-patch normal memory; top 10% nearest-memory distance. | Selected for cable. |
| Global density | Shrinkage-Mahalanobis distance on color, spatial-moment, and gradient features. | Strong synthetic proxy scores but rejected for hazelnut/screw due to noise sensitivity. |
| PCA reconstruction | Category-wise 32px grayscale low-rank reconstruction residual. | More stable in some cases but lower proxy separation than selected alternatives. |

Frozen pretrained embeddings and pretrained PatchCore were not run: no pretrained weights were cached, no compatible package was installed, and the competition's pretrained-weight policy is unresolved. No external model was downloaded.

## Evidence-based selection

The comparison first used three deterministic category-specific folds with each scoring image outside its fitted normal bank. Synthetic `cutout`, `scratch`, and `texture_patch` images are proxy positives only. The final screen also tested brightness and Gaussian-noise shifts on held-out normal images. A method had to keep both benign-shift false-positive fractions at or below 20% at its out-of-fold 95th-percentile normal threshold, when an alternative satisfied that screen.

| Category | Selected method | Proxy AUROC | Proxy AP | Proxy F1 | Worst benign-shift FPR | Normal-calibrated threshold |
|---|---|---:|---:|---:|---:|---:|
| bottle | pixel_knn | 0.9974 | 0.9993 | 0.9913 | 17.8% | 0.191436 |
| cable | patch_memory | 0.9395 | 0.9782 | 0.9374 | 8.2% | 1.540608 |
| hazelnut | pixel_knn | 0.9938 | 0.9981 | 0.9773 | 5.9% | 0.213427 |
| screw | pixel_knn | 0.9683 | 0.9902 | 0.9487 | 15.7% | 0.321100 |

Global density was rejected despite synthetic-proxy AUROC above 0.999 for hazelnut and screw: Gaussian noise crossed its 5%-normal threshold for 100% of the hazelnut check and 77.1% of the screw check. This is exactly why synthetic performance alone was not used for selection.

The 95th-percentile thresholds target a nominal 5% normal false-positive rate. They do not encode an assumed test defect prevalence and are not selected to yield an attractive number of defective predictions. Threshold bootstrap intervals, every proxy metric, runtime, method configuration, and per-family results are recorded in [experiment_results.csv](experiment_results.csv).

## Final artefacts

- [Candidate selection ledger](candidate_selection.csv)
- [Machine-readable run summary](../artifacts/phase3/summary.json)
- [Continuous test scores and local threshold decisions](../artifacts/phase3/candidate_test_scores.csv)
- `artifacts/phase3/models/` — selected category models. Saved/reload checks produced max absolute score error of 0 for all four models.
- `artifacts/phase3/robustness_all_methods.csv` and `robustness_ablation.csv` — method-wide and selected-method robustness evidence.
- `artifacts/phase3/plots/` — category score-distribution plots.

## Limitations and recommendation

The candidate is defensible as a local normal-only baseline, but no real-defect F1 is measurable: synthetic proxy scores are not hidden-test F1. The first justified improvement path is a frozen pretrained patch-feature benchmark after the Rules page explicitly confirms permitted pretrained weights. Until then, finalize the competition label mapping and metric contract before any submission; no final submission has been generated.

## Re-run

```powershell
python -m src.phase3_benchmark --data-root . --output-dir artifacts/phase3
python -m unittest discover -s tests -v
```
