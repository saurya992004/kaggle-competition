# Final Model Report

## Selected final pipeline

The final normal-only anomaly detector is category-specific:

| Category | Model | Final threshold calibration |
|---|---|---|
| bottle | Pixel nearest-normal residual | 95th percentile of leave-one-out normal scores |
| cable | Patch-statistic memory | 95th percentile of leave-one-out normal scores |
| hazelnut | Pixel nearest-normal residual | 95th percentile of leave-one-out normal scores |
| screw | Pixel nearest-normal residual | 95th percentile of leave-one-out normal scores |

Pixel KNN finds the closest normal image in low-resolution luminance space and averages the largest 1% RGB residuals. Cable uses a bounded 512-descriptor local colour/texture patch memory with a top-10% nearest-patch aggregation. Images are decoded with EXIF handling and RGB conversion, then resized consistently to 64x64.

## Evidence for selection

The four local alternatives were assessed in leakage-safe, category-specific, three-fold normal-only cross-validation: pixel KNN, patch memory, global Mahalanobis density, and PCA reconstruction. The selection uses synthetic-proxy separation **and** benign-shift robustness. Global density had excellent synthetic proxy performance but was rejected for hazelnut and screw because Gaussian-noise false-positive rates reached 100% and 77.1% in the held-out robustness check. Full evidence is in [MODEL_COMPARISON.md](MODEL_COMPARISON.md) and [experiment_results.csv](experiment_results.csv).

The selected models' synthetic-proxy AUROC values were bottle 0.9974, cable 0.9395, hazelnut 0.9938, and screw 0.9683. These are not real-defect or competition F1 values.

## Final fit and calibration

Each final model was fitted on all normal training images in its category. Its threshold was recalculated with leave-one-out (LOO) normal scoring at the final reference-bank scale: the image being scored is absent from its reference bank. The operating point is the LOO 95th-percentile score, targeting a nominal 5% normal false-positive rate. This is a normal-behaviour policy, not an F1-optimal threshold and it does not use test labels or test score-distribution tuning.

Exact calibration thresholds and bootstrap intervals are stored in `outputs/final_calibration.csv`. The final process took 639.0 seconds on the available CPU, mainly due to self-excluding patch-memory calibration.

## Limitations

No real defective labels are available, so real precision, recall, F1, AUROC, and the true optimum threshold are unknown. The local sample submission is numeric and train labels are all `0`; the candidate maps high anomaly score to `1`, but the live Evaluation tab's allowed-label contract was not accessible here. Confirm that mapping before uploading. No pretrained model was used because allowed external/pretrained-weight policy remains unverified.
