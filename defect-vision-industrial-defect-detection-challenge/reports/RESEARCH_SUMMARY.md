# Research Summary

## Problem and hypothesis

Training contains only normal examples, so standard binary supervised learning is unsupported. We hypothesized that category-specific normality models can flag deviations while preserving a clear boundary between measurable normal behaviour and unknown real-defect performance.

## Methods and findings

The data audit established 742 normal training images, 503 unlabeled test images, four categories, valid paths, no exact duplicates, mixed PNG/JPEG sources, multiple resolutions, and visually grayscale RGB files. The pipeline applies robust decoding, EXIF orientation handling, and RGB canonicalization.

Experiments compared pixel nearest-neighbour residuals, local patch memory, handcrafted global density, and PCA reconstruction. Each cross-validation score used a reference bank excluding the scored normal image. Synthetic cutout, scratch, and texture perturbations supported proxy AUROC/AP/F1 comparisons but were never called competition metrics. Robustness checks under benign brightness/noise shifts rejected fragile density choices despite high proxy scores.

The final candidate is pixel KNN for bottle/hazelnut/screw and patch memory for cable, with category-specific leave-one-out 95th-percentile normal thresholds. It has saved model artefacts, exact reload checks, continuous test scores, a strict submission validator, and an immutable CSV backup.

## What is supported vs unknown

Supported: all data rows decode; final output has 503 ordered IDs; score/model reload/CSV checks pass; selected models have strong synthetic-proxy separation and are less fragile than rejected alternatives under the tested benign shifts.

Unknown: real-defect F1, precision, recall, prevalence, and whether the synthetic corruption families reflect actual defects. The numeric label mapping is inferred from local artefacts and must be confirmed on Kaggle before upload. A rules-approved pretrained feature/patch encoder is the most justified next research step.
