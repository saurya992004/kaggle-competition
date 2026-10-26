# Pretrained Feature Experiment Plan

The supplied competition description explicitly recommends transfer learning and states that the test set is defect-heavy. The previous 0.45771 public F1 came from a local handcrafted normal-only candidate; it is recorded as limited public feedback, not as a source of labels.

## Methods

1. **Frozen ResNet18 global KNN:** ImageNet-pretrained official torchvision ResNet18 average-pooled embeddings, category-specific normal reference bank, cosine-distance score.
2. **Frozen ResNet18 spatial anomaly score:** Intermediate `layer2` feature grid; category- and location-specific standardized normal features; image score is the mean largest 1% patch deviations. This is a compact PaDiM-style approximation, not full PatchCore.

Images are EXIF-normalized/RGB-converted, resized to 224px, and ImageNet-normalized. Models remain frozen: no test or defective label is used for fitting.

## Evaluation and thresholds

Both approaches use the existing category-wise three-fold, self-excluding normal-validation protocol plus synthetic proxy families. Test candidates will use a small, predeclared grid of global defect proportions (25%, 35%, 45%, 55%, 65%), generated from continuous category-normalized ranks. They are for controlled public-leaderboard comparison only; they are not ground truth nor a private-score guarantee. No automatic upload will occur.

## Feasibility

Official ResNet18 ImageNet weights will be downloaded through torchvision if not cached. This is a 44.7MB pretrained weight artifact; it is recorded locally with its source/version and is not external defect data. CPU-only feature extraction is expected to be the main cost.
