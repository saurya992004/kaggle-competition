# Experiment Plan — Normal-only Anomaly Benchmark

Status: written before Phase 3 feature extraction/training on 2026-10-19.

## Scientific boundary

All 742 training images are normal. The benchmark will estimate leakage-safe out-of-fold normal-score distributions and use deliberately created synthetic corruptions only as a proxy task. It will not produce, claim, or optimize true-defect F1, AUROC, precision, or recall. Thresholds used for the eventual candidate will be calibrated from out-of-fold normal quantiles, not test labels or a desired test-positive count.

The Phase 1 dHash scan produced many visual-similarity candidates but no exact duplicates; its low threshold is collision-prone for these aligned product views. Exact SHA-256 duplicates are absent. CV therefore keeps each ID independent, records this limitation, and always excludes the image being scored from its reference bank.

## Planned experiments

| ID | Hypothesis / method | Features and fitting | Validation / score | Cost and failure mode |
|---|---|---|---|---|
| E1 | Pixel nearest-normal residual detects large localized deviations in aligned categories. | Existing 64px RGB, per-category normal reference bank. | 3-fold OOF; nearest luminance reference then largest 1% RGB residual. | Low CPU / can mistake pose or lighting for defects. |
| E2 | Global normal-feature density detects product-level shifts that pixel matching misses. | RGB histograms, 4x4 colour moments, and luminance-gradient histogram; category-wise shrinkage Mahalanobis model. | 3-fold OOF; squared Mahalanobis distance. | Low CPU / may suppress small localized defects. |
| E3 | Patch-memory scoring captures local texture/material changes without external weights. | 8x8 patch colour/texture statistics; category-specific standardized normal patch memory with a deterministic 512-descriptor coreset. | 3-fold OOF; nearest patch distance, image score = top 10% patch distances. | Moderate CPU / limited semantic representation; not a pretrained PatchCore implementation. |
| E4 | A compact low-rank reconstruction model gives a complementary reconstruction signal. | 32px grayscale PCA reconstruction, category-wise retained-energy model. | 3-fold OOF; largest 1% reconstruction residual. | Low/moderate CPU / can reconstruct simple anomalies or penalize rotations. |
| E5 | Frozen pretrained CNN embedding / PatchCore feature extraction may improve semantic matching. | ResNet/ViT pretrained weights, only if already locally cached and rules allow. | Same folds and proxy suite. | **Preflight-gated**: no cached weights found; rules unresolved; do not download or run now. |

## Shared protocol

- Categories are fitted and calibrated separately.
- Three deterministic category-stratified folds use seed `20261019`; each model scores only images outside its reference fit set.
- Synthetic proxy families: cutout, scratch, and texture patch at deterministic randomized placements. They are reported separately by family.
- Measurable outputs: normal score quantiles; normal FPR at OOF 1%, 2%, 5%, and 10% thresholds; synthetic AUROC, AP, and F1; runtime; finite-score and deterministic checks.
- Robustness ablation: 64px versus 96px input for the selected scoring family where feasible, and benign brightness/JPEG-like noise perturbations of held-out normal images.
- Candidate selection: compare all completed methods by cross-family synthetic rank, OOF normal-score stability, normal FPR threshold behaviour, runtime, and known failure modes. The candidate threshold is the OOF 95th normal-score percentile (nominal 5% FPR), with uncertainty documented.

## Reproduction

```powershell
python -m src.phase3_benchmark --data-root . --output-dir artifacts/phase3
python -m unittest discover -s tests -v
```

No final competition submission will be created in this phase.
