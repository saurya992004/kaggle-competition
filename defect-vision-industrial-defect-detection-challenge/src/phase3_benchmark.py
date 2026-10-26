"""Leakage-safe local model portfolio for DefectVision normal-only training.

All synthetic metrics produced here are proxy metrics.  No test labels are
loaded or inferred, and the output is deliberately not a Kaggle submission.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.data_loader import load_pil_image, resolve_image_path
from src.phase2_baseline import add_synthetic_defect, best_f1_threshold, luminance, mean_top_fraction


@dataclass(frozen=True)
class BenchmarkConfig:
    seed: int = 20261019
    folds: int = 3
    image_size: int = 64
    reconstruction_size: int = 32
    tail_fraction: float = 0.01
    patch_size: int = 8
    patch_stride: int = 8
    patch_tail_fraction: float = 0.10
    patch_memory_size: int = 512
    pca_components: int = 24
    target_normal_fpr: float = 0.05
    synthetic_variants: tuple[str, ...] = ("cutout", "scratch", "texture_patch")
    methods: tuple[str, ...] = ("pixel_knn", "global_density", "patch_memory", "pca_reconstruction")


def load_config(path: Path) -> BenchmarkConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    defaults = asdict(BenchmarkConfig())
    defaults.update(raw)
    defaults["synthetic_variants"] = tuple(defaults["synthetic_variants"])
    defaults["methods"] = tuple(defaults["methods"])
    return BenchmarkConfig(**defaults)


def load_image(data_root: Path, relative_path: str, size: int) -> np.ndarray:
    image = load_pil_image(resolve_image_path(data_root, relative_path))
    return np.asarray(image.resize((size, size)), dtype=np.float32) / 255.0


def global_features(images: np.ndarray) -> np.ndarray:
    """Colour, spatial-moment, and gradient features; independent of test labels."""
    features: list[np.ndarray] = []
    for image in images:
        hist = np.concatenate([np.histogram(image[..., channel], bins=16, range=(0.0, 1.0), density=True)[0]
                               for channel in range(3)]).astype(np.float32)
        blocks = []
        for y in np.array_split(np.arange(image.shape[0]), 4):
            for x in np.array_split(np.arange(image.shape[1]), 4):
                block = image[np.ix_(y, x)]
                blocks.extend(block.mean(axis=(0, 1)))
                blocks.extend(block.std(axis=(0, 1)))
        gray = luminance(image[None])[0]
        gy, gx = np.gradient(gray)
        magnitude = np.hypot(gx, gy)
        gradient_hist = np.histogram(magnitude, bins=16, range=(0.0, 0.75), density=True)[0]
        features.append(np.concatenate([hist, np.asarray(blocks), gradient_hist]).astype(np.float32))
    return np.stack(features)


def patch_features(images: np.ndarray, size: int, stride: int) -> tuple[np.ndarray, int]:
    """Return [n_images, n_patches, 10] local appearance descriptors."""
    all_images: list[np.ndarray] = []
    for image in images:
        descriptors = []
        for top in range(0, image.shape[0] - size + 1, stride):
            for left in range(0, image.shape[1] - size + 1, stride):
                patch = image[top:top + size, left:left + size]
                gray = luminance(patch[None])[0]
                gy, gx = np.gradient(gray)
                descriptors.append(np.concatenate([
                    patch.mean(axis=(0, 1)), patch.std(axis=(0, 1)),
                    [gray.mean(), gray.std(), np.hypot(gx, gy).mean(), np.hypot(gx, gy).std()],
                ]))
        all_images.append(np.asarray(descriptors, dtype=np.float32))
    return np.stack(all_images), len(all_images[0])


class Detector:
    name = "base"

    def fit(self, images: np.ndarray) -> "Detector":
        raise NotImplementedError

    def score(self, images: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def save(self, path: Path) -> None:
        raise NotImplementedError


class PixelKNN(Detector):
    name = "pixel_knn"

    def __init__(self, tail_fraction: float) -> None:
        self.tail_fraction = tail_fraction
        self.reference: np.ndarray | None = None
        self.reference_gray: np.ndarray | None = None

    def fit(self, images: np.ndarray) -> "PixelKNN":
        self.reference = images.copy()
        self.reference_gray = luminance(images)
        return self

    def score(self, images: np.ndarray) -> np.ndarray:
        assert self.reference is not None and self.reference_gray is not None
        distances = ((luminance(images)[:, None] - self.reference_gray[None]) ** 2).mean(axis=(2, 3))
        nearest = self.reference[distances.argmin(axis=1)]
        residual = np.abs(images - nearest).mean(axis=3)
        return mean_top_fraction(residual, self.tail_fraction)

    def save(self, path: Path) -> None:
        assert self.reference is not None
        np.savez_compressed(path, name=self.name, tail_fraction=self.tail_fraction, reference=self.reference)


class GlobalDensity(Detector):
    name = "global_density"

    def __init__(self, shrinkage: float = 0.10) -> None:
        self.shrinkage = shrinkage
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.precision: np.ndarray | None = None

    def fit(self, images: np.ndarray) -> "GlobalDensity":
        data = global_features(images)
        self.mean = data.mean(axis=0)
        self.std = np.maximum(data.std(axis=0), 1e-5)
        standardized = (data - self.mean) / self.std
        covariance = np.cov(standardized, rowvar=False)
        covariance = (1 - self.shrinkage) * covariance + self.shrinkage * np.eye(covariance.shape[0])
        self.precision = np.linalg.pinv(covariance, rcond=1e-5)
        return self

    def score(self, images: np.ndarray) -> np.ndarray:
        assert self.mean is not None and self.std is not None and self.precision is not None
        delta = (global_features(images) - self.mean) / self.std
        return np.einsum("ni,ij,nj->n", delta, self.precision, delta)

    def save(self, path: Path) -> None:
        assert self.mean is not None and self.std is not None and self.precision is not None
        np.savez_compressed(path, name=self.name, shrinkage=self.shrinkage, mean=self.mean, std=self.std, precision=self.precision)


class PatchMemory(Detector):
    name = "patch_memory"

    def __init__(self, patch_size: int, stride: int, tail_fraction: float, memory_size: int = 2048) -> None:
        self.patch_size, self.stride, self.tail_fraction, self.memory_size = patch_size, stride, tail_fraction, memory_size
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.memory: np.ndarray | None = None

    def fit(self, images: np.ndarray) -> "PatchMemory":
        patches, _ = patch_features(images, self.patch_size, self.stride)
        flattened = patches.reshape(-1, patches.shape[-1])
        self.mean, self.std = flattened.mean(axis=0), np.maximum(flattened.std(axis=0), 1e-5)
        standardized = (flattened - self.mean) / self.std
        # Deterministic uniform coreset: avoids an unbounded patch bank.
        index = np.linspace(0, len(standardized) - 1, min(self.memory_size, len(standardized)), dtype=int)
        self.memory = standardized[index]
        return self

    def score(self, images: np.ndarray) -> np.ndarray:
        assert self.mean is not None and self.std is not None and self.memory is not None
        patches, _ = patch_features(images, self.patch_size, self.stride)
        standardized = (patches - self.mean) / self.std
        output = []
        for sample in standardized:
            squared = ((sample[:, None, :] - self.memory[None, :, :]) ** 2).sum(axis=2)
            output.append(mean_top_fraction(np.sqrt(squared.min(axis=1))[None, :, None], self.tail_fraction)[0])
        return np.asarray(output)

    def save(self, path: Path) -> None:
        assert self.mean is not None and self.std is not None and self.memory is not None
        np.savez_compressed(path, name=self.name, patch_size=self.patch_size, stride=self.stride,
                            tail_fraction=self.tail_fraction, memory_size=self.memory_size,
                            mean=self.mean, std=self.std, memory=self.memory)


class PCAReconstruction(Detector):
    name = "pca_reconstruction"

    def __init__(self, reconstruction_size: int, components: int, tail_fraction: float) -> None:
        self.reconstruction_size, self.components, self.tail_fraction = reconstruction_size, components, tail_fraction
        self.mean: np.ndarray | None = None
        self.basis: np.ndarray | None = None

    def _vectors(self, images: np.ndarray) -> np.ndarray:
        from PIL import Image

        resized = []
        for image in images:
            gray = (luminance(image[None])[0] * 255).astype(np.uint8)
            resized.append(np.asarray(Image.fromarray(gray).resize((self.reconstruction_size, self.reconstruction_size)), dtype=np.float32) / 255.0)
        return np.stack(resized).reshape(len(images), -1)

    def fit(self, images: np.ndarray) -> "PCAReconstruction":
        vectors = self._vectors(images)
        self.mean = vectors.mean(axis=0)
        _, _, vh = np.linalg.svd(vectors - self.mean, full_matrices=False)
        self.basis = vh[:min(self.components, len(vh))]
        return self

    def score(self, images: np.ndarray) -> np.ndarray:
        assert self.mean is not None and self.basis is not None
        centered = self._vectors(images) - self.mean
        reconstruction = (centered @ self.basis.T) @ self.basis
        residual = np.abs(centered - reconstruction)
        return mean_top_fraction(residual.reshape(len(images), self.reconstruction_size, self.reconstruction_size), self.tail_fraction)

    def save(self, path: Path) -> None:
        assert self.mean is not None and self.basis is not None
        np.savez_compressed(path, name=self.name, reconstruction_size=self.reconstruction_size,
                            components=self.components, tail_fraction=self.tail_fraction, mean=self.mean, basis=self.basis)


def build_detector(name: str, config: BenchmarkConfig) -> Detector:
    if name == "pixel_knn": return PixelKNN(config.tail_fraction)
    if name == "global_density": return GlobalDensity()
    if name == "patch_memory": return PatchMemory(config.patch_size, config.patch_stride, config.patch_tail_fraction, config.patch_memory_size)
    if name == "pca_reconstruction": return PCAReconstruction(config.reconstruction_size, config.pca_components, config.tail_fraction)
    raise ValueError(f"Unsupported detector: {name}")


def load_detector(path: Path) -> Detector:
    state = np.load(path, allow_pickle=False)
    name = str(state["name"])
    if name == PixelKNN.name:
        model = PixelKNN(float(state["tail_fraction"])); model.reference = state["reference"]; model.reference_gray = luminance(model.reference)
    elif name == GlobalDensity.name:
        model = GlobalDensity(float(state["shrinkage"])); model.mean, model.std, model.precision = state["mean"], state["std"], state["precision"]
    elif name == PatchMemory.name:
        model = PatchMemory(int(state["patch_size"]), int(state["stride"]), float(state["tail_fraction"]), int(state["memory_size"])); model.mean, model.std, model.memory = state["mean"], state["std"], state["memory"]
    elif name == PCAReconstruction.name:
        model = PCAReconstruction(int(state["reconstruction_size"]), int(state["components"]), float(state["tail_fraction"])); model.mean, model.basis = state["mean"], state["basis"]
    else:
        raise ValueError(f"Unknown saved detector: {name}")
    return model


def folds(n: int, count: int, seed: int) -> list[np.ndarray]:
    order = np.random.default_rng(seed).permutation(n)
    return list(np.array_split(order, count))


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive, negative = labels == 1, labels == 0
    if not positive.any() or not negative.any(): return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float); ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[positive].sum() - positive.sum() * (positive.sum() + 1) / 2) / (positive.sum() * negative.sum()))


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(scores)[::-1]
    ordered = labels[order]
    cumulative = np.cumsum(ordered)
    precision = cumulative / np.arange(1, len(labels) + 1)
    return float((precision * ordered).sum() / max(1, ordered.sum()))


def normal_quantile_ci(scores: np.ndarray, quantile: float, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    bootstraps = [np.quantile(rng.choice(scores, len(scores), replace=True), quantile) for _ in range(300)]
    return float(np.quantile(scores, quantile)), float(np.quantile(bootstraps, 0.025)), float(np.quantile(bootstraps, 0.975))


def run_cv(data_root: Path, train: pd.DataFrame, images: np.ndarray, config: BenchmarkConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    runtime: dict[tuple[str, str], float] = {}
    for category in sorted(train.category.unique()):
        positions = np.flatnonzero(train.category.to_numpy() == category)
        cat_images, cat_frame = images[positions], train.iloc[positions].reset_index(drop=True)
        category_folds = folds(len(cat_images), config.folds, config.seed + sum(map(ord, category)))
        for fold_id, validation in enumerate(category_folds):
            fitting = np.setdiff1d(np.arange(len(cat_images)), validation, assume_unique=True)
            rng = np.random.default_rng(config.seed + fold_id + sum(map(ord, category)))
            synthetic = {variant: np.stack([add_synthetic_defect(image, variant, rng) for image in cat_images[validation]]) for variant in config.synthetic_variants}
            for method in config.methods:
                started = time.perf_counter()
                model = build_detector(method, config).fit(cat_images[fitting])
                normal_scores = model.score(cat_images[validation])
                variant_scores = {variant: model.score(value) for variant, value in synthetic.items()}
                runtime[(category, method)] = runtime.get((category, method), 0.0) + time.perf_counter() - started
                for local_index, score in enumerate(normal_scores):
                    rows.append({"category": category, "fold": fold_id, "id": cat_frame.iloc[validation[local_index]].id,
                                 "method": method, "source": "heldout_normal", "variant": "none", "label_proxy": 0, "score": float(score)})
                for variant, scores in variant_scores.items():
                    for local_index, score in enumerate(scores):
                        rows.append({"category": category, "fold": fold_id, "id": cat_frame.iloc[validation[local_index]].id,
                                     "method": method, "source": "synthetic", "variant": variant, "label_proxy": 1, "score": float(score)})
    score_frame = pd.DataFrame(rows)
    metrics: list[dict[str, Any]] = []
    for (category, method), group in score_frame.groupby(["category", "method"], sort=True):
        normal = group[group.label_proxy.eq(0)].score.to_numpy()
        combined = group[group.variant.isin(["none", *config.synthetic_variants])]
        threshold_proxy, proxy_f1 = best_f1_threshold(combined.score.to_numpy(), combined.label_proxy.to_numpy())
        entry = {"run_id": "phase3-local-20261019", "timestamp_utc": datetime.now(timezone.utc).isoformat(), "category": category,
                 "method": method, "variant": "all_synthetic", "normal_count": len(normal),
                 "synthetic_count": int((combined.label_proxy == 1).sum()), "proxy_auroc": roc_auc(combined.label_proxy.to_numpy(), combined.score.to_numpy()),
                 "proxy_ap": average_precision(combined.label_proxy.to_numpy(), combined.score.to_numpy()), "proxy_f1": proxy_f1,
                 "proxy_f1_threshold": threshold_proxy, "runtime_seconds": runtime[(category, method)]}
        for rate in (0.01, 0.02, 0.05, 0.10):
            threshold, low, high = normal_quantile_ci(normal, 1 - rate, config.seed)
            entry[f"normal_threshold_fpr_{int(rate * 100)}pct"] = threshold
            entry[f"estimated_fpr_{int(rate * 100)}pct"] = float((normal >= threshold).mean())
            if rate == config.target_normal_fpr:
                entry["selected_normal_threshold"], entry["threshold_ci_low"], entry["threshold_ci_high"] = threshold, low, high
        metrics.append(entry)
        for variant in config.synthetic_variants:
            subset = group[(group.variant == "none") | (group.variant == variant)]
            metrics.append({**entry, "variant": variant, "synthetic_count": int((subset.label_proxy == 1).sum()),
                            "proxy_auroc": roc_auc(subset.label_proxy.to_numpy(), subset.score.to_numpy()),
                            "proxy_ap": average_precision(subset.label_proxy.to_numpy(), subset.score.to_numpy()),
                            "proxy_f1": best_f1_threshold(subset.score.to_numpy(), subset.label_proxy.to_numpy())[1]})
    return score_frame, pd.DataFrame(metrics)


def select_candidate(metrics: pd.DataFrame) -> pd.DataFrame:
    primary = metrics[metrics.variant.eq("all_synthetic")].copy()
    selected = []
    for category, group in primary.groupby("category", sort=True):
        # Geometric mean rewards methods that are consistently strong in ranking and precision-recall.
        group = group.assign(selection_score=np.sqrt(group.proxy_auroc.clip(0, 1) * group.proxy_ap.clip(0, 1)))
        selected.append(group.sort_values(["selection_score", "runtime_seconds"], ascending=[False, True]).iloc[0])
    return pd.DataFrame(selected)


def final_fit_score(data_root: Path, train: pd.DataFrame, test: pd.DataFrame, images: np.ndarray, config: BenchmarkConfig,
                    selection: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    model_dir = output_dir / "models"; model_dir.mkdir(parents=True, exist_ok=True)
    test_rows: list[pd.DataFrame] = []; reload_checks: list[dict[str, Any]] = []
    for row in selection.itertuples(index=False):
        category, method = row.category, row.method
        positions = np.flatnonzero(train.category.to_numpy() == category)
        model = build_detector(method, config).fit(images[positions])
        path = model_dir / f"{category}_{method}.npz"; model.save(path)
        test_cat = test[test.category.eq(category)].copy()
        test_images = np.stack([load_image(data_root, item, config.image_size) for item in test_cat.path])
        score = model.score(test_images)
        reloaded = load_detector(path).score(test_images[:min(3, len(test_images))])
        reload_checks.append({"category": category, "method": method, "model_path": str(path),
                              "reload_max_abs_error": float(np.max(np.abs(score[:len(reloaded)] - reloaded)))})
        test_cat["selected_method"] = method
        test_cat["anomaly_score"] = score
        test_cat["normal_calibrated_threshold"] = float(row.selected_normal_threshold)
        test_cat["candidate_anomaly_at_5pct_normal_fpr"] = (score >= float(row.selected_normal_threshold)).astype(int)
        test_rows.append(test_cat)
    full = pd.concat(test_rows).set_index("id").loc[test.id].reset_index()
    return full, reload_checks


def run_robustness_ablation(data_root: Path, train: pd.DataFrame, selection: pd.DataFrame,
                            config: BenchmarkConfig, resolutions: tuple[int, ...] | None = None) -> pd.DataFrame:
    """Fixed-split resolution and benign-shift check; separate from the main CV ledger."""
    records: list[dict[str, Any]] = []
    for picked in selection.itertuples(index=False):
        category, method = picked.category, picked.method
        cat = train[train.category.eq(category)].reset_index(drop=True)
        split = folds(len(cat), config.folds, config.seed + sum(map(ord, category)))[0]
        fit = np.setdiff1d(np.arange(len(cat)), split, assume_unique=True)
        for resolution in (resolutions or (config.image_size, 96)):
            adjusted = BenchmarkConfig(**{**asdict(config), "image_size": resolution,
                                         "reconstruction_size": min(config.reconstruction_size, resolution)})
            images = np.stack([load_image(data_root, path, resolution) for path in cat.path])
            model = build_detector(method, adjusted).fit(images[fit])
            normal = model.score(images[split])
            rng = np.random.default_rng(config.seed + resolution + sum(map(ord, category)))
            synthetic = np.concatenate([model.score(np.stack([add_synthetic_defect(image, variant, rng) for image in images[split]]))
                                        for variant in config.synthetic_variants])
            labels = np.concatenate([np.zeros(len(normal), dtype=int), np.ones(len(synthetic), dtype=int)])
            scores = np.concatenate([normal, synthetic])
            brightness = model.score(np.clip(images[split] * 0.85 + 0.03, 0, 1))
            noise = model.score(np.clip(images[split] + rng.normal(0, 0.025, images[split].shape), 0, 1).astype(np.float32))
            records.append({"category": category, "method": method, "resolution": resolution,
                            "validation_normals": len(normal), "synthetic_proxy_auroc": roc_auc(labels, scores),
                            "synthetic_proxy_ap": average_precision(labels, scores),
                            "normal_median_score": float(np.median(normal)),
                            "brightness_median_score": float(np.median(brightness)),
                            "noise_median_score": float(np.median(noise)),
                            "brightness_above_main_5pct_threshold": float((brightness >= picked.selected_normal_threshold).mean()) if resolution == config.image_size else float("nan"),
                            "noise_above_main_5pct_threshold": float((noise >= picked.selected_normal_threshold).mean()) if resolution == config.image_size else float("nan")})
    return pd.DataFrame(records)


def select_robust_candidate(metrics: pd.DataFrame, robustness: pd.DataFrame) -> pd.DataFrame:
    """Select with both proxy separation and measured benign-shift sensitivity.

    A candidate is robust when neither tested benign shift pushes more than 20%
    of fixed-split held-out normals over its OOF 5% threshold. If no method for
    a category meets this deliberately lenient screen, choose the least fragile
    one, then break ties with proxy rank. This never uses test images/labels.
    """
    primary = metrics[metrics.variant.eq("all_synthetic")].copy()
    robust = robustness[robustness.resolution.eq(64)][["category", "method", "brightness_above_main_5pct_threshold", "noise_above_main_5pct_threshold"]]
    table = primary.merge(robust, on=["category", "method"], how="left")
    table["selection_score"] = np.sqrt(table.proxy_auroc.clip(0, 1) * table.proxy_ap.clip(0, 1))
    table["benign_shift_max_fpr"] = table[["brightness_above_main_5pct_threshold", "noise_above_main_5pct_threshold"]].max(axis=1)
    chosen = []
    for _, group in table.groupby("category", sort=True):
        safe = group[group.benign_shift_max_fpr <= 0.20]
        if not safe.empty:
            chosen.append(safe.sort_values(["selection_score", "benign_shift_max_fpr", "runtime_seconds"], ascending=[False, True, True]).iloc[0])
        else:
            chosen.append(group.sort_values(["benign_shift_max_fpr", "selection_score", "runtime_seconds"], ascending=[True, False, True]).iloc[0])
    return pd.DataFrame(chosen)


def save_plots(scores: pd.DataFrame, selection: pd.DataFrame, test_scores: pd.DataFrame, output_dir: Path) -> None:
    plot_dir = output_dir / "plots"; plot_dir.mkdir(parents=True, exist_ok=True)
    for category in sorted(selection.category):
        figure, axis = plt.subplots(figsize=(8, 4))
        for method in selection[selection.category.eq(category)].method:
            subset = scores[(scores.category == category) & (scores.method == method)]
            axis.hist(subset[subset.label_proxy == 0].score, bins=28, density=True, alpha=.55, label=f"{method}: normal")
            axis.hist(subset[subset.label_proxy == 1].score, bins=28, density=True, alpha=.35, label=f"{method}: synthetic")
        axis.set(title=f"{category}: selected model proxy distributions", xlabel="anomaly score", ylabel="density")
        axis.legend(fontsize=7); figure.tight_layout(); figure.savefig(plot_dir / f"proxy_distribution_{category}.png", dpi=160); plt.close(figure)
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for axis, category in zip(axes.ravel(), sorted(test_scores.category.unique())):
        subset = test_scores[test_scores.category.eq(category)]
        axis.hist(subset.anomaly_score, bins=24, alpha=.75)
        axis.axvline(subset.normal_calibrated_threshold.iloc[0], color="crimson", label="5% OOF-normal threshold")
        axis.set_title(category); axis.legend(fontsize=7)
    figure.suptitle("Unlabeled test-score distributions — selected local candidate")
    figure.savefig(plot_dir / "selected_test_score_distributions.png", dpi=160); plt.close(figure)


def report(selection: pd.DataFrame, reload_checks: list[dict[str, Any]], ablation: pd.DataFrame) -> str:
    rows = "\n".join(f"| {x.category} | {x.method} | {x.proxy_auroc:.4f} | {x.proxy_ap:.4f} | {x.proxy_f1:.4f} | {x.selected_normal_threshold:.6f} |" for x in selection.itertuples(index=False))
    checks = "\n".join(f"| {x['category']} | {x['method']} | {x['reload_max_abs_error']:.2e} |" for x in reload_checks)
    ablations = "\n".join(f"| {x.category} | {x.method} | {x.resolution} | {x.synthetic_proxy_auroc:.4f} | {x.synthetic_proxy_ap:.4f} | {x.brightness_above_main_5pct_threshold if np.isfinite(x.brightness_above_main_5pct_threshold) else 'n/a'} | {x.noise_above_main_5pct_threshold if np.isfinite(x.noise_above_main_5pct_threshold) else 'n/a'} |" for x in ablation.itertuples(index=False))
    return f"""# Phase 3 Model Comparison and Candidate Selection

## Completed portfolio

Four local normal-only methods were executed in three leakage-safe category-specific folds: pixel nearest-neighbour residual, global shrinkage-Mahalanobis density, patch-statistic memory, and low-rank PCA reconstruction. Frozen pretrained CNN/ViT and pretrained PatchCore were not executed: no weights were cached locally, no compatible package was installed, and Kaggle's pretrained-weight policy remains unresolved. No external model was downloaded.

## Selection evidence

| Category | Selected method | Synthetic proxy AUROC | Synthetic proxy AP | Synthetic proxy F1 | 95th-percentile OOF normal threshold |
|---|---|---:|---:|---:|---:|
{rows}

The selected threshold is the 95th percentile of out-of-fold normal scores, a nominal 5% normal false-positive operating point. It is deliberately not the synthetic-F1-optimal threshold and was not calibrated from the test score distribution. Its sampling uncertainty is stored in `experiment_results.csv`.

## Candidate and caveats

`candidate_test_scores.csv` stores continuous scores plus a local binary decision at the normal-calibrated threshold. It is **not a submission** and does not establish real-defect F1. The final test-label encoding and metric must be confirmed before submission.

The two important weaknesses are (1) synthetic anomalies need not match real defects and (2) handcrafted local features lack the semantic invariance a permitted pretrained patch encoder could provide. Pretrained experiments are the justified next escalation only after the competition Rules page confirms their allowance.

## Compact robustness ablation

This fixed, held-out normal split changes resolution only (64 versus 96). Brightness and Gaussian-noise perturbations are applied only at 64 because the main 5%-normal threshold is defined at that resolution. Values shown are the fraction of benignly perturbed held-out normals exceeding the original 64px threshold; they are robustness diagnostics, not defect labels.

| Category | Method | Resolution | Synthetic proxy AUROC | Synthetic proxy AP | Brightness above 64px threshold | Noise above 64px threshold |
|---|---|---:|---:|---:|---:|---:|
{ablations}

## Saved-model reload verification

| Category | Method | Max absolute score error after reload |
|---|---|---:|
{checks}
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/phase3"))
    parser.add_argument("--config", type=Path, default=Path("config/phase3_benchmark.yaml"))
    parser.add_argument("--ablation-only", action="store_true", help="Run the compact robustness check from existing candidate selection.")
    parser.add_argument("--refine-selection", action="store_true", help="Robustness-screen all completed methods and refit only the selected candidate.")
    args = parser.parse_args()
    root, output = args.data_root.resolve(), args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    train, test = pd.read_csv(root / "train.csv"), pd.read_csv(root / "test.csv")
    if train.label.nunique() != 1: raise ValueError("This benchmark is only valid for one-class training data")
    if args.ablation_only or args.refine_selection:
        required = (output / "candidate_selection.csv", output / "candidate_test_scores.csv", output / "reload_checks.json", output / "oof_and_synthetic_scores.csv", output / "experiment_results.csv")
        if not all(path.is_file() for path in required):
            raise FileNotFoundError("continuation requires prior Phase 3 candidate artefacts")
        selection, test_scores = pd.read_csv(required[0]), pd.read_csv(required[1])
        scores, metrics = pd.read_csv(required[3]), pd.read_csv(required[4])
        reload_checks = json.loads(required[2].read_text(encoding="utf-8"))
        elapsed = 0.0
        if args.refine_selection:
            all_candidates = metrics[metrics.variant.eq("all_synthetic")].copy()
            all_robustness = run_robustness_ablation(root, train, all_candidates, config, resolutions=(config.image_size,))
            all_robustness.to_csv(output / "robustness_all_methods.csv", index=False)
            selection = select_robust_candidate(metrics, all_robustness)
            images = np.stack([load_image(root, item, config.image_size) for item in train.path])
            test_scores, reload_checks = final_fit_score(root, train, test, images, config, selection, output)
            selection.to_csv(output / "candidate_selection.csv", index=False)
            test_scores.to_csv(output / "candidate_test_scores.csv", index=False)
            selected_high_res = run_robustness_ablation(root, train, selection, config, resolutions=(96,))
            ablation = pd.concat([all_robustness.merge(selection[["category", "method"]], on=["category", "method"]), selected_high_res], ignore_index=True)
        else:
            ablation = run_robustness_ablation(root, train, selection, config)
    else:
        images = np.stack([load_image(root, item, config.image_size) for item in train.path])
        started = time.perf_counter(); scores, metrics = run_cv(root, train, images, config); elapsed = time.perf_counter() - started
        selection = select_candidate(metrics)
        test_scores, reload_checks = final_fit_score(root, train, test, images, config, selection, output)
        scores.to_csv(output / "oof_and_synthetic_scores.csv", index=False)
        metrics.to_csv(output / "experiment_results.csv", index=False)
        selection.to_csv(output / "candidate_selection.csv", index=False)
        test_scores.to_csv(output / "candidate_test_scores.csv", index=False)
        ablation = run_robustness_ablation(root, train, selection, config)
    ablation.to_csv(output / "robustness_ablation.csv", index=False)
    save_plots(scores, selection, test_scores, output)
    (output / "reload_checks.json").write_text(json.dumps(reload_checks, indent=2), encoding="utf-8")
    (output / "MODEL_COMPARISON.generated.md").write_text(report(selection, reload_checks, ablation), encoding="utf-8")
    summary = {"run_id": "phase3-local-20261019", "elapsed_seconds": elapsed, "config": asdict(config),
               "selection": selection[["category", "method", "proxy_auroc", "proxy_ap", "selected_normal_threshold"]].to_dict(orient="records"),
               "candidate_test_rows": len(test_scores), "reload_checks": reload_checks,
               "robustness_ablation_rows": len(ablation),
               "pretrained_branch": "not_run_no_cached_weights_and_rules_unresolved"}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
