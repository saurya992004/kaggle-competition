"""Normal-only anomaly baseline and honest proxy evaluation for DefectVision.

This module deliberately never calls synthetic-corruption F1 a competition F1.
It uses only the supplied normal training split to fit category-specific models.

Usage:
    python -m src.phase2_baseline --data-root . --output-dir artifacts/phase2
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageDraw, ImageFilter

from src.data_loader import load_pil_image, resolve_image_path


@dataclass(frozen=True)
class Phase2Config:
    seed: int = 20261019
    image_size: int = 64
    folds: int = 3
    tail_fraction: float = 0.01
    synthetic_variants: tuple[str, ...] = ("cutout", "scratch", "texture_patch")
    methods: tuple[str, ...] = ("knn_tail", "template_tail")
    candidate_good_label: int = 0
    candidate_defective_label: int = 1


def load_config(path: Path) -> Phase2Config:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return Phase2Config(**{**Phase2Config().__dict__, **raw,
                           "synthetic_variants": tuple(raw.get("synthetic_variants", Phase2Config().synthetic_variants)),
                           "methods": tuple(raw.get("methods", Phase2Config().methods))})


def image_array(data_root: Path, relative_path: str, image_size: int) -> np.ndarray:
    """Load RGB image into [0, 1] HWC without ImageNet normalization for scoring."""
    image = load_pil_image(resolve_image_path(data_root, relative_path))
    image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def luminance(images: np.ndarray) -> np.ndarray:
    return images[..., 0] * 0.299 + images[..., 1] * 0.587 + images[..., 2] * 0.114


def mean_top_fraction(values: np.ndarray, fraction: float) -> np.ndarray:
    """Mean of a sample's largest fraction; first axis is sample."""
    flattened = values.reshape(values.shape[0], -1)
    count = max(1, int(np.ceil(flattened.shape[1] * fraction)))
    partitioned = np.partition(flattened, flattened.shape[1] - count, axis=1)
    return partitioned[:, -count:].mean(axis=1)


class CategoryAnomalyModel:
    """Nearest-normal residual and robust normal-template residual.

    Nearest neighbour selects a pose/illumination-compatible normal reference
    using low-resolution luminance MSE. The final score emphasizes localized
    residuals, suitable for defects that occupy a small region.
    """

    def __init__(self, tail_fraction: float) -> None:
        self.tail_fraction = tail_fraction
        self.reference: np.ndarray | None = None
        self.reference_luminance: np.ndarray | None = None
        self.template: np.ndarray | None = None
        self.scale: np.ndarray | None = None

    def fit(self, images: np.ndarray) -> "CategoryAnomalyModel":
        if images.ndim != 4 or len(images) < 2:
            raise ValueError("fit requires at least two RGB HWC reference images")
        self.reference = images.astype(np.float32, copy=True)
        self.reference_luminance = luminance(self.reference)
        self.template = np.median(self.reference, axis=0)
        mad = np.median(np.abs(self.reference - self.template), axis=0)
        # Bound prevents textureless regions from creating unstable ratios.
        self.scale = np.maximum(1.4826 * mad, 0.04)
        return self

    def score(self, images: np.ndarray) -> dict[str, np.ndarray]:
        if self.reference is None or self.reference_luminance is None or self.template is None or self.scale is None:
            raise RuntimeError("Model must be fitted before scoring")
        gray = luminance(images)
        reference_distances = ((gray[:, None] - self.reference_luminance[None]) ** 2).mean(axis=(2, 3))
        nearest_indices = reference_distances.argmin(axis=1)
        nearest = self.reference[nearest_indices]
        nearest_residual = np.abs(images - nearest).mean(axis=3)
        template_residual = (np.abs(images - self.template) / self.scale).mean(axis=3)
        return {
            "knn_tail": mean_top_fraction(nearest_residual, self.tail_fraction),
            "template_tail": mean_top_fraction(template_residual, self.tail_fraction),
        }


def add_synthetic_defect(image: np.ndarray, variant: str, rng: np.random.Generator) -> np.ndarray:
    """Make an explicitly marked proxy anomaly; never used as a real label."""
    result = Image.fromarray(np.clip(image * 255.0, 0, 255).astype(np.uint8), mode="RGB")
    width, height = result.size
    draw = ImageDraw.Draw(result)
    if variant == "cutout":
        radius_x = int(rng.integers(max(3, width // 12), max(5, width // 5)))
        radius_y = int(rng.integers(max(3, height // 12), max(5, height // 5)))
        center_x = int(rng.integers(radius_x, width - radius_x))
        center_y = int(rng.integers(radius_y, height - radius_y))
        color = tuple(int(x) for x in rng.choice([0, 20, 235, 255], size=3))
        draw.ellipse((center_x - radius_x, center_y - radius_y, center_x + radius_x, center_y + radius_y), fill=color)
    elif variant == "scratch":
        start = (int(rng.integers(0, width)), int(rng.integers(0, height)))
        end = (int(rng.integers(0, width)), int(rng.integers(0, height)))
        color = tuple(int(x) for x in rng.choice([0, 235, 255], size=3))
        draw.line((start, end), fill=color, width=int(rng.integers(1, 4)))
    elif variant == "texture_patch":
        patch_width = int(rng.integers(max(5, width // 8), max(7, width // 3)))
        patch_height = int(rng.integers(max(5, height // 8), max(7, height // 3)))
        left = int(rng.integers(0, width - patch_width + 1))
        top = int(rng.integers(0, height - patch_height + 1))
        noise = rng.integers(0, 256, (patch_height, patch_width, 3), dtype=np.uint8)
        noise_image = Image.fromarray(noise, mode="RGB").filter(ImageFilter.GaussianBlur(radius=0.35))
        result.paste(noise_image, (left, top))
    else:
        raise ValueError(f"Unknown synthetic variant: {variant}")
    return np.asarray(result, dtype=np.float32) / 255.0


def stratified_fold_indices(frame: pd.DataFrame, folds: int, seed: int) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    """Yield deterministic category-stratified normal-only train/validation indices."""
    rng = np.random.default_rng(seed)
    buckets: list[list[np.ndarray]] = []
    for _, group in frame.groupby("category", sort=True):
        indices = group.index.to_numpy().copy()
        rng.shuffle(indices)
        buckets.append(list(np.array_split(indices, folds)))
    all_indices = frame.index.to_numpy()
    for fold in range(folds):
        validation = np.concatenate([bucket[fold] for bucket in buckets])
        training = np.setdiff1d(all_indices, validation, assume_unique=True)
        yield training, validation


def best_f1_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Return threshold and F1 for proxy labels only; predictions use >= threshold."""
    order = np.argsort(scores)[::-1]
    sorted_scores, sorted_labels = scores[order], labels[order].astype(np.int64)
    total_positive = int(sorted_labels.sum())
    true_positive = 0
    best_f1, best_threshold = -1.0, float(sorted_scores[0])
    for index, label in enumerate(sorted_labels):
        true_positive += int(label)
        predicted_positive = index + 1
        denominator = predicted_positive + total_positive
        f1 = (2 * true_positive / denominator) if denominator else 0.0
        if f1 > best_f1:
            best_f1, best_threshold = f1, float(sorted_scores[index])
    return best_threshold, best_f1


def run_proxy_cv(data_root: Path, train: pd.DataFrame, config: Phase2Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    images = np.stack([image_array(data_root, path, config.image_size) for path in train.path])
    records: list[dict[str, object]] = []
    threshold_records: list[dict[str, object]] = []
    for category in sorted(train.category.unique()):
        cat_positions = np.flatnonzero(train.category.to_numpy() == category)
        cat_frame = train.iloc[cat_positions].reset_index(drop=True)
        cat_images = images[cat_positions]
        for fold, (fit_idx, validation_idx) in enumerate(stratified_fold_indices(cat_frame, config.folds, config.seed)):
            model = CategoryAnomalyModel(config.tail_fraction).fit(cat_images[fit_idx])
            normal_scores = model.score(cat_images[validation_idx])
            rng = np.random.default_rng(config.seed + fold + sum(map(ord, category)))
            for variant in config.synthetic_variants:
                synthetic = np.stack([add_synthetic_defect(image, variant, rng) for image in cat_images[validation_idx]])
                anomaly_scores = model.score(synthetic)
                for method in config.methods:
                    for score in normal_scores[method]:
                        records.append({"category": category, "fold": fold, "method": method, "source": "heldout_normal", "variant": "none", "label_proxy": 0, "score": float(score)})
                    for score in anomaly_scores[method]:
                        records.append({"category": category, "fold": fold, "method": method, "source": "synthetic", "variant": variant, "label_proxy": 1, "score": float(score)})
    scores = pd.DataFrame.from_records(records)
    for (category, method), group in scores.groupby(["category", "method"], sort=True):
        threshold, f1 = best_f1_threshold(group.score.to_numpy(), group.label_proxy.to_numpy())
        threshold_records.append({"category": category, "method": method, "proxy_threshold": threshold,
                                  "proxy_f1": f1, "rows": len(group),
                                  "normal_rows": int((group.label_proxy == 0).sum()),
                                  "synthetic_rows": int((group.label_proxy == 1).sum())})
    return scores, pd.DataFrame.from_records(threshold_records)


def fit_and_score_test(data_root: Path, train: pd.DataFrame, test: pd.DataFrame, config: Phase2Config,
                       thresholds: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for category in sorted(train.category.unique()):
        train_category = train[train.category.eq(category)].reset_index(drop=True)
        test_category = test[test.category.eq(category)].reset_index(drop=True)
        train_images = np.stack([image_array(data_root, path, config.image_size) for path in train_category.path])
        test_images = np.stack([image_array(data_root, path, config.image_size) for path in test_category.path])
        scores = CategoryAnomalyModel(config.tail_fraction).fit(train_images).score(test_images)
        frame = test_category[["id", "path", "category"]].copy()
        for method in config.methods:
            frame[f"score_{method}"] = scores[method]
            threshold = float(thresholds.query("category == @category and method == @method").proxy_threshold.iloc[0])
            frame[f"pred_{method}"] = (scores[method] >= threshold).astype(int)
        rows.append(frame)
    return pd.concat(rows, ignore_index=True).set_index("id").loc[test.id].reset_index()


def save_plots(proxy_scores: pd.DataFrame, test_scores: pd.DataFrame, output_dir: Path) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for method in sorted(proxy_scores.method.unique()):
        figure, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
        for axis, category in zip(axes.ravel(), sorted(proxy_scores.category.unique())):
            subset = proxy_scores[(proxy_scores.method == method) & (proxy_scores.category == category)]
            axis.hist(subset[subset.label_proxy == 0].score, bins=30, alpha=0.65, label="held-out normal")
            axis.hist(subset[subset.label_proxy == 1].score, bins=30, alpha=0.65, label="synthetic proxy")
            axis.set_title(category); axis.set_xlabel("anomaly score"); axis.legend(fontsize=7)
        figure.suptitle(f"Proxy score distributions — {method}")
        figure.savefig(plot_dir / f"proxy_score_distributions_{method}.png", dpi=160)
        plt.close(figure)
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for axis, category in zip(axes.ravel(), sorted(test_scores.category.unique())):
        subset = test_scores[test_scores.category == category]
        for method in ("knn_tail", "template_tail"):
            axis.hist(subset[f"score_{method}"], bins=25, alpha=0.55, label=method)
        axis.set_title(category); axis.set_xlabel("test anomaly score"); axis.legend(fontsize=7)
    figure.suptitle("Unlabeled test-score distributions")
    figure.savefig(plot_dir / "test_score_distributions.png", dpi=160)
    plt.close(figure)


def save_ranked_montages(data_root: Path, test_scores: pd.DataFrame, output_dir: Path, image_size: int) -> None:
    """Save diagnostic top-score images; these are never used to revise test labels."""
    plot_dir = output_dir / "plots"
    for category in sorted(test_scores.category.unique()):
        subset = test_scores[test_scores.category.eq(category)].nlargest(8, "selected_proxy_score")
        figure, axes = plt.subplots(2, 4, figsize=(10, 5), constrained_layout=True)
        for axis, row in zip(axes.ravel(), subset.itertuples(index=False)):
            axis.imshow(image_array(data_root, row.path, image_size))
            axis.set_title(f"{row.id}\n{row.selected_proxy_score:.3f}", fontsize=8)
            axis.axis("off")
        figure.suptitle(f"{category}: eight highest selected-proxy scores (unlabeled test diagnostics)")
        figure.savefig(plot_dir / f"top_test_scores_{category}.png", dpi=160)
        plt.close(figure)


def render_report(config: Phase2Config, threshold_table: pd.DataFrame, test_scores: pd.DataFrame) -> str:
    summary_rows = []
    for row in threshold_table.sort_values(["category", "method"]).itertuples(index=False):
        predicted = int(test_scores[f"pred_{row.method}"].sum())
        summary_rows.append(f"| {row.category} | {row.method} | {row.proxy_f1:.4f} | {row.proxy_threshold:.6f} | {predicted} |")
    return """# Phase 2: Normal-only Anomaly Baselines

## What was run

Two category-specific normal-only baselines were fit exclusively on the 742 training images:

1. **`knn_tail`**: find the closest normal image by low-resolution luminance MSE and average the largest 1% of pixel residuals.
2. **`template_tail`**: form a category-wise RGB median normal template, scale residuals by robust per-pixel MAD, and average the largest 1%.

The benchmark uses three deterministic folds. Each held-out normal is scored both as normal and after three marked **synthetic proxy corruptions** (`cutout`, `scratch`, and `texture_patch`). The threshold maximizes F1 only for those proxy labels. This is useful for a reproducible ordering experiment; it is not a test-set F1 estimate and must not be reported as one.

## Proxy results and candidate counts

| Category | Method | Proxy F1 | Proxy threshold | Test rows predicted defective |
|---|---|---:|---:|---:|
""" + "\n".join(summary_rows) + """

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
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/phase2"))
    parser.add_argument("--config", type=Path, default=Path("config/phase2_baseline.yaml"))
    args = parser.parse_args()
    data_root, output_dir = args.data_root.resolve(), args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    train, test = pd.read_csv(data_root / "train.csv"), pd.read_csv(data_root / "test.csv")
    if train.label.nunique() != 1:
        raise ValueError("Phase 2 normal-only baseline requires a single training label")
    proxy_scores, thresholds = run_proxy_cv(data_root, train, config)
    test_scores = fit_and_score_test(data_root, train, test, config, thresholds)
    proxy_scores.to_csv(output_dir / "proxy_validation_scores.csv", index=False)
    thresholds.to_csv(output_dir / "proxy_thresholds.csv", index=False)
    # Select a method per category by proxy result, since normal image geometry differs by category.
    selected = thresholds.sort_values(["category", "proxy_f1"], ascending=[True, False]).groupby("category", as_index=False).first()
    selected_by_category = dict(zip(selected.category.astype(str), selected.method.astype(str)))
    test_scores["selected_proxy_method"] = test_scores.category.map(selected_by_category)
    test_scores["selected_proxy_score"] = [row[f"score_{row.selected_proxy_method}"] for _, row in test_scores.iterrows()]
    test_scores["pred_selected_proxy_method"] = [row[f"pred_{row.selected_proxy_method}"] for _, row in test_scores.iterrows()]
    test_scores.to_csv(output_dir / "test_anomaly_scores.csv", index=False)
    candidate = test_scores[["id"]].copy()
    candidate["label"] = np.where(test_scores["pred_selected_proxy_method"].to_numpy() == 1,
                                  config.candidate_defective_label, config.candidate_good_label)
    candidate.to_csv(output_dir / "submission_candidate_numeric_0_good_1_defective.csv", index=False)
    save_plots(proxy_scores, test_scores, output_dir)
    save_ranked_montages(data_root, test_scores, output_dir, config.image_size)
    summary = {"config": config.__dict__, "selected_proxy_method_by_category": selected_by_category,
               "thresholds": thresholds.to_dict(orient="records"),
               "candidate_predicted_defective": int(candidate.label.eq(config.candidate_defective_label).sum()),
               "test_rows": int(len(test))}
    (output_dir / "phase2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = render_report(config, thresholds, test_scores)
    (output_dir / "PHASE2_PROXY_EXPERIMENTS.generated.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
