"""Frozen ImageNet ResNet18 normal-only anomaly benchmark for DefectVision."""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from PIL import Image
from torchvision.models import ConvNeXt_Tiny_Weights, ResNet18_Weights, ResNet50_Weights, ViT_B_16_Weights, convnext_tiny, resnet18, resnet50, vit_b_16
from torchvision.models.feature_extraction import create_feature_extractor

from src.data_loader import load_pil_image, resolve_image_path
from src.phase2_baseline import add_synthetic_defect, best_f1_threshold
from src.phase3_benchmark import average_precision, folds, normal_quantile_ci, roc_auc


@dataclass(frozen=True)
class Config:
    seed: int = 20261020
    image_size: int = 224
    batch_size: int = 16
    folds: int = 3
    tail_fraction: float = .01
    synthetic_variants: tuple[str, ...] = ("cutout", "scratch", "texture_patch")
    candidate_defect_rates: tuple[float, ...] = (.25, .35, .45, .55, .65)
    backbone: str = "resnet18"


def read_config(path: Path) -> Config:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    data = asdict(Config()); data.update(raw)
    data["synthetic_variants"] = tuple(data["synthetic_variants"]); data["candidate_defect_rates"] = tuple(data["candidate_defect_rates"])
    return Config(**data)


class FrozenImageNetResNet:
    """Official frozen ImageNet ResNet/ViT feature extractor, CPU/GPU agnostic."""
    def __init__(self, backbone: str) -> None:
        if backbone == "resnet18":
            self.weights = ResNet18_Weights.IMAGENET1K_V1
            base = resnet18(weights=self.weights).eval()
        elif backbone == "resnet50":
            self.weights = ResNet50_Weights.IMAGENET1K_V2
            base = resnet50(weights=self.weights).eval()
        elif backbone == "vit_b16":
            self.weights = ViT_B_16_Weights.IMAGENET1K_V1
            base = vit_b_16(weights=self.weights).eval()
        elif backbone == "convnext_tiny":
            self.weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
            base = convnext_tiny(weights=self.weights).eval()
        else:
            raise ValueError(f"unsupported backbone: {backbone}")
        for parameter in base.parameters(): parameter.requires_grad_(False)
        self.backbone = backbone
        if backbone == "vit_b16":
            nodes = {"encoder.ln": "tokens"}
        elif backbone == "convnext_tiny":
            nodes = {"avgpool": "global", "features.5": "patch"}
        else:
            nodes = {"avgpool": "global", "layer2": "patch"}
        self.model = create_feature_extractor(base, return_nodes=nodes).eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.mean = torch.tensor(self.weights.transforms().mean).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor(self.weights.transforms().std).view(1, 3, 1, 1).to(self.device)

    def extract(self, images: np.ndarray, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        global_vectors, patches = [], []
        with torch.inference_mode():
            for start in range(0, len(images), batch_size):
                batch = torch.from_numpy(images[start:start + batch_size]).permute(0, 3, 1, 2).to(self.device)
                batch = (batch - self.mean) / self.std
                output = self.model(batch)
                if self.backbone == "vit_b16":
                    tokens = output["tokens"]
                    global_vectors.append(tokens[:, 0].cpu().numpy().astype(np.float32))
                    patches.append(tokens[:, 1:].reshape(-1, 14, 14, tokens.shape[-1]).cpu().numpy().astype(np.float32))
                else:
                    global_vectors.append(output["global"].flatten(1).cpu().numpy().astype(np.float32))
                    patches.append(output["patch"].permute(0, 2, 3, 1).cpu().numpy().astype(np.float32))
        return np.concatenate(global_vectors), np.concatenate(patches)


def images_from_paths(root: Path, paths: pd.Series, size: int) -> np.ndarray:
    result = []
    for relative_path in paths:
        image = load_pil_image(resolve_image_path(root, str(relative_path))).resize((size, size), Image.Resampling.BILINEAR)
        result.append(np.asarray(image, dtype=np.float32) / 255.0)
    return np.stack(result)


def normalise_rows(features: np.ndarray) -> np.ndarray:
    return features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-8)


def global_knn_score(reference: np.ndarray, query: np.ndarray) -> np.ndarray:
    reference, query = normalise_rows(reference), normalise_rows(query)
    return 1.0 - (query @ reference.T).max(axis=1)


def patch_score(reference: np.ndarray, query: np.ndarray, tail_fraction: float) -> np.ndarray:
    """Spatial diagonal-Mahalanobis score, fitting per location on normals only."""
    mean = reference.mean(axis=0)
    std = np.maximum(reference.std(axis=0), 0.05)
    deviation = np.abs((query - mean) / std).mean(axis=3).reshape(len(query), -1)
    count = max(1, int(np.ceil(deviation.shape[1] * tail_fraction)))
    return np.partition(deviation, deviation.shape[1] - count, axis=1)[:, -count:].mean(axis=1)


def score_method(method: str, reference_global: np.ndarray, reference_patch: np.ndarray, query_global: np.ndarray, query_patch: np.ndarray, config: Config) -> np.ndarray:
    if method.endswith("_global_knn"): return global_knn_score(reference_global, query_global)
    if method.endswith("_spatial"): return patch_score(reference_patch, query_patch, config.tail_fraction)
    raise ValueError(method)


def proxy_cv(train: pd.DataFrame, global_features: np.ndarray, patch_features: np.ndarray, synthetic_features: dict[str, tuple[np.ndarray, np.ndarray]], config: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, Any]] = []
    methods = (f"{config.backbone}_global_knn", f"{config.backbone}_spatial")
    for category in sorted(train.category.unique()):
        position = np.flatnonzero(train.category.to_numpy() == category)
        cat = train.iloc[position].reset_index(drop=True)
        cf, cp = global_features[position], patch_features[position]
        variants = {key: (value[0][position], value[1][position]) for key, value in synthetic_features.items()}
        for fold_index, validation in enumerate(folds(len(cat), config.folds, config.seed + sum(map(ord, category)))):
            fitting = np.setdiff1d(np.arange(len(cat)), validation, assume_unique=True)
            for method in methods:
                normal = score_method(method, cf[fitting], cp[fitting], cf[validation], cp[validation], config)
                for index, score in enumerate(normal): records.append({"category": category, "fold": fold_index, "id": cat.iloc[validation[index]].id, "method": method, "variant": "none", "label_proxy": 0, "score": float(score)})
                for variant, (vg, vp) in variants.items():
                    anomalous = score_method(method, cf[fitting], cp[fitting], vg[validation], vp[validation], config)
                    for index, score in enumerate(anomalous): records.append({"category": category, "fold": fold_index, "id": cat.iloc[validation[index]].id, "method": method, "variant": variant, "label_proxy": 1, "score": float(score)})
    scores = pd.DataFrame(records); metrics = []
    for (category, method), group in scores.groupby(["category", "method"], sort=True):
        normal = group[group.label_proxy.eq(0)].score.to_numpy()
        threshold, f1 = best_f1_threshold(group.score.to_numpy(), group.label_proxy.to_numpy())
        normal_threshold, lo, hi = normal_quantile_ci(normal, .95, config.seed)
        base = {"run_id": f"pretrained-{config.backbone}-20261020", "timestamp_utc": datetime.now(timezone.utc).isoformat(), "category": category, "method": method,
                "proxy_auroc": roc_auc(group.label_proxy.to_numpy(), group.score.to_numpy()), "proxy_ap": average_precision(group.label_proxy.to_numpy(), group.score.to_numpy()),
                "proxy_f1": f1, "proxy_threshold": threshold, "normal_95pct_threshold": normal_threshold, "threshold_ci_low": lo, "threshold_ci_high": hi,
                "normal_rows": len(normal), "synthetic_rows": int((group.label_proxy == 1).sum())}
        metrics.append({**base, "variant": "all_synthetic"})
        for variant in config.synthetic_variants:
            split = group[(group.variant == "none") | (group.variant == variant)]
            metrics.append({**base, "variant": variant, "proxy_auroc": roc_auc(split.label_proxy.to_numpy(), split.score.to_numpy()),
                            "proxy_ap": average_precision(split.label_proxy.to_numpy(), split.score.to_numpy()),
                            "proxy_f1": best_f1_threshold(split.score.to_numpy(), split.label_proxy.to_numpy())[1]})
    return scores, pd.DataFrame(metrics)


def final_test_scores(train: pd.DataFrame, test: pd.DataFrame, train_global: np.ndarray, train_patch: np.ndarray, test_global: np.ndarray, test_patch: np.ndarray, metrics: pd.DataFrame, config: Config) -> pd.DataFrame:
    output = test[["id", "path", "category"]].copy()
    for method in (f"{config.backbone}_global_knn", f"{config.backbone}_spatial"):
        all_scores = np.zeros(len(test), dtype=np.float32)
        for category in sorted(train.category.unique()):
            tr = np.flatnonzero(train.category.to_numpy() == category); te = np.flatnonzero(test.category.to_numpy() == category)
            all_scores[te] = score_method(method, train_global[tr], train_patch[tr], test_global[te], test_patch[te], config)
        output[f"score_{method}"] = all_scores
        # Category percentile rank removes incomparable raw score scales for controlled threshold candidates.
        output[f"rank_{method}"] = output.groupby("category")[f"score_{method}"].rank(pct=True, method="average")
    return output


def write_candidates(scores: pd.DataFrame, output: Path, config: Config) -> None:
    sample = pd.read_csv(output.parent / "sample_submission.csv") if (output.parent / "sample_submission.csv").exists() else None
    for method in (f"{config.backbone}_global_knn", f"{config.backbone}_spatial"):
        for rate in config.candidate_defect_rates:
            candidate = scores[["id"]].copy(); candidate["label"] = (scores[f"rank_{method}"] > 1 - rate).astype(int)
            candidate.to_csv(output / f"candidate_{method}_top_{int(rate * 100):02d}pct.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--data-root", type=Path, default=Path(".")); parser.add_argument("--output-dir", type=Path, default=Path("artifacts/pretrained_resnet18")); parser.add_argument("--config", type=Path, default=Path("config/pretrained_resnet18.yaml"))
    args = parser.parse_args(); root, output, config = args.data_root.resolve(), args.output_dir.resolve(), read_config(args.config); output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.seed); np.random.seed(config.seed); started = time.perf_counter()
    train, test = pd.read_csv(root / "train.csv"), pd.read_csv(root / "test.csv")
    extractor = FrozenImageNetResNet(config.backbone)
    train_images = images_from_paths(root, train.path, config.image_size); test_images = images_from_paths(root, test.path, config.image_size)
    train_global, train_patch = extractor.extract(train_images, config.batch_size); test_global, test_patch = extractor.extract(test_images, config.batch_size)
    synthetic: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    rng = np.random.default_rng(config.seed)
    for variant in config.synthetic_variants:
        altered = np.stack([add_synthetic_defect(image, variant, rng) for image in train_images])
        synthetic[variant] = extractor.extract(altered, config.batch_size)
    proxy_scores, metrics = proxy_cv(train, train_global, train_patch, synthetic, config)
    test_scores = final_test_scores(train, test, train_global, train_patch, test_global, test_patch, metrics, config)
    np.savez_compressed(output / "feature_references.npz", train_global=train_global, train_patch=train_patch, test_global=test_global, test_patch=test_patch)
    proxy_scores.to_csv(output / "proxy_scores.csv", index=False); metrics.to_csv(output / "experiment_results.csv", index=False); test_scores.to_csv(output / "test_feature_scores.csv", index=False)
    write_candidates(test_scores, output, config)
    summary = {"config": asdict(config), "weight_source": extractor.weights.url, "torch": torch.__version__, "torchvision": __import__("torchvision").__version__, "device": str(extractor.device), "elapsed_seconds": time.perf_counter() - started,
               "metrics": metrics[metrics.variant.eq("all_synthetic")].to_dict(orient="records"), "candidate_files": sorted(path.name for path in output.glob("candidate_*.csv"))}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8"); print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
