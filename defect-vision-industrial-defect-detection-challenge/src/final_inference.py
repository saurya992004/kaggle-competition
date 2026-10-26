"""Generate the audited final DefectVision prediction package.

Uses saved Phase 3 models and recalibrates category thresholds from leave-one-out
normal scores so calibration never self-matches an image in its reference bank.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.phase3_benchmark import (BenchmarkConfig, PatchMemory, PixelKNN, build_detector, load_config,
                                  load_image, normal_quantile_ci)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def loo_scores(images: np.ndarray, method: str, config: BenchmarkConfig) -> np.ndarray:
    """Normal calibration scores where the scored image is excluded from references."""
    values = []
    for index in range(len(images)):
        keep = np.concatenate([np.arange(index), np.arange(index + 1, len(images))])
        model = build_detector(method, config).fit(images[keep])
        values.append(float(model.score(images[index:index + 1])[0]))
    return np.asarray(values)


def finalise(data_root: Path, output_dir: Path, config_path: Path) -> dict[str, object]:
    started = time.perf_counter()
    config = load_config(config_path)
    train, test = pd.read_csv(data_root / "train.csv"), pd.read_csv(data_root / "test.csv")
    selection = pd.read_csv(data_root / "artifacts" / "phase3" / "candidate_selection.csv")
    if train.label.nunique() != 1:
        raise ValueError("Expected normal-only train labels")
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration_rows, score_rows = [], []
    for chosen in selection.itertuples(index=False):
        category, method = str(chosen.category), str(chosen.method)
        train_cat = train[train.category.eq(category)].reset_index(drop=True)
        test_cat = test[test.category.eq(category)].copy()
        train_images = np.stack([load_image(data_root, path, config.image_size) for path in train_cat.path])
        calibration = loo_scores(train_images, method, config)
        if not np.isfinite(calibration).all():
            raise ValueError(f"Non-finite LOO calibration score for {category}")
        threshold, ci_low, ci_high = normal_quantile_ci(calibration, 1 - config.target_normal_fpr, config.seed)
        # Fit exactly once on all permitted normal images for test inference.
        model = build_detector(method, config).fit(train_images)
        test_images = np.stack([load_image(data_root, path, config.image_size) for path in test_cat.path])
        scores = model.score(test_images)
        if not np.isfinite(scores).all():
            raise ValueError(f"Non-finite test score for {category}")
        test_cat["selected_method"] = method
        test_cat["anomaly_score"] = scores
        test_cat["loo_normal_threshold_95pct"] = threshold
        test_cat["candidate_label_numeric"] = (scores >= threshold).astype(int)
        score_rows.append(test_cat)
        calibration_rows.append({"category": category, "method": method, "normal_train_rows": len(train_cat),
                                 "calibration": "leave-one-out normal scores; 95th percentile", "threshold": threshold,
                                 "threshold_ci_low": ci_low, "threshold_ci_high": ci_high,
                                 "loo_estimated_normal_fpr": float((calibration >= threshold).mean()),
                                 "loo_score_median": float(np.median(calibration))})
    scores = pd.concat(score_rows).set_index("id").loc[test.id].reset_index()
    if not scores.id.equals(test.id):
        raise ValueError("Internal test-order mismatch")
    calibration_frame = pd.DataFrame(calibration_rows)
    scores.to_csv(output_dir / "final_prediction_scores.csv", index=False)
    calibration_frame.to_csv(output_dir / "final_calibration.csv", index=False)
    # The sample submission's actual downloaded vocabulary is numeric 0, and the
    # normal-only training label is 0; numeric 1 remains a documented inference.
    submission = scores[["id"]].copy(); submission["label"] = scores.candidate_label_numeric.astype(int)
    submission_path = data_root / "submission.csv"; submission.to_csv(submission_path, index=False)
    digest = sha256(submission_path)
    backup_dir = output_dir / "immutable_submissions"; backup_dir.mkdir(exist_ok=True)
    backup_path = backup_dir / f"submission_{digest}.csv"
    if not backup_path.exists(): shutil.copy2(submission_path, backup_path)
    summary = {"created_utc": datetime.now(timezone.utc).isoformat(), "submission_path": str(submission_path),
               "submission_sha256": digest, "immutable_backup": str(backup_path), "test_rows": len(scores),
               "prediction_counts": {str(key): int(value) for key, value in submission.label.value_counts().sort_index().items()},
               "by_category": scores.groupby("category").candidate_label_numeric.agg(["count", "sum"]).rename(columns={"sum": "predicted_1"}).to_dict(orient="index"),
               "runtime_seconds": time.perf_counter() - started}
    (output_dir / "final_run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--config", type=Path, default=Path("config/phase3_benchmark.yaml"))
    args = parser.parse_args()
    finalise(args.data_root.resolve(), args.output_dir.resolve(), args.config)


if __name__ == "__main__":
    main()
