"""Perform the reproducible, read-only Phase 1 DefectVision audit.

Usage (from the project directory):
    python -m src.data_audit --data-root . --output-dir artifacts/audit
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from src.data_loader import DataConfig, load_pil_image, preprocess_image, resolve_image_path


SPLITS = ("train", "test")
EXPECTED_CATEGORIES = {"bottle", "cable", "screw", "hazelnut"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def difference_hash(image: Image.Image) -> int:
    """64-bit dHash; comparable with Hamming distance without a third-party package."""
    gray = ImageOps.exif_transpose(image).convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray, dtype=np.int16)
    bits = pixels[:, 1:] > pixels[:, :-1]
    value = 0
    for bit in bits.ravel():
        value = (value << 1) | int(bit)
    return value


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def write_frame(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def read_metadata(data_root: Path) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    metadata: dict[str, pd.DataFrame] = {}
    for split in SPLITS:
        path = data_root / f"{split}.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        needed = {"id", "path", "category"}
        missing = needed - set(frame.columns)
        if missing:
            raise ValueError(f"{path.name} missing columns: {sorted(missing)}")
        frame["split"] = split
        metadata[split] = frame
    sample_path = data_root / "sample_submission.csv"
    if not sample_path.is_file():
        raise FileNotFoundError(sample_path)
    sample = pd.read_csv(sample_path)
    if set(sample.columns) != {"id", "label"}:
        raise ValueError(f"Unexpected sample_submission columns: {sample.columns.tolist()}")
    return metadata, sample


def inspect_images(data_root: Path, frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        path = resolve_image_path(data_root, str(row.path))
        record: dict[str, Any] = {
            "id": str(row.id), "path": str(row.path), "category": str(row.category),
            "split": str(row.split), "exists": path.is_file(), "status": "ok",
            "format": None, "mode": None, "width": None, "height": None, "raw_channels": None,
            "sha256": None, "dhash_hex": None, "mean_intensity": None, "std_intensity": None,
            "mean_r": None, "mean_g": None, "mean_b": None, "appears_grayscale": None,
        }
        if not record["exists"]:
            record["status"] = "missing"
            records.append(record)
            continue
        try:
            with Image.open(path) as raw:
                raw.load()
                original_mode = raw.mode
                canonical = ImageOps.exif_transpose(raw).convert("RGB")
                array = np.asarray(canonical, dtype=np.float32)
                record.update({
                    "format": raw.format, "mode": original_mode, "width": canonical.width,
                    "height": canonical.height, "raw_channels": len(raw.getbands()),
                    "sha256": sha256_file(path), "dhash_hex": f"{difference_hash(canonical):016x}",
                    "mean_intensity": round(float(array.mean()), 6), "std_intensity": round(float(array.std()), 6),
                    "mean_r": round(float(array[..., 0].mean()), 6),
                    "mean_g": round(float(array[..., 1].mean()), 6), "mean_b": round(float(array[..., 2].mean()), 6),
                    # RGB-encoded monochrome images remain RGB on disk; this reports pixels, not file mode.
                    "appears_grayscale": bool(np.array_equal(array[..., 0], array[..., 1]) and np.array_equal(array[..., 1], array[..., 2])),
                })
        except Exception as exc:
            record["status"] = f"unreadable: {type(exc).__name__}: {exc}"
        records.append(record)
    return pd.DataFrame.from_records(records)


def exact_duplicate_groups(manifest: pd.DataFrame) -> pd.DataFrame:
    usable = manifest[manifest.status.eq("ok")].copy()
    rows: list[dict[str, Any]] = []
    for sha, group in usable.groupby("sha256", dropna=True):
        if len(group) > 1:
            rows.append({
                "hash": sha, "n_images": len(group), "ids": "|".join(group.id.astype(str)),
                "splits": "|".join(sorted(group.split.unique())),
                "categories": "|".join(sorted(group.category.unique())),
            })
    return pd.DataFrame(rows, columns=["hash", "n_images", "ids", "splits", "categories"])


def near_duplicate_pairs(manifest: pd.DataFrame, maximum_distance: int = 4) -> pd.DataFrame:
    """Exhaustive within-category dHash comparison; findings are candidates, not proof."""
    usable = manifest[manifest.status.eq("ok")].copy()
    usable["dhash"] = usable.dhash_hex.map(lambda value: int(str(value), 16))
    matches: list[dict[str, Any]] = []
    for category, group in usable.groupby("category"):
        data = list(group[["id", "split", "dhash"]].itertuples(index=False, name=None))
        for left, right in itertools.combinations(data, 2):
            distance = hamming(int(left[2]), int(right[2]))
            if distance <= maximum_distance:
                matches.append({
                    "category": category, "id_a": left[0], "split_a": left[1],
                    "id_b": right[0], "split_b": right[1], "hamming_distance": distance,
                    "cross_split": left[1] != right[1],
                })
    return pd.DataFrame(matches, columns=["category", "id_a", "split_a", "id_b", "split_b", "hamming_distance", "cross_split"])


def save_diagnostics(data_root: Path, frame: pd.DataFrame, manifest: pd.DataFrame, output_dir: Path) -> None:
    plots = output_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    good = manifest[manifest.status.eq("ok")].copy()
    plt.figure(figsize=(9, 4))
    for split, group in good.groupby("split"):
        plt.hist(group.width, bins=20, alpha=0.55, label=f"{split} width")
        plt.hist(group.height, bins=20, alpha=0.35, label=f"{split} height")
    plt.xlabel("pixels"); plt.ylabel("images"); plt.legend(); plt.tight_layout()
    plt.savefig(plots / "dimensions.png", dpi=160); plt.close()

    plt.figure(figsize=(8, 4))
    for split, group in good.groupby("split"):
        plt.hist(group.mean_intensity, bins=30, alpha=0.6, label=split)
    plt.xlabel("mean RGB intensity (0–255)"); plt.ylabel("images"); plt.legend(); plt.tight_layout()
    plt.savefig(plots / "mean_intensity.png", dpi=160); plt.close()

    for split in SPLITS:
        subset = frame[frame.split.eq(split)]
        categories = sorted(subset.category.unique())
        figure, axes = plt.subplots(len(categories), 5, figsize=(12, 2.7 * len(categories)))
        axes = np.atleast_2d(axes)
        for row_index, category in enumerate(categories):
            examples = subset[subset.category.eq(category)].head(5)
            for col, row in enumerate(examples.itertuples(index=False)):
                image = load_pil_image(resolve_image_path(data_root, str(row.path)))
                axes[row_index, col].imshow(image)
                axes[row_index, col].set_title(str(row.id), fontsize=7)
                axes[row_index, col].axis("off")
            for col in range(len(examples), 5):
                axes[row_index, col].axis("off")
            axes[row_index, 0].set_ylabel(category, fontsize=9)
        figure.suptitle(f"{split.title()} representative images (first five per category)")
        figure.tight_layout()
        figure.savefig(plots / f"{split}_montage.png", dpi=160)
        plt.close(figure)


def serialise_counts(series: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in series.sort_index().items()}


def build_summary(metadata: dict[str, pd.DataFrame], sample: pd.DataFrame, manifest: pd.DataFrame,
                  exact: pd.DataFrame, near: pd.DataFrame, data_root: Path) -> dict[str, Any]:
    combined = pd.concat(metadata.values(), ignore_index=True)
    listed_paths = set(combined.path.astype(str))
    files_on_disk = {path.name for path in (data_root / "clips").iterdir() if path.is_file()}
    sample_ids_match = set(sample.id.astype(str)) == set(metadata["test"].id.astype(str))
    config_check = preprocess_image(load_pil_image(resolve_image_path(data_root, str(combined.iloc[0].path))), DataConfig())
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input_root": str(data_root.resolve()),
        "metadata_rows": {split: int(len(df)) for split, df in metadata.items()},
        "metadata_columns": {split: list(df.columns) for split, df in metadata.items()},
        "missing_values": {split: serialise_counts(df.isna().sum()) for split, df in metadata.items()},
        "category_counts": {split: serialise_counts(df.category.value_counts()) for split, df in metadata.items()},
        "label_counts_train": serialise_counts(metadata["train"].label.value_counts(dropna=False)) if "label" in metadata["train"] else {},
        "unique_ids": {split: bool(df.id.is_unique) for split, df in metadata.items()},
        "train_test_id_overlap": int(len(set(metadata["train"].id) & set(metadata["test"].id))),
        "train_test_path_overlap": int(len(set(metadata["train"].path) & set(metadata["test"].path))),
        "sample_submission": {"columns": list(sample.columns), "rows": int(len(sample)), "ids_match_test_set": sample_ids_match,
                              "label_values": [str(value) for value in sorted(sample.label.dropna().unique())]},
        "image_status_counts": serialise_counts(manifest.status.value_counts()),
        "formats": serialise_counts(manifest.format.value_counts(dropna=False)),
        "original_modes": serialise_counts(manifest["mode"].value_counts(dropna=False)),
        "raw_channel_counts": serialise_counts(manifest.raw_channels.value_counts(dropna=False)),
        "pixel_grayscale_count": int(manifest.appears_grayscale.fillna(False).sum()),
        "dimensions_by_pair": {f"{int(w)}x{int(h)}": int(count) for (w, h), count in manifest.groupby(["width", "height"]).size().sort_index().items()},
        "dimensions": {"unique": int(manifest[["width", "height"]].dropna().drop_duplicates().shape[0]),
                       "width_range": [int(manifest.width.min()), int(manifest.width.max())],
                       "height_range": [int(manifest.height.min()), int(manifest.height.max())]},
        "clips_files": int(len(files_on_disk)), "unreferenced_clips_files": int(len(files_on_disk - listed_paths)),
        "referenced_files_missing": int((~manifest.exists).sum()), "unreadable_files": int((manifest.status != "ok").sum()),
        "exact_duplicate_groups": int(len(exact)), "exact_duplicate_images": int(exact.n_images.sum()) if not exact.empty else 0,
        "near_duplicate_pairs_dhash_le_4": int(len(near)), "cross_split_near_duplicate_pairs": int(near.cross_split.sum()) if not near.empty else 0,
        "preprocessing_check_shape": list(config_check.shape), "preprocessing_check_dtype": str(config_check.dtype),
    }


def render_markdown(summary: dict[str, Any]) -> str:
    rows = summary["metadata_rows"]
    dimensions = summary["dimensions"]
    return f"""# Data Audit Results

Generated by `python -m src.data_audit --data-root . --output-dir artifacts/audit` on {summary['generated_utc']}.

## Reconciliation

| Check | Result |
|---|---:|
| Training metadata rows | {rows['train']} |
| Test metadata rows | {rows['test']} |
| Image files in `clips/` | {summary['clips_files']} |
| Referenced files missing | {summary['referenced_files_missing']} |
| Unreadable/missing references | {summary['unreadable_files']} |
| Unreferenced files in `clips/` | {summary['unreferenced_clips_files']} |
| Train/test ID overlap | {summary['train_test_id_overlap']} |
| Train/test path overlap | {summary['train_test_path_overlap']} |
| Sample IDs exactly match test IDs | {summary['sample_submission']['ids_match_test_set']} |

## Metadata

Training labels observed: `{summary['label_counts_train']}`. This is numeric metadata, so the human-readable `good`/`defective` vocabulary has **not** been inferred from it.

| Split | Category counts |
|---|---|
| Train | `{summary['category_counts']['train']}` |
| Test | `{summary['category_counts']['test']}` |

The sample submission has columns `{summary['sample_submission']['columns']}`, {summary['sample_submission']['rows']} rows, and observed placeholder label values `{summary['sample_submission']['label_values']}`.

## Image integrity

All referenced paths were decoded with Pillow, retaining their original format/mode in `artifacts/audit/image_manifest.csv`; preprocessing canonicalizes images using EXIF transpose then RGB conversion. Formats: `{summary['formats']}`. Original modes: `{summary['original_modes']}`; raw-channel counts: `{summary['raw_channel_counts']}`; pixels that are exactly monochrome across RGB channels: {summary['pixel_grayscale_count']}. There are {dimensions['unique']} distinct decoded `(width, height)` pairs, with width range {dimensions['width_range']} and height range {dimensions['height_range']}: `{summary['dimensions_by_pair']}`.

Measured global mean intensity and standard deviation are diagnostic signals, not evidence by themselves of sensor noise, compression, or defects. See `artifacts/audit/plots/` for distributions and category montages.

## Duplicates and leakage

Exact byte-identical groups: {summary['exact_duplicate_groups']} ({summary['exact_duplicate_images']} participating images). Candidate perceptual near-duplicate pairs (64-bit dHash Hamming distance <=4, compared only within category): {summary['near_duplicate_pairs_dhash_le_4']}; cross-split candidates: {summary['cross_split_near_duplicate_pairs']}. dHash candidates are not confirmed duplicates and should be visually reviewed before any validation decision. See `exact_duplicate_groups.csv` and `near_duplicate_pairs_dhash_le_4.csv`.

## Preprocessing verification

The configured baseline decoded an actual metadata-referenced image and produced CHW `{summary['preprocessing_check_dtype']}` output with shape `{summary['preprocessing_check_shape']}`. Phase 1 default is 256px letterboxing: it preserves full framing; center crop and stretch remain explicit controlled alternatives for Phase 2. No random augmentations are used here.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/audit"))
    parser.add_argument("--skip-plots", action="store_true", help="Reuse existing diagnostics when refreshing tabular audit outputs.")
    args = parser.parse_args()
    data_root, output_dir = args.data_root.resolve(), args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata, sample = read_metadata(data_root)
    combined = pd.concat(metadata.values(), ignore_index=True)
    manifest = inspect_images(data_root, combined)
    exact = exact_duplicate_groups(manifest)
    near = near_duplicate_pairs(manifest)
    write_frame(manifest, output_dir / "image_manifest.csv")
    write_frame(manifest[manifest.status.ne("ok")], output_dir / "rejected_or_unreadable_files.csv")
    write_frame(exact, output_dir / "exact_duplicate_groups.csv")
    write_frame(near, output_dir / "near_duplicate_pairs_dhash_le_4.csv")
    if not args.skip_plots:
        save_diagnostics(data_root, combined, manifest, output_dir)
    summary = build_summary(metadata, sample, manifest, exact, near, data_root)
    (output_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "DATA_AUDIT.generated.md").write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
