"""Robust metadata-driven image loading and preprocessing for DefectVision."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from PIL import Image, ImageOps


ResizeMode = Literal["letterbox", "center_crop", "stretch"]


@dataclass(frozen=True)
class DataConfig:
    """Explicit, serializable preprocessing settings.

    ``letterbox`` is the Phase 1 default because it retains the complete object
    and avoids discarding potential defects at an edge.  Other modes are made
    explicit for controlled Phase 2 experiments.
    """

    image_size: int = 256
    resize_mode: ResizeMode = "letterbox"
    interpolation: str = "bilinear"
    pad_value: int = 0
    normalize: bool = True
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)


_INTERPOLATION = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
}


def resolve_image_path(data_root: Path, relative_path: str) -> Path:
    """Resolve a metadata path below ``clips`` without assuming filename IDs."""
    path = (data_root / "clips" / relative_path).resolve()
    clips_root = (data_root / "clips").resolve()
    if clips_root not in path.parents:
        raise ValueError(f"Path escapes clips directory: {relative_path!r}")
    return path


def load_pil_image(path: Path) -> Image.Image:
    """Decode an image, apply EXIF orientation, and return canonical RGB pixels."""
    try:
        with Image.open(path) as source:
            source.load()  # Surface corruption before returning a copied object.
            return ImageOps.exif_transpose(source).convert("RGB")
    except Exception as exc:  # Keep source filename visible to callers and logs.
        raise RuntimeError(f"Unable to decode image: {path}") from exc


def preprocess_image(image: Image.Image, config: DataConfig = DataConfig()) -> np.ndarray:
    """Return a CHW float32 image using the configured deterministic transform."""
    if config.image_size <= 0:
        raise ValueError("image_size must be positive")
    if config.interpolation not in _INTERPOLATION:
        raise ValueError(f"Unsupported interpolation: {config.interpolation}")
    image = ImageOps.exif_transpose(image).convert("RGB")
    target = config.image_size
    resample = _INTERPOLATION[config.interpolation]
    width, height = image.size

    if config.resize_mode == "stretch":
        processed = image.resize((target, target), resample=resample)
    elif config.resize_mode == "center_crop":
        scale = max(target / width, target / height)
        resized = image.resize((round(width * scale), round(height * scale)), resample=resample)
        left = (resized.width - target) // 2
        top = (resized.height - target) // 2
        processed = resized.crop((left, top, left + target, top + target))
    elif config.resize_mode == "letterbox":
        scale = min(target / width, target / height)
        resized = image.resize((max(1, round(width * scale)), max(1, round(height * scale))), resample=resample)
        processed = Image.new("RGB", (target, target), (config.pad_value,) * 3)
        processed.paste(resized, ((target - resized.width) // 2, (target - resized.height) // 2))
    else:
        raise ValueError(f"Unsupported resize_mode: {config.resize_mode}")

    array = np.asarray(processed, dtype=np.float32).transpose(2, 0, 1) / 255.0
    if config.normalize:
        mean = np.asarray(config.mean, dtype=np.float32)[:, None, None]
        std = np.asarray(config.std, dtype=np.float32)[:, None, None]
        array = (array - mean) / std
    return array


class MetadataImageDataset:
    """Small dependency-free dataset facade that preserves metadata row order."""

    def __init__(self, csv_path: Path, data_root: Path, config: DataConfig = DataConfig()) -> None:
        self.frame = pd.read_csv(csv_path)
        required = {"id", "path", "category"}
        missing = required - set(self.frame.columns)
        if missing:
            raise ValueError(f"Missing required metadata columns: {sorted(missing)}")
        if not self.frame["id"].is_unique:
            raise ValueError(f"Duplicate IDs in {csv_path}")
        self.data_root = Path(data_root)
        self.config = config

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.frame.iloc[index]
        path = resolve_image_path(self.data_root, str(row.path))
        item: dict[str, object] = {
            "id": str(row.id),
            "category": str(row.category),
            "image": preprocess_image(load_pil_image(path), self.config),
        }
        if "label" in self.frame.columns:
            item["label"] = row.label
        return item
