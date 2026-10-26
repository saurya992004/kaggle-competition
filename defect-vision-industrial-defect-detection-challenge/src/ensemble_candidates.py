"""Rank-average anomaly scores from independently pretrained feature extractors."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.candidate_sweep import make_candidate


def ensemble_scores(first: pd.DataFrame, second: pd.DataFrame) -> pd.DataFrame:
    """Average category-normalized ranks while preserving the original test order."""
    required = {"id", "category", "rank_resnet18_spatial"}
    if missing := required.difference(first.columns):
        raise ValueError(f"first score file missing: {sorted(missing)}")
    required = {"id", "category", "rank_resnet50_spatial"}
    if missing := required.difference(second.columns):
        raise ValueError(f"second score file missing: {sorted(missing)}")
    merged = first[["id", "category", "rank_resnet18_spatial"]].merge(
        second[["id", "category", "rank_resnet50_spatial"]], on=["id", "category"], how="inner", validate="one_to_one"
    )
    if len(merged) != len(first) or not merged.id.equals(first.id):
        raise ValueError("score files must contain the same IDs in the same order")
    merged["score_spatial_rank_ensemble"] = (merged.rank_resnet18_spatial + merged.rank_resnet50_spatial) / 2.0
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resnet18", type=Path, required=True)
    parser.add_argument("--resnet50", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rates", type=float, nargs="+", default=[.70, .75, .80])
    args = parser.parse_args()
    combined = ensemble_scores(pd.read_csv(args.resnet18), pd.read_csv(args.resnet50))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output_dir / "ensemble_test_scores.csv", index=False)
    for rate in args.rates:
        candidate = make_candidate(combined, "score_spatial_rank_ensemble", rate)
        path = args.output_dir / f"candidate_resnet18_resnet50_spatial_ensemble_top_{int(rate * 100):02d}pct.csv"
        candidate.to_csv(path, index=False)
        print(f"{path.name}: {int(candidate.label.sum())}/{len(candidate)} defective")


if __name__ == "__main__":
    main()
