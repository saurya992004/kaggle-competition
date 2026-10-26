"""Generate additional threshold candidates from saved, reproducible anomaly scores."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def make_candidate(scores: pd.DataFrame, score_column: str, defect_rate: float) -> pd.DataFrame:
    """Mark the highest-scoring fraction within each product category as defective."""
    if not 0.0 < defect_rate < 1.0:
        raise ValueError("defect_rate must lie strictly between 0 and 1")
    required = {"id", "category", score_column}
    if missing := required.difference(scores.columns):
        raise ValueError(f"missing required columns: {sorted(missing)}")

    labels = pd.Series(0, index=scores.index, dtype="int64")
    for _, group in scores.groupby("category", sort=False):
        count = round(len(group) * defect_rate)
        selected = group.nlargest(count, score_column).index
        labels.loc[selected] = 1
    return pd.DataFrame({"id": scores["id"], "label": labels})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method", default="resnet18_spatial")
    parser.add_argument("--rates", type=float, nargs="+", required=True)
    args = parser.parse_args()

    scores = pd.read_csv(args.scores)
    score_column = f"score_{args.method}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for rate in args.rates:
        candidate = make_candidate(scores, score_column, rate)
        destination = args.output_dir / f"candidate_{args.method}_top_{int(rate * 100):02d}pct.csv"
        candidate.to_csv(destination, index=False)
        print(f"{destination.name}: {int(candidate.label.sum())}/{len(candidate)} defective")


if __name__ == "__main__":
    main()
