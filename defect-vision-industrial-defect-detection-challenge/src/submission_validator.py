"""Independent strict validator for DefectVision submission CSVs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def validate(data_root: Path, submission_path: Path, score_path: Path | None = None) -> dict[str, object]:
    test, sample, submission = (pd.read_csv(data_root / "test.csv"), pd.read_csv(data_root / "sample_submission.csv"), pd.read_csv(submission_path))
    expected_columns = list(sample.columns)
    issues: list[str] = []
    if list(submission.columns) != expected_columns: issues.append(f"columns={submission.columns.tolist()}, expected={expected_columns}")
    if len(submission) != len(test): issues.append(f"rows={len(submission)}, expected={len(test)}")
    if not submission.id.is_unique: issues.append("duplicate submission IDs")
    if submission.isna().any().any(): issues.append("missing submission values")
    if not submission.id.equals(test.id): issues.append("IDs do not exactly match test order")
    # The actual downloaded template and training metadata show numeric 0; 1 is
    # the only inferred counterpart and is recorded as a contract caveat.
    if not set(submission.label.dropna().unique()).issubset({0, 1}): issues.append("labels outside numeric {0,1}")
    if score_path is not None:
        scores = pd.read_csv(score_path)
        if len(scores) != len(test) or not scores.id.equals(test.id): issues.append("score audit file does not trace test order")
        elif not np.isfinite(scores.anomaly_score.to_numpy(dtype=float)).all(): issues.append("non-finite anomaly score")
        elif not np.array_equal(scores.candidate_label_numeric.to_numpy(), submission.label.to_numpy()): issues.append("score labels do not match submission")
    return {"valid": not issues, "issues": issues, "rows": len(submission), "columns": list(submission.columns),
            "label_counts": {str(key): int(value) for key, value in submission.label.value_counts().sort_index().items()}}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--data-root", type=Path, default=Path(".")); parser.add_argument("--submission", type=Path, default=Path("submission.csv")); parser.add_argument("--scores", type=Path, default=Path("outputs/final_prediction_scores.csv"))
    args = parser.parse_args(); result = validate(args.data_root.resolve(), args.submission.resolve(), args.scores.resolve())
    print(json.dumps(result, indent=2))
    if not result["valid"]: raise SystemExit(1)


if __name__ == "__main__": main()
