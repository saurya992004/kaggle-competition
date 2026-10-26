"""Create score-based audit plots for the final DefectVision prediction run."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.phase3_benchmark import load_image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--scores", type=Path, default=Path("outputs/final_prediction_scores.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/final_audit_plots"))
    args = parser.parse_args(); root, output = args.data_root.resolve(), args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    scores = pd.read_csv(args.scores)
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for axis, category in zip(axes.ravel(), sorted(scores.category.unique())):
        subset = scores[scores.category.eq(category)]
        axis.hist(subset.anomaly_score, bins=22, alpha=.75)
        axis.axvline(subset.loo_normal_threshold_95pct.iloc[0], color="crimson", label="LOO 95th normal percentile")
        axis.set_title(category); axis.set_xlabel("anomaly score"); axis.legend(fontsize=7)
    figure.suptitle("Final unlabeled test-score distributions")
    figure.savefig(output / "final_score_distributions.png", dpi=160); plt.close(figure)
    for category in sorted(scores.category.unique()):
        subset = scores[scores.category.eq(category)].copy()
        subset["abs_margin"] = (subset.anomaly_score - subset.loo_normal_threshold_95pct).abs()
        views = [("highest", subset.nlargest(4, "anomaly_score")), ("borderline", subset.nsmallest(4, "abs_margin"))]
        figure, axes = plt.subplots(2, 4, figsize=(10, 5), constrained_layout=True)
        for row_index, (label, entries) in enumerate(views):
            for axis, item in zip(axes[row_index], entries.itertuples(index=False)):
                axis.imshow(load_image(root, item.path, 64))
                axis.set_title(f"{item.id}\nscore={item.anomaly_score:.3f}, y={item.candidate_label_numeric}", fontsize=7)
                axis.axis("off")
            axes[row_index, 0].set_ylabel(label, fontsize=9)
        figure.suptitle(f"{category}: score-based cases, not ground truth")
        figure.savefig(output / f"{category}_highest_and_borderline.png", dpi=160); plt.close(figure)


if __name__ == "__main__": main()
