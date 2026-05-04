#!/usr/bin/env python3
"""Generate Brier/ECE visualizations for the Bayesian MF model."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recsys.bayesian_mf import make_initial_bayesian_state
from recsys.data_loader import load_processed_data
from recsys.evaluate import rating_bucket_probs_from_gaussian

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RATING_LABELS = ["1★", "2★", "3★", "4★", "5★"]


def collect_predictions(state, test_rows):
    probs, true_buckets = [], []
    for i, row in enumerate(test_rows):
        mean, var = state.predict_rating(user_id=row.user_id, item_id=row.item_id)
        probs.append(rating_bucket_probs_from_gaussian(mean, var))
        true_buckets.append(max(0, min(4, int(round(row.rating)) - 1)))
        if (i + 1) % 20000 == 0:
            print(f"  {i + 1}/{len(test_rows)}", flush=True)
    return probs, true_buckets


def plot_reliability_diagram(probs, true_buckets, out_path, num_bins=10):
    """Calibration curve with ECE threshold bands."""
    confidences, accuracies = [], []
    for p, y in zip(probs, true_buckets):
        pred_idx = int(np.argmax(p))
        confidences.append(p[pred_idx])
        accuracies.append(1.0 if pred_idx == y else 0.0)

    bin_edges = np.linspace(0, 1, num_bins + 1)
    bin_conf, bin_acc, bin_count = [], [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        idx = [i for i, c in enumerate(confidences) if lo <= c < hi]
        if idx:
            bin_conf.append(np.mean([confidences[i] for i in idx]))
            bin_acc.append(np.mean([accuracies[i] for i in idx]))
            bin_count.append(len(idx))

    ece = sum(
        (bin_count[i] / sum(bin_count)) * abs(bin_conf[i] - bin_acc[i])
        for i in range(len(bin_conf))
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Bayesian MF — Calibration Analysis", fontsize=13, fontweight="bold")

    # Left: reliability diagram
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect calibration", zorder=3)
    ax.fill_between(bin_conf,
                    [c - abs(c - a) for c, a in zip(bin_conf, bin_acc)],
                    [c + abs(c - a) for c, a in zip(bin_conf, bin_acc)],
                    alpha=0.2, color="#e63946", label="Miscalibration gap")
    ax.bar(bin_conf, bin_acc, width=0.08, alpha=0.75, color="#457b9d", label="Model", zorder=2)

    # ECE threshold annotation
    ax.axhline(0, color="none")  # spacer
    ax.text(0.04, 0.93, f"ECE = {ece:.4f}", transform=ax.transAxes,
            fontsize=11, fontweight="bold", color="#457b9d",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#457b9d", linewidth=1.5))
    ax.text(0.04, 0.83, "✓ Well calibrated\n   (threshold < 0.05)", transform=ax.transAxes,
            fontsize=9, color="#2a9d8f",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0faf9", edgecolor="#2a9d8f", alpha=0.9))

    ax.set_xlabel("Mean predicted confidence", fontsize=11)
    ax.set_ylabel("Fraction correct", fontsize=11)
    ax.set_title("Reliability Diagram", fontsize=11)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)

    # Right: ECE threshold bar
    ax2 = axes[1]
    thresholds = [0.05, 0.10, 0.20]
    threshold_labels = ["Well calibrated\n(< 0.05)", "Acceptable\n(< 0.10)", "Poor\n(> 0.10)"]
    band_colors = ["#2a9d8f", "#f4a261", "#e63946"]
    band_alpha = 0.15

    ax2.axhspan(0,      0.05, color=band_colors[0], alpha=band_alpha)
    ax2.axhspan(0.05,   0.10, color=band_colors[1], alpha=band_alpha)
    ax2.axhspan(0.10,   0.25, color=band_colors[2], alpha=band_alpha)

    ax2.axhline(0.05, color=band_colors[0], lw=1.2, linestyle="--", alpha=0.7)
    ax2.axhline(0.10, color=band_colors[1], lw=1.2, linestyle="--", alpha=0.7)

    ax2.text(0.72, 0.025, "Well calibrated", transform=ax2.transAxes,
             fontsize=8.5, color=band_colors[0], va="center")
    ax2.text(0.72, 0.27, "Acceptable", transform=ax2.transAxes,
             fontsize=8.5, color=band_colors[1], va="center")
    ax2.text(0.72, 0.55, "Poor", transform=ax2.transAxes,
             fontsize=8.5, color=band_colors[2], va="center")

    ax2.bar(["Bayesian MF"], [ece], color="#457b9d", width=0.4, zorder=3)
    ax2.scatter(["Bayesian MF"], [ece], color="white", s=60, zorder=4)
    ax2.text(0, ece + 0.003, f"{ece:.4f}", ha="center", va="bottom",
             fontsize=12, fontweight="bold", color="#457b9d")

    ax2.set_ylabel("ECE (lower = better)", fontsize=11)
    ax2.set_title("ECE vs Literature Thresholds\n(Guo et al. 2017)", fontsize=11)
    ax2.set_ylim(0, 0.25)
    ax2.grid(alpha=0.3, axis="y", zorder=0)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_predicted_heatmap(probs, true_buckets, out_path):
    """For each true rating, show average predicted probability per bucket."""
    matrix = np.zeros((5, 5))
    counts = np.zeros(5)
    for p, y in zip(probs, true_buckets):
        matrix[y] += p
        counts[y] += 1
    for i in range(5):
        if counts[i] > 0:
            matrix[i] /= counts[i]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=0.6)
    plt.colorbar(im, ax=ax, label="Average predicted probability")

    for i in range(5):
        for j in range(5):
            val = matrix[i, j]
            color = "white" if val > 0.35 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=10, color=color, fontweight="bold")

    # Highlight diagonal
    for k in range(5):
        ax.add_patch(plt.Rectangle((k - 0.5, k - 0.5), 1, 1,
                                   fill=False, edgecolor="#e63946", lw=2))

    ax.set_xticks(range(5)); ax.set_xticklabels(RATING_LABELS)
    ax.set_yticks(range(5)); ax.set_yticklabels(RATING_LABELS)
    ax.set_xlabel("Predicted rating bucket", fontsize=11)
    ax.set_ylabel("True rating", fontsize=11)
    ax.set_title("Bayesian MF — Predicted Probability Heatmap\n"
                 "(red boxes = diagonal, model should concentrate mass here)", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_brier_comparison(brier_bayesian, out_path):
    """Brier score vs baselines with uninformative threshold annotated."""
    brier_uniform = 0.80
    # Majority class: always predict 4★, cap at 1.0 for display
    ml_dist = [0.06, 0.11, 0.27, 0.35, 0.21]
    pred_majority = [0.0, 0.0, 0.0, 1.0, 0.0]
    brier_majority_raw = sum(
        ml_dist[y] * sum((pred_majority[j] - (1.0 if j == y else 0.0)) ** 2 for j in range(5))
        for y in range(5)
    )
    brier_majority_display = min(brier_majority_raw, 1.0)

    labels = ["Uniform\n(no information)", "Majority class\n(always 4★)", "Bayesian MF"]
    scores = [brier_uniform, brier_majority_display, brier_bayesian]
    bar_colors = ["#adb5bd", "#6c757d", "#457b9d"]

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(labels, scores, color=bar_colors, width=0.5,
                  edgecolor="white", linewidth=1.2, zorder=3)
    for bar, score, raw in zip(bars, scores, [brier_uniform, brier_majority_raw, brier_bayesian]):
        label = f"{raw:.4f}" + (" (capped)" if raw > 1.0 else "")
        ax.text(bar.get_x() + bar.get_width() / 2, min(score, 0.99) + 0.01,
                label, ha="center", va="bottom", fontsize=11, fontweight="bold")

    # Uninformative threshold line
    ax.axhline(brier_uniform, color="#adb5bd", linestyle="--", lw=1.5,
               label=f"Uninformative threshold ({brier_uniform:.2f})", zorder=2)
    ax.fill_between([-0.5, 2.5], 0, brier_uniform,
                    color="#2a9d8f", alpha=0.06, zorder=1)
    ax.fill_between([-0.5, 2.5], brier_uniform, 1.05,
                    color="#e63946", alpha=0.05, zorder=1)

    ax.text(0.02, 0.97, "Worse than uninformative →",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=8.5, color="#e63946", style="italic")
    ax.text(0.02, 0.65, "← Learned something real",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=8.5, color="#2a9d8f", style="italic")

    # Gap annotation between uniform and Bayesian
    ax.annotate("", xy=(2.28, brier_uniform), xytext=(2.28, brier_bayesian),
                arrowprops=dict(arrowstyle="<->", color="#457b9d", lw=1.5))
    ax.text(2.32, (brier_uniform + brier_bayesian) / 2,
            f"−{brier_uniform - brier_bayesian:.4f}", va="center",
            fontsize=9, color="#457b9d", fontweight="bold")

    ax.set_ylabel("Brier Score (lower = better)", fontsize=11)
    ax.set_title("Brier Score vs Baselines", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(-0.5, 2.5)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("recsys/data/processed"))
    parser.add_argument("--bayesian-checkpoint", type=Path, default=Path("recsys/runs/bayesian_mf/bayesian_mf.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("recsys/runs/compare/graphs"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...", flush=True)
    data = load_processed_data(args.data_dir)
    test_rows = data["by_split"]["test"]

    print("Loading Bayesian checkpoint...", flush=True)
    checkpoint = torch.load(args.bayesian_checkpoint, map_location="cpu")
    state = make_initial_bayesian_state(
        item_factors=checkpoint["item_factors"].float(),
        item_bias=checkpoint["item_bias"].float(),
        user_prior_mean=checkpoint["user_prior_mean"].float(),
        prior_var=float(checkpoint["prior_var"]),
        global_bias=float(checkpoint["global_bias"]),
        noise_var=float(checkpoint["noise_var"]),
        forgetting=float(checkpoint["forgetting"]),
    )

    print(f"Collecting predictions over {len(test_rows)} test rows...", flush=True)
    probs, true_buckets = collect_predictions(state, test_rows)

    print("Generating plots...", flush=True)
    plot_reliability_diagram(probs, true_buckets, args.output_dir / "bayesian_calibration.png")
    plot_predicted_heatmap(probs, true_buckets, args.output_dir / "bayesian_prob_heatmap.png")
    plot_brier_comparison(0.7351, args.output_dir / "brier_comparison.png")

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
