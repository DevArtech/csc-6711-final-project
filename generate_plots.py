from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

COLORS = {
    "sequential":    "#2196F3",   # blue
    "popularity":    "#FF9800",   # orange
    "static_mf":     "#9E9E9E",   # grey
    "static_mf_bpr": "#E91E63",   # pink/red
    "bayesian_mf":   "#4CAF50",   # green
}
LABELS = {
    "sequential":    "Sequential (GRU)",
    "popularity":    "Popularity",
    "static_mf":     "Static MF (MSE)",
    "static_mf_bpr": "Static MF (BPR)",
    "bayesian_mf":   "Bayesian MF",
}


def _bar_chart(
    ax: plt.Axes,
    models: list[str],
    values: list[float],
    title: str,
    ylabel: str,
    lower_is_better: bool = False,
) -> None:
    bars = ax.bar(
        range(len(models)),
        values,
        color=[COLORS[m] for m in models],
        edgecolor="white",
        linewidth=0.8,
        width=0.55,
    )
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([LABELS[m] for m in models], fontsize=8, rotation=15, ha="right")
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.015,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=8, fontweight="bold",
        )

    label = "lower is better ↓" if lower_is_better else "higher is better ↑"
    ax.text(0.98, 0.97, label, transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color="#888888", style="italic")


def plot_ranking_comparison(summary: dict, bpr: dict, pop_full: dict, output_dir: Path) -> None:
    all_models = ["sequential", "static_mf_bpr", "popularity", "static_mf", "bayesian_mf"]

    hr10_vals = {
        "sequential":    summary["sequential"]["hr@10"],
        "static_mf":     summary["static_mf"]["hr@10"],
        "static_mf_bpr": bpr["static_mf"]["hr@10"],
        "bayesian_mf":   summary["bayesian_mf"]["hr@10"],
        "popularity":    pop_full["hr@10"],
    }
    ndcg10_vals = {
        "sequential":    summary["sequential"]["ndcg@10"],
        "static_mf":     summary["static_mf"]["ndcg@10"],
        "static_mf_bpr": bpr["static_mf"]["ndcg@10"],
        "bayesian_mf":   summary["bayesian_mf"]["ndcg@10"],
        "popularity":    pop_full["ndcg@10"],
    }
    rmse_models = ["static_mf", "bayesian_mf", "popularity"]
    rmse_vals = {
        "static_mf":   summary["static_mf"]["rmse"],
        "bayesian_mf": summary["bayesian_mf"]["rmse"],
        "popularity":  pop_full["rmse"],
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Model Comparison — Full Test Set (102,759 ratings)", fontsize=13, fontweight="bold", y=1.01)

    _bar_chart(axes[0], all_models, [hr10_vals[m] for m in all_models], "HR@10", "Hit Rate @ 10")
    _bar_chart(axes[1], all_models, [ndcg10_vals[m] for m in all_models], "NDCG@10", "NDCG @ 10")
    _bar_chart(axes[2], rmse_models, [rmse_vals[m] for m in rmse_models],
               "RMSE (MSE-trained models only)", "RMSE", lower_is_better=True)

    fig.tight_layout()
    path = output_dir / "ranking_comparison.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def plot_drift_comparison(summary: dict, drift_summary: dict, bpr: dict, bpr_drift: dict,
                          pop_overall: dict, pop_drift: dict, output_dir: Path) -> None:
    models = ["sequential", "static_mf_bpr", "popularity", "static_mf", "bayesian_mf"]

    overall_hr = {
        "sequential":    summary["sequential"]["hr@10"],
        "static_mf":     summary["static_mf"]["hr@10"],
        "static_mf_bpr": bpr["static_mf"]["hr@10"],
        "bayesian_mf":   summary["bayesian_mf"]["hr@10"],
        "popularity":    pop_overall["hr@10"],
    }
    drift_hr = {
        "sequential":    drift_summary["sequential"]["hr@10"],
        "static_mf":     drift_summary["static_mf"]["hr@10"],
        "static_mf_bpr": bpr_drift["static_mf"]["hr@10"],
        "bayesian_mf":   drift_summary["bayesian_mf"]["hr@10"],
        "popularity":    pop_drift["hr@10"],
    }

    x = np.arange(len(models))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 5))

    bars1 = ax.bar(x - width/2, [overall_hr[m] for m in models], width,
                   label="All users", color=[COLORS[m] for m in models],
                   edgecolor="white", alpha=0.9)
    bars2 = ax.bar(x + width/2, [drift_hr[m] for m in models], width,
                   label="Drift users only", color=[COLORS[m] for m in models],
                   edgecolor="white", alpha=0.5, hatch="///")

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[m] for m in models], fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("HR@10", fontsize=11)
    ax.set_title("HR@10: All Users vs Drift Users\n(Drift users had preferences swapped mid-stream)",
                 fontsize=11, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend(fontsize=10)

    for bar in list(bars1) + list(bars2):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=7.5)

    fig.tight_layout()
    path = output_dir / "drift_comparison.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def plot_streaming_curves(curves: dict, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.suptitle("Model Performance Over Stream (HR@10 and NDCG@10 at each checkpoint)",
                 fontsize=12, fontweight="bold", y=1.01)

    stream_models = ["sequential", "static_mf", "bayesian_mf"]
    metrics = [("hr10", "HR@10"), ("ndcg10", "NDCG@10")]

    for ax, (metric_key, metric_label) in zip(axes, metrics):
        for m in stream_models:
            points = curves.get(m, [])
            if not points:
                continue
            x = [p["stream_step"] / 1000 for p in points]
            y = [p[metric_key] for p in points]
            ax.plot(x, y, marker="o", markersize=4, linewidth=2,
                    color=COLORS[m], label=LABELS[m])

        ax.set_xlabel("Stream Steps (thousands)", fontsize=10)
        ax.set_ylabel(metric_label, fontsize=10)
        ax.set_title(f"{metric_label} Over Stream", fontsize=11, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.3, linestyle="--")
        ax.legend(fontsize=9)

    fig.tight_layout()
    path = output_dir / "streaming_curves.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def plot_rmse_mae(summary: dict, pop_overall: dict, output_dir: Path) -> None:
    models = ["static_mf", "bayesian_mf", "popularity"]
    vals = {m: summary[m] for m in ["static_mf", "bayesian_mf"]}
    vals["popularity"] = {"rmse": pop_overall["rmse"], "mae": pop_overall["mae"]}

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    fig.suptitle("Rating Prediction Error (MSE-trained models only)", fontsize=12, fontweight="bold")

    _bar_chart(axes[0], models, [vals[m]["rmse"] for m in models],
               "RMSE", "RMSE", lower_is_better=True)
    _bar_chart(axes[1], models, [vals[m]["mae"] for m in models],
               "MAE", "MAE", lower_is_better=True)

    fig.tight_layout()
    path = output_dir / "rating_error.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def plot_bpr_vs_mse(summary: dict, bpr: dict, pop_full: dict, output_dir: Path) -> None:
    models = ["static_mf", "static_mf_bpr", "popularity"]
    hr10_vals = {
        "static_mf":     summary["static_mf"]["hr@10"],
        "static_mf_bpr": bpr["static_mf"]["hr@10"],
        "popularity":    pop_full["hr@10"],
    }
    ndcg_vals = {
        "static_mf":     summary["static_mf"]["ndcg@10"],
        "static_mf_bpr": bpr["static_mf"]["ndcg@10"],
        "popularity":    pop_full["ndcg@10"],
    }

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.suptitle("Effect of Training Objective: MSE vs BPR on Static MF",
                 fontsize=12, fontweight="bold")

    _bar_chart(axes[0], models, [hr10_vals[m] for m in models], "HR@10", "Hit Rate @ 10")
    _bar_chart(axes[1], models, [ndcg_vals[m] for m in models], "NDCG@10", "NDCG @ 10")

    fig.tight_layout()
    path = output_dir / "bpr_vs_mse.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def plot_bayesian_tuning(tuning_results: dict, output_dir: Path) -> None:
    configs = ["f=1.00 (baseline)", "f=0.98", "f=0.95", "f=0.95 nv=0.25", "f=0.90"]
    hr10   = [tuning_results[c]["hr10"]  for c in configs]
    ndcg10 = [tuning_results[c]["ndcg10"] for c in configs]

    x = np.arange(len(configs))
    width = 0.35
    fig, ax = plt.subplots(figsize=(11, 5))

    bars1 = ax.bar(x - width/2, hr10,   width, label="HR@10",   color="#4CAF50", alpha=0.9)
    bars2 = ax.bar(x + width/2, ndcg10, width, label="NDCG@10", color="#4CAF50", alpha=0.5, hatch="///")

    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=10)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Bayesian MF — Forgetting Factor Tuning\n(higher is better)",
                 fontsize=11, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend(fontsize=10)

    for bar in list(bars1) + list(bars2):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.axvline(x=0.5, color="#888888", linestyle="--", linewidth=1, alpha=0.5)
    ax.text(0.5, ax.get_ylim()[1] * 0.97, "← best config", ha="center",
            fontsize=8, color="#888888", style="italic")

    fig.tight_layout()
    path = output_dir / "bayesian_tuning.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def plot_ranking_original(summary: dict, pop_full: dict, drift: dict, pop_drift: dict, output_dir: Path) -> None:
    """Ranking comparison with original 4 models only (no BPR)."""
    all_models = ["sequential", "popularity", "static_mf", "bayesian_mf"]

    hr10_vals = {
        "sequential":  summary["sequential"]["hr@10"],
        "static_mf":   summary["static_mf"]["hr@10"],
        "bayesian_mf": summary["bayesian_mf"]["hr@10"],
        "popularity":  pop_full["hr@10"],
    }
    ndcg10_vals = {
        "sequential":  summary["sequential"]["ndcg@10"],
        "static_mf":   summary["static_mf"]["ndcg@10"],
        "bayesian_mf": summary["bayesian_mf"]["ndcg@10"],
        "popularity":  pop_full["ndcg@10"],
    }
    rmse_models = ["static_mf", "bayesian_mf", "popularity"]
    rmse_vals = {
        "static_mf":   summary["static_mf"]["rmse"],
        "bayesian_mf": summary["bayesian_mf"]["rmse"],
        "popularity":  pop_full["rmse"],
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle("Model Comparison — Full Test Set (102,759 ratings)", fontsize=13, fontweight="bold", y=1.01)

    _bar_chart(axes[0], all_models, [hr10_vals[m] for m in all_models], "HR@10", "Hit Rate @ 10")
    _bar_chart(axes[1], all_models, [ndcg10_vals[m] for m in all_models], "NDCG@10", "NDCG @ 10")
    _bar_chart(axes[2], rmse_models, [rmse_vals[m] for m in rmse_models],
               "RMSE (rating prediction)", "RMSE", lower_is_better=True)

    fig.tight_layout()
    path = output_dir / "ranking_comparison_original.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def _drift_side_by_side(ax: plt.Axes, models: list[str], overall: dict, drift: dict,
                        metric: str, ylabel: str, title: str) -> None:
    x = np.arange(len(models))
    width = 0.38
    bars1 = ax.bar(x - width/2, [overall[m] for m in models], width,
                   label="All users", color=[COLORS[m] for m in models],
                   edgecolor="white", alpha=0.9)
    bars2 = ax.bar(x + width/2, [drift[m] for m in models], width,
                   label="Drift users only", color=[COLORS[m] for m in models],
                   edgecolor="white", alpha=0.5, hatch="///")
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[m] for m in models], fontsize=10)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend(fontsize=10)
    for bar in list(bars1) + list(bars2):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=8)


def plot_drift_original(summary: dict, drift: dict, pop_full: dict, pop_drift: dict, output_dir: Path) -> None:
    """Drift comparison with original 4 models only (no BPR) — HR@10 and NDCG@10."""
    models = ["sequential", "popularity", "static_mf", "bayesian_mf"]

    overall_hr   = {m: summary[m]["hr@10"]   if m != "popularity" else pop_full["hr@10"]   for m in models}
    overall_ndcg = {m: summary[m]["ndcg@10"] if m != "popularity" else pop_full["ndcg@10"] for m in models}
    drift_hr     = {m: drift[m]["hr@10"]     if m != "popularity" else pop_drift["hr@10"]   for m in models}
    drift_ndcg   = {m: drift[m]["ndcg@10"]   if m != "popularity" else pop_drift["ndcg@10"] for m in models}

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("All Users vs Drift Users\n(Drift users had preferences swapped mid-stream)",
                 fontsize=12, fontweight="bold", y=1.02)

    _drift_side_by_side(axes[0], models, overall_hr,   drift_hr,   "hr@10",   "HR@10",   "HR@10")
    _drift_side_by_side(axes[1], models, overall_ndcg, drift_ndcg, "ndcg@10", "NDCG@10", "NDCG@10")

    fig.tight_layout()
    path = output_dir / "drift_comparison_original.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def main() -> None:
    base = Path(__file__).resolve().parent.parent / "recsys/runs"
    proj = Path(__file__).resolve().parent
    output_dir = base / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary       = json.loads((base / "compare/summary.json").read_text())
    drift         = json.loads((base / "compare/drift_subset_summary.json").read_text())
    curves        = json.loads((base / "compare/streaming_curves.json").read_text())
    pop_full      = json.loads((base / "popularity/summary.json").read_text())["popularity"]
    pop_drift     = json.loads((base / "popularity/drift_subset_summary.json").read_text())["popularity"]

    bpr_base  = proj / "runs/compare_bpr_fast"
    bpr       = json.loads((bpr_base / "summary.json").read_text())
    bpr_drift = json.loads((bpr_base / "drift_subset_summary.json").read_text())

    tuning_results = {
        "f=1.00 (baseline)": {"hr10": summary["bayesian_mf"]["hr@10"],  "ndcg10": summary["bayesian_mf"]["ndcg@10"]},
        "f=0.98":  {"hr10": 0.20895, "ndcg10": 0.10953},
        "f=0.95":  {"hr10": 0.19923, "ndcg10": 0.10449},
        "f=0.95 nv=0.25": {"hr10": 0.19007, "ndcg10": 0.09849},
        "f=0.90":  {"hr10": 0.18775, "ndcg10": 0.09781},
    }

    plot_ranking_original(summary, pop_full, drift, pop_drift, output_dir)
    plot_drift_original(summary, drift, pop_full, pop_drift, output_dir)
    plot_ranking_comparison(summary, bpr, pop_full, output_dir)
    plot_drift_comparison(summary, drift, bpr, bpr_drift, pop_full, pop_drift, output_dir)
    plot_streaming_curves(curves, output_dir)
    plot_rmse_mae(summary, pop_full, output_dir)
    plot_bpr_vs_mse(summary, bpr, pop_full, output_dir)
    plot_bayesian_tuning(tuning_results, output_dir)
    print(f"\nAll plots written to {output_dir}")


if __name__ == "__main__":
    main()
