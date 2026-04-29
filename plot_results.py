from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def generate_plots(
    summary: dict[str, dict[str, float]],
    curves: dict[str, list[dict[str, float | int | None]]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    models = list(summary.keys())
    hr10 = [float(summary[m].get("hr@10", 0.0)) for m in models]
    ndcg10 = [float(summary[m].get("ndcg@10", 0.0)) for m in models]

    fig, ax = plt.subplots(figsize=(8, 4))
    x = list(range(len(models)))
    ax.bar([i - 0.18 for i in x], hr10, width=0.35, label="HR@10")
    ax.bar([i + 0.18 for i in x], ndcg10, width=0.35, label="NDCG@10")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Head-to-Head Ranking Metrics")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "head_to_head_bar.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    for model_name, points in curves.items():
        if not points:
            continue
        x_steps = [int(p["stream_step"]) for p in points]
        y = [float(p["ndcg10"]) for p in points]
        ax.plot(x_steps, y, marker="o", label=model_name)
    ax.set_xlabel("Stream Step")
    ax.set_ylabel("NDCG@10")
    ax.set_title("Streaming NDCG@10")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "streaming_ndcg10.png", dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot recommender comparison outputs.")
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--curve-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.summary_json.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    with args.curve_json.open("r", encoding="utf-8") as handle:
        curves = json.load(handle)
    generate_plots(summary=summary, curves=curves, output_dir=args.output_dir)
    print(f"Wrote plots to {args.output_dir}")


if __name__ == "__main__":
    main()
