#!/usr/bin/env python3
"""Compute Brier score and ECE for the Bayesian MF model on the test split."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recsys.bayesian_mf import make_initial_bayesian_state
from recsys.data_loader import load_processed_data
from recsys.evaluate import brier_and_ece_from_probabilities, rating_bucket_probs_from_gaussian


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--bayesian-checkpoint", type=Path, default=Path("runs/bayesian_mf/bayesian_mf.pt"))
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    print(f"[brier_ece] loading data from {args.data_dir}", flush=True)
    data = load_processed_data(args.data_dir)
    test_rows = data["by_split"]["test"]
    print(f"[brier_ece] test rows: {len(test_rows)}", flush=True)

    print(f"[brier_ece] loading Bayesian checkpoint from {args.bayesian_checkpoint}", flush=True)
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

    print("[brier_ece] computing Brier/ECE over test set...", flush=True)
    probs: list[list[float]] = []
    true_buckets: list[int] = []
    for i, row in enumerate(test_rows):
        mean, var = state.predict_rating(user_id=row.user_id, item_id=row.item_id)
        probs.append(rating_bucket_probs_from_gaussian(mean, var))
        true_buckets.append(max(0, min(4, int(round(row.rating)) - 1)))
        if (i + 1) % 20000 == 0:
            print(f"[brier_ece]   {i + 1}/{len(test_rows)}", flush=True)

    result = brier_and_ece_from_probabilities(probs, true_buckets)
    print(f"\n[brier_ece] brier={result['brier']:.6f}  ece={result['ece']:.6f}")

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w") as f:
            json.dump(result, f, indent=2)
        print(f"[brier_ece] saved to {args.output_json}")


if __name__ == "__main__":
    main()
