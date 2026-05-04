from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from recsys.bayesian_mf import make_initial_bayesian_state
from recsys.data_loader import Interaction, load_processed_data
from recsys.evaluate import evaluate_ranking_and_rating
from recsys.plot_results import generate_plots
from recsys.sequential_model import SequentialRecConfig, SequentialRecommender
from recsys.streaming_eval import run_streaming_evaluation


class StaticRuntime:
    def __init__(self, checkpoint_path: Path) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state = checkpoint["state"]
        self.global_bias = float(state["global_bias"])
        self.user_factors = state["user_factors"].float()
        self.item_factors = state["item_factors"].float()
        self.user_bias = state["user_bias"].float()
        self.item_bias = state["item_bias"].float()

    def update(self, row: Interaction) -> None:
        _ = row

    def score_items(self, user_id: int, item_ids: torch.Tensor) -> torch.Tensor:
        p = self.user_factors[user_id]
        return (
            self.global_bias
            + self.user_bias[user_id]
            + self.item_bias[item_ids]
            + (self.item_factors[item_ids] @ p)
        )

    def predict_rating(self, user_id: int, item_id: int) -> float:
        return float(
            self.global_bias
            + self.user_bias[user_id]
            + self.item_bias[item_id]
            + torch.dot(self.user_factors[user_id], self.item_factors[item_id]).item()
        )


class SequentialRuntime:
    def __init__(self, checkpoint_path: Path, warm_rows: list[Interaction]) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        cfg = checkpoint["config"]
        self.model = SequentialRecommender(
            SequentialRecConfig(
                num_items=int(cfg["num_items"]),
                num_genres=int(cfg["num_genres"]),
                item_embedding_dim=int(cfg["item_embedding_dim"]),
                hidden_dim=int(cfg["hidden_dim"]),
            )
        )
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

        self.history_items: dict[int, list[int]] = defaultdict(list)
        self.history_ratings: dict[int, list[float]] = defaultdict(list)
        for row in warm_rows:
            self.history_items[row.user_id].append(row.item_id)
            self.history_ratings[row.user_id].append(row.rating)

    def update(self, row: Interaction) -> None:
        self.history_items[row.user_id].append(row.item_id)
        self.history_ratings[row.user_id].append(row.rating)

    def score_items(self, user_id: int, item_ids: torch.Tensor) -> torch.Tensor:
        items = self.history_items.get(user_id, [])
        ratings = self.history_ratings.get(user_id, [])
        if not items:
            return torch.zeros_like(item_ids, dtype=torch.float32)
        hist_items = torch.tensor(items, dtype=torch.long)
        hist_ratings = torch.tensor(ratings, dtype=torch.float32)
        return self.model.score_candidates(hist_items, hist_ratings, item_ids)

    def predict_rating(self, user_id: int, item_id: int) -> float:
        score = self.score_items(user_id, torch.tensor([item_id], dtype=torch.long))[0].item()
        # Map latent score to MovieLens rating range.
        return float(1.0 + 4.0 * torch.sigmoid(torch.tensor(score)).item())


class BayesianRuntime:
    def __init__(self, checkpoint_path: Path) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.state = make_initial_bayesian_state(
            item_factors=checkpoint["item_factors"].float(),
            item_bias=checkpoint["item_bias"].float(),
            user_prior_mean=checkpoint["user_prior_mean"].float(),
            prior_var=float(checkpoint["prior_var"]),
            global_bias=float(checkpoint["global_bias"]),
            noise_var=float(checkpoint["noise_var"]),
            forgetting=float(checkpoint["forgetting"]),
        )

    def update(self, row: Interaction) -> None:
        self.state.update(user_id=row.user_id, item_id=row.item_id, rating=row.rating, user_bias=0.0)

    def score_items(self, user_id: int, item_ids: torch.Tensor) -> torch.Tensor:
        return self.state.score_items(user_id=user_id, item_ids=item_ids)

    def predict_rating(self, user_id: int, item_id: int) -> float:
        mean, _var = self.state.predict_rating(user_id=user_id, item_id=item_id)
        return mean

    def predict_rating_with_uncertainty(self, user_id: int, item_id: int) -> tuple[float, float]:
        return self.state.predict_rating(user_id=user_id, item_id=item_id)


def select_drift_users(drift_meta_path: Path) -> set[int]:
    if not drift_meta_path.exists():
        return set()
    with drift_meta_path.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)
    return {int(x) for x in meta.get("actual_drift_users", [])}


def per_user_csv(
    output_path: Path,
    runtimes: dict[str, object],
    test_rows: list[Interaction],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model", "user_id", "item_id", "true_rating", "pred_rating", "abs_error"],
        )
        writer.writeheader()
        for model_name, runtime in runtimes.items():
            for row in test_rows:
                pred = float(runtime.predict_rating(row.user_id, row.item_id))
                writer.writerow(
                    {
                        "model": model_name,
                        "user_id": row.user_id,
                        "item_id": row.item_id,
                        "true_rating": row.rating,
                        "pred_rating": pred,
                        "abs_error": abs(pred - row.rating),
                    }
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Head-to-head comparison of recommender variants.")
    parser.add_argument("--data-dir", type=Path, default=Path("recsys/data/processed"))
    parser.add_argument("--interactions-file", type=str, default="interactions.csv")
    parser.add_argument("--drift-meta", type=Path, default=Path("recsys/data/processed/drift_meta.json"))
    parser.add_argument("--static-checkpoint", type=Path, required=True)
    parser.add_argument("--sequential-checkpoint", type=Path, required=True)
    parser.add_argument("--bayesian-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("recsys/runs/compare"))
    parser.add_argument("--eval-every", type=int, default=20000)
    parser.add_argument("--n-negatives", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--models", type=str, default=None,
        help="Comma-separated subset of models to evaluate, e.g. 'bayesian_mf' or 'static_mf,bayesian_mf'. "
             "Defaults to all three.",
    )
    return parser.parse_args()


def main() -> None:
    t0 = time.time()
    args = parse_args()
    print(f"[compare] starting with args={vars(args)}", flush=True)
    bundle = load_processed_data(args.data_dir, interactions_filename=args.interactions_file)
    metadata = bundle["metadata"]
    by_split = bundle["by_split"]
    warm_rows: list[Interaction] = by_split.get("warm", [])
    stream_rows: list[Interaction] = by_split.get("stream", [])
    test_rows: list[Interaction] = by_split.get("test", [])

    base_seen_by_user: dict[int, set[int]] = defaultdict(set)
    for row in warm_rows:
        base_seen_by_user[row.user_id].add(row.item_id)

    enabled = {m.strip() for m in args.models.split(",")} if args.models else {"static_mf", "sequential", "bayesian_mf"}
    all_runtimes = {
        "static_mf":   lambda: StaticRuntime(args.static_checkpoint),
        "sequential":  lambda: SequentialRuntime(args.sequential_checkpoint, warm_rows=warm_rows),
        "bayesian_mf": lambda: BayesianRuntime(args.bayesian_checkpoint),
    }
    runtimes = {name: factory() for name, factory in all_runtimes.items() if name in enabled}
    print(f"[compare] evaluating models: {list(runtimes.keys())}", flush=True)
    print(
        f"[compare] loaded data: warm={len(warm_rows)} stream={len(stream_rows)} test={len(test_rows)} "
        f"num_items={metadata['num_items']}",
        flush=True,
    )

    print("[compare] running streaming evaluation...", flush=True)
    curves = run_streaming_evaluation(
        runtimes=runtimes,
        stream_rows=stream_rows,
        test_rows=test_rows,
        base_seen_items_by_user=base_seen_by_user,
        num_items=int(metadata["num_items"]),
        eval_every=args.eval_every,
        ks=(5, 10, 20),
        n_negatives=args.n_negatives,
        seed=args.seed,
    )
    print("[compare] finished streaming evaluation", flush=True)

    summary: dict[str, dict[str, float]] = {}
    for idx, (name, runtime) in enumerate(runtimes.items()):
        print(f"[compare] computing final summary for model={name}", flush=True)
        seen = defaultdict(set, {u: set(items) for u, items in base_seen_by_user.items()})
        for row in stream_rows:
            seen[row.user_id].add(row.item_id)
        uncertainty_fn = getattr(runtime, "predict_rating_with_uncertainty", None)
        result = evaluate_ranking_and_rating(
            runtime=runtime,
            test_rows=test_rows,
            seen_items_by_user=seen,
            num_items=int(metadata["num_items"]),
            ks=(5, 10, 20),
            n_negatives=args.n_negatives,
            seed=args.seed + 777 + idx,
            uncertainty_fn=uncertainty_fn,
        )
        summary[name] = {
            "hr@5": result.hr[5],
            "hr@10": result.hr[10],
            "hr@20": result.hr[20],
            "ndcg@5": result.ndcg[5],
            "ndcg@10": result.ndcg[10],
            "ndcg@20": result.ndcg[20],
            "precision@10": result.precision[10],
            "recall@10": result.recall[10],
            "rmse": result.rmse,
            "mae": result.mae,
            "count": float(result.count),
            "brier": result.brier,
            "ece": result.ece,
        }
        print(
            f"[compare] final summary model={name} hr10={summary[name]['hr@10']:.4f} "
            f"ndcg10={summary[name]['ndcg@10']:.4f} rmse={summary[name]['rmse']:.4f}"
            + (f" brier={result.brier:.4f} ece={result.ece:.4f}" if uncertainty_fn else ""),
            flush=True,
        )

    drift_users = select_drift_users(args.drift_meta)
    drift_test_rows = [r for r in test_rows if r.user_id in drift_users]
    drift_summary: dict[str, dict[str, float]] = {}
    if drift_test_rows:
        print(
            f"[compare] computing drift subset summary for users={len(drift_users)} "
            f"rows={len(drift_test_rows)}",
            flush=True,
        )
        for idx, (name, runtime) in enumerate(runtimes.items()):
            seen = defaultdict(set, {u: set(items) for u, items in base_seen_by_user.items()})
            for row in stream_rows:
                seen[row.user_id].add(row.item_id)
            uncertainty_fn = getattr(runtime, "predict_rating_with_uncertainty", None)
            result = evaluate_ranking_and_rating(
                runtime=runtime,
                test_rows=drift_test_rows,
                seen_items_by_user=seen,
                num_items=int(metadata["num_items"]),
                ks=(5, 10, 20),
                n_negatives=args.n_negatives,
                seed=args.seed + 991 + idx,
                uncertainty_fn=uncertainty_fn,
            )
            drift_summary[name] = {
                "hr@10": result.hr[10],
                "ndcg@10": result.ndcg[10],
                "rmse": result.rmse,
                "mae": result.mae,
                "count": float(result.count),
                "brier": result.brier,
                "ece": result.ece,
            }
            print(
                f"[compare] drift summary model={name} hr10={drift_summary[name]['hr@10']:.4f} "
                f"ndcg10={drift_summary[name]['ndcg@10']:.4f}"
                + (f" brier={result.brier:.4f} ece={result.ece:.4f}" if uncertainty_fn else ""),
                flush=True,
            )
    else:
        print("[compare] no drift subset rows found; skipping drift summary", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    curves_path = args.output_dir / "streaming_curves.json"
    drift_path = args.output_dir / "drift_subset_summary.json"
    per_user_path = args.output_dir / "per_user_results.csv"

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    with curves_path.open("w", encoding="utf-8") as handle:
        json.dump(curves, handle, indent=2, sort_keys=True)
    with drift_path.open("w", encoding="utf-8") as handle:
        json.dump(drift_summary, handle, indent=2, sort_keys=True)

    # Flatten curve JSON into CSV for easier spreadsheet/debug workflows.
    with (args.output_dir / "streaming_curve.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "stream_step",
                "hr10",
                "ndcg10",
                "precision10",
                "recall10",
                "rmse",
                "mae",
                "count",
                "brier",
                "ece",
            ],
        )
        writer.writeheader()
        for model_name, points in curves.items():
            for p in points:
                writer.writerow({"model": model_name, **p})

    per_user_csv(per_user_path, runtimes=runtimes, test_rows=test_rows)
    generate_plots(summary=summary, curves=curves, output_dir=args.output_dir / "graphs")
    print(
        f"[compare] wrote outputs to {args.output_dir} total_elapsed_s={(time.time() - t0):.1f}",
        flush=True,
    )

    print(json.dumps({"summary_path": str(summary_path), "drift_path": str(drift_path)}, indent=2))


if __name__ == "__main__":
    main()
