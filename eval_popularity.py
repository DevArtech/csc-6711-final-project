from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from recsys.data_loader import Interaction, load_processed_data
from recsys.evaluate import evaluate_ranking_and_rating


class PopularityRuntime:
    """Recommends the globally most popular items. No training required."""

    def __init__(self, warm_rows: list[Interaction], global_mean: float) -> None:
        counts = Counter(row.item_id for row in warm_rows)
        # pre-sorted list of (item_id, count) descending
        self._ranked: list[int] = [item_id for item_id, _ in counts.most_common()]
        self._global_mean = global_mean

    def update(self, row: Interaction) -> None:
        pass  # popularity baseline never updates

    def score_items(self, user_id: int, item_ids: torch.Tensor) -> torch.Tensor:
        # score = negative rank in popularity list (most popular = highest score)
        rank_of = {item_id: rank for rank, item_id in enumerate(self._ranked)}
        scores = torch.tensor(
            [-rank_of.get(int(i), len(self._ranked)) for i in item_ids],
            dtype=torch.float32,
        )
        return scores

    def predict_rating(self, user_id: int, item_id: int) -> float:
        return self._global_mean


def select_drift_users(drift_meta_path: Path) -> set[int]:
    if not drift_meta_path.exists():
        return set()
    with drift_meta_path.open(encoding="utf-8") as f:
        meta = json.load(f)
    return set(meta.get("actual_drift_users", []))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate popularity baseline.")
    parser.add_argument("--data-dir", type=Path, default=Path("recsys/data/processed"))
    parser.add_argument("--interactions-file", type=str, default="interactions.csv")
    parser.add_argument("--drift-meta", type=Path, default=Path("recsys/data/processed/drift_meta.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("recsys/runs/popularity"))
    parser.add_argument("--n-negatives", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"[popularity] loading data from {args.data_dir}", flush=True)

    bundle = load_processed_data(args.data_dir, interactions_filename=args.interactions_file)
    metadata = bundle["metadata"]
    by_split = bundle["by_split"]
    warm_rows: list[Interaction] = by_split.get("warm", [])
    stream_rows: list[Interaction] = by_split.get("stream", [])
    test_rows: list[Interaction] = by_split.get("test", [])
    num_items = int(metadata["num_items"])
    global_mean = float(metadata["global_mean_rating"])

    print(
        f"[popularity] warm={len(warm_rows)} stream={len(stream_rows)} "
        f"test={len(test_rows)} num_items={num_items}",
        flush=True,
    )

    runtime = PopularityRuntime(warm_rows, global_mean)

    # seen items = warm + stream (same as compare_recommenders final eval)
    seen: dict[int, set[int]] = defaultdict(set)
    for row in warm_rows:
        seen[row.user_id].add(row.item_id)
    for row in stream_rows:
        seen[row.user_id].add(row.item_id)

    print("[popularity] evaluating on full test set...", flush=True)
    result = evaluate_ranking_and_rating(
        runtime=runtime,
        test_rows=test_rows,
        seen_items_by_user=seen,
        num_items=num_items,
        ks=(5, 10, 20),
        n_negatives=args.n_negatives,
        seed=args.seed,
    )

    summary = {
        "popularity": {
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
        }
    }
    print(
        f"[popularity] hr@10={summary['popularity']['hr@10']:.4f} "
        f"ndcg@10={summary['popularity']['ndcg@10']:.4f} "
        f"rmse={summary['popularity']['rmse']:.4f}",
        flush=True,
    )

    # drift subset
    drift_users = select_drift_users(args.drift_meta)
    drift_test_rows = [r for r in test_rows if r.user_id in drift_users]
    drift_summary: dict = {}
    if drift_test_rows:
        print(
            f"[popularity] evaluating drift subset: users={len(drift_users)} rows={len(drift_test_rows)}",
            flush=True,
        )
        drift_result = evaluate_ranking_and_rating(
            runtime=runtime,
            test_rows=drift_test_rows,
            seen_items_by_user=seen,
            num_items=num_items,
            ks=(5, 10, 20),
            n_negatives=args.n_negatives,
            seed=args.seed + 1,
        )
        drift_summary["popularity"] = {
            "hr@10": drift_result.hr[10],
            "ndcg@10": drift_result.ndcg[10],
            "rmse": drift_result.rmse,
            "mae": drift_result.mae,
            "count": float(drift_result.count),
        }
        print(
            f"[popularity] drift hr@10={drift_summary['popularity']['hr@10']:.4f} "
            f"ndcg@10={drift_summary['popularity']['ndcg@10']:.4f}",
            flush=True,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    drift_path = args.output_dir / "drift_subset_summary.json"

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[popularity] wrote {summary_path}", flush=True)

    if drift_summary:
        with drift_path.open("w", encoding="utf-8") as f:
            json.dump(drift_summary, f, indent=2)
        print(f"[popularity] wrote {drift_path}", flush=True)

    print("[popularity] done.", flush=True)


if __name__ == "__main__":
    main()
