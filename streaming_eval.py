from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass
from typing import Protocol

from recsys.data_loader import Interaction
from recsys.evaluate import (
    brier_and_ece_from_probabilities,
    evaluate_ranking_and_rating,
    rating_bucket_probs_from_gaussian,
)


class OnlineRuntime(Protocol):
    def update(self, row: Interaction) -> None: ...
    def predict_rating(self, user_id: int, item_id: int) -> float: ...


@dataclass
class StreamingPoint:
    stream_step: int
    hr10: float
    ndcg10: float
    precision10: float
    recall10: float
    rmse: float
    mae: float
    count: int
    brier: float | None = None
    ece: float | None = None


def _maybe_bayesian_calibration(runtime: OnlineRuntime, test_rows: list[Interaction]) -> dict[str, float] | None:
    if not hasattr(runtime, "predict_rating_with_uncertainty"):
        return None
    probs = []
    true_buckets = []
    for row in test_rows:
        mean, var = runtime.predict_rating_with_uncertainty(row.user_id, row.item_id)  # type: ignore[attr-defined]
        probs.append(rating_bucket_probs_from_gaussian(mean, var))
        true_bucket = max(0, min(4, int(round(row.rating) - 1)))
        true_buckets.append(true_bucket)
    return brier_and_ece_from_probabilities(probs=probs, true_buckets=true_buckets)


def run_streaming_evaluation(
    runtimes: dict[str, OnlineRuntime],
    stream_rows: list[Interaction],
    test_rows: list[Interaction],
    base_seen_items_by_user: dict[int, set[int]],
    num_items: int,
    eval_every: int = 20000,
    ks: tuple[int, ...] = (5, 10, 20),
    n_negatives: int = 100,
    seed: int = 0,
) -> dict[str, list[dict[str, float | int | None]]]:
    start = time.time()
    seen_by_model = {name: copy.deepcopy(base_seen_items_by_user) for name in runtimes}
    curves: dict[str, list[dict[str, float | int | None]]] = {name: [] for name in runtimes}
    total_steps = len(stream_rows)
    print(
        f"[compare] streaming evaluation start: total_stream_rows={total_steps}, "
        f"eval_every={eval_every}, models={list(runtimes.keys())}",
        flush=True,
    )

    for step, row in enumerate(stream_rows, start=1):
        for name, runtime in runtimes.items():
            runtime.update(row)
            seen_by_model[name].setdefault(row.user_id, set()).add(row.item_id)

        if step % eval_every != 0 and step != len(stream_rows):
            continue

        elapsed = time.time() - start
        print(
            f"[compare] checkpoint step={step}/{total_steps} "
            f"({(100.0 * step / max(1, total_steps)):.1f}%) elapsed_s={elapsed:.1f}",
            flush=True,
        )

        for idx, (name, runtime) in enumerate(runtimes.items()):
            model_t0 = time.time()
            print(f"[compare] evaluating model={name} at step={step}", flush=True)
            summary = evaluate_ranking_and_rating(
                runtime=runtime,
                test_rows=test_rows,
                seen_items_by_user=seen_by_model[name],
                num_items=num_items,
                ks=ks,
                n_negatives=n_negatives,
                seed=seed + idx + step,
            )
            point = StreamingPoint(
                stream_step=step,
                hr10=summary.hr.get(10, 0.0),
                ndcg10=summary.ndcg.get(10, 0.0),
                precision10=summary.precision.get(10, 0.0),
                recall10=summary.recall.get(10, 0.0),
                rmse=summary.rmse,
                mae=summary.mae,
                count=summary.count,
            )
            cal = _maybe_bayesian_calibration(runtime, test_rows)
            if cal is not None:
                point.brier = cal["brier"]
                point.ece = cal["ece"]
            curves[name].append(asdict(point))
            model_elapsed = time.time() - model_t0
            print(
                f"[compare] done model={name} step={step} "
                f"hr10={point.hr10:.4f} ndcg10={point.ndcg10:.4f} rmse={point.rmse:.4f} "
                f"elapsed_s={model_elapsed:.1f}",
                flush=True,
            )

    print(
        f"[compare] streaming evaluation complete elapsed_s={(time.time() - start):.1f}",
        flush=True,
    )
    return curves
