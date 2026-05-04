from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Protocol

import torch

from recsys.data_loader import Interaction


@dataclass
class RankingSummary:
    hr: dict[int, float]
    ndcg: dict[int, float]
    precision: dict[int, float]
    recall: dict[int, float]
    rmse: float
    mae: float
    count: int
    brier: float = field(default=0.0)
    ece: float = field(default=0.0)


class RecommenderRuntime(Protocol):
    def score_items(self, user_id: int, item_ids: torch.Tensor) -> torch.Tensor: ...
    def predict_rating(self, user_id: int, item_id: int) -> float: ...


def _ndcg_at_k(rank: int, k: int) -> float:
    if rank <= k:
        return 1.0 / math.log2(rank + 1.0)
    return 0.0


def sample_negative_items(
    rng: random.Random,
    num_items: int,
    seen_items: set[int],
    positive_item: int,
    n_negatives: int,
) -> list[int]:
    negatives: list[int] = []
    attempts = 0
    max_attempts = n_negatives * 50 + 1000
    while len(negatives) < n_negatives and attempts < max_attempts:
        attempts += 1
        candidate = rng.randrange(num_items)
        if candidate == positive_item or candidate in seen_items:
            continue
        negatives.append(candidate)
    return negatives


def evaluate_ranking_and_rating(
    runtime: RecommenderRuntime,
    test_rows: list[Interaction],
    seen_items_by_user: dict[int, set[int]],
    num_items: int,
    ks: tuple[int, ...] = (5, 10, 20),
    n_negatives: int = 100,
    seed: int = 0,
    uncertainty_fn: Callable[[int, int], tuple[float, float]] | None = None,
) -> RankingSummary:
    rng = random.Random(seed)
    hits = {k: 0.0 for k in ks}
    ndcgs = {k: 0.0 for k in ks}
    precisions = {k: 0.0 for k in ks}
    recalls = {k: 0.0 for k in ks}

    sq_error = 0.0
    abs_error = 0.0
    count = 0

    bucket_probs: list[list[float]] = []
    true_buckets: list[int] = []

    for row in test_rows:
        positives = row.item_id
        seen = seen_items_by_user.get(row.user_id, set())
        negatives = sample_negative_items(rng, num_items, seen, positives, n_negatives=n_negatives)
        if len(negatives) < max(ks):
            continue

        candidates = [positives] + negatives
        candidate_tensor = torch.tensor(candidates, dtype=torch.long)
        scores = runtime.score_items(row.user_id, candidate_tensor).detach().cpu()
        sorted_indices = torch.argsort(scores, descending=True)

        rank = 1 + int((sorted_indices == 0).nonzero(as_tuple=False)[0].item())
        for k in ks:
            hit = 1.0 if rank <= k else 0.0
            hits[k] += hit
            ndcgs[k] += _ndcg_at_k(rank, k)
            precisions[k] += hit / k
            recalls[k] += hit

        pred = runtime.predict_rating(row.user_id, row.item_id)
        err = pred - row.rating
        sq_error += err * err
        abs_error += abs(err)
        count += 1

        if uncertainty_fn is not None:
            mean, var = uncertainty_fn(row.user_id, row.item_id)
            bucket_probs.append(rating_bucket_probs_from_gaussian(mean, var))
            true_buckets.append(max(0, min(4, int(round(row.rating)) - 1)))

    denom = max(1, count)
    cal = brier_and_ece_from_probabilities(bucket_probs, true_buckets) if bucket_probs else {"brier": 0.0, "ece": 0.0}
    return RankingSummary(
        hr={k: hits[k] / denom for k in ks},
        ndcg={k: ndcgs[k] / denom for k in ks},
        precision={k: precisions[k] / denom for k in ks},
        recall={k: recalls[k] / denom for k in ks},
        rmse=math.sqrt(sq_error / denom),
        mae=abs_error / denom,
        count=count,
        brier=cal["brier"],
        ece=cal["ece"],
    )


def rating_bucket_probs_from_gaussian(mean: float, var: float) -> list[float]:
    std = math.sqrt(max(var, 1e-8))
    dist = torch.distributions.Normal(loc=torch.tensor(mean), scale=torch.tensor(std))
    # 5 rating bins centered around integer ratings 1..5 with 0.5 boundaries.
    bounds = [1.5, 2.5, 3.5, 4.5]
    cdfs = [float(dist.cdf(torch.tensor(b)).item()) for b in bounds]
    p1 = cdfs[0]
    p2 = cdfs[1] - cdfs[0]
    p3 = cdfs[2] - cdfs[1]
    p4 = cdfs[3] - cdfs[2]
    p5 = 1.0 - cdfs[3]
    probs = [max(0.0, p) for p in [p1, p2, p3, p4, p5]]
    s = sum(probs)
    if s <= 0:
        return [0.2] * 5
    return [p / s for p in probs]


def brier_and_ece_from_probabilities(
    probs: list[list[float]],
    true_buckets: list[int],
    num_bins: int = 10,
) -> dict[str, float]:
    if not probs:
        return {"brier": 0.0, "ece": 0.0}

    brier = 0.0
    confidences: list[float] = []
    accuracies: list[float] = []
    for p, y in zip(probs, true_buckets):
        one_hot = [0.0] * len(p)
        one_hot[y] = 1.0
        brier += sum((pi - yi) ** 2 for pi, yi in zip(p, one_hot))
        pred_idx = max(range(len(p)), key=lambda i: p[i])
        confidences.append(p[pred_idx])
        accuracies.append(1.0 if pred_idx == y else 0.0)
    brier /= len(probs)

    ece = 0.0
    for b in range(num_bins):
        lo = b / num_bins
        hi = (b + 1) / num_bins
        idx = [i for i, c in enumerate(confidences) if (lo <= c < hi) or (b == num_bins - 1 and c == hi)]
        if not idx:
            continue
        avg_conf = sum(confidences[i] for i in idx) / len(idx)
        avg_acc = sum(accuracies[i] for i in idx) / len(idx)
        ece += (len(idx) / len(confidences)) * abs(avg_conf - avg_acc)

    return {"brier": brier, "ece": ece}
