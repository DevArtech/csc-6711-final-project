from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Interaction:
    user_id: int
    item_id: int
    rating: float
    timestamp: int
    split: str
    user_order_idx: int


def load_processed_data(data_dir: Path, interactions_filename: str = "interactions.csv") -> dict[str, object]:
    interactions_path = data_dir / interactions_filename
    metadata_path = data_dir / "metadata.json"
    item_genres_path = data_dir / "item_genres.json"

    if not interactions_path.exists():
        raise FileNotFoundError(f"Missing {interactions_path}. Run preprocess first.")

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    with item_genres_path.open("r", encoding="utf-8") as handle:
        item_genres = json.load(handle)

    interactions: list[Interaction] = []
    with interactions_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            interactions.append(
                Interaction(
                    user_id=int(row["user_id"]),
                    item_id=int(row["item_id"]),
                    rating=float(row["rating"]),
                    timestamp=int(row["timestamp"]),
                    split=row["split"],
                    user_order_idx=int(row["user_order_idx"]),
                )
            )

    by_split: dict[str, list[Interaction]] = defaultdict(list)
    by_user: dict[int, list[Interaction]] = defaultdict(list)
    seen_items_by_user: dict[int, set[int]] = defaultdict(set)
    for x in interactions:
        by_split[x.split].append(x)
        by_user[x.user_id].append(x)
        seen_items_by_user[x.user_id].add(x.item_id)

    for seq in by_user.values():
        seq.sort(key=lambda r: (r.timestamp, r.item_id))
    for seq in by_split.values():
        seq.sort(key=lambda r: (r.timestamp, r.user_id, r.item_id))

    item_to_genres = {int(k): [int(g) for g in v] for k, v in item_genres["item_to_genres"].items()}

    return {
        "metadata": metadata,
        "interactions": interactions,
        "by_split": dict(by_split),
        "by_user": dict(by_user),
        "seen_items_by_user": dict(seen_items_by_user),
        "item_to_genres": item_to_genres,
        "genre_names": item_genres["genre_names"],
        "num_genres": len(item_genres["genre_names"]),
    }


class MFTrainDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, interactions: list[Interaction]) -> None:
        self.rows = interactions

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.rows[idx]
        return (
            torch.tensor(row.user_id, dtype=torch.long),
            torch.tensor(row.item_id, dtype=torch.long),
            torch.tensor(row.rating, dtype=torch.float32),
        )


class BPRTrainDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Yields (user, pos_item, neg_item) triplets for BPR loss training."""

    def __init__(self, interactions: list[Interaction], num_items: int) -> None:
        self.rows = interactions
        self.num_items = num_items
        self.user_positives: dict[int, set[int]] = defaultdict(set)
        for row in interactions:
            self.user_positives[row.user_id].add(row.item_id)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.rows[idx]
        pos_set = self.user_positives[row.user_id]
        neg = random.randint(0, self.num_items - 1)
        while neg in pos_set:
            neg = random.randint(0, self.num_items - 1)
        return (
            torch.tensor(row.user_id, dtype=torch.long),
            torch.tensor(row.item_id, dtype=torch.long),
            torch.tensor(neg, dtype=torch.long),
        )


class NextItemSequenceDataset(Dataset[dict[str, torch.Tensor]]):
    """
    Creates training windows from chronological user histories.
    """

    def __init__(self, by_user: dict[int, list[Interaction]], min_prefix: int = 2) -> None:
        self.examples: list[tuple[int, list[int], list[float], int, float]] = []
        for _user, seq in by_user.items():
            if len(seq) < min_prefix + 1:
                continue
            for t in range(min_prefix, len(seq)):
                prefix = seq[:t]
                target = seq[t]
                self.examples.append(
                    (
                        target.user_id,
                        [x.item_id for x in prefix],
                        [x.rating for x in prefix],
                        target.item_id,
                        target.rating,
                    )
                )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        user_id, item_hist, rating_hist, target_item, target_rating = self.examples[idx]
        return {
            "user_id": torch.tensor(user_id, dtype=torch.long),
            "item_hist": torch.tensor(item_hist, dtype=torch.long),
            "rating_hist": torch.tensor(rating_hist, dtype=torch.float32),
            "target_item": torch.tensor(target_item, dtype=torch.long),
            "target_rating": torch.tensor(target_rating, dtype=torch.float32),
        }


def collate_sequence_batch(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    batch_size = len(batch)
    lengths = [len(x["item_hist"]) for x in batch]
    max_len = max(lengths)

    item_hist = torch.zeros((batch_size, max_len), dtype=torch.long)
    rating_hist = torch.zeros((batch_size, max_len), dtype=torch.float32)
    padding_mask = torch.ones((batch_size, max_len), dtype=torch.bool)

    for i, example in enumerate(batch):
        n = len(example["item_hist"])
        item_hist[i, :n] = example["item_hist"]
        rating_hist[i, :n] = example["rating_hist"]
        padding_mask[i, :n] = False

    return {
        "user_id": torch.stack([x["user_id"] for x in batch], dim=0),
        "item_hist": item_hist,
        "rating_hist": rating_hist,
        "padding_mask": padding_mask,
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "target_item": torch.stack([x["target_item"] for x in batch], dim=0),
        "target_rating": torch.stack([x["target_rating"] for x in batch], dim=0),
    }
