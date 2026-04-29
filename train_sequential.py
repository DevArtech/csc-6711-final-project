from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from recsys.data_loader import (
    Interaction,
    NextItemSequenceDataset,
    collate_sequence_batch,
    load_processed_data,
)
from recsys.sequential_model import SequentialRecConfig, SequentialRecommender, ratings_to_bucket


def build_warm_histories(warm_rows: list[Interaction]) -> dict[int, list[Interaction]]:
    by_user: dict[int, list[Interaction]] = defaultdict(list)
    for row in warm_rows:
        by_user[row.user_id].append(row)
    for seq in by_user.values():
        seq.sort(key=lambda x: (x.timestamp, x.item_id))
    return dict(by_user)


def genre_target_matrix(item_to_genres: dict[int, list[int]], num_items: int, num_genres: int) -> torch.Tensor:
    target = torch.zeros((num_items, num_genres), dtype=torch.float32)
    for item_id, genres in item_to_genres.items():
        if 0 <= item_id < num_items:
            for g in genres:
                if 0 <= g < num_genres:
                    target[item_id, g] = 1.0
    return target


def train_sequential(
    data_dir: Path,
    output_dir: Path,
    item_embedding_dim: int,
    hidden_dim: int,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    aux_genre_weight: float,
    aux_rating_weight: float,
    device: str,
) -> dict[str, object]:
    bundle = load_processed_data(data_dir)
    metadata = bundle["metadata"]
    warm_rows = bundle["by_split"]["warm"]
    warm_histories = build_warm_histories(warm_rows)

    dataset = NextItemSequenceDataset(warm_histories, min_prefix=2)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_sequence_batch,
    )

    num_items = int(metadata["num_items"])
    num_genres = int(bundle["num_genres"])
    model = SequentialRecommender(
        SequentialRecConfig(
            num_items=num_items,
            num_genres=num_genres,
            item_embedding_dim=item_embedding_dim,
            hidden_dim=hidden_dim,
        )
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    item_genre_targets = genre_target_matrix(bundle["item_to_genres"], num_items, num_genres).to(device)

    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        loss_total = 0.0
        n_examples = 0
        for batch in loader:
            item_hist = batch["item_hist"].to(device)
            rating_hist = batch["rating_hist"].to(device)
            lengths = batch["lengths"].to(device)
            target_item = batch["target_item"].to(device)
            target_rating = batch["target_rating"].to(device)

            out = model(item_hist=item_hist, rating_hist=rating_hist, lengths=lengths)
            item_loss = F.cross_entropy(out["item_logits"], target_item)

            genre_target = item_genre_targets[target_item]
            genre_loss = F.binary_cross_entropy_with_logits(out["genre_logits"], genre_target)

            rating_bucket = ratings_to_bucket(target_rating).to(device)
            rating_loss = F.cross_entropy(out["rating_logits"], rating_bucket)

            loss = item_loss + aux_genre_weight * genre_loss + aux_rating_weight * rating_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            bs = target_item.shape[0]
            loss_total += float(loss.item()) * bs
            n_examples += bs

        epoch_loss = loss_total / max(1, n_examples)
        history.append({"epoch": float(epoch), "train_loss": epoch_loss})
        print(f"[sequential] epoch={epoch} train_loss={epoch_loss:.4f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_type": "sequential",
        "config": {
            "num_items": num_items,
            "num_genres": num_genres,
            "item_embedding_dim": item_embedding_dim,
            "hidden_dim": hidden_dim,
        },
        "state_dict": model.state_dict(),
        "history": history,
    }
    ckpt_path = output_dir / "sequential.pt"
    torch.save(checkpoint, ckpt_path)

    with (output_dir / "history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    summary = {
        "checkpoint": str(ckpt_path),
        "num_examples": len(dataset),
        "epochs": epochs,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train sequential recommender baseline/control.")
    parser.add_argument("--data-dir", type=Path, default=Path("recsys/data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("recsys/runs/sequential"))
    parser.add_argument("--item-embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--aux-genre-weight", type=float, default=0.1)
    parser.add_argument("--aux-rating-weight", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = train_sequential(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        item_embedding_dim=args.item_embedding_dim,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        aux_genre_weight=args.aux_genre_weight,
        aux_rating_weight=args.aux_rating_weight,
        device=args.device,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
