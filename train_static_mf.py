from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from recsys.data_loader import MFTrainDataset, load_processed_data
from recsys.static_mf import StaticMF, StaticMFConfig


def train_static_mf(
    data_dir: Path,
    output_dir: Path,
    embedding_dim: int,
    batch_size: int,
    epochs: int,
    lr: float,
    l2: float,
    device: str,
) -> dict[str, object]:
    bundle = load_processed_data(data_dir)
    metadata = bundle["metadata"]
    warm_rows = bundle["by_split"]["warm"]

    model = StaticMF(
        StaticMFConfig(
            num_users=int(metadata["num_users"]),
            num_items=int(metadata["num_items"]),
            embedding_dim=embedding_dim,
            global_mean=float(metadata["global_mean_rating"]),
        )
    ).to(device)

    dataset = MFTrainDataset(warm_rows)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse = torch.nn.MSELoss()

    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        count = 0
        for user_ids, item_ids, ratings in loader:
            user_ids = user_ids.to(device)
            item_ids = item_ids.to(device)
            ratings = ratings.to(device)

            preds = model(user_ids, item_ids)
            loss = mse(preds, ratings)

            # Standard L2 penalty over embeddings and bias parameters.
            reg = (
                model.user_factors.weight.norm(2).pow(2)
                + model.item_factors.weight.norm(2).pow(2)
                + model.user_bias.weight.norm(2).pow(2)
                + model.item_bias.weight.norm(2).pow(2)
            )
            loss = loss + l2 * reg / max(1, user_ids.shape[0])

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * user_ids.shape[0]
            count += user_ids.shape[0]

        epoch_loss = running_loss / max(1, count)
        history.append({"epoch": float(epoch), "train_loss": epoch_loss})
        print(f"[static-mf] epoch={epoch} train_loss={epoch_loss:.4f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_type": "static_mf",
        "history": history,
        "state": model.export_state(),
    }
    checkpoint_path = output_dir / "static_mf.pt"
    torch.save(checkpoint, checkpoint_path)

    with (output_dir / "history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    summary = {
        "checkpoint": str(checkpoint_path),
        "epochs": epochs,
        "num_train_rows": len(warm_rows),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train static matrix factorization baseline.")
    parser.add_argument("--data-dir", type=Path, default=Path("recsys/data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("recsys/runs/static_mf"))
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--l2", type=float, default=1e-6)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = train_static_mf(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        embedding_dim=args.embedding_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        l2=args.l2,
        device=args.device,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
