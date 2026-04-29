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


def fit_warm_static_mf(
    num_users: int,
    num_items: int,
    global_mean: float,
    warm_rows,
    embedding_dim: int,
    batch_size: int,
    epochs: int,
    lr: float,
    l2: float,
    device: str,
) -> tuple[StaticMF, list[dict[str, float]]]:
    model = StaticMF(
        StaticMFConfig(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=embedding_dim,
            global_mean=global_mean,
        )
    ).to(device)

    dataset = MFTrainDataset(warm_rows)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse = torch.nn.MSELoss()

    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        count = 0
        for user_ids, item_ids, ratings in loader:
            user_ids = user_ids.to(device)
            item_ids = item_ids.to(device)
            ratings = ratings.to(device)

            preds = model(user_ids, item_ids)
            loss = mse(preds, ratings)
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

            bs = user_ids.shape[0]
            epoch_loss += float(loss.item()) * bs
            count += bs

        train_loss = epoch_loss / max(1, count)
        history.append({"epoch": float(epoch), "train_loss": train_loss})
        print(f"[bayes-warm] epoch={epoch} train_loss={train_loss:.4f}")
    return model, history


def train_bayesian_mf(
    data_dir: Path,
    output_dir: Path,
    embedding_dim: int,
    batch_size: int,
    warm_epochs: int,
    lr: float,
    l2: float,
    prior_var: float,
    noise_var: float,
    forgetting: float,
    device: str,
) -> dict[str, object]:
    bundle = load_processed_data(data_dir)
    metadata = bundle["metadata"]
    warm_rows = bundle["by_split"]["warm"]

    model, history = fit_warm_static_mf(
        num_users=int(metadata["num_users"]),
        num_items=int(metadata["num_items"]),
        global_mean=float(metadata["global_mean_rating"]),
        warm_rows=warm_rows,
        embedding_dim=embedding_dim,
        batch_size=batch_size,
        epochs=warm_epochs,
        lr=lr,
        l2=l2,
        device=device,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_type": "bayesian_mf",
        "embedding_dim": embedding_dim,
        "global_bias": float(model.global_bias.item()),
        "item_factors": model.item_factors.weight.detach().cpu(),
        "item_bias": model.item_bias.weight.detach().cpu().squeeze(-1),
        "user_prior_mean": model.user_factors.weight.detach().cpu(),
        "prior_var": float(prior_var),
        "noise_var": float(noise_var),
        "forgetting": float(forgetting),
        "history": history,
    }
    ckpt_path = output_dir / "bayesian_mf.pt"
    torch.save(checkpoint, ckpt_path)

    with (output_dir / "history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    summary = {
        "checkpoint": str(ckpt_path),
        "num_warm_rows": len(warm_rows),
        "warm_epochs": warm_epochs,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train warm-start parameters for Bayesian online MF.")
    parser.add_argument("--data-dir", type=Path, default=Path("recsys/data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("recsys/runs/bayesian_mf"))
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--warm-epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--l2", type=float, default=1e-6)
    parser.add_argument("--prior-var", type=float, default=1.0)
    parser.add_argument("--noise-var", type=float, default=0.5)
    parser.add_argument("--forgetting", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = train_bayesian_mf(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        embedding_dim=args.embedding_dim,
        batch_size=args.batch_size,
        warm_epochs=args.warm_epochs,
        lr=args.lr,
        l2=args.l2,
        prior_var=args.prior_var,
        noise_var=args.noise_var,
        forgetting=args.forgetting,
        device=args.device,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
