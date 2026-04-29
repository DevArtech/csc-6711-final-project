from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class StaticMFConfig:
    num_users: int
    num_items: int
    embedding_dim: int = 64
    global_mean: float = 3.5


class StaticMF(nn.Module):
    def __init__(self, config: StaticMFConfig) -> None:
        super().__init__()
        self.config = config
        self.user_factors = nn.Embedding(config.num_users, config.embedding_dim)
        self.item_factors = nn.Embedding(config.num_items, config.embedding_dim)
        self.user_bias = nn.Embedding(config.num_users, 1)
        self.item_bias = nn.Embedding(config.num_items, 1)
        self.global_bias = nn.Parameter(torch.tensor(config.global_mean, dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.user_factors.weight, mean=0.0, std=0.05)
        nn.init.normal_(self.item_factors.weight, mean=0.0, std=0.05)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, user_ids: Tensor, item_ids: Tensor) -> Tensor:
        p = self.user_factors(user_ids)
        q = self.item_factors(item_ids)
        dot = (p * q).sum(dim=-1)
        bu = self.user_bias(user_ids).squeeze(-1)
        bi = self.item_bias(item_ids).squeeze(-1)
        return self.global_bias + bu + bi + dot

    @torch.no_grad()
    def predict_all_items(self, user_id: int, device: torch.device) -> Tensor:
        user = torch.tensor([user_id], dtype=torch.long, device=device)
        p = self.user_factors(user).squeeze(0)
        all_q = self.item_factors.weight
        scores = all_q @ p
        scores = scores + self.item_bias.weight.squeeze(-1) + self.user_bias(user).squeeze(0) + self.global_bias
        return scores

    @torch.no_grad()
    def export_state(self) -> dict[str, Tensor | float | int]:
        return {
            "num_users": self.config.num_users,
            "num_items": self.config.num_items,
            "embedding_dim": self.config.embedding_dim,
            "global_bias": float(self.global_bias.item()),
            "user_factors": self.user_factors.weight.detach().cpu(),
            "item_factors": self.item_factors.weight.detach().cpu(),
            "user_bias": self.user_bias.weight.detach().cpu().squeeze(-1),
            "item_bias": self.item_bias.weight.detach().cpu().squeeze(-1),
        }
