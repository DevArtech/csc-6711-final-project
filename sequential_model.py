from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class SequentialRecConfig:
    num_items: int
    num_genres: int
    item_embedding_dim: int = 64
    hidden_dim: int = 128
    rating_buckets: int = 5
    dropout: float = 0.1


class SequentialRecommender(nn.Module):
    def __init__(self, config: SequentialRecConfig) -> None:
        super().__init__()
        self.config = config
        self.item_embedding = nn.Embedding(config.num_items, config.item_embedding_dim)
        self.rating_proj = nn.Linear(1, config.item_embedding_dim)
        self.gru = nn.GRU(
            input_size=config.item_embedding_dim * 2,
            hidden_size=config.hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.next_item_proj = nn.Linear(config.hidden_dim, config.item_embedding_dim)
        self.genre_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_genres),
        )
        self.rating_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.rating_buckets),
        )

    def encode_history(self, item_hist: Tensor, rating_hist: Tensor, lengths: Tensor) -> Tensor:
        item_emb = self.item_embedding(item_hist)
        rating_emb = self.rating_proj(rating_hist.unsqueeze(-1))
        x = torch.cat([item_emb, rating_emb], dim=-1)
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            x,
            lengths=lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _out, h_n = self.gru(packed)
        hidden = h_n[-1]
        return self.dropout(hidden)

    def forward(self, item_hist: Tensor, rating_hist: Tensor, lengths: Tensor) -> dict[str, Tensor]:
        hidden = self.encode_history(item_hist=item_hist, rating_hist=rating_hist, lengths=lengths)
        query = self.next_item_proj(hidden)
        item_logits = query @ self.item_embedding.weight.t()
        genre_logits = self.genre_head(hidden)
        rating_logits = self.rating_head(hidden)
        return {
            "hidden": hidden,
            "item_logits": item_logits,
            "genre_logits": genre_logits,
            "rating_logits": rating_logits,
        }

    @torch.no_grad()
    def score_candidates(self, history_items: Tensor, history_ratings: Tensor, candidate_items: Tensor) -> Tensor:
        if history_items.ndim == 1:
            history_items = history_items.unsqueeze(0)
        if history_ratings.ndim == 1:
            history_ratings = history_ratings.unsqueeze(0)
        lengths = torch.tensor([history_items.shape[1]], dtype=torch.long, device=history_items.device)
        out = self.forward(history_items, history_ratings, lengths)
        query = self.next_item_proj(out["hidden"]).squeeze(0)
        cand_emb = self.item_embedding(candidate_items)
        return cand_emb @ query


def ratings_to_bucket(ratings: Tensor) -> Tensor:
    """
    MovieLens ratings are 1..5 in 0.5 increments. We bucket to 5 classes:
    [1], [2], [3], [4], [5] by nearest integer after clamping.
    """
    clipped = ratings.clamp(1.0, 5.0)
    return torch.round(clipped).long() - 1
