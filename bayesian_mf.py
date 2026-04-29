from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class BayesianMFState:
    item_factors: Tensor  # [num_items, dim]
    item_bias: Tensor  # [num_items]
    global_bias: float
    user_precision: Tensor  # [num_users, dim, dim]
    user_eta: Tensor  # [num_users, dim]
    noise_var: float
    forgetting: float

    @property
    def device(self) -> torch.device:
        return self.item_factors.device

    def user_mean(self, user_id: int) -> Tensor:
        precision = self.user_precision[user_id]
        eta = self.user_eta[user_id]
        return torch.linalg.solve(precision, eta)

    def predict_rating(self, user_id: int, item_id: int) -> tuple[float, float]:
        q = self.item_factors[item_id]
        b = self.item_bias[item_id]
        precision = self.user_precision[user_id]
        eta = self.user_eta[user_id]
        mu = torch.linalg.solve(precision, eta)
        mean = self.global_bias + b + torch.dot(mu, q)
        cov_q = torch.linalg.solve(precision, q)
        var = torch.dot(q, cov_q) + self.noise_var
        return float(mean.item()), float(var.item())

    def score_items(self, user_id: int, item_ids: Tensor) -> Tensor:
        mu = self.user_mean(user_id)
        q = self.item_factors[item_ids]
        b = self.item_bias[item_ids]
        return self.global_bias + b + (q @ mu)

    def update(self, user_id: int, item_id: int, rating: float, user_bias: float = 0.0) -> None:
        q = self.item_factors[item_id]
        b = self.item_bias[item_id]
        precision_prev = self.user_precision[user_id]
        eta_prev = self.user_eta[user_id]

        precision = self.forgetting * precision_prev + (1.0 / self.noise_var) * torch.outer(q, q)
        residual = rating - self.global_bias - b.item() - user_bias
        eta = self.forgetting * eta_prev + (1.0 / self.noise_var) * residual * q

        self.user_precision[user_id] = precision
        self.user_eta[user_id] = eta


def make_initial_bayesian_state(
    item_factors: Tensor,
    item_bias: Tensor,
    user_prior_mean: Tensor,
    prior_var: float,
    global_bias: float,
    noise_var: float,
    forgetting: float,
) -> BayesianMFState:
    num_users, dim = user_prior_mean.shape
    eye = torch.eye(dim, dtype=item_factors.dtype, device=item_factors.device)
    prior_precision = (1.0 / prior_var) * eye
    user_precision = prior_precision.unsqueeze(0).repeat(num_users, 1, 1).clone()
    user_eta = torch.einsum("ij,uj->ui", prior_precision, user_prior_mean)
    return BayesianMFState(
        item_factors=item_factors,
        item_bias=item_bias,
        global_bias=global_bias,
        user_precision=user_precision,
        user_eta=user_eta,
        noise_var=noise_var,
        forgetting=forgetting,
    )
