"""Euclidean distances and norms."""

from __future__ import annotations

import torch
from torch import Tensor


def pairwise_dist(x: Tensor, y: Tensor, eps: float = 1e-8) -> Tensor:
    """`(B1, D)`, `(B2, D)` -> `(B1, B2)`."""
    sq_dist = (
        torch.sum(x**2, dim=-1, keepdim=True)
        + torch.sum(y**2, dim=-1, keepdim=True).transpose(-1, -2)
        - 2 * x @ y.transpose(-1, -2)
    )
    # The sqrt gradient is undefined at coincident points.
    return torch.sqrt(torch.clamp(sq_dist, min=eps))


def pairwise_dist_batch(x: Tensor, y: Tensor, eps: float = 1e-8) -> Tensor:
    """`(B1, N, D)`, `(B2, N, D)` -> `(N, B1, B2)`."""
    return pairwise_dist(x.transpose(0, 1), y.transpose(0, 1), eps=eps)


def get_space_norm(x: Tensor) -> Tensor:
    """`(..., D)` -> `(..., 1)`."""
    return torch.sqrt(torch.sum(x**2, dim=-1, keepdim=True))


def get_distance_from_origin(x: Tensor) -> Tensor:
    """`(B, N, D)` -> `(B, N, 1)`."""
    return get_space_norm(x)
