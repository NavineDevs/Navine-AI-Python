"""Noise schedule helpers for diffusion training and sampling."""

from __future__ import annotations

import math
from typing import List

import torch


def make_beta_schedule(
    timesteps: int,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
    schedule: str = "linear",
) -> torch.Tensor:
    """Build betas without native CUDA helpers (those can collapse alpha_bar)."""
    t = max(1, int(timesteps))
    schedule = (schedule or "linear").lower()
    if schedule == "cosine":
        s = 0.008

        def alpha_bar(u: float) -> float:
            return math.cos((u + s) / (1.0 + s) * math.pi * 0.5) ** 2

        betas = []
        for i in range(t):
            t1 = i / t
            t2 = (i + 1) / t
            a1 = max(alpha_bar(t1), 1e-8)
            a2 = max(alpha_bar(t2), 1e-8)
            betas.append(min(max(1.0 - a2 / a1, 1e-6), 0.999))
        return torch.tensor(betas, dtype=torch.float32)
    if schedule == "sigmoid":
        xs = torch.linspace(0, 1, t)
        sig = torch.sigmoid(12.0 * (xs - 0.5))
        sig = (sig - sig.min()) / (sig.max() - sig.min() + 1e-12)
        return (beta_start + (beta_end - beta_start) * sig).to(torch.float32)
    if t == 1:
        return torch.tensor([float(beta_start)], dtype=torch.float32)
    return torch.linspace(beta_start, beta_end, t, dtype=torch.float32)


def alphas_from_betas(betas: torch.Tensor) -> torch.Tensor:
    alphas = 1.0 - betas
    return torch.cumprod(alphas, dim=0)


def schedule_names() -> List[str]:
    return ["linear", "cosine", "sigmoid"]
