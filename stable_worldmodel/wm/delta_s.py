"""Proprioception ``delta_s`` head (LEQ / OfflineRL-Kit style).

Predicts a diagonal Gaussian over ``s' - s`` from latent features (e.g. LeWM
embeddings). Training uses Gaussian NLL; eval uses mean MSE/MAE.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_clamp(
    x: torch.Tensor,
    _min: torch.Tensor | None = None,
    _max: torch.Tensor | None = None,
) -> torch.Tensor:
    """Clamp while preserving gradients (OfflineRL-Kit)."""
    if _max is not None:
        x = _max - F.softplus(_max - x)
    if _min is not None:
        x = _min + F.softplus(x - _min)
    return x


class DeltaSDist:
    """Diagonal Gaussian over proprioception state deltas."""

    def __init__(self, mean: torch.Tensor, logvar: torch.Tensor):
        if mean.shape != logvar.shape:
            raise ValueError(
                f'mean shape {tuple(mean.shape)} != logvar shape {tuple(logvar.shape)}'
            )
        self.mean = mean
        self.logvar = logvar

    def nll(self, target: torch.Tensor) -> torch.Tensor:
        """Per-sample Gaussian NLL (mean over last dim)."""
        target = target.to(dtype=self.mean.dtype)
        inv_var = torch.exp(-self.logvar)
        mse = (self.mean - target).pow(2) * inv_var
        return mse.mean(dim=-1) + self.logvar.mean(dim=-1)

    def mse(self, target: torch.Tensor) -> torch.Tensor:
        target = target.to(dtype=self.mean.dtype)
        return (self.mean - target).pow(2).mean(dim=-1)


class DeltaSHead(nn.Module):
    """MLP predicting ``(mean, logvar)`` of proprio ``Δs`` from features."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 1024,
        layers: int = 2,
        act: str = 'silu',
    ):
        super().__init__()
        if output_dim < 1:
            raise ValueError(f'output_dim must be >= 1, got {output_dim}')
        self.output_dim = output_dim
        act_fn = {'silu': nn.SiLU, 'gelu': nn.GELU, 'relu': nn.ReLU}[act]

        mods: list[nn.Module] = []
        dim = input_dim
        for _ in range(layers):
            mods.extend([nn.Linear(dim, hidden_dim), nn.LayerNorm(hidden_dim), act_fn()])
            dim = hidden_dim
        mods.append(nn.Linear(dim, 2 * output_dim))
        self.net = nn.Sequential(*mods)
        self.register_parameter(
            'max_logvar', nn.Parameter(torch.ones(output_dim) * 0.5)
        )
        self.register_parameter(
            'min_logvar', nn.Parameter(torch.ones(output_dim) * -10.0)
        )

    def forward(self, feat: torch.Tensor) -> DeltaSDist:
        """
        feat: (..., D) latent features (e.g. predicted LeWM embeddings).
        Returns ``DeltaSDist`` with mean/logvar shaped ``(..., output_dim)``.
        """
        out = self.net(feat)
        mean, logvar = out.chunk(2, dim=-1)
        logvar = soft_clamp(logvar, self.min_logvar, self.max_logvar)
        return DeltaSDist(mean, logvar)

    def nll(self, feat: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.forward(feat).nll(target)
