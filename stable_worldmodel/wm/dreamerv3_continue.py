"""DreamerV3 binary continue distribution (PyTorch).

Matches LEQ_DV3 / DreamerV3 ``conhead``:
- Bernoulli logits from an MLP (``outscale=1.0`` — normal init, unlike reward).
- Training uses ``-log_prob(cont)``; with ``contdisc=True`` targets are soft:
  ``cont * (1 - 1 / horizon)``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryDist:
    """Bernoulli distribution over continuation probability.

    Supports soft targets in ``[0, 1]`` (DreamerV3 ``contdisc``).
    """

    def __init__(self, logits: torch.Tensor):
        self.logits = logits

    def mean(self) -> torch.Tensor:
        return torch.sigmoid(self.logits)

    def log_prob(self, cont: torch.Tensor) -> torch.Tensor:
        """Binary cross-entropy with soft or hard targets."""
        cont = cont.to(dtype=self.logits.dtype)
        log_p = F.logsigmoid(self.logits)
        log_not_p = F.logsigmoid(-self.logits)
        return cont * log_p + (1 - cont) * log_not_p

    def nll(self, cont: torch.Tensor) -> torch.Tensor:
        return -self.log_prob(cont)


class DreamerV3ContinueHead(nn.Module):
    """MLP continue head ending in Bernoulli logits (DreamerV3 conhead)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 1024,
        layers: int = 1,
        act: str = 'silu',
    ):
        super().__init__()
        act_fn = {'silu': nn.SiLU, 'gelu': nn.GELU, 'relu': nn.ReLU}[act]

        mods: list[nn.Module] = []
        dim = input_dim
        for _ in range(layers):
            mods.extend([nn.Linear(dim, hidden_dim), nn.LayerNorm(hidden_dim), act_fn()])
            dim = hidden_dim
        mods.append(nn.Linear(dim, 1))
        # DreamerV3 conhead uses outscale=1.0 → keep default linear init.
        self.net = nn.Sequential(*mods)

    def forward(self, feat: torch.Tensor) -> BinaryDist:
        """
        feat: (..., D) latent features (e.g. LEWM predicted embeddings).
        Returns a ``BinaryDist``; logits have shape ``(..., 1)``.
        """
        logits = self.net(feat).squeeze(-1)
        return BinaryDist(logits)

    def nll(self, feat: torch.Tensor, cont: torch.Tensor) -> torch.Tensor:
        return self.forward(feat).nll(cont)
