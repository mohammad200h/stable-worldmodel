"""DreamerV3 ``symexp_twohot`` reward distribution (PyTorch).

Matches LEQ_DV3 / DreamerV3:
- Bin centers live in reward space via ``symexp(linspace(-20, 0))``.
- Training uses soft two-hot targets + cross-entropy (``log_prob``).
- Does **not** use a soft-mean decode for the training loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


def make_symexp_bins(num_bins: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """Build DreamerV3 ``symexp_twohot`` bin centers in reward space."""
    if num_bins < 2:
        raise ValueError(f'num_bins must be >= 2, got {num_bins}')
    if num_bins % 2 == 1:
        half = torch.linspace(-20.0, 0.0, (num_bins - 1) // 2 + 1, dtype=dtype)
        half = symexp(half)
        bins = torch.cat([half, -half[:-1].flip(0)], dim=0)
    else:
        half = torch.linspace(-20.0, 0.0, num_bins // 2, dtype=dtype)
        half = symexp(half)
        bins = torch.cat([half, -half.flip(0)], dim=0)
    assert bins.numel() == num_bins, (bins.numel(), num_bins)
    return bins.to(device=device)


class SymexpTwoHotDist:
    """Categorical two-hot over DreamerV3 ``symexp_twohot`` bins.

    Loss path: ``-log_prob(reward)`` (soft two-hot CE).
    No soft-mean decode is used for training.
    """

    def __init__(self, logits: torch.Tensor, bins: torch.Tensor):
        if logits.shape[-1] != bins.numel():
            raise ValueError(
                f'logits last dim {logits.shape[-1]} != num_bins {bins.numel()}'
            )
        self.logits = logits
        self.bins = bins.to(device=logits.device, dtype=logits.dtype)

    def log_prob(self, reward: torch.Tensor) -> torch.Tensor:
        """Soft two-hot cross-entropy against scalar rewards (DreamerV3)."""
        reward = reward.to(dtype=self.logits.dtype)
        flat = reward.reshape(-1)
        logits = self.logits.reshape(-1, self.bins.numel())

        # Find neighboring bins (same logic as jaxutils.TwoHotDist.log_prob).
        below = (self.bins <= flat[:, None]).to(torch.int64).sum(-1) - 1
        above = self.bins.numel() - (self.bins > flat[:, None]).to(torch.int64).sum(-1)
        below = below.clamp(0, self.bins.numel() - 1)
        above = above.clamp(0, self.bins.numel() - 1)

        equal = below == above
        dist_to_below = torch.where(
            equal, torch.ones_like(flat), (self.bins[below] - flat).abs()
        )
        dist_to_above = torch.where(
            equal, torch.ones_like(flat), (self.bins[above] - flat).abs()
        )
        total = dist_to_below + dist_to_above
        weight_below = dist_to_above / total
        weight_above = dist_to_below / total

        target = (
            F.one_hot(below, self.bins.numel()).to(logits.dtype) * weight_below[:, None]
            + F.one_hot(above, self.bins.numel()).to(logits.dtype) * weight_above[:, None]
        )
        log_pred = F.log_softmax(logits, dim=-1)
        logp = (target * log_pred).sum(-1)
        return logp.reshape(reward.shape)

    def nll(self, reward: torch.Tensor) -> torch.Tensor:
        return -self.log_prob(reward)

    def mean(self) -> torch.Tensor:
        """Expected reward under the two-hot distribution (DreamerV3 ``.mean()``)."""
        probs = F.softmax(self.logits, dim=-1)
        return (probs * self.bins).sum(dim=-1)

    def mode(self) -> torch.Tensor:
        return self.mean()

    def twohot_targets(self, reward: torch.Tensor) -> torch.Tensor:
        """Soft two-hot targets with the same shape as ``logits``."""
        reward = reward.to(dtype=self.logits.dtype)
        flat = reward.reshape(-1)
        below = (self.bins <= flat[:, None]).to(torch.int64).sum(-1) - 1
        above = self.bins.numel() - (self.bins > flat[:, None]).to(torch.int64).sum(-1)
        below = below.clamp(0, self.bins.numel() - 1)
        above = above.clamp(0, self.bins.numel() - 1)

        equal = below == above
        dist_to_below = torch.where(
            equal, torch.ones_like(flat), (self.bins[below] - flat).abs()
        )
        dist_to_above = torch.where(
            equal, torch.ones_like(flat), (self.bins[above] - flat).abs()
        )
        total = dist_to_below + dist_to_above
        weight_below = dist_to_above / total
        weight_above = dist_to_below / total
        target = (
            F.one_hot(below, self.bins.numel()).to(self.logits.dtype)
            * weight_below[:, None]
            + F.one_hot(above, self.bins.numel()).to(self.logits.dtype)
            * weight_above[:, None]
        )
        return target.reshape(*reward.shape, self.bins.numel())

    def primary_bin_indices(self, reward: torch.Tensor) -> torch.Tensor:
        """Bin with largest two-hot mass (for eval accuracy)."""
        return self.twohot_targets(reward).argmax(dim=-1)

    def pred_bin_indices(self) -> torch.Tensor:
        return self.logits.argmax(dim=-1)

    def bin_cross_entropy(self, reward: torch.Tensor) -> torch.Tensor:
        """Soft two-hot CE; same as ``nll`` but exposed for eval metrics."""
        return self.nll(reward)


class DreamerV3RewardHead(nn.Module):
    """MLP reward head ending in ``symexp_twohot`` logits (DreamerV3 rewhead)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 1024,
        num_bins: int = 255,
        layers: int = 1,
        act: str = 'silu',
    ):
        super().__init__()
        self.num_bins = num_bins
        act_fn = {'silu': nn.SiLU, 'gelu': nn.GELU, 'relu': nn.ReLU}[act]

        mods: list[nn.Module] = []
        dim = input_dim
        for _ in range(layers):
            mods.extend([nn.Linear(dim, hidden_dim), nn.LayerNorm(hidden_dim), act_fn()])
            dim = hidden_dim
        mods.append(nn.Linear(dim, num_bins))
        # DreamerV3 uses outscale=0.0 on the reward Dist linear → zero-init logits.
        nn.init.zeros_(mods[-1].weight)
        nn.init.zeros_(mods[-1].bias)
        self.net = nn.Sequential(*mods)
        self.register_buffer('bins', make_symexp_bins(num_bins), persistent=True)

    def forward(self, feat: torch.Tensor) -> SymexpTwoHotDist:
        """
        feat: (..., D) latent features (e.g. LEWM embeddings).
        Returns a ``SymexpTwoHotDist`` over the trailing batch dims.
        """
        logits = self.net(feat)
        return SymexpTwoHotDist(logits, self.bins)

    def nll(self, feat: torch.Tensor, reward: torch.Tensor) -> torch.Tensor:
        return self.forward(feat).nll(reward)
