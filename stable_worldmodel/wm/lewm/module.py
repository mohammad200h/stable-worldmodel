from enum import Enum

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


class RewardPredictionMode(str, Enum):
    """Input features for the reward predictor head."""

    CURRENT_EMB = 'current_emb'
    CURRENT_AND_PRED_EMB = 'current_and_pred_emb'
    CURRENT_PRED_EMB_AND_ACTION = 'current_pred_emb_and_action'

    @classmethod
    def from_str(cls, mode: str) -> 'RewardPredictionMode':
        try:
            return cls(mode)
        except ValueError as exc:
            valid = ', '.join(m.value for m in cls)
            raise ValueError(
                f'Unknown reward prediction mode {mode!r}. '
                f'Expected one of: {valid}.'
            ) from exc


def detach_clone(v):
    return v.detach().clone() if torch.is_tensor(v) else v


def modulate(x, shift, scale):
    """AdaLN-zero modulation"""
    return x * (1 + scale) + shift


class FeedForward(nn.Module):
    """FeedForward network used in Transformers"""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Scaled dot-product attention with causal masking"""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, causal=True):
        """
        x : (B, T, D)
        """
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(
            3, dim=-1
        )  # q, k, v: (B, heads, T, dim_head)
        q, k, v = (
            rearrange(t, 'b t (h d) -> b h t d', h=self.heads) for t in qkv
        )
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=drop, is_causal=causal
        )
        out = rearrange(out, 'b h t d -> b t (h d)')
        return self.to_out(out)


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(
            dim, heads=heads, dim_head=dim_head, dropout=dropout
        )
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )

        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        x = x + gate_mlp * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class Block(nn.Module):
    """Standard Transformer block"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(
            dim, heads=heads, dim_head=dim_head, dropout=dropout
        )
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    """Standard Transformer with support for AdaLN-zero blocks"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        block_class=Block,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([])

        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.cond_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.output_proj = (
            nn.Linear(hidden_dim, output_dim)
            if hidden_dim != output_dim
            else nn.Identity()
        )

        for _ in range(depth):
            self.layers.append(
                block_class(hidden_dim, heads, dim_head, mlp_dim, dropout)
            )

    def forward(self, x, c=None):
        x = self.input_proj(x)

        if c is not None:
            c = self.cond_proj(c)

        for block in self.layers:
            x = block(x) if isinstance(block, Block) else block(x, c)
        x = self.norm(x)
        x = self.output_proj(x)
        return x


class Embedder(nn.Module):
    def __init__(
        self,
        input_dim=10,
        smoothed_dim=10,
        emb_dim=10,
        mlp_scale=4,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.smoothed_dim = smoothed_dim
        self.emb_dim = emb_dim
        self.mlp_scale = mlp_scale
        self.patch_embed = nn.Conv1d(
            input_dim, smoothed_dim, kernel_size=1, stride=1
        )
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x):
        """
        x: (B, T, D)
        """
        x = x.float()
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        x = self.embed(x)
        return x


class MLP(nn.Module):
    """Simple MLP with optional normalization and activation"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=None,
        norm_fn=nn.LayerNorm,
        act_fn=nn.GELU,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim or input_dim
        norm_fn = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm_fn,
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x):
        """
        x: (*, D) — any leading batch/time dims, last dim is input_dim
        """
        lead = x.shape[:-1]
        if len(lead) > 1:
            x = x.reshape(-1, x.shape[-1])
            return self.net(x).view(*lead, -1)
        return self.net(x)


class StateEncoder(nn.Module):
    """MLP encoder for numeric state vectors (no pixels)."""

    def __init__(
        self,
        input_dim,
        emb_dim,
        hidden_dim=None,
        norm_fn=nn.LayerNorm,
        act_fn=nn.GELU,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.encoder = MLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim or 4 * emb_dim,
            output_dim=emb_dim,
            norm_fn=norm_fn,
            act_fn=act_fn,
        )

    def forward(self, x):
        """
        x: (B, T, D) numeric state
        returns: (B, T, emb_dim)
        """
        return self.encoder(x.float())


class Predictor(nn.Module):
    """Autoregressive predictor for next-step embedding prediction."""

    def __init__(
        self,
        *,
        num_frames,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        output_dim=None,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim or input_dim
        self.depth = depth
        self.heads = heads
        self.dim_head = dim_head
        self.mlp_dim = mlp_dim
        self.emb_dropout = emb_dropout
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_frames, input_dim)
        )
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            input_dim,
            hidden_dim,
            output_dim or input_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            block_class=ConditionalBlock,
        )

    def forward(self, x, c):
        """
        x: (B, T, d)
        c: (B, T, act_dim)
        """
        T = x.size(1)
        x = x + self.pos_embedding[:, :T]
        x = self.dropout(x)
        x = self.transformer(x, c)
        return x


class RewardPredictionMode(str, Enum):
    """Input features for latent transition heads (reward, continue, ...)."""

    CURRENT_EMB = 'current_emb'
    CURRENT_AND_PRED_EMB = 'current_and_pred_emb'
    CURRENT_PRED_EMB_AND_ACTION = 'current_pred_emb_and_action'

    @classmethod
    def from_str(cls, mode: str) -> 'RewardPredictionMode':
        try:
            return cls(mode)
        except ValueError as exc:
            valid = ', '.join(m.value for m in cls)
            raise ValueError(
                f'Unknown latent head mode {mode!r}. '
                f'Expected one of: {valid}.'
            ) from exc


def latent_head_input_dim(
    mode: RewardPredictionMode,
    embed_dim: int,
    action_emb_dim: int,
) -> int:
    return {
        RewardPredictionMode.CURRENT_EMB: embed_dim,
        RewardPredictionMode.CURRENT_AND_PRED_EMB: 2 * embed_dim,
        RewardPredictionMode.CURRENT_PRED_EMB_AND_ACTION: (
            2 * embed_dim + action_emb_dim
        ),
    }[mode]


def build_latent_head_features(
    mode: RewardPredictionMode,
    z_cur: torch.Tensor,
    z_pred: torch.Tensor | None = None,
    act_emb: torch.Tensor | None = None,
) -> torch.Tensor:
    if mode == RewardPredictionMode.CURRENT_EMB:
        return z_cur
    if mode == RewardPredictionMode.CURRENT_AND_PRED_EMB:
        if z_pred is None:
            raise ValueError(f'mode={mode.value} requires z_pred.')
        return torch.cat([z_cur, z_pred], dim=-1)
    if z_pred is None or act_emb is None:
        raise ValueError(f'mode={mode.value} requires z_pred and act_emb.')
    return torch.cat([z_cur, z_pred, act_emb], dim=-1)


class LatentTransitionHead(nn.Module):
    """MLP head over latent transition features (shared by reward / continue)."""

    def __init__(
        self,
        mode: str,
        embed_dim: int,
        action_emb_dim: int,
        hidden_dim: int | None = None,
    ):
        super().__init__()
        self.mode = RewardPredictionMode.from_str(mode)
        hidden_dim = hidden_dim or 4 * embed_dim
        input_dim = latent_head_input_dim(
            self.mode, embed_dim, action_emb_dim
        )
        self.head = MLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            norm_fn=nn.LayerNorm,
        )
        nn.init.zeros_(self.head.net[-1].weight)
        nn.init.zeros_(self.head.net[-1].bias)

    def forward(
        self,
        z_cur: torch.Tensor,
        z_pred: torch.Tensor | None = None,
        act_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features = build_latent_head_features(
            self.mode, z_cur, z_pred=z_pred, act_emb=act_emb
        )
        return self.head(features).squeeze(-1)


class RewardPredictor(LatentTransitionHead):
    """Predict scalar reward from latent transition features."""


class ContinuePredictor(LatentTransitionHead):
    """Predict episode-continue logit from latent transition features."""
