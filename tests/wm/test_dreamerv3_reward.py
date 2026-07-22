"""Unit tests for DreamerV3 symexp_twohot reward head."""

import torch

from stable_worldmodel.wm.dreamerv3_reward import (
    DreamerV3RewardHead,
    SymexpTwoHotDist,
    make_symexp_bins,
)
from stable_worldmodel.wm.lewm.lewm import LeWM


def test_symexp_bins_odd_count_centered_at_zero():
    bins = make_symexp_bins(255)
    assert bins.shape == (255,)
    assert torch.isclose(bins[127], torch.tensor(0.0), atol=1e-5)


def test_reward_head_nll_shape_and_grad():
    head = DreamerV3RewardHead(input_dim=8, hidden_dim=16, num_bins=31, layers=1)
    feat = torch.randn(2, 4, 8, requires_grad=True)
    rew = torch.randn(2, 4)
    loss = head.nll(feat, rew).mean()
    loss.backward()
    assert feat.grad is not None
    assert loss.ndim == 0


def test_zero_logits_nll_near_log_num_bins():
    bins = make_symexp_bins(31)
    logits = torch.zeros(5, 31)
    nll = SymexpTwoHotDist(logits, bins).nll(torch.zeros(5))
    assert torch.allclose(nll, torch.full_like(nll, torch.log(torch.tensor(31.0))), atol=1e-4)


def test_lewm_predict_reward_optional():
    model = object.__new__(LeWM)
    object.__setattr__(model, 'reward_head', None)
    assert model.predict_reward(torch.randn(1, 2, 3)) is None

    head = DreamerV3RewardHead(3, hidden_dim=8, num_bins=15)
    object.__setattr__(model, 'reward_head', head)
    dist = model.predict_reward(torch.randn(2, 3, 3))
    assert isinstance(dist, SymexpTwoHotDist)
    assert dist.logits.shape == (2, 3, 15)


def test_reward_bin_helpers():
    head = DreamerV3RewardHead(input_dim=4, hidden_dim=8, num_bins=15, layers=1)
    feat = torch.randn(3, 4)
    rew = torch.zeros(3)
    dist = head(feat)
    true_bins = dist.primary_bin_indices(rew)
    pred_bins = dist.pred_bin_indices()
    assert true_bins.shape == (3,)
    assert pred_bins.shape == (3,)
    assert dist.twohot_targets(rew).shape == (3, 15)
    assert dist.bin_cross_entropy(rew).shape == (3,)
