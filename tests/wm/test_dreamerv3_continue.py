"""Unit tests for DreamerV3 binary continue head."""

import torch

from stable_worldmodel.wm.dreamerv3_continue import (
    BinaryDist,
    DreamerV3ContinueHead,
)
from stable_worldmodel.wm.lewm.lewm import LeWM


def test_binary_dist_soft_target():
    logits = torch.zeros(4)
    dist = BinaryDist(logits)
    # soft target 0.997 → BCE should be finite and positive
    soft = torch.full((4,), 0.997)
    nll = dist.nll(soft)
    assert nll.shape == (4,)
    assert (nll > 0).all()


def test_continue_head_nll_shape_and_grad():
    head = DreamerV3ContinueHead(input_dim=8, hidden_dim=16, layers=1)
    feat = torch.randn(2, 4, 8, requires_grad=True)
    cont = torch.ones(2, 4)
    loss = head.nll(feat, cont).mean()
    loss.backward()
    assert feat.grad is not None
    assert loss.ndim == 0


def test_continue_head_mean_is_sigmoid():
    head = DreamerV3ContinueHead(input_dim=4, hidden_dim=8, layers=1)
    feat = torch.randn(3, 4)
    dist = head(feat)
    assert torch.allclose(dist.mean(), torch.sigmoid(dist.logits))


def test_lewm_predict_continue_optional():
    model = object.__new__(LeWM)
    object.__setattr__(model, 'continue_head', None)
    assert model.predict_continue(torch.randn(1, 2, 3)) is None

    head = DreamerV3ContinueHead(3, hidden_dim=8)
    object.__setattr__(model, 'continue_head', head)
    dist = model.predict_continue(torch.randn(2, 3, 3))
    assert isinstance(dist, BinaryDist)
    assert dist.logits.shape == (2, 3)
