"""Unit tests for proprio delta_s head."""

import torch

from stable_worldmodel.wm.delta_s import DeltaSDist, DeltaSHead
from stable_worldmodel.wm.lewm.lewm import LeWM


def test_delta_s_head_nll_shape_and_grad():
    head = DeltaSHead(input_dim=8, output_dim=5, hidden_dim=16, layers=1)
    feat = torch.randn(2, 4, 8, requires_grad=True)
    target = torch.randn(2, 4, 5)
    loss = head.nll(feat, target).mean()
    loss.backward()
    assert feat.grad is not None
    assert loss.ndim == 0
    dist = head(feat)
    assert dist.mean.shape == (2, 4, 5)
    assert dist.logvar.shape == (2, 4, 5)


def test_lewm_predict_delta_s_optional():
    model = object.__new__(LeWM)
    object.__setattr__(model, 'delta_s_head', None)
    assert model.predict_delta_s(torch.randn(1, 2, 3)) is None

    head = DeltaSHead(3, output_dim=4, hidden_dim=8, layers=1)
    object.__setattr__(model, 'delta_s_head', head)
    dist = model.predict_delta_s(torch.randn(2, 3, 3))
    assert isinstance(dist, DeltaSDist)
    assert dist.mean.shape == (2, 3, 4)
