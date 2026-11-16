"""Tests du contrat `BaseTransform`."""

import torch

import pytest

from ia_xp_plateform.transform.base import BaseTransform


class _Identity(BaseTransform):
    name = "identity"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _NeedsFit(BaseTransform):
    name = "needs"
    stateful = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def test_stateless_transform_returns_its_input_and_an_empty_state():
    x = torch.tensor([[1.0, 2.0]])
    transform = _Identity()

    assert torch.equal(transform(x), x)
    assert transform.state_dict() == {}


def test_stateful_transform_refuses_to_run_before_fit():
    with pytest.raises(RuntimeError, match="needs"):
        _NeedsFit()(torch.zeros(2, 2))
