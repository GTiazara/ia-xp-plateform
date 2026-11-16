"""Tests unitaires de `MLP`.

On vérifie la forme des sorties, que chaque sortie est un logit, la
largeur de la couche cachée, et le refus d'une architecture vide.
"""

import pytest
import torch
from torch import nn

from ia_xp_plateform.models.base import BaseModel
from ia_xp_plateform.models.mlp import MLP


def test_default_output_is_one_logit_per_example():
    model = MLP()
    output = model(torch.randn(5, 2))

    assert isinstance(model, BaseModel)
    assert output.shape == (5, 1)


def test_output_features_sets_the_number_of_logits():
    model = MLP(input_features=4, output_features=3, hidden_features=8)
    output = model(torch.randn(2, 4))

    assert output.shape == (2, 3)


def test_the_last_layer_can_leave_the_unit_interval():
    model = MLP(input_features=2, output_features=1, hidden_features=4)
    last = model.network[-1]
    with torch.no_grad():
        last.weight.zero_()
        last.bias.fill_(2.0)

    output = model(torch.zeros(1, 2))

    assert isinstance(last, nn.Linear)
    assert output.item() > 1.0


def test_hidden_width_changes_the_parameter_count():
    narrow = MLP(input_features=2, output_features=1, hidden_features=4)
    wide = MLP(input_features=2, output_features=1, hidden_features=8)

    # Couche cachée 2 -> 4 puis sortie 4 -> 1 : 8 + 4 biais, 4 + 1 biais.
    assert sum(parameter.numel() for parameter in narrow.parameters()) == 17
    assert sum(parameter.numel() for parameter in wide.parameters()) > 17


def test_hidden_features_must_be_at_least_one():
    with pytest.raises(ValueError, match="hidden_features"):
        MLP(hidden_features=0)


def test_hidden_layers_stack_one_relu_per_extra_depth():
    model = MLP(input_features=2, output_features=1, hidden_features=4, hidden_layers=2)
    output = model(torch.randn(3, 2))

    assert model.hidden_layers == 2
    assert output.shape == (3, 1)
    assert sum(isinstance(block, torch.nn.ReLU) for block in model.network) == 2
    # 2 -> 4, 4 -> 4, puis 4 -> 1.
    assert sum(parameter.numel() for parameter in model.parameters()) == 37


def test_hidden_layers_must_be_at_least_one():
    with pytest.raises(ValueError, match="hidden_layers"):
        MLP(hidden_layers=0)
