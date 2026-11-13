"""Tests unitaires de la factory de modèles."""

import pytest
import torch

from ia_xp_plateform.factory.model_factory import MODELS, create_model
from ia_xp_plateform.models.base import BaseModel
from ia_xp_plateform.models.mlp import MLP
from ia_xp_plateform.models.perceptron import Perceptron


def test_create_perceptron_with_default_arguments():
    model = create_model("perceptron")

    assert isinstance(model, Perceptron)
    assert isinstance(model, BaseModel)


def test_create_perceptron_forwards_constructor_arguments():
    model = create_model("perceptron", input_features=4, output_features=2)
    output = model(torch.randn(3, 4))

    assert output.shape == (3, 2)


def test_create_mlp_forwards_the_number_of_outputs():
    model = create_model(
        "mlp",
        input_features=4,
        output_features=3,
        hidden_features=5,
    )
    output = model(torch.randn(2, 4))

    assert isinstance(model, MLP)
    assert output.shape == (2, 3)


def test_unknown_model_name_lists_the_available_ones():
    with pytest.raises(ValueError, match="mlp"):
        create_model("inconnu")


def test_registry_maps_perceptron_to_its_class():
    assert MODELS["perceptron"] is Perceptron
    assert MODELS["mlp"] is MLP
