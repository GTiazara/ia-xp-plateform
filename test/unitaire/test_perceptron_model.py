"""Tests unitaires de `Perceptron`.

On vérifie la forme de la sortie, que la sigmoïde renvoie bien une
probabilité entre 0 et 1, la configuration du nombre d'entrées/sorties, et
que le contrat `BaseModel` est respecté.
"""

import torch

from ia_xp_plateform.models.base import BaseModel
from ia_xp_plateform.models.perceptron import Perceptron


def test_default_output_shape():
    model = Perceptron()
    x = torch.randn(5, 2)

    output = model(x)

    assert output.shape == (5, 1)


def test_output_is_a_probability():
    model = Perceptron()
    x = torch.randn(20, 2)

    output = model(x)

    assert torch.all(output >= 0.0)
    assert torch.all(output <= 1.0)


def test_custom_input_and_output_features():
    model = Perceptron(input_features=5, output_features=3)
    x = torch.randn(2, 5)

    output = model(x)

    assert output.shape == (2, 3)


def test_respects_the_base_model_contract():
    model = Perceptron()

    assert isinstance(model, BaseModel)
    # Une couche linéaire 2 -> 1 : 2 poids + 1 biais.
    assert sum(p.numel() for p in model.parameters()) == 3
