"""Tests du modèle XP-01 (Perceptron)."""
import torch

from xp.train.models.perceptron import DEFAULT_IN_FEATURES
from xp.train.models.perceptron import Perceptron


def test_perceptron_output_shape():
    """Le perceptron doit renvoyer un logit par échantillon."""
    model = Perceptron(in_features=DEFAULT_IN_FEATURES)
    batch = torch.rand(16, DEFAULT_IN_FEATURES)

    logits = model(batch)

    assert logits.shape == (16,)


def test_perceptron_single_linear_layer():
    """Un perceptron = une seule couche linéaire (pas de couche cachée)."""
    model = Perceptron(in_features=4)

    linear_layers = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]

    assert len(linear_layers) == 1
    assert linear_layers[0].in_features == 4
    assert linear_layers[0].out_features == 1


def test_perceptron_is_differentiable():
    """Les gradients doivent se propager jusqu'aux poids du perceptron."""
    model = Perceptron(in_features=4)
    batch = torch.rand(8, 4, requires_grad=False)

    logits = model(batch)
    loss = logits.sum()
    loss.backward()

    assert model.linear.weight.grad is not None
    assert not torch.all(model.linear.weight.grad == 0)
