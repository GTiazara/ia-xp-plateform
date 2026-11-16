"""Tests unitaires de `ClassificationModule`.

Le module doit accepter n'importe quel `BaseModel` (pas seulement
`Perceptron`), déléguer `forward`, calculer une perte positive et
construire un optimiseur sur les poids du modèle.
"""

import pytest
import torch
from torch import nn

from ia_xp_plateform.models.base import BaseModel
from ia_xp_plateform.models.mlp import MLP
from ia_xp_plateform.models.perceptron import Perceptron
from ia_xp_plateform.module import ClassificationModule
from ia_xp_plateform.task.multiclass import MulticlassTask


class _TwoLayerClassifier(BaseModel):
    """Un autre `BaseModel` que `Perceptron`, pour vérifier le découplage."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def test_forward_delegates_to_the_wrapped_model():
    perceptron = Perceptron()
    module = ClassificationModule(model=perceptron)
    x = torch.randn(4, 2)

    assert torch.allclose(module(x), torch.sigmoid(perceptron(x)))


def test_accepts_any_base_model_not_only_perceptron():
    module = ClassificationModule(model=_TwoLayerClassifier())
    x = torch.randn(4, 2)
    y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])

    loss = module.training_step((x, y), batch_idx=0)

    assert loss.ndim == 0
    assert loss.item() > 0.0


def test_multiclass_forward_is_the_softmax_of_the_logits():
    model = MLP(input_features=2, output_features=3, hidden_features=4)
    module = ClassificationModule(model=model, task=MulticlassTask())
    x = torch.randn(4, 2)

    assert torch.allclose(module(x), torch.softmax(model(x), dim=-1))


def test_multiclass_refuses_a_class_index_past_the_last_one():
    module = ClassificationModule(
        model=MLP(input_features=2, output_features=3, hidden_features=4),
        task=MulticlassTask(),
    )

    with pytest.raises(ValueError, match="0 .. 2"):
        module.training_step((torch.zeros(1, 2), torch.tensor([[3.0]])), batch_idx=0)


def test_a_target_of_two_is_refused_before_the_loss():
    module = ClassificationModule(model=Perceptron())
    x = torch.zeros(1, 2)
    y = torch.tensor([[2.0]])

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        module.training_step((x, y), batch_idx=0)


def test_rejects_an_object_that_is_not_a_base_model():
    with pytest.raises(TypeError):
        ClassificationModule(model=nn.Linear(2, 1))


def test_rejects_non_positive_learning_rate():
    with pytest.raises(ValueError):
        ClassificationModule(model=Perceptron(), learning_rate=0.0)


def test_rejects_threshold_outside_zero_and_one():
    with pytest.raises(ValueError):
        ClassificationModule(model=Perceptron(), threshold=1.0)


def test_configure_optimizers_returns_adam_with_the_configured_lr():
    module = ClassificationModule(model=Perceptron(), learning_rate=0.05)

    optimizer = module.configure_optimizers()

    assert isinstance(optimizer, torch.optim.Adam)
    assert optimizer.param_groups[0]["lr"] == 0.05
