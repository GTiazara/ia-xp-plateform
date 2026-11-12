"""Tests unitaires de `Predictor`.

Aucun `Trainer` ici : les checkpoints sont fabriqués à la main, au format
que `ModelCheckpoint` écrit pour un `ClassificationModule` (poids sous le
préfixe `model.`).
"""

import pytest
import torch
from torch import nn

from ia_xp_plateform.inference import Predictor
from ia_xp_plateform.models.base import BaseModel
from ia_xp_plateform.models.perceptron import Perceptron


def _perceptron_with(weights: list[float], bias: float) -> Perceptron:
    """Perceptron à 2 entrées dont les poids sont connus."""
    model = Perceptron(input_features=2, output_features=1)
    with torch.no_grad():
        model.linear.weight.copy_(torch.tensor([weights]))
        model.linear.bias.copy_(torch.tensor([bias]))
    return model


def _write_checkpoint(path, weights: list[float], bias: float) -> None:
    """Écrit un checkpoint minimal, au format de `ClassificationModule`."""
    torch.save(
        {
            "state_dict": {
                "model.linear.weight": torch.tensor([weights]),
                "model.linear.bias": torch.tensor([bias]),
            }
        },
        path,
    )


# --- construction ---------------------------------------------------------


def test_puts_the_model_in_evaluation_mode():
    model = Perceptron()
    model.train()

    predictor = Predictor(model)

    assert predictor.model.training is False


def test_rejects_an_object_that_is_not_a_base_model():
    with pytest.raises(TypeError):
        Predictor(nn.Linear(2, 1))


@pytest.mark.parametrize("threshold", [0.0, 1.0, -0.1, 1.5])
def test_rejects_threshold_outside_zero_and_one(threshold):
    with pytest.raises(ValueError):
        Predictor(Perceptron(), threshold=threshold)


# --- predict_proba / predict ----------------------------------------------


def test_predict_proba_matches_the_model_output():
    model = _perceptron_with([1.0, 1.0], 0.0)
    predictor = Predictor(model)
    x = torch.tensor([[1.0, 2.0], [-1.0, -2.0]])

    probabilities = predictor.predict_proba(x)

    assert probabilities.shape == (2, 1)
    assert torch.allclose(probabilities, model(x))
    assert probabilities.requires_grad is False


def test_predict_returns_zero_or_one_with_the_decision_rule():
    # Poids (1, 1) et biais 0 : classe 1 si la somme des entrées est > 0.
    predictor = Predictor(_perceptron_with([1.0, 1.0], 0.0))
    x = torch.tensor([[1.0, 2.0], [-1.0, -2.0], [3.0, -1.0]])

    classes = predictor.predict(x)

    assert classes.dtype == torch.long
    assert classes.tolist() == [[1], [0], [1]]


def test_threshold_changes_the_predicted_class():
    # Probabilité de ~0.73 : classe 1 à 0.5, classe 0 à 0.9.
    model = _perceptron_with([1.0, 0.0], 0.0)
    x = torch.tensor([[1.0, 0.0]])

    assert Predictor(model, threshold=0.5).predict(x).item() == 1
    assert Predictor(model, threshold=0.9).predict(x).item() == 0


def test_predict_proba_rejects_a_non_tensor():
    predictor = Predictor(Perceptron())

    with pytest.raises(TypeError):
        predictor.predict_proba([[1.0, 2.0]])


def test_predict_works_with_any_base_model():
    class TwoLayers(BaseModel):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 1), nn.Sigmoid()
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    predictor = Predictor(TwoLayers())

    assert predictor.predict(torch.randn(5, 2)).shape == (5, 1)


# --- from_checkpoint --------------------------------------------------------


def test_from_checkpoint_loads_the_weights(tmp_path):
    path = tmp_path / "best-model.ckpt"
    _write_checkpoint(path, weights=[2.0, -3.0], bias=0.5)

    predictor = Predictor.from_checkpoint(
        path, "perceptron", model_kwargs={"input_features": 2}
    )

    assert isinstance(predictor.model, Perceptron)
    assert predictor.model.linear.weight.tolist() == [[2.0, -3.0]]
    assert predictor.model.linear.bias.tolist() == [0.5]


def test_from_checkpoint_gives_the_same_prediction_as_the_trained_model(
    tmp_path,
):
    path = tmp_path / "best-model.ckpt"
    _write_checkpoint(path, weights=[1.0, 1.0], bias=0.0)
    x = torch.tensor([[1.0, 2.0], [-1.0, -2.0]])

    predictor = Predictor.from_checkpoint(path, "perceptron")

    expected = _perceptron_with([1.0, 1.0], 0.0)(x)
    assert torch.allclose(predictor.predict_proba(x), expected)


def test_from_checkpoint_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        Predictor.from_checkpoint(tmp_path / "absent.ckpt", "perceptron")


def test_from_checkpoint_rejects_a_file_without_state_dict(tmp_path):
    path = tmp_path / "other.ckpt"
    torch.save({"weights": torch.zeros(1)}, path)

    with pytest.raises(ValueError, match="state_dict"):
        Predictor.from_checkpoint(path, "perceptron")


def test_from_checkpoint_rejects_weights_without_the_model_prefix(tmp_path):
    path = tmp_path / "raw.ckpt"
    torch.save(
        {
            "state_dict": {
                "linear.weight": torch.zeros(1, 2),
                "linear.bias": torch.zeros(1),
            }
        },
        path,
    )

    with pytest.raises(ValueError, match="model."):
        Predictor.from_checkpoint(path, "perceptron")


def test_from_checkpoint_rejects_an_architecture_mismatch(tmp_path):
    path = tmp_path / "best-model.ckpt"
    _write_checkpoint(path, weights=[1.0, 1.0], bias=0.0)

    with pytest.raises(ValueError, match="ne correspondent pas"):
        Predictor.from_checkpoint(
            path, "perceptron", model_kwargs={"input_features": 5}
        )


def test_from_checkpoint_unknown_model_name_raises(tmp_path):
    path = tmp_path / "best-model.ckpt"
    _write_checkpoint(path, weights=[1.0, 1.0], bias=0.0)

    with pytest.raises(ValueError, match="perceptron"):
        Predictor.from_checkpoint(path, "inconnu")
