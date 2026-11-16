"""Tests des métriques multiclasses : macro, MCC, matrice."""

import pytest
import torch

from ia_xp_plateform.metric.multiclass import MulticlassClassificationMetrics


def _probabilities(classes: list[int], count: int) -> torch.Tensor:
    """Une probabilité 1 sur la classe annoncée, 0 ailleurs."""
    scores = torch.zeros(len(classes), count)
    for row, class_index in enumerate(classes):
        scores[row, class_index] = 1.0
    return scores


def test_macro_scores_include_an_absent_class():
    metrics = MulticlassClassificationMetrics()
    metrics.update(_probabilities([0, 0, 1], count=3), torch.tensor([0, 1, 1]))
    values = metrics.compute()

    assert "ap" not in values
    assert values["acc"].item() == pytest.approx(2 / 3)
    assert values["precision"].item() == pytest.approx(0.5)
    assert values["recall"].item() == pytest.approx(0.5)
    assert values["f1"].item() == pytest.approx(4 / 9)
    assert torch.equal(
        metrics.confusion_matrix(),
        torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),
    )


def test_perfect_predictions_have_mcc_one():
    metrics = MulticlassClassificationMetrics()
    metrics.update(_probabilities([0, 1, 2], count=3), torch.tensor([0, 1, 2]))

    assert metrics.compute()["mcc"].item() == pytest.approx(1.0)


def test_a_single_class_on_both_sides_has_mcc_zero():
    metrics = MulticlassClassificationMetrics()
    metrics.update(_probabilities([0, 0, 0], count=3), torch.tensor([0, 0, 0]))

    assert metrics.compute()["mcc"].item() == pytest.approx(0.0)


def test_reset_forgets_the_epoch():
    metrics = MulticlassClassificationMetrics()
    metrics.update(_probabilities([0, 1, 2], count=3), torch.tensor([0, 1, 2]))
    metrics.reset()

    assert metrics.compute()["acc"].item() == pytest.approx(0.0)
