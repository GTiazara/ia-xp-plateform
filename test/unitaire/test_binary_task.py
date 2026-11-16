"""Tests de la tâche binaire : sigmoïde, perte sur les logits, seuil."""

import math

import pytest
import torch

from ia_xp_plateform.task.binary import BinaryTask


def test_zero_logit_and_positive_target_give_log_two():
    task = BinaryTask()
    logits = torch.tensor([[0.0]])
    targets = torch.tensor([[1.0]])

    assert task.probabilities(logits).item() == pytest.approx(0.5)
    assert task.loss(logits, targets).item() == pytest.approx(math.log(2))


def test_a_large_logit_against_a_negative_target_stays_finite():
    task = BinaryTask()
    loss = task.loss(torch.tensor([[80.0]]), torch.tensor([[0.0]]))

    assert math.isfinite(loss.item())
    assert loss.item() > 0.0


def test_a_fractional_target_is_accepted():
    task = BinaryTask()
    loss = task.loss(torch.tensor([[0.0]]), torch.tensor([[0.3]]))

    assert math.isfinite(loss.item())


def test_threshold_is_applied_to_the_probability():
    task = BinaryTask()
    logits = torch.tensor([[1.0]])

    assert task.predict(logits, threshold=0.5).item() == 1
    assert task.predict(logits, threshold=0.9).item() == 0


def test_a_target_outside_the_unit_interval_is_refused():
    task = BinaryTask()

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        task.loss(torch.tensor([[0.0]]), torch.tensor([[2.0]]))


def test_different_shapes_are_named_in_the_error():
    task = BinaryTask()
    logits = torch.zeros(2, 3)
    targets = torch.zeros(2, 1)

    with pytest.raises(ValueError, match=r"\(2, 3\).*\(2, 1\)"):
        task.loss(logits, targets)


def test_an_unknown_loss_names_bce():
    with pytest.raises(ValueError, match="bce"):
        BinaryTask(loss="mse")
