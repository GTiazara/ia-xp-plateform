"""Tests de la tâche multiclasse : softmax, argmax, entropie croisée."""

import math

import pytest
import torch

from ia_xp_plateform.task.multiclass import MulticlassTask


def test_equal_logits_are_a_uniform_softmax_and_cost_log_k():
    task = MulticlassTask()
    logits = torch.zeros(1, 3)
    probabilities = task.probabilities(logits)

    assert torch.allclose(probabilities, torch.full((1, 3), 1 / 3))
    assert probabilities.sum(dim=-1).item() == pytest.approx(1.0)
    assert task.loss(logits, torch.tensor([[0.0]])).item() == pytest.approx(math.log(3))


def test_argmax_ignores_the_threshold_and_keeps_the_smallest_index():
    task = MulticlassTask()
    dominant = torch.tensor([[0.0, 5.0, 1.0]])
    tied = torch.tensor([[2.0, 2.0, 0.0]])

    assert task.predict(dominant, threshold=0.1).item() == 1
    assert task.predict(dominant, threshold=0.9).item() == 1
    assert task.predict(tied, threshold=0.5).item() == 0


def test_a_large_logit_on_the_wrong_class_stays_finite():
    task = MulticlassTask()
    loss = task.loss(torch.tensor([[80.0, 0.0, 0.0]]), torch.tensor([[1.0]]))

    assert math.isfinite(loss.item())
    assert loss.item() > 0.0


def test_a_whole_float_is_a_class_index():
    task = MulticlassTask()
    loss = task.loss(torch.zeros(1, 3), torch.tensor([[2.0]]))

    assert math.isfinite(loss.item())


def test_an_index_past_the_last_class_names_the_interval():
    task = MulticlassTask()

    with pytest.raises(ValueError, match=r"0 .. 2") as error:
        task.loss(torch.zeros(1, 3), torch.tensor([[3.0]]))

    message = str(error.value)
    assert "0" in message
    assert "2" in message


def test_a_fractional_target_is_refused():
    task = MulticlassTask()

    with pytest.raises(ValueError):
        task.loss(torch.zeros(1, 3), torch.tensor([[1.5]]))


def test_a_single_logit_names_that_count():
    task = MulticlassTask()

    with pytest.raises(ValueError, match="1"):
        task.loss(torch.zeros(2, 1), torch.zeros(2, 1))


def test_an_unknown_loss_names_cross_entropy():
    with pytest.raises(ValueError, match="cross_entropy"):
        MulticlassTask(loss="bce")
