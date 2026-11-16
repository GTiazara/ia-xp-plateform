"""Tests de la factory de tâches."""

import pytest

from ia_xp_plateform.factory import TASKS, create_task
from ia_xp_plateform.task.binary import BinaryTask
from ia_xp_plateform.task.multiclass import MulticlassTask


def test_binary_and_multiclass_are_registered():
    assert set(TASKS) == {"binary", "multiclass"}
    binary = create_task("binary")
    multiclass = create_task("multiclass")

    assert isinstance(binary, BinaryTask)
    assert binary.loss_name == "bce"
    assert isinstance(multiclass, MulticlassTask)
    assert multiclass.loss_name == "cross_entropy"


def test_an_unknown_task_names_binary_and_multiclass():
    with pytest.raises(ValueError, match="binary") as error:
        create_task("inconnu")

    assert "multiclass" in str(error.value)


def test_an_unknown_loss_names_the_accepted_one():
    with pytest.raises(ValueError, match="bce"):
        create_task("binary", loss="mse")
    with pytest.raises(ValueError, match="cross_entropy"):
        create_task("multiclass", loss="bce")
