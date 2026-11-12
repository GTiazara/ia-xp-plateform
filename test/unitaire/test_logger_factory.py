"""Tests unitaires de la factory de loggers."""

import pytest
from pytorch_lightning.loggers import LitLogger, TensorBoardLogger

from ia_xp_plateform.factory.logger_factory import (
    LOGGER_DIR_KEYS,
    LOGGERS,
    create_logger,
)


def test_create_tensorboard_forwards_the_experiment_folder(tmp_path):
    logger = create_logger("tensorboard", save_dir=tmp_path, name="perceptron")

    assert isinstance(logger, TensorBoardLogger)
    assert logger.name == "perceptron"
    assert logger.save_dir == str(tmp_path)


def test_tensorboard_numbers_each_run_so_past_runs_are_kept(tmp_path):
    first = create_logger("tensorboard", save_dir=tmp_path, name="perceptron")
    first.log_metrics({"val_loss": 0.5}, step=0)
    first.save()
    second = create_logger("tensorboard", save_dir=tmp_path, name="perceptron")

    assert first.version == 0
    assert second.version == 1


def test_create_litlogger_forwards_the_experiment_folder(tmp_path):
    logger = create_logger("litlogger", root_dir=tmp_path, name="perceptron")

    assert isinstance(logger, LitLogger)
    assert logger.name == "perceptron"
    assert logger.root_dir == str(tmp_path)


def test_unknown_logger_name_lists_the_available_ones():
    with pytest.raises(ValueError, match="tensorboard") as caught:
        create_logger("clearml")

    assert "litlogger" in str(caught.value)


def test_registry_maps_names_to_their_classes():
    assert LOGGERS["tensorboard"] is TensorBoardLogger
    assert LOGGERS["litlogger"] is LitLogger


def test_every_logger_declares_the_name_of_its_folder_argument():
    assert set(LOGGER_DIR_KEYS) == set(LOGGERS)
    assert LOGGER_DIR_KEYS["tensorboard"] == "save_dir"
    assert LOGGER_DIR_KEYS["litlogger"] == "root_dir"
