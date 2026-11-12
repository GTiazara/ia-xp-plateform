"""Tests unitaires de `create_callbacks`."""

from pathlib import Path

import pytest
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

from ia_xp_plateform.callback import CallbackConfig, create_callbacks


def test_returns_the_three_callbacks_in_order(tmp_path):
    callbacks = create_callbacks(
        CallbackConfig(output_dir=tmp_path, early_stopping_patience=3)
    )

    assert [type(callback) for callback in callbacks] == [
        ModelCheckpoint,
        EarlyStopping,
        LearningRateMonitor,
    ]


def test_checkpoint_watches_val_loss_and_keeps_one_file(tmp_path):
    checkpoint = create_callbacks(
        CallbackConfig(output_dir=tmp_path, early_stopping_patience=2)
    )[0]

    assert checkpoint.dirpath == str(Path(tmp_path) / "checkpoints")
    assert checkpoint.filename == "best-model"
    assert checkpoint.monitor == "val_loss"
    assert checkpoint.mode == "min"
    assert checkpoint.save_top_k == 1


def test_early_stopping_uses_the_configured_patience(tmp_path):
    early_stopping = create_callbacks(
        CallbackConfig(output_dir=tmp_path, early_stopping_patience=5)
    )[1]

    assert early_stopping.monitor == "val_loss"
    assert early_stopping.patience == 5
    assert early_stopping.mode == "min"


def test_learning_rate_is_logged_every_epoch(tmp_path):
    lr_monitor = create_callbacks(
        CallbackConfig(output_dir=tmp_path, early_stopping_patience=1)
    )[2]

    assert lr_monitor.logging_interval == "epoch"


def test_patience_must_be_at_least_one(tmp_path):
    with pytest.raises(ValueError):
        CallbackConfig(output_dir=tmp_path, early_stopping_patience=0)
