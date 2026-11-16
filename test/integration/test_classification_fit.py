"""Tests d'intégration : `ClassificationModule` entraîné de bout en bout.

On branche ensemble les factories, le datamodule tabulaire, les
callbacks et `ClassificationModule` (avec `Perceptron`) dans un
`Trainer.fit`.
"""

from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

from ia_xp_plateform.callback import CallbackConfig, create_callbacks
from ia_xp_plateform.factory import create_datamodule, create_model
from ia_xp_plateform.module import ClassificationModule


def test_classification_module_trains_perceptron_with_table_datamodule(
    tmp_path, sample_dataframe
):
    data_path = tmp_path / "data.csv"
    sample_dataframe.to_csv(data_path, index=False)

    datamodule = create_datamodule(
        "table",
        data_path=data_path,
        target_columns="target",
        batch_size=4,
    )
    model = create_model("perceptron", input_features=2, output_features=1)
    module = ClassificationModule(model=model, learning_rate=0.05)
    callbacks = create_callbacks(
        CallbackConfig(output_dir=tmp_path, early_stopping_patience=2)
    )

    trainer = pl.Trainer(
        max_epochs=2,
        callbacks=callbacks,
        logger=CSVLogger(save_dir=tmp_path, name="logs"),
        accelerator="cpu",
        devices=1,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(module, datamodule=datamodule)

    assert "val_loss" in trainer.callback_metrics
    assert "val_acc" in trainer.callback_metrics
    assert "val_precision" in trainer.callback_metrics
    assert "val_recall" in trainer.callback_metrics
    assert "val_f1" in trainer.callback_metrics
    assert 0.0 <= trainer.callback_metrics["val_precision"] <= 1.0
    assert 0.0 <= trainer.callback_metrics["val_recall"] <= 1.0
    assert 0.0 <= trainer.callback_metrics["val_f1"] <= 1.0
    assert 0.0 <= trainer.callback_metrics["val_ap"] <= 1.0
    assert -1.0 <= trainer.callback_metrics["val_mcc"] <= 1.0
    assert (Path(tmp_path) / "checkpoints" / "best-model.ckpt").is_file()


def test_mlp_trains_as_a_binary_classifier(tmp_path, sample_dataframe):
    data_path = tmp_path / "data.csv"
    sample_dataframe.to_csv(data_path, index=False)
    datamodule = create_datamodule(
        "table",
        data_path=data_path,
        target_columns="target",
        batch_size=4,
    )
    module = ClassificationModule(
        model=create_model(
            "mlp",
            input_features=2,
            output_features=1,
            hidden_features=4,
        ),
        learning_rate=0.05,
    )
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(module, datamodule=datamodule)

    assert "val_loss" in trainer.callback_metrics
    assert "val_f1" in trainer.callback_metrics
    assert 0.0 <= trainer.callback_metrics["val_f1"] <= 1.0


def test_confusion_matrix_image_is_written_each_epoch(tmp_path, sample_dataframe):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    from pytorch_lightning.loggers import TensorBoardLogger

    data_path = tmp_path / "data.csv"
    sample_dataframe.to_csv(data_path, index=False)
    datamodule = create_datamodule(
        "table",
        data_path=data_path,
        target_columns="target",
        batch_size=4,
    )
    module = ClassificationModule(
        model=create_model("perceptron", input_features=2, output_features=1)
    )
    trainer = pl.Trainer(
        max_epochs=1,
        logger=TensorBoardLogger(save_dir=tmp_path, name="tb"),
        accelerator="cpu",
        devices=1,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(module, datamodule=datamodule)

    events = EventAccumulator(str(tmp_path / "tb" / "version_0"))
    events.Reload()

    images = events.Tags()["images"]
    assert "val_confusion_matrix" in images
    assert "train_confusion_matrix" in images
    assert "val_confusion_matrix_normalized" in images
    assert "train_confusion_matrix_normalized" in images
