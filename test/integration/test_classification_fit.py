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
    assert (Path(tmp_path) / "checkpoints" / "best-model.ckpt").is_file()
