"""Un MLP à trois logits s'entraîne comme une tâche multiclasse."""

from types import SimpleNamespace

import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

from ia_xp_plateform.callback import CallbackConfig, create_callbacks
from ia_xp_plateform.cli.command.predict import run_predict
from ia_xp_plateform.factory import create_datamodule, create_model, create_task
from ia_xp_plateform.module import ClassificationModule


def _three_class_frame() -> pd.DataFrame:
    """Trente lignes, une colonne cible qui vaut 0, 1 ou 2."""
    return pd.DataFrame(
        {
            "id": list(range(30)),
            "a": [float(index) for index in range(30)],
            "b": [float(index % 3) for index in range(30)],
            "target": [float(index % 3) for index in range(30)],
        }
    )


def test_multiclass_fit_logs_macro_scores_and_writes_one_prediction(tmp_path):
    data_path = tmp_path / "data.csv"
    _three_class_frame().to_csv(data_path, index=False)
    datamodule = create_datamodule(
        "table",
        data_path=data_path,
        target_columns="target",
        batch_size=8,
    )
    module = ClassificationModule(
        model=create_model("mlp", input_features=2, output_features=3),
        learning_rate=0.05,
        task=create_task("multiclass"),
    )
    trainer = pl.Trainer(
        max_epochs=1,
        callbacks=create_callbacks(
            CallbackConfig(output_dir=tmp_path, early_stopping_patience=2)
        ),
        accelerator="cpu",
        devices=1,
        logger=CSVLogger(save_dir=tmp_path, name="logs"),
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(module, datamodule=datamodule)
    metrics = trainer.callback_metrics

    assert 0.0 <= metrics["val_acc"] <= 1.0
    assert 0.0 <= metrics["val_f1"] <= 1.0
    assert -1.0 <= metrics["val_mcc"] <= 1.0
    assert "val_ap" not in metrics

    predictions = tmp_path / "predictions.csv"
    run_predict(
        SimpleNamespace(
            data=str(data_path),
            hdf_key=None,
            features=["a", "b"],
            target="target",
            id_column="id",
            input_features=2,
            output_features=3,
            hidden_features=None,
            hidden_layers=None,
            model="mlp",
            threshold=0.5,
            checkpoint=trainer.checkpoint_callback.best_model_path,
            output=str(predictions),
            task="multiclass",
            loss="cross_entropy",
        )
    )
    written = pd.read_csv(predictions)

    assert list(written.columns) == [
        "id",
        "prediction",
        "probability_0",
        "probability_1",
        "probability_2",
    ]
