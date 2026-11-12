"""Tests d'intégration : prédire avec un checkpoint produit par un vrai fit.

On entraîne un `Perceptron` avec `ClassificationModule` et les callbacks,
puis on relit le checkpoint avec `Predictor`, qui n'utilise ni `Trainer`,
ni datamodule, ni `ClassificationModule`.
"""

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import CSVLogger

from ia_xp_plateform.callback import CallbackConfig, create_callbacks
from ia_xp_plateform.factory import create_datamodule, create_model
from ia_xp_plateform.inference import Predictor
from ia_xp_plateform.module import ClassificationModule


def _train(tmp_path, sample_dataframe) -> str:
    """Entraîne 2 époques, renvoie le chemin du meilleur checkpoint."""
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
    return trainer.checkpoint_callback.best_model_path


def test_predictor_reads_a_checkpoint_written_by_a_real_fit(
    tmp_path, sample_dataframe
):
    checkpoint_path = _train(tmp_path, sample_dataframe)
    x = torch.randn(6, 2)

    predictor = Predictor.from_checkpoint(
        checkpoint_path,
        "perceptron",
        model_kwargs={"input_features": 2, "output_features": 1},
    )

    assert predictor.predict_proba(x).shape == (6, 1)
    assert predictor.predict(x).shape == (6, 1)


def test_predictor_matches_the_module_loaded_the_lightning_way(
    tmp_path, sample_dataframe
):
    checkpoint_path = _train(tmp_path, sample_dataframe)
    x = torch.randn(6, 2)

    # Côté entraînement : on recharge le checkpoint dans un
    # `ClassificationModule` vierge.
    reloaded = ClassificationModule(model=create_model("perceptron"))
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    reloaded.load_state_dict(state["state_dict"])
    reloaded.eval()
    with torch.no_grad():
        expected = reloaded(x)

    # Côté inférence : seulement le modèle et les poids.
    predictor = Predictor.from_checkpoint(checkpoint_path, "perceptron")

    assert torch.allclose(predictor.predict_proba(x), expected)
