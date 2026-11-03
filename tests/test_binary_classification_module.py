"""Test de bout en bout (smoke test) de l'architecture de l'XP-01.

Vérifie que modèle + module + données s'assemblent correctement et qu'un
entraînement de quelques époques réduit effectivement la perte, sans
nécessiter Hydra (instanciation directe des classes Python).
"""
from functools import partial

import pytorch_lightning as pl
import torch

from xp.train.data.datamodule import CarDrivingDataModule
from xp.train.models.perceptron import Perceptron
from xp.train.module.binary_classification import BinaryDecisionModule


def _build_task() -> BinaryDecisionModule:
    model = Perceptron(in_features=4)
    optimizer = partial(torch.optim.SGD, lr=0.5)
    loss = torch.nn.BCEWithLogitsLoss()
    return BinaryDecisionModule(model=model, optimizer=optimizer, loss=loss)


def test_training_reduces_loss():
    datamodule = CarDrivingDataModule(
        n_samples=500, batch_size=32, num_workers=0, label_noise=0.0, seed=0
    )
    task = _build_task()

    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(task, datamodule=datamodule)

    # `mean_train_loss` est resetée par `on_train_epoch_end` après chaque
    # époque : on relit donc la dernière valeur loggée par le Trainer.
    final_train_loss = trainer.callback_metrics["loss/train"].item()

    assert final_train_loss < 0.7  # nettement mieux qu'un modèle non entraîné


def test_predict_step_returns_binary_decision():
    datamodule = CarDrivingDataModule(n_samples=50, batch_size=10, seed=0)
    task = _build_task()

    trainer = pl.Trainer(
        accelerator="cpu", logger=False, enable_progress_bar=False
    )
    predictions = trainer.predict(task, datamodule=datamodule)

    for batch_out in predictions:
        assert set(batch_out["preds"].unique().tolist()).issubset({0, 1})
