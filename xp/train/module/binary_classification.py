"""Tâche d'apprentissage pour une décision binaire (ex: XP-01 : accélérer ou non).

Joue le même rôle que `bdf.train.module.classification.ClassificationModel`
dans l'architecture `bdf-model` : assembler le modèle, l'optimiseur, la
fonction de perte et les métriques, et implémenter les étapes
d'entraînement / validation / test / prédiction attendues par PyTorch
Lightning.

Le module est volontairement agnostique du modèle utilisé (il peut recevoir
un `Perceptron`, mais aussi un futur `MLP` ou `CNN` pour les XP suivantes) :
il attend seulement un modèle qui prend `(batch, in_features)` et renvoie
`(batch,)` logits, et un batch sous la forme `{"features": ..., "label": ...}`.
"""
from typing import Literal

import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim import Optimizer
from torchmetrics import MeanMetric
from torchmetrics.classification import BinaryAccuracy
from torchmetrics.classification import BinaryF1Score
from torchmetrics.classification import BinaryPrecision
from torchmetrics.classification import BinaryRecall

from xp.cli.utils import get_pylogger

log = get_pylogger(__name__)


class BinaryDecisionModule(pl.LightningModule):
    """Module Lightning générique pour une tâche de décision binaire.

    Parameters
    ----------
    model : nn.Module
        Le modèle à entraîner (ex: `xp.train.models.perceptron.Perceptron`).
        Doit renvoyer un logit par échantillon (sortie de forme `(batch,)`).
    optimizer : Optimizer
        Optimiseur partiellement instancié par Hydra (`_partial_: true`),
        auquel il ne manque que les paramètres du modèle.
    loss : nn.Module
        Fonction de perte, typiquement `torch.nn.BCEWithLogitsLoss`.
    threshold : float, optional
        Seuil de décision appliqué à `sigmoid(logit)`, par défaut 0.5.
    optim_metric : str, optional
        Nom de la métrique à surveiller (pour les callbacks), par défaut "loss/val".
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        loss: nn.Module,
        threshold: float = 0.5,
        optim_metric: str = "loss/val",
    ) -> None:
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss
        self.threshold = threshold
        self.optim_metric = optim_metric

    def setup(self, stage: Literal["fit", "validate", "test", "predict"]) -> None:
        """Initialise les métriques nécessaires au `stage` demandé."""
        if stage in ("fit", "validate"):
            self.mean_train_loss = MeanMetric()
            self.mean_val_loss = MeanMetric()
            self.train_metrics = self._build_metrics()
            self.val_metrics = self._build_metrics()
        elif stage == "test":
            self.mean_test_loss = MeanMetric()
            self.test_metrics = self._build_metrics()

    def _build_metrics(self) -> nn.ModuleDict:
        """Construit le jeu de métriques standard pour une décision binaire."""
        return nn.ModuleDict(
            {
                "accuracy": BinaryAccuracy(threshold=self.threshold),
                "precision": BinaryPrecision(threshold=self.threshold),
                "recall": BinaryRecall(threshold=self.threshold),
                "f1": BinaryF1Score(threshold=self.threshold),
            }
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Calcule les logits pour un batch d'entrées."""
        return self.model(features)

    def step(self, batch: dict) -> dict:
        """Calcule la perte, les probabilités et les prédictions pour un batch.

        Parameters
        ----------
        batch : dict
            Dictionnaire `{"features": Tensor, "label": Tensor}`.

        Returns
        -------
        dict
            `{"loss", "probs", "preds", "targets"}`.
        """
        logits = self.model(batch["features"])
        targets = batch["label"].float()
        loss = self.loss_fn(logits, targets)
        probs = torch.sigmoid(logits)
        preds = (probs > self.threshold).float()
        return {"loss": loss, "probs": probs, "preds": preds, "targets": targets}

    def training_step(self, batch: dict, *args, **kwargs) -> dict:
        out = self.step(batch)
        self.mean_train_loss.update(out["loss"])
        for metric in self.train_metrics.values():
            metric.update(out["probs"], out["targets"].int())
        return out

    def on_train_epoch_end(self) -> None:
        """Log et reset les métriques d'entraînement en fin d'époque."""
        metrics = {f"{name}/train": metric.compute() for name, metric in self.train_metrics.items()}
        metrics["loss/train"] = self.mean_train_loss.compute()
        self.log_dict(metrics, on_epoch=True, prog_bar=True)

        self.mean_train_loss.reset()
        for metric in self.train_metrics.values():
            metric.reset()

    def validation_step(self, batch: dict, *args, **kwargs) -> dict:
        out = self.step(batch)
        self.mean_val_loss.update(out["loss"])
        for metric in self.val_metrics.values():
            metric.update(out["probs"], out["targets"].int())
        return out

    def on_validation_epoch_end(self) -> None:
        """Log et reset les métriques de validation en fin d'époque."""
        metrics = {f"{name}/val": metric.compute() for name, metric in self.val_metrics.items()}
        metrics["loss/val"] = self.mean_val_loss.compute()
        self.log_dict(metrics, on_epoch=True, prog_bar=True)

        self.mean_val_loss.reset()
        for metric in self.val_metrics.values():
            metric.reset()

    def test_step(self, batch: dict, *args, **kwargs) -> dict:
        out = self.step(batch)
        self.mean_test_loss.update(out["loss"])
        for metric in self.test_metrics.values():
            metric.update(out["probs"], out["targets"].int())
        return out

    def on_test_epoch_end(self) -> None:
        """Log les métriques de test en fin d'époque."""
        metrics = {f"{name}/test": metric.compute() for name, metric in self.test_metrics.items()}
        metrics["loss/test"] = self.mean_test_loss.compute()
        self.log_dict(metrics, on_epoch=True, prog_bar=True)

    def predict_step(self, batch: dict, *args, **kwargs) -> dict:
        """Renvoie la probabilité d'accélérer et la décision associée."""
        logits = self.model(batch["features"])
        probs = torch.sigmoid(logits)
        preds = (probs > self.threshold).long()
        return {"probs": probs, "preds": preds}

    def configure_optimizers(self) -> Optimizer:
        """Instancie l'optimiseur (partiel) avec les paramètres du modèle."""
        return self.optimizer(params=self.model.parameters())
