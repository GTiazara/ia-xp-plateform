"""

Un modèle (`BaseModel`) ne sait faire qu'une chose : transformer un batch
d'entrées en une sortie. Il ne sait pas comment on mesure son erreur, ni
avec quel optimiseur on met ses poids à jour, ni quelles métriques on
journalise. C'est le rôle du `LightningModule`.

`ClassificationModule` porte cette tâche de classification pour n'importe
quel `BaseModel`. Aujourd'hui on lui donne un `Perceptron`, plus tard un
MLP ou un autre réseau : le module d'entraînement ne change pas.

"""

import pytorch_lightning as pl
import torch

from ia_xp_plateform.factory.task_factory import create_task
from ia_xp_plateform.metric import (
    confusion_matrix_figure,
    normalize_confusion_matrix,
)
from ia_xp_plateform.models.base import BaseModel
from ia_xp_plateform.task.binary import BinaryTask
from ia_xp_plateform.task.multiclass import MulticlassTask


class ClassificationModule(pl.LightningModule):
    """Entraînement d'un `BaseModel` pour la tâche reçue.

    Le modèle reçu doit renvoyer des logits. La tâche en fait une
    probabilité, une perte et une classe. Sans tâche nommée, c'est la
    tâche binaire qui est utilisée.
    """

    def __init__(
        self,
        model: BaseModel,
        learning_rate: float = 1e-2,
        threshold: float = 0.5,
        task: BinaryTask | MulticlassTask | None = None,
    ) -> None:
        """Initialise la tâche de classification.

        Parameters
        ----------
        model : BaseModel
            N'importe quel modèle qui respecte le contrat `BaseModel`
            et renvoie des logits.
        learning_rate : float, optional
            Taux d'apprentissage de l'optimiseur Adam. Par défaut `1e-2`.
        threshold : float, optional
            Seuil au-dessus duquel la probabilité devient la classe `1`.
            Par défaut `0.5`.
        task : BinaryTask | MulticlassTask | None, optional
            Tâche qui calcule la perte, la probabilité et les métriques.
            Par défaut, la tâche binaire et sa perte `bce`.
        """
        super().__init__()
        if not isinstance(model, BaseModel):
            raise TypeError(
                "model doit hériter de BaseModel "
                f"(reçu : {type(model).__name__})."
            )
        if learning_rate <= 0:
            raise ValueError(
                f"learning_rate doit être strictement positif ({learning_rate})."
            )
        if not 0.0 < threshold < 1.0:
            raise ValueError(
                f"threshold doit être entre 0 et 1 ({threshold})."
            )

        self.model = model
        self.learning_rate = learning_rate
        self.threshold = threshold
        # La tâche porte la sigmoïde et la perte. Le critère est aussi
        # un attribut du module : le résumé Lightning continue de l'afficher.
        self.task = create_task("binary") if task is None else task
        self.criterion = self.task.criterion
        # Un compteur par étape, choisi par la tâche. Le module
        # journalise chaque clef de `compute()` : `val_ap` n'existe que
        # si la tâche binaire la fournit.
        self.metrics = {
            stage: self.task.new_metrics()
            for stage in ("train", "val", "test")
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Probabilité de la tâche appliquée aux logits du modèle."""
        return self.task.probabilities(self.model(x))

    def _step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        stage: str,
    ) -> torch.Tensor:
        """Calcule la perte et alimente les métriques de l'étape.

        Les noms journalisés sont ceux que `compute()` de la tâche
        renvoie, préfixés par l'étape.
        """
        x, y = batch
        logits = self.model(x)
        loss = self.task.loss(logits, y)
        probabilities = self.task.probabilities(logits)
        self.task.observe(self.metrics[stage], probabilities, y, self.threshold)

        # `val_loss` est le nom attendu par `ModelCheckpoint` et
        # `EarlyStopping` dans `create_callbacks`.
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True)
        return loss

    def _log_metrics(self, stage: str) -> None:
        """Publie les métriques et l'image de la matrice de confusion."""
        for name, value in self.metrics[stage].compute().items():
            self.log(f"{stage}_{name}", value, on_step=False, on_epoch=True)
        self._log_confusion_matrix(stage)

    def _log_confusion_matrix(self, stage: str) -> None:
        """Envoie la matrice de l'époque aux loggers qui acceptent une figure.

        TensorBoard range les effectifs sous `{stage}_confusion_matrix`
        et la part de chaque classe réelle sous
        `{stage}_confusion_matrix_normalized`, une image par époque.
        Un logger sans `add_figure` (le CSV, par exemple) est ignoré.
        """
        counts = self.metrics[stage].confusion_matrix()
        figures = {
            f"{stage}_confusion_matrix": confusion_matrix_figure(
                counts,
                title=f"Matrice de confusion ({stage})",
            ),
            f"{stage}_confusion_matrix_normalized": confusion_matrix_figure(
                normalize_confusion_matrix(counts),
                title=f"Matrice de confusion normalisée ({stage})",
                normalized=True,
            ),
        }
        step = int(self.current_epoch)
        for logger in self.loggers:
            experiment = getattr(logger, "experiment", None)
            add_figure = getattr(experiment, "add_figure", None)
            if add_figure is None:
                continue
            for tag, figure in figures.items():
                add_figure(tag, figure, global_step=step)
        for figure in figures.values():
            figure.clear()

    def on_train_epoch_start(self) -> None:
        self.metrics["train"].reset()

    def on_validation_epoch_start(self) -> None:
        self.metrics["val"].reset()

    def on_test_epoch_start(self) -> None:
        self.metrics["test"].reset()

    def on_train_epoch_end(self) -> None:
        self._log_metrics("train")

    def on_validation_epoch_end(self) -> None:
        self._log_metrics("val")

    def on_test_epoch_end(self) -> None:
        self._log_metrics("test")

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        return self._step(batch, "val")

    def test_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        return self._step(batch, "test")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Optimiseur Adam sur les poids du modèle."""
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
