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
from torch import nn

from ia_xp_plateform.models.base import BaseModel


class ClassificationModule(pl.LightningModule):
    """Tâche de classification binaire autour d'un `BaseModel`.

    Le modèle reçu doit renvoyer une probabilité entre 0 et 1, de même
    forme que la cible `y` (comme le fait `Perceptron` avec sa sigmoïde).
    """

    def __init__(
        self,
        model: BaseModel,
        learning_rate: float = 1e-2,
        threshold: float = 0.5,
    ) -> None:
        """Initialise la tâche de classification.

        Parameters
        ----------
        model : BaseModel
            N'importe quel modèle qui respecte le contrat `BaseModel`.
        learning_rate : float, optional
            Taux d'apprentissage de l'optimiseur Adam. Par défaut `1e-2`.
        threshold : float, optional
            Seuil au-dessus duquel la probabilité devient la classe `1`.
            Par défaut `0.5`.
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
        # `BCELoss` attend une probabilité entre 0 et 1, ce que renvoie
        # déjà `Perceptron` après sa sigmoïde.
        self.criterion = nn.BCELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Délègue le calcul au modèle sous-jacent."""
        return self.model(x)

    def _step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        stage: str,
    ) -> torch.Tensor:
        """Calcule la perte et la précision pour `train`, `val` ou `test`."""
        x, y = batch
        probabilities = self(x)
        loss = self.criterion(probabilities, y)

        predictions = (probabilities >= self.threshold).float()
        accuracy = (predictions == y).float().mean()

        # `val_loss` est le nom attendu par `ModelCheckpoint` et
        # `EarlyStopping` dans `create_callbacks`.
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True)
        self.log(f"{stage}_acc", accuracy, on_step=False, on_epoch=True)
        return loss

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
