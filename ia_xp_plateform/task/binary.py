"""Tâche de classification binaire.

Le modèle renvoie un logit. Cette tâche en fait une probabilité, une
perte et une classe. Elle refuse une cible que la perte ne sait pas
mesurer, avant d'appeler le critère.
"""

import torch
from torch import nn

from ia_xp_plateform.metric.binary_classification import BinaryClassificationMetrics


class BinaryTask:
    """Sigmoïde, entropie croisée binaire et seuil, pour des logits.

    `loss` est le nom de la perte. Aujourd'hui seule `bce` existe :
    l'entropie croisée binaire calculée sur les logits, qui reste finie
    quand la sigmoïde sature.
    """

    def __init__(self, loss: str = "bce") -> None:
        """Choisit la perte de la tâche binaire.

        Parameters
        ----------
        loss : str, optional
            Nom de la perte. Seule `bce` est acceptée. Par défaut `bce`.

        Raises
        ------
        ValueError
            Si `loss` n'est pas `bce`.
        """
        if loss != "bce":
            raise ValueError(
                f"Perte inconnue pour la tâche binaire : '{loss}'. "
                "Pertes disponibles : bce."
            )
        self.loss_name = loss
        self.criterion = nn.BCEWithLogitsLoss()

    def probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        """Probabilité de la classe 1, une sigmoïde par logit.

        Parameters
        ----------
        logits : torch.Tensor
            Sortie du modèle, sans activation.

        Returns
        -------
        torch.Tensor
            Même forme, chaque valeur dans `(0, 1)`.
        """
        return torch.sigmoid(logits)

    def loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Entropie croisée binaire des logits.

        La forme des logits et celle de la cible doivent être identiques.
        Chaque cible doit être dans `[0, 1]`. Sinon une `ValueError` est
        levée et le critère n'est pas appelé.

        Parameters
        ----------
        logits : torch.Tensor
            Sortie du modèle, sans activation.
        targets : torch.Tensor
            Cibles dans `[0, 1]`, de même forme que `logits`.

        Returns
        -------
        torch.Tensor
            Perte scalaire, moyenne sur le lot.
        """
        if logits.shape != targets.shape:
            raise ValueError(
                "La tâche binaire attend des logits et une cible de même "
                f"forme (logits {tuple(logits.shape)}, "
                f"cible {tuple(targets.shape)})."
            )
        outside = (targets < 0) | (targets > 1)
        if bool(outside.any()):
            raise ValueError(
                "La tâche binaire attend une cible dans [0, 1]."
            )
        return self.criterion(logits, targets)

    def predict(self, logits: torch.Tensor, threshold: float) -> torch.Tensor:
        """Classe 1 si la probabilité atteint le seuil, 0 sinon.

        Parameters
        ----------
        logits : torch.Tensor
            Sortie du modèle, sans activation.
        threshold : float
            Seuil appliqué à la probabilité, pas au logit.

        Returns
        -------
        torch.Tensor
            Entiers `0` ou `1`, de même forme que `logits`.
        """
        return (self.probabilities(logits) >= threshold).long()

    def new_metrics(self) -> BinaryClassificationMetrics:
        """Compteurs binaires d'une époque, encore vides."""
        return BinaryClassificationMetrics()

    def observe(
        self,
        metrics: BinaryClassificationMetrics,
        probabilities: torch.Tensor,
        targets: torch.Tensor,
        threshold: float,
    ) -> None:
        """Compte le lot au seuil reçu."""
        metrics.update(probabilities, targets, threshold)
