"""Tâche de classification multiclasse.

Le modèle renvoie un logit par classe. Cette tâche en fait une
distribution, une perte et un indice de classe. Elle refuse une cible
qui n'est pas un indice entier avant d'appeler le critère.
"""

import torch
from torch import nn

from ia_xp_plateform.metric.multiclass import MulticlassClassificationMetrics

_INTEGER_TOLERANCE = 1e-5


class MulticlassTask:
    """Softmax, entropie croisée et argmax, pour des logits.

    `loss` est le nom de la perte. Seule `cross_entropy` existe :
    l'entropie croisée calculée sur les logits et un indice de classe.
    Le nombre de classes est le nombre de logits.
    """

    def __init__(self, loss: str = "cross_entropy") -> None:
        """Choisit la perte de la tâche multiclasse.

        Parameters
        ----------
        loss : str, optional
            Nom de la perte. Seule `cross_entropy` est acceptée.
            Par défaut `cross_entropy`.

        Raises
        ------
        ValueError
            Si `loss` n'est pas `cross_entropy`.
        """
        if loss != "cross_entropy":
            raise ValueError(
                f"Perte inconnue pour la tâche multiclasse : '{loss}'. "
                "Pertes disponibles : cross_entropy."
            )
        self.loss_name = loss
        self.criterion = nn.CrossEntropyLoss()

    def probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        """Distribution de probabilités, un softmax par ligne.

        Parameters
        ----------
        logits : torch.Tensor
            Sortie du modèle, sans activation, de forme `(batch, classes)`.

        Returns
        -------
        torch.Tensor
            Même forme. Chaque ligne est strictement positive et somme à 1.
        """
        return torch.softmax(logits, dim=-1)

    def loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Entropie croisée des logits et de l'indice de classe.

        Les logits ont la forme `(N, K)` avec `K >= 2`. La cible a la
        forme `(N, 1)` ou `(N,)` et chaque valeur est un entier de
        `0` à `K - 1`. Sinon une `ValueError` est levée et le critère
        n'est pas appelé.

        Parameters
        ----------
        logits : torch.Tensor
            Sortie du modèle, sans activation.
        targets : torch.Tensor
            Indices de classe.

        Returns
        -------
        torch.Tensor
            Perte scalaire, moyenne sur le lot.
        """
        indices = self.class_indices(logits, targets)
        return self.criterion(logits, indices)

    def predict(self, logits: torch.Tensor, threshold: float) -> torch.Tensor:
        """Indice de la plus grande coordonnée.

        Parameters
        ----------
        logits : torch.Tensor
            Sortie du modèle, sans activation.
        threshold : float
            Accepté pour la même signature que la tâche binaire, et ignoré.

        Returns
        -------
        torch.Tensor
            Entiers, forme `(batch, 1)`. Deux logits égaux gardent le
            plus petit indice.
        """
        del threshold
        return logits.argmax(dim=-1, keepdim=True)

    def new_metrics(self) -> MulticlassClassificationMetrics:
        """Compteurs multiclasses d'une époque, encore vides."""
        return MulticlassClassificationMetrics()

    def observe(
        self,
        metrics: MulticlassClassificationMetrics,
        probabilities: torch.Tensor,
        targets: torch.Tensor,
        threshold: float,
    ) -> None:
        """Compte le lot. Le seuil ne change pas la classe."""
        del threshold
        metrics.update(probabilities, self.class_indices(probabilities, targets))

    def class_indices(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Indices `long` de forme `(N,)`, ou une erreur avant la perte.

        Une valeur à moins de `1e-5` de son entier le plus proche est
        cet entier. `2.0` est donc l'indice 2.
        """
        if logits.ndim != 2 or logits.shape[1] < 2:
            received = int(logits.shape[-1]) if logits.ndim > 0 else 0
            raise ValueError(
                "La tâche multiclasse attend au moins 2 logits "
                f"(reçu : {received})."
            )
        original_shape = tuple(targets.shape)
        if targets.ndim == 2 and targets.shape[1] == 1:
            targets = targets[:, 0]
        rows = logits.shape[0]
        if targets.ndim != 1 or targets.shape[0] != rows:
            raise ValueError(
                "La tâche multiclasse attend une cible de forme (N, 1) "
                f"ou (N,) (logits {tuple(logits.shape)}, "
                f"cible {original_shape})."
            )
        if targets.numel() == 0:
            return targets.round().long()
        nearest = targets.round()
        if bool((targets - nearest).abs().max() > _INTEGER_TOLERANCE):
            raise ValueError(
                "La tâche multiclasse attend un indice de classe entier."
            )
        indices = nearest.long()
        last = int(logits.shape[1]) - 1
        if bool(((indices < 0) | (indices > last)).any()):
            raise ValueError(
                "La tâche multiclasse attend un indice de classe "
                f"dans 0 .. {last}."
            )
        return indices
