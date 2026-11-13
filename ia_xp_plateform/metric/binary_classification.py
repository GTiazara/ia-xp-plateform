"""Métriques de classification binaire, indépendantes du module.

Le module d'entraînement décide de la perte et de l'optimiseur. Cet
objet ne fait que mesurer : il accumule les comptes d'une époque, puis
rend la justesse, la précision, le rappel et le F1. Un autre module pourra
brancher un autre objet de métriques sans recopier ces comptes.
"""

import torch


class BinaryClassificationMetrics:
    """Justesse, précision, rappel et F1 d'une classification binaire.

    On lui donne des probabilités et un seuil. Il compte sur toute
    l'époque, pas lot par lot : `reset` au début, `update` à chaque
    lot, `compute` à la fin.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Remet les compteurs à zéro pour une nouvelle époque."""
        self._true_positive = 0.0
        self._false_positive = 0.0
        self._false_negative = 0.0
        self._correct = 0.0
        self._total = 0.0

    def update(
        self,
        probabilities: torch.Tensor,
        targets: torch.Tensor,
        threshold: float,
    ) -> None:
        """Ajoute un lot aux compteurs de l'époque.

        Parameters
        ----------
        probabilities : torch.Tensor
            Probabilités entre 0 et 1, même forme que `targets`.
        targets : torch.Tensor
            Cibles valant 0 ou 1.
        threshold : float
            Seuil au-dessus duquel la probabilité devient la classe 1.
        """
        predicted_positive = probabilities >= threshold
        actual_positive = targets == 1
        self._true_positive += float((predicted_positive & actual_positive).sum())
        self._false_positive += float((predicted_positive & ~actual_positive).sum())
        self._false_negative += float((~predicted_positive & actual_positive).sum())
        self._correct += float((predicted_positive == actual_positive).sum())
        self._total += float(targets.numel())

    def compute(self) -> dict[str, torch.Tensor]:
        """Justesse, précision, rappel et F1 de l'époque en cours.

        La précision est la part de vrais positifs parmi les annonces
        positives. Le rappel est la part des positifs réels retrouvés.
        Le F1 est leur moyenne harmonique :
        `2 * vrais positifs / (2 * vrais positifs + faux positifs + faux négatifs)`.
        Un dénominateur nul donne 0.

        Returns
        -------
        dict[str, torch.Tensor]
            Les clefs sont `acc`, `precision`, `recall` et `f1`.
        """
        predicted_positive = self._true_positive + self._false_positive
        actual_positive = self._true_positive + self._false_negative
        f1_denominator = (
            2 * self._true_positive + self._false_positive + self._false_negative
        )
        accuracy = self._correct / self._total if self._total > 0 else 0.0
        precision = (
            self._true_positive / predicted_positive if predicted_positive > 0 else 0.0
        )
        recall = self._true_positive / actual_positive if actual_positive > 0 else 0.0
        f1 = 2 * self._true_positive / f1_denominator if f1_denominator > 0 else 0.0
        return {
            "acc": torch.tensor(accuracy),
            "precision": torch.tensor(precision),
            "recall": torch.tensor(recall),
            "f1": torch.tensor(f1),
        }

    def confusion_matrix(self) -> torch.Tensor:
        """Matrice 2×2 de l'époque : lignes = classe réelle, colonnes = classe prédite.

        Returns
        -------
        torch.Tensor
            `[[vrais négatifs, faux positifs], [faux négatifs, vrais positifs]]`.
        """
        true_negative = (
            self._total
            - self._true_positive
            - self._false_positive
            - self._false_negative
        )
        return torch.tensor(
            [
                [true_negative, self._false_positive],
                [self._false_negative, self._true_positive],
            ]
        )
