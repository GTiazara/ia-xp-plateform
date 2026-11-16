"""Métriques de classification multiclasse, indépendantes du module.

On lui donne les probabilités d'une époque et l'indice de classe réel.
Il compte la matrice entière, puis rend la justesse, la précision, le
rappel et le F1 macro, ainsi que le coefficient de Matthews. Il n'y a
pas d'average precision : chaque ligne n'a qu'une classe.
"""

import torch


class MulticlassClassificationMetrics:
    """Justesse, scores macro et MCC sur les K classes du modèle.

    `reset` au début de l'époque, `update` à chaque lot, `compute` à la
    fin. `compute` rend `acc`, `precision`, `recall`, `f1` et `mcc`.
    Une classe absente entre dans la moyenne avec un score de 0.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Oublie la matrice de l'époque précédente."""
        self._matrix: torch.Tensor | None = None

    def update(self, probabilities: torch.Tensor, targets: torch.Tensor) -> None:
        """Ajoute un lot à la matrice de l'époque.

        Parameters
        ----------
        probabilities : torch.Tensor
            Probabilités de forme `(batch, classes)`. La classe prédite
            est l'indice le plus grand de chaque ligne.
        targets : torch.Tensor
            Indices de classe, forme `(batch,)`.
        """
        probabilities = probabilities.detach().cpu()
        targets = targets.detach().cpu().reshape(-1).long()
        predicted = probabilities.argmax(dim=-1).reshape(-1)
        classes = int(probabilities.shape[-1])
        if self._matrix is None:
            self._matrix = torch.zeros(classes, classes)
        flat = targets * classes + predicted
        counts = torch.bincount(flat, minlength=classes * classes)
        self._matrix += counts.reshape(classes, classes).float()

    def confusion_matrix(self) -> torch.Tensor:
        """Effectifs de l'époque : ligne = classe réelle, colonne = classe prédite."""
        if self._matrix is None:
            return torch.zeros(0, 0)
        return self._matrix.clone()

    def compute(self) -> dict[str, torch.Tensor]:
        """Scores macro et coefficient de Matthews de l'époque.

        Returns
        -------
        dict[str, torch.Tensor]
            Les clefs sont `acc`, `precision`, `recall`, `f1` et `mcc`.
        """
        matrix = self._matrix
        if matrix is None or matrix.numel() == 0:
            return {name: torch.tensor(0.0) for name in ("acc", "precision", "recall", "f1", "mcc")}
        true_positive = matrix.diag()
        predicted = matrix.sum(dim=0)
        actual = matrix.sum(dim=1)
        false_positive = predicted - true_positive
        false_negative = actual - true_positive
        precision = torch.where(predicted > 0, true_positive / predicted, torch.zeros_like(predicted))
        recall = torch.where(actual > 0, true_positive / actual, torch.zeros_like(actual))
        f1_denominator = 2 * true_positive + false_positive + false_negative
        f1 = torch.where(f1_denominator > 0, 2 * true_positive / f1_denominator, torch.zeros_like(f1_denominator))
        total = float(matrix.sum())
        accuracy = float(true_positive.sum()) / total if total > 0 else 0.0
        return {
            "acc": torch.tensor(accuracy),
            "precision": torch.tensor(float(precision.mean())),
            "recall": torch.tensor(float(recall.mean())),
            "f1": torch.tensor(float(f1.mean())),
            "mcc": torch.tensor(self._matthews(matrix)),
        }

    def _matthews(self, matrix: torch.Tensor) -> float:
        """MCC multiclasse. Un dénominateur nul donne 0."""
        correct = float(matrix.diag().sum())
        total = float(matrix.sum())
        predicted = matrix.sum(dim=0)
        actual = matrix.sum(dim=1)
        numerator = correct * total - float((predicted * actual).sum())
        denominator = (total ** 2 - float((predicted ** 2).sum())) * (
            total ** 2 - float((actual ** 2).sum())
        )
        if denominator <= 0:
            return 0.0
        return numerator / (denominator ** 0.5)
