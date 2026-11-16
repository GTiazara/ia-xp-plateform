"""Métriques de classification binaire, indépendantes du module.

Le module d'entraînement décide de la perte et de l'optimiseur. Cet
objet ne fait que mesurer : il accumule les comptes d'une époque, puis
rend la justesse, la précision, le rappel, le F1, l'average precision
(`ap`) et le coefficient de Matthews (`mcc`). Un autre module pourra
brancher un autre objet de métriques sans recopier ces comptes.
"""

import torch


class BinaryClassificationMetrics:
    """Justesse, précision, rappel, F1, average precision et MCC.

    On lui donne des probabilités et un seuil. Il compte sur toute
    l'époque, pas lot par lot : `reset` au début, `update` à chaque
    lot, `compute` à la fin. `compute` rend `acc`, `precision`,
    `recall`, `f1`, `ap` et `mcc`, journalisés `{stage}_ap` et
    `{stage}_mcc`.
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
        self._probabilities: list[torch.Tensor] = []
        self._targets: list[torch.Tensor] = []

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
        probabilities = probabilities.detach().cpu().reshape(-1)
        targets = targets.detach().cpu().reshape(-1)
        predicted_positive = probabilities >= threshold
        actual_positive = targets == 1
        self._true_positive += float((predicted_positive & actual_positive).sum())
        self._false_positive += float((predicted_positive & ~actual_positive).sum())
        self._false_negative += float((~predicted_positive & actual_positive).sum())
        self._correct += float((predicted_positive == actual_positive).sum())
        self._total += float(targets.numel())
        self._probabilities.append(probabilities)
        self._targets.append(targets)

    def compute(self) -> dict[str, torch.Tensor]:
        """Métriques de l'époque en cours.

        La précision est la part de vrais positifs parmi les annonces
        positives. Le rappel est la part des positifs réels retrouvés.
        Le F1 est leur moyenne harmonique :
        `2 * vrais positifs / (2 * vrais positifs + faux positifs + faux négatifs)`.
        Un dénominateur nul donne 0. `ap` ignore le seuil. `mcc` utilise
        les quatre cases de la matrice au seuil reçu par `update`.

        Returns
        -------
        dict[str, torch.Tensor]
            Les clefs sont `acc`, `precision`, `recall`, `f1`, `ap` et `mcc`.
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
            "ap": torch.tensor(self._average_precision()),
            "mcc": torch.tensor(self._matthews_correlation()),
        }

    def _average_precision(self) -> float:
        """Moyenne des précisions à chaque positif, tri décroissant stable.

        Le seuil n'intervient pas. Aucun positif, ou aucun exemple, donne 0.
        Deux probabilités égales gardent l'ordre d'arrivée des lots.
        """
        if not self._probabilities:
            return 0.0
        scores = torch.cat(self._probabilities)
        labels = torch.cat(self._targets) == 1
        if not bool(labels.any()):
            return 0.0
        order = torch.argsort(scores, descending=True, stable=True)
        ranked_positive = labels[order]
        seen = ranked_positive.cumsum(dim=0)
        rank = torch.arange(1, ranked_positive.numel() + 1)
        return float((seen[ranked_positive] / rank[ranked_positive]).mean())

    def _matthews_correlation(self) -> float:
        """Coefficient de Matthews au seuil déjà utilisé pour les comptes.

        `(VP * VN - FP * FN) / sqrt((VP + FP) * (VP + FN) * (VN + FP) * (VN + FN))`.
        Un dénominateur nul, par exemple une prédiction toujours négative,
        donne 0.
        """
        true_positive = self._true_positive
        false_positive = self._false_positive
        false_negative = self._false_negative
        true_negative = (
            self._total - true_positive - false_positive - false_negative
        )
        denominator = (
            (true_positive + false_positive)
            * (true_positive + false_negative)
            * (true_negative + false_positive)
            * (true_negative + false_negative)
        )
        if denominator <= 0:
            return 0.0
        numerator = true_positive * true_negative - false_positive * false_negative
        return numerator / (denominator ** 0.5)

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
