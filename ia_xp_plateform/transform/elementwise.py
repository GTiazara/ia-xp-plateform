"""Transformations qui agissent valeur par valeur, sans statistique."""

from __future__ import annotations

import torch

from ia_xp_plateform.transform.base import BaseTransform


class Clip(BaseTransform):
    """Borne chaque valeur entre `min` et `max`.

    Au moins une des deux bornes est obligatoire. Rien n'est appris :
    les bornes sont des paramètres.
    """

    name = "clip"

    def __init__(
        self,
        min: float | None = None,
        max: float | None = None,
    ) -> None:
        super().__init__()
        if min is None and max is None:
            raise ValueError(
                "clip : indiquez au moins une borne, min ou max."
            )
        if min is not None and max is not None and min > max:
            raise ValueError(
                f"clip : min ({min}) doit être inférieur ou égal à max ({max})."
            )
        self.minimum = min
        self.maximum = max

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.clamp(min=self.minimum, max=self.maximum)


class GaussianNoise(BaseTransform):
    """Ajoute un bruit normal centré, d'écart-type `std`.

    Deux appels sur la même entrée diffèrent. La graine de PyTorch les
    rend reproductibles. Pensée pour l'entraînement : un bruit sur la
    validation ou le test rend les métriques instables.
    """

    name = "gaussian_noise"

    def __init__(self, std: float) -> None:
        super().__init__()
        if std <= 0:
            raise ValueError(
                "gaussian_noise : std doit être strictement positif "
                f"({std})."
            )
        self.std = float(std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.randn_like(x) * self.std
