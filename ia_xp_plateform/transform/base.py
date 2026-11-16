"""

Contrat d'une transformation d'entrée.

Une transformation reçoit un tenseur et en renvoie un de même forme.
Elle ne sait pas si ce tenseur est une ligne de tableau ou un canal
d'image : le prétraitement reste le même pour les données tabulaires
d'aujourd'hui et les données raster à venir.

Certaines transformations apprennent des statistiques (`stateful`).
Elles refusent de s'appliquer tant qu'on ne les a pas apprises, ou
tant qu'on ne leur a pas rendu un état déjà appris.

"""

from __future__ import annotations

import torch


class BaseTransform:
    """Transformation d'un tenseur vers un tenseur de même forme.

    `name` est à la fois le nom de la factory et la clé qui relie une
    transformation de validation, de test ou de prédiction à celle
    apprise pendant l'entraînement. `stateful` dit si `fit` doit tourner
    avant le premier appel.
    """

    name = "transform"
    stateful = False

    def __init__(self) -> None:
        self._fitted = False

    def fit(self, x: torch.Tensor) -> "BaseTransform":
        """Apprend sur `x`. Par défaut, il n'y a rien à apprendre."""
        self._fitted = True
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applique la transformation. Chaque sous-classe la définit."""
        raise NotImplementedError

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Applique la transformation si elle est prête."""
        if self.stateful and not self._fitted:
            raise RuntimeError(
                f"Apprenez la transformation '{self.name}' ou chargez son "
                "état avant de l'appliquer."
            )
        return self.forward(x)

    def state_dict(self) -> dict:
        """État appris, fait de nombres et de listes. Vide par défaut."""
        return {}

    def load_state_dict(self, state: dict) -> None:
        """Reprend un état exporté par `state_dict`.

        Une transformation sans état n'accepte qu'un dictionnaire vide.
        Une transformation à statistiques exige un état à son nom : c'est
        la sous-classe qui le recharge, après cette vérification.
        """
        if not self.stateful:
            if state:
                raise ValueError(
                    f"La transformation '{self.name}' n'a pas d'état "
                    f"(reçu : {sorted(state)})."
                )
            return
        received = state.get("name")
        if received != self.name:
            raise ValueError(
                f"État incompatible : '{self.name}' a reçu un état "
                f"de '{received}'."
            )
        self._fitted = True
