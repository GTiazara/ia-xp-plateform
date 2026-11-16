"""Transformations qui apprennent une statistique par colonne."""

from __future__ import annotations

import torch

from ia_xp_plateform.transform.base import BaseTransform


def _columns(x: torch.Tensor) -> torch.Tensor:
    """Ramène un exemple isolé `(f,)` à un lot d'une ligne `(1, f)`."""
    if x.ndim == 1:
        return x.unsqueeze(0)
    return x


class Standardize(BaseTransform):
    """Centre et réduit chaque colonne : `(x - moyenne) / écart-type`.

    L'écart-type nul d'une colonne constante est remplacé par 1 : la
    colonne transformée vaut 0, sans division par zéro.
    """

    name = "standardize"
    stateful = True

    def fit(self, x: torch.Tensor) -> "Standardize":
        columns = _columns(x)
        self.mean = columns.mean(dim=0)
        self.std = columns.std(dim=0, unbiased=False)
        self.std = torch.where(self.std == 0, torch.ones_like(self.std), self.std)
        self._fitted = True
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def state_dict(self) -> dict:
        return {
            "name": self.name,
            "mean": self.mean.detach().cpu().tolist(),
            "std": self.std.detach().cpu().tolist(),
        }

    def load_state_dict(self, state: dict) -> None:
        super().load_state_dict(state)
        self.mean = torch.tensor(state["mean"], dtype=torch.float32)
        self.std = torch.tensor(state["std"], dtype=torch.float32)


class MinMax(BaseTransform):
    """Ramène chaque colonne apprise dans `[0, 1]`.

    Une valeur hors de la plage apprise dépasse 0 ou 1 : elle n'est pas
    tronquée. Une colonne constante, d'étendue nulle, devient 0.
    """

    name = "min_max"
    stateful = True

    def fit(self, x: torch.Tensor) -> "MinMax":
        columns = _columns(x)
        self.minimum = columns.min(dim=0).values
        self.maximum = columns.max(dim=0).values
        span = self.maximum - self.minimum
        self.span = torch.where(span == 0, torch.ones_like(span), span)
        self._fitted = True
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.minimum) / self.span

    def state_dict(self) -> dict:
        return {
            "name": self.name,
            "minimum": self.minimum.detach().cpu().tolist(),
            "maximum": self.maximum.detach().cpu().tolist(),
            "span": self.span.detach().cpu().tolist(),
        }

    def load_state_dict(self, state: dict) -> None:
        super().load_state_dict(state)
        self.minimum = torch.tensor(state["minimum"], dtype=torch.float32)
        self.maximum = torch.tensor(state["maximum"], dtype=torch.float32)
        self.span = torch.tensor(state["span"], dtype=torch.float32)
