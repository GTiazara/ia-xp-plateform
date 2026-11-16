"""Enchaînement des transformations, un pipeline par split."""

from __future__ import annotations

import torch

from ia_xp_plateform.transform.base import BaseTransform

_SPLITS = ("train", "val", "test", "predict")


class Pipeline:
    """Transformations appliquées dans l'ordre. Vide, il ne change rien."""

    def __init__(self, transforms: list[BaseTransform] | None = None) -> None:
        self.transforms = list(transforms or [])

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for transform in self.transforms:
            x = transform(x)
        return x

    def __iter__(self):
        return iter(self.transforms)


class Preprocessing:
    """Quatre pipelines : train, val, test et predict.

    `fit` apprend chaque transformation à statistiques du pipeline
    `train` sur la sortie de la précédente, puis recopie ces états dans
    les autres pipelines, par nom et par rang d'apparition. `val`, `test`
    et `predict` n'apprennent jamais.
    """

    def __init__(
        self,
        train: Pipeline | None = None,
        val: Pipeline | None = None,
        test: Pipeline | None = None,
        predict: Pipeline | None = None,
    ) -> None:
        self.pipelines = {
            "train": train or Pipeline(),
            "val": val or Pipeline(),
            "test": test or Pipeline(),
            "predict": predict or Pipeline(),
        }
        self.fitted = False

    def pipeline(self, split: str) -> Pipeline:
        """Pipeline du split demandé."""
        try:
            return self.pipelines[split]
        except KeyError as error:
            known = ", ".join(_SPLITS)
            raise ValueError(
                f"Split inconnu : '{split}'. Splits disponibles : {known}."
            ) from error

    def requires_state(self, split: str) -> bool:
        """Vrai si ce split contient une transformation à statistiques."""
        return any(transform.stateful for transform in self.pipeline(split))

    def fit(self, x: torch.Tensor) -> "Preprocessing":
        """Apprend sur le jeu d'entraînement, puis partage les états."""
        current = x
        for transform in self.pipelines["train"]:
            if transform.stateful:
                transform.fit(current)
            current = transform(current)
        self.load_states(self.states())
        return self

    def states(self) -> list[dict]:
        """États des transformations à statistiques du pipeline train."""
        return [
            {"name": transform.name, "state": transform.state_dict()}
            for transform in self.pipelines["train"]
            if transform.stateful
        ]

    def load_states(self, states: list[dict]) -> None:
        """Charge `states` dans les quatre pipelines, sans réapprendre."""
        for split in _SPLITS:
            self._load_pipeline(self.pipelines[split], states, split)
        self.fitted = True

    def _load_pipeline(self, pipeline: Pipeline, states: list[dict], split: str) -> None:
        found: dict[str, list[dict]] = {}
        for item in states:
            found.setdefault(item["name"], []).append(item["state"])
        seen: dict[str, int] = {}
        for transform in pipeline:
            if not transform.stateful:
                continue
            index = seen.get(transform.name, 0)
            seen[transform.name] = index + 1
            matches = found.get(transform.name, [])
            if index >= len(matches):
                raise ValueError(
                    f"La transformation '{transform.name}' du split "
                    f"'{split}' n'a pas d'état appris correspondant."
                )
            transform.load_state_dict(matches[index])
