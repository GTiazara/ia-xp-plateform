"""Enveloppe commune à tous les modèles de la plateforme ia-xp-plateform.

Inspirée de `bdf.train.models.base.BaseModel`, très simplifiée : chaque XP
définit son propre modèle en héritant de `BaseModel`, qui ajoute uniquement
un utilitaire générique pour recharger les poids d'une XP précédente.
"""
from pathlib import Path
from typing import Optional

import torch
from torch import nn


class BaseModel(nn.Module):
    """`nn.Module` enrichi d'un utilitaire de chargement de poids pré-entraînés.

    Parameters
    ----------
    pretrained_weights : Path, optional
        Chemin vers un checkpoint (`.pt` / `.ckpt`) dont on veut reprendre les
        poids. Si `None` (par défaut), le modèle est initialisé normalement.
    """

    def __init__(self, pretrained_weights: Optional[Path] = None) -> None:
        super().__init__()
        self.pretrained_weights = pretrained_weights

    def load_pretrained_weights(self) -> None:
        """Charge les poids d'une XP précédente, si `pretrained_weights` est défini.

        Le chargement se fait en mode non strict (`strict=False`) afin de
        pouvoir réutiliser partiellement les poids d'une architecture proche
        mais pas strictement identique.
        """
        if self.pretrained_weights is None:
            return

        checkpoint = torch.load(self.pretrained_weights, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        self.load_state_dict(state_dict, strict=False)
