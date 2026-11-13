"""Modèles : `BaseModel` (contrat commun) et ses implémentations."""

from ia_xp_plateform.models.base import BaseModel
from ia_xp_plateform.models.mlp import MLP
from ia_xp_plateform.models.perceptron import Perceptron

__all__ = ["BaseModel", "MLP", "Perceptron"]
