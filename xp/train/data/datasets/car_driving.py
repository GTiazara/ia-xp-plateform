"""Jeu de données synthétique pour l'XP-01 : accélérer ou non en conduite.

Reproduit, sous forme d'une règle "experte" déterministe (légèrement
bruitée), le tableau d'entrées décrit dans la fiche de l'XP-01 (voir
`docs/xp-01-perceptron-conduite.md`) :

=========================  ===================
Entrée                     Exemple de valeur
=========================  ===================
distance à l'obstacle (m)  2
présence d'un obstacle     Oui / Non
position de la voiture     Gauche / Droite
vitesse (km/h)             30
=========================  ===================

La règle de décision simulée (« accélérer si et seulement si il n'y a pas
d'obstacle trop proche compte tenu de la vitesse ») est volontairement
(quasi) linéairement séparable : c'est exactement ce qu'un perceptron simple
est capable d'apprendre. Un peu de bruit est ajouté sur les étiquettes pour
illustrer que même ce problème simple n'est jamais parfaitement résolu — une
mise en bouche des limites détaillées dans la fiche de l'XP.
"""
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

#: Ordre canonique des 4 entrées utilisées par l'XP-01. Doit correspondre à
#: `xp.train.models.perceptron.DEFAULT_IN_FEATURES`.
FEATURES = ("distance_obstacle_m", "obstacle_present", "position", "speed_kmh")

#: 0 = voiture à gauche de la voie, 1 = voiture à droite de la voie.
POSITION_LABELS = {0: "gauche", 1: "droite"}


@dataclass(frozen=True)
class CarDrivingConfig:
    """Paramètres de génération du jeu de données synthétique de l'XP-01.

    Parameters
    ----------
    n_samples : int, optional
        Nombre d'échantillons à générer, par défaut 2000.
    max_distance_m : float, optional
        Distance maximale à l'obstacle simulée (mètres), par défaut 20.0.
    max_speed_kmh : float, optional
        Vitesse maximale simulée (km/h), par défaut 90.0.
    safety_margin_s : float, optional
        Marge de sécurité, exprimée en "secondes à la vitesse actuelle",
        utilisée par la règle experte pour décider si la distance à
        l'obstacle est suffisante. Par défaut 1.5 s.
    label_noise : float, optional
        Proportion d'étiquettes volontairement inversées pour simuler des
        situations ambiguës, par défaut 0.05 (5%).
    seed : int, optional
        Graine aléatoire pour la reproductibilité, par défaut 42.
    """

    n_samples: int = 2000
    max_distance_m: float = 20.0
    max_speed_kmh: float = 90.0
    safety_margin_s: float = 1.5
    label_noise: float = 0.05
    seed: int = 42


def _expert_rule(
    distance_m: np.ndarray,
    obstacle_present: np.ndarray,
    speed_kmh: np.ndarray,
    safety_margin_s: float,
) -> np.ndarray:
    """Règle "experte" simulée : accélérer seulement si c'est prudent.

    On considère qu'il est dangereux d'accélérer si un obstacle est présent
    et que la distance restante est inférieure à la distance qui serait
    parcourue en `safety_margin_s` secondes à la vitesse actuelle (avec un
    plancher de 1 mètre pour les très faibles vitesses).

    Parameters
    ----------
    distance_m : np.ndarray
        Distance à l'obstacle, en mètres.
    obstacle_present : np.ndarray
        Présence d'un obstacle (0 ou 1).
    speed_kmh : np.ndarray
        Vitesse de la voiture, en km/h.
    safety_margin_s : float
        Marge de sécurité en secondes.

    Returns
    -------
    np.ndarray
        1.0 si l'on peut accélérer en sécurité, 0.0 sinon.
    """
    speed_ms = speed_kmh / 3.6
    safe_distance_m = np.maximum(safety_margin_s * speed_ms, 1.0)
    is_dangerous = obstacle_present.astype(bool) & (distance_m < safe_distance_m)
    return (~is_dangerous).astype(np.float32)


class CarDrivingDataset(Dataset):
    """Dataset synthétique "voiture virtuelle" pour l'XP-01 (perceptron).

    Chaque échantillon associe les 4 entrées de `FEATURES` (normalisées dans
    `[0, 1]`) à une décision binaire : 1 = accélérer, 0 = ne pas accélérer.

    Parameters
    ----------
    config : CarDrivingConfig, optional
        Paramètres de génération du jeu de données. Si `None`, utilise les
        valeurs par défaut de `CarDrivingConfig`.
    """

    def __init__(self, config: CarDrivingConfig | None = None) -> None:
        self.config = config or CarDrivingConfig()
        rng = np.random.default_rng(self.config.seed)
        n = self.config.n_samples

        distance_m = rng.uniform(0.0, self.config.max_distance_m, size=n)
        obstacle_present = rng.integers(0, 2, size=n)
        position = rng.integers(0, 2, size=n)  # 0 = gauche, 1 = droite
        speed_kmh = rng.uniform(0.0, self.config.max_speed_kmh, size=n)

        labels = _expert_rule(
            distance_m, obstacle_present, speed_kmh, self.config.safety_margin_s
        )

        if self.config.label_noise > 0:
            flip_mask = rng.random(n) < self.config.label_noise
            labels = np.where(flip_mask, 1.0 - labels, labels)

        features = np.stack(
            [
                distance_m / self.config.max_distance_m,
                obstacle_present.astype(np.float32),
                position.astype(np.float32),
                speed_kmh / self.config.max_speed_kmh,
            ],
            axis=1,
        ).astype(np.float32)

        self.features = torch.from_numpy(features)
        self.labels = torch.from_numpy(labels.astype(np.float32))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict:
        return {"features": self.features[index], "label": self.labels[index]}
