"""Charge la configuration Hydra du dossier ``conf/``.

Hydra ne construit pas les objets. Il assemble les groupes
(``data``, ``model``, ``trainer``, ``module``, ``callbacks``) en une
seule configuration. Les factories lisent ensuite les noms.
"""

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig


def conf_directory() -> Path:
    """Dossier ``conf/`` à la racine du projet."""
    return Path(__file__).resolve().parents[1] / "conf"


def load_config(overrides: list[str] | None = None) -> DictConfig:
    """Assemble ``config.yaml`` et applique les surcharges Hydra.

    Parameters
    ----------
    overrides : list[str] | None, optional
        Surcharges de la forme ``"trainer.max_epochs=5"``. Par défaut
        aucune : on obtient l'expérience décrite dans les fichiers.

    Returns
    -------
    DictConfig
        Configuration résolue (les interpolations comme ``${seed}``
        sont déjà remplacées).
    """
    with initialize_config_dir(
        config_dir=str(conf_directory()),
        version_base="1.3",
    ):
        return compose(config_name="config", overrides=list(overrides or []))
