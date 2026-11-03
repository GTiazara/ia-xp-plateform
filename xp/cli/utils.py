"""Utilitaires génériques pour le CLI (logging, hyperparamètres).

Repris et simplifié de `bdf.cli.utils` (bdf-model) : la logique est générique
et ne dépend d'aucun détail propre à une XP en particulier.
"""
import logging
from typing import Any
from typing import Dict

from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import OmegaConf


def get_pylogger(name: str = __name__) -> logging.Logger:
    """Initialise un logger Python compatible multi-GPU (rank zero only).

    Parameters
    ----------
    name : str, optional
        Nom du logger, par défaut `__name__`.

    Returns
    -------
    logging.Logger
        Le logger configuré.
    """
    logger = logging.getLogger(name)

    # Marque chaque niveau de log comme "rank zero only" pour éviter les logs
    # dupliqués en cas d'exécution multi-GPU (utile pour rester cohérent avec
    # l'architecture bdf-model, même si ia-xp-plateform tourne surtout en CPU).
    logging_levels = ("debug", "info", "warning", "error", "exception", "fatal", "critical")
    for level in logging_levels:
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


log = get_pylogger(__name__)


@rank_zero_only
def log_hyperparameters(object_dict: Dict[str, Any]) -> None:
    """Envoie la configuration complète de l'XP aux loggers de Lightning.

    Parameters
    ----------
    object_dict : dict
        Dictionnaire contenant :
        - `"config"` : la configuration Hydra (DictConfig) de l'XP.
        - `"trainer"` : le `Trainer` Lightning.
    """
    cfg = OmegaConf.to_container(object_dict["config"], resolve=True)
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Pas de logger configuré, les hyperparamètres ne seront pas sauvegardés.")
        return

    for logger_ in trainer.loggers:
        logger_.log_hyperparams(cfg)
