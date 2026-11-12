"""Factories : construire un modèle, un datamodule ou un logger à partir de son nom."""

from ia_xp_plateform.factory.datamodule_factory import (
    DATAMODULES,
    create_datamodule,
)
from ia_xp_plateform.factory.logger_factory import LOGGERS, create_logger
from ia_xp_plateform.factory.model_factory import MODELS, create_model

__all__ = [
    "DATAMODULES",
    "LOGGERS",
    "MODELS",
    "create_datamodule",
    "create_logger",
    "create_model",
]
