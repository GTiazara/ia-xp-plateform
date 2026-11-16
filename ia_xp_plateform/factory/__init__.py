"""Factories : construire un modèle, un datamodule, un logger, une transformation ou une tâche à partir de son nom."""

from ia_xp_plateform.factory.datamodule_factory import (
    DATAMODULES,
    create_datamodule,
)
from ia_xp_plateform.factory.logger_factory import LOGGERS, create_logger
from ia_xp_plateform.factory.model_factory import MODELS, create_model
from ia_xp_plateform.factory.task_factory import TASKS, create_task
from ia_xp_plateform.factory.transform_factory import (
    TRANSFORMS,
    create_preprocessing,
    create_transform,
)

__all__ = [
    "DATAMODULES",
    "LOGGERS",
    "MODELS",
    "TASKS",
    "TRANSFORMS",
    "create_datamodule",
    "create_logger",
    "create_model",
    "create_preprocessing",
    "create_task",
    "create_transform",
]
