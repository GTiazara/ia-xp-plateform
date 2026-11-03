"""Orchestration d'une XP, inspirée de `bdf.cli.launch_experiment`.

Reprend la même architecture que `bdf-model` :

    1. `setup_datamodule`  -> instancie le `LightningDataModule` de l'XP.
    2. `setup_callbacks`   -> instancie les callbacks Lightning.
    3. `setup_trainer`     -> instancie le logger puis le `Trainer`.
    4. `setup_task`        -> instancie le `LightningModule` (modèle + perte + optim).
    5. `run_mode`          -> lance l'entraînement / la validation / la prédiction.

Simplifiée par rapport à `bdf-model` : pas de ClearML, pas de stratégie multi-GPU,
pas de fusion de modalités. On garde volontairement la même squelette pour
pouvoir faire grossir la plateforme XP après XP sans tout réécrire.
"""
import warnings
from typing import List

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning import Callback
from pytorch_lightning import LightningDataModule
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.model_summary import ModelSummary
from torch import set_float32_matmul_precision

from xp.cli.utils import get_pylogger
from xp.cli.utils import log_hyperparameters

log = get_pylogger(__name__)


def setup_datamodule(config: DictConfig) -> LightningDataModule:
    """Instancie le `LightningDataModule` décrit dans `config.datamodule`.

    Parameters
    ----------
    config : DictConfig
        Configuration Hydra complète de l'XP.

    Returns
    -------
    LightningDataModule
        Le datamodule instancié.
    """
    log.debug(f"Instanciation du datamodule <{config.datamodule._target_}>")
    return hydra.utils.instantiate(config.datamodule)


def setup_callbacks(config: DictConfig) -> List[Callback]:
    """Instancie tous les callbacks Lightning listés dans `config.callbacks`.

    Parameters
    ----------
    config : DictConfig
        Configuration Hydra complète de l'XP.

    Returns
    -------
    list[Callback]
        La liste des callbacks instanciés.
    """
    callbacks: List[Callback] = []
    for name, cb_conf in config.get("callbacks", {}).items():
        if "_target_" in cb_conf:
            log.debug(f"Instanciation du callback <{name}: {cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))
    return callbacks


def setup_task(config: DictConfig) -> pl.LightningModule:
    """Instancie la tâche d'apprentissage (modèle + perte + optimiseur).

    Parameters
    ----------
    config : DictConfig
        Configuration Hydra complète de l'XP.

    Returns
    -------
    pl.LightningModule
        La tâche d'apprentissage instanciée.
    """
    log.debug(f"Instanciation de la tâche <{config.task._target_}>")
    task: pl.LightningModule = hydra.utils.instantiate(config.task, _convert_="partial")
    log.info(ModelSummary(task, max_depth=2))
    return task


def setup_trainer(config: DictConfig, callbacks: List[Callback]) -> Trainer:
    """Instancie le logger puis le `Trainer` Lightning.

    Parameters
    ----------
    config : DictConfig
        Configuration Hydra complète de l'XP.
    callbacks : list[Callback]
        Callbacks déjà instanciés à passer au `Trainer`.

    Returns
    -------
    Trainer
        Le `Trainer` Lightning instancié.
    """
    log.debug(f"Instanciation du logger <{config.logger._target_}>")
    logger = hydra.utils.instantiate(config.logger)

    log.debug(f"Instanciation du trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer,
        callbacks=callbacks,
        logger=logger,
    )
    return trainer


def run_mode(
    config: DictConfig,
    trainer: Trainer,
    task: pl.LightningModule,
    datamodule: LightningDataModule,
) -> None:
    """Exécute le mode demandé (train / test / predict).

    Parameters
    ----------
    config : DictConfig
        Configuration Hydra complète de l'XP.
    trainer : Trainer
        Le `Trainer` Lightning instancié.
    task : pl.LightningModule
        La tâche d'apprentissage instanciée.
    datamodule : LightningDataModule
        Le datamodule instancié.

    Raises
    ------
    ValueError
        Si `config.mode` n'est ni "train", ni "test", ni "predict".
    """
    checkpoint_path = config.paths.get("checkpoint_path") or None

    if config.mode == "train":
        resume_ckpt = config.paths.get("resume_checkpoint_path") or None
        if resume_ckpt is not None:
            log.info(f"Reprise de l'entraînement depuis le checkpoint : {resume_ckpt}")
        trainer.fit(task, datamodule=datamodule, ckpt_path=resume_ckpt)
        log.info("Entraînement terminé, validation finale avec le meilleur checkpoint :")
        trainer.validate(task, datamodule=datamodule, ckpt_path="best")
    elif config.mode == "test":
        log.info("Lancement du test.")
        trainer.test(task, datamodule=datamodule, ckpt_path=checkpoint_path)
    elif config.mode == "predict":
        log.info("Lancement de la prédiction.")
        trainer.predict(task, datamodule=datamodule, ckpt_path=checkpoint_path)
    else:
        raise ValueError(f"Mode inconnu : {config.mode} (attendu: train, test ou predict)")


def launch_experiment(config: DictConfig) -> None:
    """Lance une XP de bout en bout à partir de sa configuration Hydra.

    Parameters
    ----------
    config : DictConfig
        Configuration Hydra complète de l'XP (cf. `conf/`).
    """
    log.info(f"Lancement de l'expérience : {config.task_name}")

    set_float32_matmul_precision("high")
    if config.get("ignore_warnings"):
        log.info("Désactivation des warnings Python <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    datamodule = setup_datamodule(config)
    callbacks = setup_callbacks(config)
    trainer = setup_trainer(config, callbacks)
    task = setup_task(config)

    if trainer.logger:
        log.info("Enregistrement des hyperparamètres de l'XP.")
        log_hyperparameters({"config": config, "trainer": trainer})

    run_mode(config, trainer, task, datamodule)
