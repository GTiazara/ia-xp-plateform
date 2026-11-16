"""Lance train, test ou predict à partir de la configuration Hydra."""

from types import SimpleNamespace

from omegaconf import DictConfig, OmegaConf

from ia_xp_plateform.cli.command.predict import run_predict
from ia_xp_plateform.cli.command.test import run_test
from ia_xp_plateform.cli.command.train import run_train
from ia_xp_plateform.configuration import load_config

_MODES = ("train", "test", "predict")


def _plain(value):
    """Passe une liste OmegaConf en liste Python. null reste None."""
    if value is None or isinstance(value, str):
        return value
    return list(value)


def namespace_from_config(cfg: DictConfig) -> SimpleNamespace:
    """Traduit la configuration assemblée vers les commandes du CLI.

    Les commandes lisent un espace de noms plat (``args.data``,
    ``args.model``). La configuration, elle, est rangée par groupe.
    """
    logger_group = OmegaConf.to_container(cfg.logger, resolve=True)
    augmentation = cfg.get("augmentation")
    if augmentation is None:
        augmentation = {}
    else:
        augmentation = OmegaConf.to_container(augmentation, resolve=True)
    input_features = cfg.model.input_features
    hidden_features = cfg.model.get("hidden_features")
    hidden_layers = cfg.model.get("hidden_layers")
    return SimpleNamespace(
        data=cfg.data.data_path,
        target=_plain(cfg.data.target_columns),
        features=_plain(cfg.data.feature_columns),
        id_column=cfg.data.id_column,
        split=cfg.data.split_path,
        split_column=cfg.data.split_column,
        hdf_key=cfg.data.hdf_key,
        split_hdf_key=cfg.data.split_hdf_key,
        batch_size=int(cfg.data.batch_size),
        num_workers=int(cfg.data.num_workers),
        val_split=float(cfg.data.val_split),
        test_split=float(cfg.data.test_split),
        seed=int(cfg.data.split_seed),
        model=cfg.model.name,
        input_features=None if input_features is None else int(input_features),
        output_features=int(cfg.model.output_features),
        hidden_features=None if hidden_features is None else int(hidden_features),
        hidden_layers=None if hidden_layers is None else int(hidden_layers),
        threshold=float(cfg.module.threshold),
        learning_rate=float(cfg.module.learning_rate),
        accelerator=str(cfg.trainer.accelerator),
        devices=int(cfg.trainer.devices),
        min_epochs=int(cfg.trainer.min_epochs),
        epochs=int(cfg.trainer.max_epochs),
        log_every_n_steps=int(cfg.trainer.log_every_n_steps),
        enable_progress_bar=bool(cfg.trainer.enable_progress_bar),
        enable_model_summary=bool(cfg.trainer.enable_model_summary),
        patience=int(cfg.callbacks.early_stopping_patience),
        output_dir=str(cfg.output_dir),
        logger=str(logger_group["name"]),
        logger_params=dict(logger_group.get("params") or {}),
        checkpoint=cfg.checkpoint,
        output=str(cfg.predictions),
        augmentation=augmentation,
    )


def run(cfg: DictConfig) -> None:
    """Exécute le mode demandé par ``cfg.mode``."""
    if cfg.mode not in _MODES:
        known = ", ".join(_MODES)
        raise ValueError(f"mode inconnu : '{cfg.mode}'. Modes disponibles : {known}.")
    if cfg.mode in ("test", "predict") and not cfg.checkpoint:
        raise ValueError(
            f"mode {cfg.mode} : indiquez le checkpoint, par exemple "
            "checkpoint=logs/xp01/checkpoints/best-model.ckpt."
        )

    args = namespace_from_config(cfg)
    if cfg.mode == "train":
        run_train(args)
    elif cfg.mode == "test":
        run_test(args)
    else:
        run_predict(args)


def launch(
    overrides: list[str] | None = None,
    config_name: str = "config",
) -> None:
    """Charge un fichier de ``conf/``, applique les surcharges, puis lance."""
    run(load_config(overrides, config_name=config_name))
