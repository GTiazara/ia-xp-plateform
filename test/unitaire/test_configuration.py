"""Tests unitaires de la configuration Hydra.

On vérifie que les groupes se composent, que les interpolations se
résolvent, et qu'une surcharge remplace une valeur sans relire le fichier.
"""

import pytest
from hydra.errors import ConfigCompositionException, MissingConfigException

from ia_xp_plateform.cli.from_config import namespace_from_config
from ia_xp_plateform.cli.hydra_main import main
from ia_xp_plateform.configuration import conf_directory, load_config


def test_help_does_not_try_to_parse_a_hydra_override(capsys):
    assert main(["--help"]) == 0
    assert "ia-xp-conf" in capsys.readouterr().out


def test_conf_directory_contains_the_root_config():
    assert (conf_directory() / "config.yaml").is_file()


def test_default_config_selects_the_perceptron_experiment():
    cfg = load_config()

    assert cfg.mode == "train"
    assert cfg.data.name == "table"
    assert cfg.data.target_columns == "y"
    assert cfg.data.split_seed == cfg.seed
    assert cfg.model.name == "perceptron"
    assert cfg.model.input_features == 2
    assert cfg.model.output_features == 1
    assert cfg.trainer.max_epochs == 10
    assert cfg.trainer.accelerator == "cpu"
    assert cfg.module.learning_rate == 0.01
    assert cfg.callbacks.early_stopping_patience == 5
    assert cfg.logger.name == "tensorboard"
    assert cfg.logger.params.name == "perceptron"
    assert cfg.logger.params.save_dir == "logs/tensorboard"


def test_litlogger_group_can_replace_the_default_logger():
    cfg = load_config(["logger=litlogger"])

    assert cfg.logger.name == "litlogger"
    assert cfg.logger.params.root_dir == "logs/xp01/logger"


def test_override_replaces_a_value_and_the_interpolation_follows():
    cfg = load_config(["seed=7", "trainer.max_epochs=3", "data.data_path=jeu.csv"])

    assert cfg.seed == 7
    assert cfg.data.split_seed == 7
    assert cfg.trainer.max_epochs == 3
    assert cfg.data.data_path == "jeu.csv"


def test_unknown_group_choice_is_rejected():
    with pytest.raises((MissingConfigException, ConfigCompositionException)):
        load_config(["model=inconnu"])


def test_namespace_flattens_the_groups_for_the_commands():
    cfg = load_config(["data.feature_columns=[a,b]"])
    args = namespace_from_config(cfg)

    assert args.model == "perceptron"
    assert args.features == ["a", "b"]
    assert args.epochs == 10
    assert args.patience == 5
    assert args.learning_rate == 0.01
    assert args.output_dir == "logs/xp01"
    assert args.logger == "tensorboard"
    assert args.logger_params["name"] == "perceptron"
    assert args.logger_params["save_dir"] == "logs/tensorboard"
