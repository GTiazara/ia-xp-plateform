"""Tests unitaires de la configuration Hydra.

On vérifie que les groupes se composent, que les interpolations se
résolvent, et qu'une surcharge remplace une valeur sans relire le fichier.
"""

import pytest
from hydra.errors import ConfigCompositionException, MissingConfigException

from ia_xp_plateform.cli.from_config import namespace_from_config
from ia_xp_plateform.cli.hydra_main import main, parse_launch_arguments
from ia_xp_plateform.configuration import available_configs, conf_directory, load_config


def test_help_does_not_try_to_parse_a_hydra_override(capsys):
    assert main(["--help"]) == 0
    output = capsys.readouterr().out
    assert "ia-xp-conf" in output
    assert "train_mlp" in output
    assert "train_perceptron" in output


def test_conf_directory_contains_the_root_config():
    assert (conf_directory() / "config.yaml").is_file()


def test_default_config_selects_the_perceptron_experiment():
    cfg = load_config()

    assert cfg.mode == "train"
    assert cfg.data.name == "table"
    assert cfg.data.target_columns == "y"
    assert cfg.data.split_seed == cfg.seed
    assert cfg.model.name == "perceptron"
    assert cfg.model.input_features == 1
    assert cfg.model.output_features == 1
    assert cfg.trainer.max_epochs == 200
    assert cfg.trainer.accelerator == "gpu"
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


def test_mlp_group_selects_the_hidden_layer_model():
    cfg = load_config(["model=mlp"])
    args = namespace_from_config(cfg)

    assert cfg.model.name == "mlp"
    assert cfg.model.output_features == 1
    assert cfg.model.hidden_features == 16
    assert cfg.model.hidden_layers == 1
    assert args.model == "mlp"
    assert args.output_features == 1
    assert args.hidden_features == 16
    assert args.hidden_layers == 1


def test_a_name_without_equals_selects_that_config_file():
    assert parse_launch_arguments(["train_mlp", "trainer.max_epochs=3"]) == (
        "train_mlp",
        ["trainer.max_epochs=3"],
    )
    assert parse_launch_arguments(["trainer.max_epochs=3"]) == (
        "config",
        ["trainer.max_epochs=3"],
    )


def test_named_configs_choose_the_model_and_their_own_log_directory():
    perceptron = load_config(config_name="train_perceptron")
    mlp = load_config(config_name="train_mlp")

    assert perceptron.model.name == "perceptron"
    assert perceptron.model.input_features is None
    assert perceptron.output_dir == "logs/train_perceptron"
    assert mlp.model.name == "mlp"
    assert mlp.model.hidden_layers == 1
    assert mlp.output_dir == "logs/train_mlp"
    assert mlp.logger.params.name == "mlp"
    assert mlp.logger.params.save_dir == "logs/tensorboard"
    assert "train_mlp" in available_configs()


def test_unknown_config_name_lists_the_files(capsys):
    assert main(["absent"]) == 1
    assert "train_mlp" in capsys.readouterr().err


def test_unknown_group_choice_is_rejected():
    with pytest.raises((MissingConfigException, ConfigCompositionException)):
        load_config(["model=inconnu"])


def test_namespace_flattens_the_groups_for_the_commands():
    cfg = load_config(["data.feature_columns=[a,b]"])
    args = namespace_from_config(cfg)

    assert args.model == "perceptron"
    assert args.hidden_features is None
    assert args.hidden_layers is None
    assert args.features == ["a", "b"]
    assert args.epochs == 200
    assert args.patience == 5
    assert args.learning_rate == 0.01
    assert args.output_dir == "logs/xp01"
    assert args.logger == "tensorboard"
    assert args.logger_params["name"] == "perceptron"
    assert args.logger_params["save_dir"] == "logs/tensorboard"
