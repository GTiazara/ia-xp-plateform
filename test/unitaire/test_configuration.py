"""Tests unitaires de la configuration Hydra.

On vérifie que les groupes se composent, que les interpolations se
résolvent, et qu'une surcharge remplace une valeur sans relire le fichier.
"""

import pytest
from hydra.errors import ConfigCompositionException, MissingConfigException

from ia_xp_plateform.cli.from_config import namespace_from_config
from ia_xp_plateform.cli.hydra_main import main, parse_launch_arguments
from ia_xp_plateform.configuration import available_configs, conf_directory, load_config


def assert_positive_int(value) -> None:
    """Entier d'expérience : présent, entier, au moins 1."""
    assert isinstance(value, int) and not isinstance(value, bool)
    assert value >= 1


def assert_positive_float(value) -> None:
    """Flottant d'expérience : présent et strictement positif."""
    assert isinstance(value, float)
    assert value > 0


def assert_optional_positive_int(value) -> None:
    """Entier d'expérience, ou None quand la commande le déduit."""
    if value is not None:
        assert_positive_int(value)


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
    assert_optional_positive_int(cfg.model.input_features)
    assert_positive_int(cfg.model.output_features)
    assert_positive_int(cfg.trainer.max_epochs)
    assert cfg.trainer.accelerator == "gpu"
    assert_positive_float(cfg.module.learning_rate)
    assert_positive_int(cfg.callbacks.early_stopping_patience)
    args = namespace_from_config(cfg)
    expected_inputs = (
        None if cfg.model.input_features is None else int(cfg.model.input_features)
    )
    assert args.input_features == expected_inputs
    assert args.output_features == int(cfg.model.output_features)
    assert args.epochs == int(cfg.trainer.max_epochs)
    assert args.learning_rate == float(cfg.module.learning_rate)
    assert args.patience == int(cfg.callbacks.early_stopping_patience)
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
    assert_positive_int(cfg.model.output_features)
    assert_positive_int(cfg.model.hidden_features)
    assert_positive_int(cfg.model.hidden_layers)
    assert args.model == "mlp"
    assert args.output_features == int(cfg.model.output_features)
    assert args.hidden_features == int(cfg.model.hidden_features)
    assert args.hidden_layers == int(cfg.model.hidden_layers)


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
    assert_positive_int(mlp.model.input_features)
    assert_positive_int(mlp.model.hidden_features)
    assert_positive_int(mlp.model.hidden_layers)
    assert mlp.output_dir == "logs/train_mlp"
    assert mlp.logger.params.name == "mlp"
    assert mlp.logger.params.save_dir == "logs/tensorboard"
    assert list(perceptron.augmentation.train) == []
    assert list(mlp.augmentation.predict) == []
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
    assert_positive_int(cfg.trainer.max_epochs)
    assert_positive_int(cfg.callbacks.early_stopping_patience)
    assert_positive_float(cfg.module.learning_rate)
    assert args.epochs == int(cfg.trainer.max_epochs)
    assert args.patience == int(cfg.callbacks.early_stopping_patience)
    assert args.learning_rate == float(cfg.module.learning_rate)
    assert args.output_dir == "logs/xp01"
    assert args.logger == "tensorboard"
    assert args.logger_params["name"] == "perceptron"
    assert args.logger_params["save_dir"] == "logs/tensorboard"


def test_default_augmentation_is_empty_on_every_split():
    cfg = load_config()
    args = namespace_from_config(cfg)

    for split in ("train", "val", "test", "predict"):
        assert list(cfg.augmentation[split]) == []
        assert args.augmentation[split] == []


def test_standardize_augmentation_is_listed_on_every_split():
    cfg = load_config(["augmentation=standardize"])
    args = namespace_from_config(cfg)

    for split in ("train", "val", "test", "predict"):
        names = [step["name"] for step in cfg.augmentation[split]]
        assert names == ["standardize"]
        assert args.augmentation[split] == [{"name": "standardize"}]
