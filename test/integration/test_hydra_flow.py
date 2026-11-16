"""Tests d'intégration : une expérience Hydra entraîne, teste, prédit."""

import pandas as pd

from ia_xp_plateform.cli.hydra_main import main


def _overrides(tmp_path, data):
    """Surcharges communes : jeu minuscule, un seul epoch, logs dans tmp."""
    return [
        f"data.data_path={data}",
        f"output_dir={tmp_path / 'run'}",
        f"logger.params.save_dir={tmp_path / 'tensorboard'}",
        "data.batch_size=4",
        "data.target_columns=target",
        "model.input_features=2",
        "model.output_features=1",
        "trainer.accelerator=cpu",
        "trainer.max_epochs=1",
        "callbacks.early_stopping_patience=2",
    ]


def test_hydra_config_trains_tests_and_predicts(tmp_path, sample_dataframe, capsys):
    data = tmp_path / "data.csv"
    sample_dataframe.to_csv(data, index=False)
    predictions = tmp_path / "predictions.csv"
    shared = _overrides(tmp_path, data)

    train_code = main([*shared, "mode=train"])
    checkpoint = capsys.readouterr().out.strip().splitlines()[-1]

    test_code = main([*shared, "mode=test", f"checkpoint={checkpoint}"])
    predict_code = main(
        [
            *shared,
            "mode=predict",
            f"checkpoint={checkpoint}",
            f"predictions={predictions}",
        ]
    )
    written = pd.read_csv(predictions)

    assert train_code == 0
    assert checkpoint.endswith("best-model.ckpt")
    assert test_code == 0
    assert predict_code == 0
    assert list(written.columns) == ["id", "probability", "prediction"]
    assert len(written) == len(sample_dataframe)


def test_named_mlp_config_trains(tmp_path, sample_dataframe, capsys):
    data = tmp_path / "data.csv"
    sample_dataframe.to_csv(data, index=False)

    code = main(
        ["train_mlp", *_overrides(tmp_path, data), "task=binary", "mode=train"]
    )
    checkpoint = capsys.readouterr().out.strip().splitlines()[-1]

    events = list((tmp_path / "tensorboard" / "mlp" / "version_0").glob("events.*"))
    assert code == 0
    assert checkpoint.endswith("best-model.ckpt")
    assert len(events) == 1


def test_tensorboard_is_the_default_logger_and_writes_event_files(
    tmp_path, sample_dataframe
):
    data = tmp_path / "data.csv"
    sample_dataframe.to_csv(data, index=False)

    code = main([*_overrides(tmp_path, data), "mode=train"])

    events = list((tmp_path / "tensorboard" / "perceptron" / "version_0").glob("events.*"))
    assert code == 0
    assert len(events) == 1


def test_past_experiments_stay_next_to_the_current_one(tmp_path, sample_dataframe):
    data = tmp_path / "data.csv"
    sample_dataframe.to_csv(data, index=False)
    shared = [*_overrides(tmp_path, data), "mode=train"]

    first = main(shared)
    second = main(shared)

    runs = sorted(p.name for p in (tmp_path / "tensorboard" / "perceptron").iterdir())
    assert first == 0
    assert second == 0
    assert runs == ["version_0", "version_1"]


def test_litlogger_remains_selectable_from_the_command_line(tmp_path, sample_dataframe):
    data = tmp_path / "data.csv"
    sample_dataframe.to_csv(data, index=False)
    overrides = [
        f"data.data_path={data}",
        f"output_dir={tmp_path / 'run'}",
        "data.batch_size=4",
        "data.target_columns=target",
        "model.input_features=2",
        "model.output_features=1",
        "trainer.accelerator=cpu",
        "trainer.max_epochs=1",
        "logger=litlogger",
        f"logger.params.root_dir={tmp_path / 'lit'}",
        "mode=train",
    ]

    code = main(overrides)

    assert code == 0
    assert not (tmp_path / "tensorboard").exists()
