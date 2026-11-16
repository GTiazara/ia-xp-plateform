"""Prétraitement Hydra de bout en bout, et CLI argparse sans transformation."""

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch

from ia_xp_plateform.cli.command.common import inputs_of, model_kwargs, preprocessing_path, read_table
from ia_xp_plateform.cli.hydra_main import main as hydra_main
from ia_xp_plateform.cli.main import main as cli_main
from ia_xp_plateform.inference import Predictor


def _overrides(tmp_path, data):
    return [
        f"data.data_path={data}",
        f"output_dir={tmp_path / 'run'}",
        f"logger.params.save_dir={tmp_path / 'tensorboard'}",
        "data.batch_size=4",
        "data.target_columns=target",
        "model.input_features=2",
        "trainer.accelerator=cpu",
        "trainer.max_epochs=1",
        "callbacks.early_stopping_patience=2",
        "augmentation=standardize",
    ]


def test_standardize_is_saved_reused_and_not_overwritten(
    tmp_path, sample_dataframe, capsys
):
    data = tmp_path / "data.csv"
    sample_dataframe.to_csv(data, index=False)
    predictions = tmp_path / "predictions.csv"
    shared = _overrides(tmp_path, data)

    train_code = hydra_main([*shared, "mode=train"])
    checkpoint = capsys.readouterr().out.strip().splitlines()[-1]
    state_path = preprocessing_path(checkpoint)
    saved = json.loads(state_path.read_text(encoding="utf-8"))

    assert train_code == 0
    assert checkpoint.endswith("best-model.ckpt")
    assert state_path.is_file()
    assert saved[0]["name"] == "standardize"
    assert len(saved[0]["state"]["mean"]) == 2
    assert len(saved[0]["state"]["std"]) == 2

    test_code = hydra_main([*shared, "mode=test", f"checkpoint={checkpoint}"])
    assert test_code == 0
    capsys.readouterr()

    predict_code = hydra_main(
        [
            *shared,
            "mode=predict",
            f"checkpoint={checkpoint}",
            f"predictions={predictions}",
        ]
    )
    assert predict_code == 0
    frame = read_table(data, None)
    raw = inputs_of(frame, ["a", "b"])
    state = saved[0]["state"]
    scaled = (raw - torch.tensor(state["mean"])) / torch.tensor(state["std"])
    manual = Predictor.from_checkpoint(
        checkpoint,
        "perceptron",
        model_kwargs(SimpleNamespace(output_features=1), len(["a", "b"])),
    ).predict_proba(scaled)
    written = pd.read_csv(predictions)
    assert torch.allclose(
        torch.tensor(written["probability"].to_numpy(), dtype=torch.float32),
        manual[:, 0],
        atol=1e-5,
    )

    original = state_path.read_bytes()
    second_code = hydra_main([*shared, "mode=train"])
    second = capsys.readouterr().out.strip().splitlines()[-1]
    assert second_code == 0
    assert state_path.read_bytes() == original
    assert preprocessing_path(second) != state_path
    assert preprocessing_path(second).is_file()

    state_path.unlink()
    missing_code = hydra_main(
        [*shared, "mode=predict", f"checkpoint={checkpoint}", f"predictions={predictions}"]
    )
    assert missing_code == 1
    assert state_path.name in capsys.readouterr().err


def test_argparse_train_does_not_transform(tmp_path, sample_dataframe, capsys):
    data = tmp_path / "data.csv"
    sample_dataframe.to_csv(data, index=False)

    code = cli_main(
        [
            "train",
            "--data",
            str(data),
            "--target",
            "target",
            "--output-dir",
            str(tmp_path / "run"),
            "--epochs",
            "1",
            "--patience",
            "2",
            "--batch-size",
            "4",
        ]
    )
    checkpoint = Path(capsys.readouterr().out.strip().splitlines()[-1])

    assert code == 0
    assert checkpoint.is_file()
    assert not preprocessing_path(checkpoint).exists()
