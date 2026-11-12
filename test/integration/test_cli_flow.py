"""Tests d'intégration : train, puis test, puis predict, par le CLI."""

import pandas as pd

from ia_xp_plateform.cli.main import main


def test_train_test_and_predict(tmp_path, sample_dataframe, capsys):
    data = tmp_path / "data.csv"
    sample_dataframe.to_csv(data, index=False)
    output_dir = tmp_path / "run"
    predictions = tmp_path / "predictions.csv"

    train_code = main(
        [
            "train",
            "--data",
            str(data),
            "--target",
            "target",
            "--output-dir",
            str(output_dir),
            "--epochs",
            "1",
            "--patience",
            "2",
            "--batch-size",
            "4",
        ]
    )
    checkpoint = capsys.readouterr().out.strip().splitlines()[-1]

    test_code = main(
        [
            "test",
            "--data",
            str(data),
            "--target",
            "target",
            "--checkpoint",
            checkpoint,
            "--batch-size",
            "4",
        ]
    )
    predict_code = main(
        [
            "predict",
            "--data",
            str(data),
            "--target",
            "target",
            "--checkpoint",
            checkpoint,
            "--output",
            str(predictions),
        ]
    )
    written = pd.read_csv(predictions)

    assert train_code == 0
    assert checkpoint.endswith("best-model.ckpt")
    assert list((output_dir / "logger" / "experience" / "version_0").glob("events.*"))
    assert test_code == 0
    assert predict_code == 0
    assert list(written.columns) == ["id", "probability", "prediction"]
    assert len(written) == len(sample_dataframe)
    assert set(written["prediction"]).issubset({0, 1})
