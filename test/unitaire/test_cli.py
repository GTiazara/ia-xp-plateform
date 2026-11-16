"""Tests unitaires du parseur du CLI.

On ne lance pas d'entraînement : on vérifie que les commandes exigées
sont refusées quand un argument manque, et qu'une donnée incohérente
renvoie le code 1 sans faire planter le programme.
"""

import pytest

from ia_xp_plateform.cli.main import main


def test_train_help_names_initialization(capsys):
    with pytest.raises(SystemExit) as caught:
        main(["train", "--help"])

    assert caught.value.code == 0
    assert "initialisation" in capsys.readouterr().out


def test_requires_a_command():
    with pytest.raises(SystemExit) as caught:
        main([])

    assert caught.value.code == 2


def test_train_requires_data_and_target():
    with pytest.raises(SystemExit) as caught:
        main(["train", "--output-dir", "logs"])

    assert caught.value.code == 2


def test_predict_does_not_require_a_target(tmp_path):
    data = tmp_path / "data.csv"
    data.write_text("id,a,b\n0,1,2\n")
    checkpoint = tmp_path / "absent.ckpt"

    code = main(
        [
            "predict",
            "--data",
            str(data),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(tmp_path / "out.csv"),
        ]
    )

    assert code == 1


def test_train_rejects_an_input_size_that_does_not_match_the_columns(tmp_path):
    data = tmp_path / "data.csv"
    data.write_text("id,a,b,target\n0,1,2,0\n1,3,4,1\n2,5,6,0\n3,7,8,1\n")

    code = main(
        [
            "train",
            "--data",
            str(data),
            "--target",
            "target",
            "--output-dir",
            str(tmp_path / "logs"),
            "--input-features",
            "9",
            "--epochs",
            "1",
        ]
    )

    assert code == 1
