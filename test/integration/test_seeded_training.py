"""Deux entraînements CPU de même graine écrivent les mêmes poids."""

import torch

from ia_xp_plateform.cli.main import main


def _train(data, output_dir, seed):
    code = main(
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
            "--batch-size",
            "4",
            "--accelerator",
            "cpu",
            "--seed",
            str(seed),
        ]
    )
    assert code == 0
    return output_dir / "checkpoints" / "best-model.ckpt"


def _model_weights(path):
    state = torch.load(path, weights_only=True)["state_dict"]
    return {name: tensor for name, tensor in state.items() if name.startswith("model.")}


def test_the_same_seed_reproduces_the_weights_and_another_seed_changes_them(
    tmp_path, sample_dataframe
):
    data = tmp_path / "data.csv"
    sample_dataframe.to_csv(data, index=False)

    first = _model_weights(_train(data, tmp_path / "seed-7-a", 7))
    second = _model_weights(_train(data, tmp_path / "seed-7-b", 7))
    third = _model_weights(_train(data, tmp_path / "seed-8", 8))

    assert first.keys() == second.keys()
    assert all(torch.equal(first[name], second[name]) for name in first)
    assert any(not torch.equal(first[name], third[name]) for name in first)
