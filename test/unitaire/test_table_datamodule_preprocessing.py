"""Tests du prétraitement branché sur `TableDataModule`."""

from pathlib import Path

import pandas as pd
import torch

from ia_xp_plateform.data.datamodule.table_datamodule import TableDataModule
from ia_xp_plateform.factory.transform_factory import create_preprocessing
from ia_xp_plateform.transform.pipeline import Pipeline, Preprocessing
from ia_xp_plateform.transform.scaling import Standardize


def _write(tmp_path: Path) -> tuple[Path, Path]:
    data = tmp_path / "data.csv"
    split = tmp_path / "split.csv"
    pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "a": [0.0, 2.0, 1000.0, 1000.0],
            "y": [0.0, 1.0, 0.0, 1.0],
        }
    ).to_csv(data, index=False)
    pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "split": ["train", "train", "val", "test"],
        }
    ).to_csv(split, index=False)
    return data, split


def _module(tmp_path: Path, preprocessing=None) -> TableDataModule:
    data, split = _write(tmp_path)
    return TableDataModule(
        data_path=data,
        split_path=split,
        target_columns="y",
        preprocessing=preprocessing,
    )


def test_statistics_are_learned_on_train_and_reused_by_val_and_test(tmp_path):
    preprocessing = create_preprocessing(
        {
            "train": [{"name": "standardize"}],
            "val": [{"name": "standardize"}],
            "test": [{"name": "standardize"}],
        }
    )
    module = _module(tmp_path, preprocessing)
    module.setup()

    mean = preprocessing.pipeline("train").transforms[0].mean
    std = preprocessing.pipeline("train").transforms[0].std
    assert torch.allclose(mean, torch.tensor([1.0]))

    for dataset in (module.val_dataset, module.test_dataset):
        transformed, target = dataset[0]
        assert torch.allclose(transformed, (dataset.x[0] - mean) / std)
        assert torch.equal(target, dataset.y[0])
    assert "id" not in module.train_dataset.feature_columns
    assert torch.equal(module.train_dataset.y, torch.tensor([[0.0], [1.0]]))


def test_a_preloaded_state_is_not_learned_again(tmp_path):
    learned = Standardize().fit(torch.tensor([[10.0], [30.0]]))
    preprocessing = Preprocessing(
        train=Pipeline([Standardize()]),
        val=Pipeline([Standardize()]),
        test=Pipeline([Standardize()]),
    )
    preprocessing.load_states([{"name": "standardize", "state": learned.state_dict()}])

    module = _module(tmp_path, preprocessing)
    module.setup()

    assert torch.allclose(
        preprocessing.pipeline("train").transforms[0].mean, learned.mean
    )


def test_without_preprocessing_the_values_are_those_of_the_file(tmp_path):
    module = _module(tmp_path)
    module.setup()

    assert torch.equal(module.train_dataset[0][0], module.train_dataset.x[0])
    assert torch.equal(module.train_dataset[0][1], module.train_dataset.y[0])
