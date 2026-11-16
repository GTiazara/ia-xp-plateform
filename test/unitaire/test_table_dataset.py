"""Tests unitaires de `TableDataset`.

`TableDataset` sépare un tableau déjà en mémoire en entrées (`x`) et
cible (`y`). On vérifie le choix des colonnes par défaut, le cas d'une
sélection explicite, les cibles multiples, et la conversion en tenseurs.
"""

import pandas as pd
import pytest
import torch

from ia_xp_plateform.data.dataset.table_dataset import TableDataset


@pytest.fixture
def dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0],
            "b": [4.0, 5.0, 6.0],
            "target": [0.0, 1.0, 0.0],
        }
    )


def test_default_feature_columns_exclude_the_target(dataframe):
    dataset = TableDataset(dataframe, target_columns="target")

    assert dataset.feature_columns == ["a", "b"]
    assert torch.equal(
        dataset.x, torch.tensor([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
    )
    assert torch.equal(dataset.y, torch.tensor([[0.0], [1.0], [0.0]]))


def test_explicit_feature_columns_are_respected(dataframe):
    dataset = TableDataset(
        dataframe, target_columns="target", feature_columns=["b"]
    )

    assert dataset.feature_columns == ["b"]
    assert dataset.x.shape == (3, 1)


def test_multiple_target_columns(dataframe):
    dataset = TableDataset(
        dataframe, target_columns=["a", "b"], feature_columns=["target"]
    )

    assert dataset.y.shape == (3, 2)
    assert dataset.x.shape == (3, 1)


def test_tensors_are_float32(dataframe):
    dataset = TableDataset(dataframe, target_columns="target")

    assert dataset.x.dtype == torch.float32
    assert dataset.y.dtype == torch.float32


def test_length_matches_the_number_of_rows(dataframe):
    dataset = TableDataset(dataframe, target_columns="target")

    assert len(dataset) == len(dataframe)


def test_transform_changes_the_features_and_leaves_the_target(dataframe):
    dataset = TableDataset(
        dataframe,
        target_columns="target",
        transform=lambda features: features + 10,
    )
    item_x, item_y = dataset[0]

    assert torch.equal(item_x, dataset.x[0] + 10)
    assert torch.equal(item_y, dataset.y[0])
    assert torch.equal(dataset.x[0], torch.tensor([1.0, 4.0]))


def test_without_a_transform_the_row_matches_the_table(dataframe):
    dataset = TableDataset(dataframe, target_columns="target")

    assert torch.equal(dataset[1][0], dataset.x[1])
    assert torch.equal(dataset[1][1], dataset.y[1])
