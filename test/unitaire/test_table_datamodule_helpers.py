"""Tests unitaires des fonctions internes de `table_datamodule`.

Ces fonctions (préfixées `_`) portent toute la logique de lecture et de
découpage ; `TableDataModule` ne fait que les enchaîner. Les tester
directement permet de couvrir chaque cas d'erreur sans construire un
datamodule complet.
"""

import pandas as pd
import pytest
import torch

from ia_xp_plateform.data.datamodule.table_datamodule import (
    _read_table,
    _require_unique_id,
    _split_dataframe,
    _split_from_file,
)


# --- _read_table ---------------------------------------------------------


def test_read_table_csv(tmp_path):
    dataframe = pd.DataFrame({"a": [1, 2]})
    path = tmp_path / "data.csv"
    dataframe.to_csv(path, index=False)

    result = _read_table(path)

    pd.testing.assert_frame_equal(result, dataframe)


def test_read_table_feather(tmp_path):
    dataframe = pd.DataFrame({"a": [1, 2]})
    path = tmp_path / "data.feather"
    dataframe.to_feather(path)

    result = _read_table(path)

    pd.testing.assert_frame_equal(result, dataframe)


def test_read_table_hdf5(tmp_path):
    dataframe = pd.DataFrame({"a": [1, 2]})
    path = tmp_path / "data.h5"
    dataframe.to_hdf(path, key="data", mode="w")

    result = _read_table(path, hdf_key="data")

    pd.testing.assert_frame_equal(result, dataframe)


def test_read_table_rejects_unknown_extension(tmp_path):
    path = tmp_path / "data.json"
    path.write_text("{}")

    with pytest.raises(ValueError):
        _read_table(path)


# --- _require_unique_id ---------------------------------------------------


def test_require_unique_id_missing_column():
    dataframe = pd.DataFrame({"a": [1, 2]})

    with pytest.raises(ValueError):
        _require_unique_id(dataframe, "id")


def test_require_unique_id_rejects_duplicates():
    dataframe = pd.DataFrame({"id": [1, 1], "a": [1, 2]})

    with pytest.raises(ValueError):
        _require_unique_id(dataframe, "id")


def test_require_unique_id_accepts_unique_ids():
    dataframe = pd.DataFrame({"id": [1, 2], "a": [1, 2]})

    _require_unique_id(dataframe, "id")  # ne doit pas lever


# --- _split_dataframe ------------------------------------------------------


def test_split_dataframe_sizes_and_no_overlap():
    dataframe = pd.DataFrame({"id": range(100), "a": range(100)})

    train, val, test = _split_dataframe(
        dataframe, val_split=0.2, test_split=0.1, seed=0
    )

    assert len(train) == 70
    assert len(val) == 20
    assert len(test) == 10

    train_ids = set(train["id"])
    val_ids = set(val["id"])
    test_ids = set(test["id"])

    assert train_ids | val_ids | test_ids == set(range(100))
    assert not (train_ids & val_ids)
    assert not (train_ids & test_ids)
    assert not (val_ids & test_ids)


def test_split_dataframe_is_reproducible_with_the_same_seed():
    dataframe = pd.DataFrame({"id": range(50), "a": range(50)})

    first_train, _, _ = _split_dataframe(
        dataframe, val_split=0.2, test_split=0.1, seed=7
    )
    second_train, _, _ = _split_dataframe(
        dataframe, val_split=0.2, test_split=0.1, seed=7
    )

    pd.testing.assert_frame_equal(first_train, second_train)


def test_split_validation_ids_ignore_the_global_generator():
    dataframe = pd.DataFrame({"id": range(50), "a": range(50)})

    torch.manual_seed(1)
    _, first_val, _ = _split_dataframe(
        dataframe, val_split=0.2, test_split=0.1, seed=7
    )
    torch.manual_seed(2)
    _, second_val, _ = _split_dataframe(
        dataframe, val_split=0.2, test_split=0.1, seed=7
    )

    assert first_val["id"].tolist() == second_val["id"].tolist()


def test_split_dataframe_changes_with_a_different_seed():
    dataframe = pd.DataFrame({"id": range(50), "a": range(50)})

    first_train, _, _ = _split_dataframe(
        dataframe, val_split=0.2, test_split=0.1, seed=1
    )
    second_train, _, _ = _split_dataframe(
        dataframe, val_split=0.2, test_split=0.1, seed=2
    )

    assert sorted(first_train["id"]) != sorted(second_train["id"])


# --- _split_from_file -------------------------------------------------------


def test_split_from_file_joins_by_id_not_by_row_order():
    dataframe = pd.DataFrame({"id": [0, 1, 2, 3], "a": [10, 11, 12, 13]})
    # Le fichier split n'est pas dans le même ordre que `dataframe`.
    split = pd.DataFrame(
        {"id": [3, 1, 0, 2], "split": ["test", "train", "train", "val"]}
    )

    train, val, test = _split_from_file(dataframe, split, id_column="id")

    assert sorted(train["id"].tolist()) == [0, 1]
    assert val["id"].tolist() == [2]
    assert test["id"].tolist() == [3]


def test_split_from_file_missing_parts_return_none():
    dataframe = pd.DataFrame({"id": [0, 1]})
    split = pd.DataFrame({"id": [0, 1], "split": ["train", "train"]})

    train, val, test = _split_from_file(dataframe, split, id_column="id")

    assert len(train) == 2
    assert val is None
    assert test is None


def test_split_from_file_requires_id_column_in_data():
    dataframe = pd.DataFrame({"a": [1, 2]})
    split = pd.DataFrame({"id": [0, 1], "split": ["train", "train"]})

    with pytest.raises(ValueError):
        _split_from_file(dataframe, split, id_column="id")


def test_split_from_file_requires_id_column_in_split():
    dataframe = pd.DataFrame({"id": [0, 1]})
    split = pd.DataFrame({"split": ["train", "train"]})

    with pytest.raises(ValueError):
        _split_from_file(dataframe, split, id_column="id")


def test_split_from_file_requires_split_column():
    dataframe = pd.DataFrame({"id": [0, 1]})
    split = pd.DataFrame({"id": [0, 1]})

    with pytest.raises(ValueError):
        _split_from_file(dataframe, split, id_column="id")


def test_split_from_file_rejects_duplicate_id_in_data():
    dataframe = pd.DataFrame({"id": [0, 0]})
    split = pd.DataFrame({"id": [0, 1], "split": ["train", "train"]})

    with pytest.raises(ValueError):
        _split_from_file(dataframe, split, id_column="id")


def test_split_from_file_rejects_duplicate_id_in_split():
    dataframe = pd.DataFrame({"id": [0, 1]})
    split = pd.DataFrame({"id": [0, 0], "split": ["train", "train"]})

    with pytest.raises(ValueError):
        _split_from_file(dataframe, split, id_column="id")


def test_split_from_file_rejects_id_missing_from_split():
    dataframe = pd.DataFrame({"id": [0, 1, 2]})
    split = pd.DataFrame({"id": [0, 1], "split": ["train", "train"]})

    with pytest.raises(ValueError):
        _split_from_file(dataframe, split, id_column="id")


def test_split_from_file_rejects_extra_id_in_split():
    dataframe = pd.DataFrame({"id": [0, 1]})
    split = pd.DataFrame(
        {"id": [0, 1, 2], "split": ["train", "train", "train"]}
    )

    with pytest.raises(ValueError):
        _split_from_file(dataframe, split, id_column="id")


def test_split_from_file_rejects_unknown_split_value():
    dataframe = pd.DataFrame({"id": [0, 1]})
    split = pd.DataFrame({"id": [0, 1], "split": ["train", "other"]})

    with pytest.raises(ValueError):
        _split_from_file(dataframe, split, id_column="id")


def test_split_from_file_requires_at_least_one_train_row():
    dataframe = pd.DataFrame({"id": [0, 1]})
    split = pd.DataFrame({"id": [0, 1], "split": ["val", "test"]})

    with pytest.raises(ValueError):
        _split_from_file(dataframe, split, id_column="id")
