"""Tests d'intégration : `TableDataModule` avec un fichier `split` compagnon.

Le fichier `split` est volontairement mélangé (ordre différent de
`data`) pour vérifier que le rapprochement se fait bien par identifiant,
et pas par position.
"""

import pandas as pd
import pytest

from ia_xp_plateform.data.datamodule import TableDataModule


def _write_split(tmp_path, sample_dataframe, labels):
    data_path = tmp_path / "data.csv"
    split_path = tmp_path / "split.csv"

    sample_dataframe.to_csv(data_path, index=False)

    split = pd.DataFrame({"id": sample_dataframe["id"], "split": labels})
    # Le fichier split n'est pas censé suivre l'ordre de data : on le
    # mélange pour le vérifier.
    split = split.sample(frac=1, random_state=0).reset_index(drop=True)
    split.to_csv(split_path, index=False)

    return data_path, split_path


def test_split_file_out_of_order_still_matches_by_id(tmp_path, sample_dataframe):
    labels = ["train"] * 14 + ["val"] * 4 + ["test"] * 2
    data_path, split_path = _write_split(tmp_path, sample_dataframe, labels)

    datamodule = TableDataModule(
        data_path=data_path, split_path=split_path, target_columns="target"
    )
    datamodule.setup()

    assert len(datamodule.train_dataset) == 14
    assert len(datamodule.val_dataset) == 4
    assert len(datamodule.test_dataset) == 2


def test_missing_split_part_gives_a_none_dataloader(tmp_path, sample_dataframe):
    labels = ["train"] * 18 + ["test"] * 2
    data_path, split_path = _write_split(tmp_path, sample_dataframe, labels)

    datamodule = TableDataModule(
        data_path=data_path, split_path=split_path, target_columns="target"
    )
    datamodule.setup()

    assert datamodule.val_dataloader() is None
    assert datamodule.test_dataloader() is not None


def test_id_missing_from_split_file_raises(tmp_path, sample_dataframe):
    labels = ["train"] * 20
    data_path, split_path = _write_split(tmp_path, sample_dataframe, labels)
    split = pd.read_csv(split_path)
    split = split[split["id"] != 5]
    split.to_csv(split_path, index=False)

    datamodule = TableDataModule(
        data_path=data_path, split_path=split_path, target_columns="target"
    )

    with pytest.raises(ValueError):
        datamodule.setup()


def test_extra_id_in_split_file_raises(tmp_path, sample_dataframe):
    labels = ["train"] * 20
    data_path, split_path = _write_split(tmp_path, sample_dataframe, labels)
    split = pd.read_csv(split_path)
    split = pd.concat(
        [split, pd.DataFrame({"id": [999], "split": ["train"]})]
    )
    split.to_csv(split_path, index=False)

    datamodule = TableDataModule(
        data_path=data_path, split_path=split_path, target_columns="target"
    )

    with pytest.raises(ValueError):
        datamodule.setup()


def test_custom_id_column(tmp_path, sample_dataframe):
    renamed = sample_dataframe.rename(columns={"id": "row_uid"})
    data_path = tmp_path / "data.csv"
    renamed.to_csv(data_path, index=False)

    split_path = tmp_path / "split.csv"
    split = pd.DataFrame(
        {"row_uid": renamed["row_uid"], "split": ["train"] * 20}
    )
    split.to_csv(split_path, index=False)

    datamodule = TableDataModule(
        data_path=data_path,
        split_path=split_path,
        target_columns="target",
        id_column="row_uid",
    )
    datamodule.setup()

    assert len(datamodule.train_dataset) == 20
    assert "row_uid" not in datamodule.train_dataset.feature_columns
