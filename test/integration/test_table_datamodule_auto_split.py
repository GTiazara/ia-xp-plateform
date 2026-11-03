"""Tests d'intégration : `TableDataModule` qui découpe lui-même un fichier.

Contrairement aux tests unitaires de `test_table_datamodule_helpers.py`,
ces tests passent par `TableDataModule` en entier : lecture du fichier
sur disque, découpage, construction des `TableDataset`, puis des
`DataLoader`.
"""

import pytest

from ia_xp_plateform.data.datamodule import TableDataModule


def _write(dataframe, path, hdf_key=None):
    suffix = path.suffix
    if suffix == ".csv":
        dataframe.to_csv(path, index=False)
    elif suffix == ".feather":
        dataframe.to_feather(path)
    elif suffix in (".h5", ".hdf5"):
        dataframe.to_hdf(path, key=hdf_key, mode="w")
    else:
        raise ValueError(suffix)


@pytest.mark.parametrize("suffix", [".csv", ".feather", ".h5"])
def test_reads_all_supported_formats(tmp_path, sample_dataframe, suffix):
    path = tmp_path / f"data{suffix}"
    _write(sample_dataframe, path, hdf_key="data")
    kwargs = {"hdf_key": "data"} if suffix == ".h5" else {}

    datamodule = TableDataModule(
        data_path=path,
        target_columns="target",
        val_split=0.2,
        test_split=0.1,
        **kwargs,
    )
    datamodule.setup()

    assert len(datamodule.train_dataset) == 14
    assert len(datamodule.val_dataset) == 4
    assert len(datamodule.test_dataset) == 2


def test_default_feature_columns_exclude_target_and_id(tmp_path, sample_dataframe):
    path = tmp_path / "data.csv"
    sample_dataframe.to_csv(path, index=False)

    datamodule = TableDataModule(data_path=path, target_columns="target")
    datamodule.setup()

    assert datamodule.train_dataset.feature_columns == ["a", "b"]


def test_train_dataloader_produces_batches_of_the_right_shape(
    tmp_path, sample_dataframe
):
    path = tmp_path / "data.csv"
    sample_dataframe.to_csv(path, index=False)

    datamodule = TableDataModule(
        data_path=path, target_columns="target", batch_size=4
    )
    datamodule.setup()

    batch_x, batch_y = next(iter(datamodule.train_dataloader()))

    assert batch_x.shape == (4, 2)
    assert batch_y.shape == (4, 1)


def test_requires_an_id_column_even_without_split_file(tmp_path, sample_dataframe):
    path = tmp_path / "data.csv"
    sample_dataframe.drop(columns=["id"]).to_csv(path, index=False)

    datamodule = TableDataModule(data_path=path, target_columns="target")

    with pytest.raises(ValueError):
        datamodule.setup()


def test_split_is_reproducible_across_setup_calls(tmp_path, sample_dataframe):
    path = tmp_path / "data.csv"
    sample_dataframe.to_csv(path, index=False)

    datamodule = TableDataModule(
        data_path=path, target_columns="target", split_seed=0
    )

    datamodule.setup()
    first_ids = datamodule.train_dataset.x[:, 0].tolist()

    datamodule.setup()
    second_ids = datamodule.train_dataset.x[:, 0].tolist()

    assert first_ids == second_ids
