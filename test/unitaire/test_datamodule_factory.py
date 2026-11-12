"""Tests unitaires de la factory de datamodules.

La factory ne lit pas le fichier : elle ne fait que choisir la classe et
lui transmettre les arguments. Le chargement est vérifié dans les tests
d'intégration.
"""

import pytest

from ia_xp_plateform.data.datamodule.table_datamodule import TableDataModule
from ia_xp_plateform.factory.datamodule_factory import (
    DATAMODULES,
    create_datamodule,
)


def test_create_table_datamodule():
    datamodule = create_datamodule(
        "table",
        data_path="data.csv",
        target_columns="y",
        batch_size=8,
    )

    assert isinstance(datamodule, TableDataModule)
    assert datamodule.data_path == "data.csv"
    assert datamodule.target_columns == "y"
    assert datamodule.batch_size == 8


def test_unknown_datamodule_name_lists_the_available_ones():
    with pytest.raises(ValueError, match="table"):
        create_datamodule("inconnu", data_path="data.csv", target_columns="y")


def test_registry_maps_table_to_its_class():
    assert DATAMODULES["table"] is TableDataModule
