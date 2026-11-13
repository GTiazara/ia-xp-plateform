"""Tests d'intégration bout en bout : datamodule -> DataLoader -> modèle.

Ces tests ne vérifient pas une brique isolée, mais que les briques
s'emboîtent correctement : la sortie d'un datamodule peut être donnée
telle quelle à un modèle `BaseModel`.
"""

import torch

from ia_xp_plateform.data.datamodule import TableDataModule
from ia_xp_plateform.models.mlp import MLP
from ia_xp_plateform.models.perceptron import Perceptron


def test_table_datamodule_feeds_the_perceptron_model(tmp_path, sample_dataframe):
    path = tmp_path / "data.csv"
    sample_dataframe.to_csv(path, index=False)

    datamodule = TableDataModule(
        data_path=path, target_columns="target", batch_size=4
    )
    datamodule.setup()

    model = Perceptron(
        input_features=len(datamodule.train_dataset.feature_columns),
        output_features=1,
    )
    batch_x, batch_y = next(iter(datamodule.train_dataloader()))
    output = model(batch_x)

    assert output.shape == batch_y.shape


def test_table_datamodule_feeds_the_mlp(tmp_path, sample_dataframe):
    path = tmp_path / "data.csv"
    sample_dataframe.to_csv(path, index=False)

    datamodule = TableDataModule(
        data_path=path, target_columns="target", batch_size=4
    )
    datamodule.setup()

    model = MLP(
        input_features=len(datamodule.train_dataset.feature_columns),
        output_features=1,
        hidden_features=4,
        hidden_layers=2,
    )
    batch_x, batch_y = next(iter(datamodule.train_dataloader()))
    output = model(batch_x)

    assert output.shape == batch_y.shape
    assert torch.all((output >= 0.0) & (output <= 1.0))
