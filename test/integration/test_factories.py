"""Tests d'intégration : les factories branchées sur un vrai fichier.

On construit le datamodule et le modèle par leur nom, puis on vérifie
qu'un batch issu du datamodule passe dans le modèle.
"""

from ia_xp_plateform.factory import create_datamodule, create_model


def test_factories_feed_a_batch_into_the_model(tmp_path, sample_dataframe):
    data_path = tmp_path / "data.csv"
    sample_dataframe.to_csv(data_path, index=False)

    datamodule = create_datamodule(
        "table",
        data_path=data_path,
        target_columns="target",
        batch_size=4,
    )
    datamodule.setup()

    model = create_model(
        "perceptron",
        input_features=len(datamodule.train_dataset.feature_columns),
        output_features=1,
    )
    batch_x, batch_y = next(iter(datamodule.train_dataloader()))
    output = model(batch_x)

    assert output.shape == batch_y.shape
