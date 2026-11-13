"""Commande `predict` : classe de nouvelles lignes avec un checkpoint."""

import argparse
from pathlib import Path

import pandas as pd

from ia_xp_plateform.cli.command.common import (
    feature_columns_of,
    inputs_of,
    model_kwargs,
    read_table,
)
from ia_xp_plateform.inference import Predictor


def run_predict(args: argparse.Namespace) -> None:
    """Écrit un CSV : identifiant, probabilité, classe prédite."""
    dataframe = read_table(args.data, hdf_key=args.hdf_key)
    columns = feature_columns_of(dataframe, args)
    if args.input_features is not None and args.input_features != len(columns):
        raise ValueError(
            f"--input-features vaut {args.input_features}, mais les colonnes "
            f"attributs sont au nombre de {len(columns)}."
        )

    predictor = Predictor.from_checkpoint(
        args.checkpoint,
        model_name=args.model,
        model_kwargs=model_kwargs(args, len(columns)),
        threshold=args.threshold,
    )
    probabilities = predictor.predict_proba(inputs_of(dataframe, columns))
    classes = (probabilities >= predictor.threshold).long()

    result = pd.DataFrame({args.id_column: dataframe[args.id_column]})
    if probabilities.shape[1] == 1:
        result["probability"] = probabilities[:, 0].numpy()
        result["prediction"] = classes[:, 0].numpy()
    else:
        for index in range(probabilities.shape[1]):
            result[f"probability_{index}"] = probabilities[:, index].numpy()
            result[f"prediction_{index}"] = classes[:, index].numpy()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(output)
