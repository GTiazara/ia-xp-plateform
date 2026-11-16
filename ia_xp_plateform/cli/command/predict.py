"""Commande `predict` : classe de nouvelles lignes avec un checkpoint."""

import argparse
from pathlib import Path

import pandas as pd
import torch

from ia_xp_plateform.cli.command.common import (
    apply_saved_preprocessing,
    build_preprocessing,
    build_task,
    feature_columns_of,
    inputs_of,
    model_kwargs,
    read_table,
)
from ia_xp_plateform.inference import Predictor


def prediction_frame(
    identifiers,
    probabilities: torch.Tensor,
    classes: torch.Tensor,
    id_column: str,
) -> pd.DataFrame:
    """Tableau identifiant, probabilités et classe, selon la forme reçue.

    Une probabilité et une classe : `probability` et `prediction`.
    Plusieurs probabilités et une seule classe : `prediction` puis
    `probability_0` … . Autant de classes que de probabilités : une
    paire `probability_k`, `prediction_k` par sortie.
    """
    result = pd.DataFrame({id_column: identifiers})
    if classes.shape[-1] == 1 and probabilities.shape[-1] == 1:
        result["probability"] = probabilities[:, 0].numpy()
        result["prediction"] = classes[:, 0].numpy()
    elif classes.shape[-1] == 1:
        result["prediction"] = classes[:, 0].numpy()
        for index in range(probabilities.shape[1]):
            result[f"probability_{index}"] = probabilities[:, index].numpy()
    else:
        for index in range(probabilities.shape[1]):
            result[f"probability_{index}"] = probabilities[:, index].numpy()
            result[f"prediction_{index}"] = classes[:, index].numpy()
    return result


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
        task=build_task(args),
    )
    features = inputs_of(dataframe, columns)
    preprocessing = build_preprocessing(args)
    apply_saved_preprocessing(preprocessing, args.checkpoint, "predict")
    if preprocessing is not None:
        features = preprocessing.pipeline("predict")(features)
    probabilities = predictor.predict_proba(features)
    classes = predictor.predict(features)
    result = prediction_frame(
        dataframe[args.id_column],
        probabilities,
        classes,
        args.id_column,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(output)
