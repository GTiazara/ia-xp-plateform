"""Réglages partagés par `train`, `test` et `predict`."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch

from ia_xp_plateform.data.datamodule.table_datamodule import (
    TableDataModule,
    _read_table,
)
from ia_xp_plateform.factory import create_datamodule, create_logger, create_model
from ia_xp_plateform.factory.logger_factory import LOGGER_DIR_KEYS
from ia_xp_plateform.models.base import BaseModel


def add_data_arguments(parser: argparse.ArgumentParser, *, require_target: bool) -> None:
    """Ajoute les arguments du fichier tabulaire."""
    parser.add_argument("--data", required=True, help="Fichier data.csv, .feather ou .h5.")
    parser.add_argument(
        "--target",
        required=require_target,
        help="Colonne cible. Obligatoire pour train et test.",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=None,
        help="Colonnes d'entrée. Par défaut, toutes sauf l'identifiant et la cible.",
    )
    parser.add_argument("--id-column", default="id", help="Colonne identifiant. Défaut : id.")
    parser.add_argument("--split", default=None, help="Fichier split compagnon, même format que data.")
    parser.add_argument("--hdf-key", default=None, help="Clé HDF5 du fichier de données.")
    parser.add_argument("--split-hdf-key", default=None, help="Clé HDF5 du fichier split.")


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    """Ajoute les arguments de l'architecture."""
    parser.add_argument("--model", default="perceptron", help="Nom dans la factory. Défaut : perceptron.")
    parser.add_argument(
        "--input-features",
        type=int,
        default=None,
        help="Nombre d'entrées. Défaut : le nombre de colonnes attributs.",
    )
    parser.add_argument("--output-features", type=int, default=1, help="Nombre de sorties. Défaut : 1.")
    parser.add_argument(
        "--hidden-features",
        type=int,
        default=None,
        help="Largeur de chaque couche cachée du MLP. Ignoré par le perceptron.",
    )
    parser.add_argument(
        "--hidden-layers",
        type=int,
        default=None,
        help="Nombre de couches cachées du MLP. Ignoré par le perceptron.",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Seuil de la classe 1. Défaut : 0.5.")


def feature_columns_of(dataframe: pd.DataFrame, args: argparse.Namespace) -> list[str]:
    """Colonnes d'entrée : celles demandées, sinon toutes sauf id et cible."""
    if args.features is not None:
        missing = [column for column in args.features if column not in dataframe.columns]
        if missing:
            raise ValueError(f"Colonnes absentes du fichier de données : {missing}.")
        return list(args.features)

    excluded = {args.id_column}
    if args.target is not None and args.target in dataframe.columns:
        excluded.add(args.target)
    columns = [column for column in dataframe.columns if column not in excluded]
    if not columns:
        raise ValueError("Aucune colonne attribut : précisez --features.")
    return columns


def build_datamodule(args: argparse.Namespace) -> TableDataModule:
    """Construit le datamodule tabulaire à partir des arguments."""
    return create_datamodule(
        "table",
        data_path=args.data,
        target_columns=args.target,
        feature_columns=args.features,
        split_path=args.split,
        id_column=args.id_column,
        hdf_key=args.hdf_key,
        split_hdf_key=args.split_hdf_key,
        batch_size=args.batch_size,
        val_split=getattr(args, "val_split", 0.2),
        test_split=getattr(args, "test_split", 0.1),
        split_seed=getattr(args, "seed", 42),
        split_column=getattr(args, "split_column", "split"),
        num_workers=getattr(args, "num_workers", 0),
    )


def model_kwargs(args: argparse.Namespace, input_features: int) -> dict:
    """Arguments d'architecture transmis à la factory de modèles.

    `hidden_features` et `hidden_layers` ne sont ajoutés que s'ils sont
    renseignés : le perceptron ne les connaît pas, le MLP oui.
    """
    kwargs = {
        "input_features": input_features,
        "output_features": args.output_features,
    }
    for key in ("hidden_features", "hidden_layers"):
        value = getattr(args, key, None)
        if value is not None:
            kwargs[key] = int(value)
    return kwargs


def build_model(args: argparse.Namespace, input_features: int) -> BaseModel:
    """Construit le modèle. `--input-features` doit coller aux colonnes."""
    if args.input_features is not None and args.input_features != input_features:
        raise ValueError(
            f"--input-features vaut {args.input_features}, mais les colonnes "
            f"attributs sont au nombre de {input_features}."
        )
    return create_model(args.model, **model_kwargs(args, input_features))


def read_table(path: str | Path, hdf_key: str | None) -> pd.DataFrame:
    """Lit un fichier tabulaire, quel que soit son format."""
    return _read_table(path, hdf_key=hdf_key)


def inputs_of(dataframe: pd.DataFrame, columns: list[str]) -> torch.Tensor:
    """Convertit les colonnes attributs en tenseur float32."""
    return torch.from_numpy(dataframe[columns].to_numpy(dtype="float32", copy=True))


def build_logger(args: argparse.Namespace, output_dir: str) -> pl.loggers.Logger:
    """Construit le logger nommé dans la configuration.

    Sans groupe Hydra (commande `ia-xp train`), on prend TensorBoard et on
    range l'expérience dans `output_dir/logger`. Les métriques viennent de
    `self.log` : ce builder ne choisit pas quoi mesurer.
    """
    logger_name = getattr(args, "logger", "tensorboard")
    params = dict(getattr(args, "logger_params", None) or {})
    params.setdefault(LOGGER_DIR_KEYS[logger_name], f"{output_dir}/logger")
    params.setdefault("name", "experience")
    return create_logger(logger_name, **params)


def make_trainer(
    args: argparse.Namespace,
    *,
    output_dir: str | None,
    max_epochs: int,
    callbacks: list | None = None,
) -> pl.Trainer:
    """Trainer CPU par défaut, avec un logger si un dossier de sortie est donné.

    Le suivi du taux d'apprentissage refuse un entraînement sans logger :
    on n'en pose un que lorsque `output_dir` est fourni (commande `train`).
    Le logger concret vient de la factory (`tensorboard` par défaut).
    Les callbacks doivent être passés ici : Lightning les enregistre à la
    construction du Trainer.
    """
    logger = False
    if output_dir is not None:
        logger = build_logger(args, output_dir)
    return pl.Trainer(
        max_epochs=max_epochs,
        min_epochs=getattr(args, "min_epochs", 1),
        log_every_n_steps=getattr(args, "log_every_n_steps", 5),
        callbacks=callbacks or [],
        logger=logger,
        accelerator=args.accelerator,
        devices=getattr(args, "devices", 1),
        enable_progress_bar=getattr(args, "enable_progress_bar", False),
        enable_model_summary=getattr(args, "enable_model_summary", False),
    )
