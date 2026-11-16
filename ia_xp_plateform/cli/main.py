"""

Le CLI est la porte d'entrée de la plateforme. Il ne contient pas de
calcul : il lit les arguments, puis appelle les briques déjà construites
(factory, datamodule, module de classification, callbacks, prédicteur).

Trois commandes :

- `train` entraîne et écrit `output_dir/checkpoints/best-model.ckpt` ;
- `test` évalue ce checkpoint sur le jeu de test ;
- `predict` classe de nouvelles lignes et écrit un CSV.

"""

from __future__ import annotations

import argparse
import sys

from ia_xp_plateform.cli.command.common import add_data_arguments, add_model_arguments
from ia_xp_plateform.cli.command.predict import run_predict
from ia_xp_plateform.cli.command.test import run_test
from ia_xp_plateform.cli.command.train import run_train
from ia_xp_plateform.factory import LOGGERS


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--batch-size", type=int, default=32, help="Taille des lots. Défaut : 32.")
    parser.add_argument(
        "--accelerator",
        default="cpu",
        choices=["cpu", "gpu", "auto"],
        help="Appareil Lightning. Défaut : cpu.",
    )


def build_parser() -> argparse.ArgumentParser:
    """Construit le parseur et ses trois sous-commandes."""
    parser = argparse.ArgumentParser(
        prog="ia-xp",
        description="Entraîner, tester et prédire avec la plateforme.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="Entraîne un modèle.")
    add_data_arguments(train, require_target=True)
    add_model_arguments(train)
    _add_runtime_arguments(train)
    train.add_argument(
        "--logger",
        default="tensorboard",
        choices=sorted(LOGGERS),
        help="Logger de la factory. Défaut : tensorboard.",
    )
    train.add_argument("--output-dir", required=True, help="Dossier des logs et du checkpoint.")
    train.add_argument("--epochs", type=int, default=10, help="Nombre d'époques. Défaut : 10.")
    train.add_argument("--learning-rate", type=float, default=1e-2, help="Taux d'apprentissage. Défaut : 0.01.")
    train.add_argument("--patience", type=int, default=5, help="Patience de l'arrêt anticipé. Défaut : 5.")
    train.add_argument("--val-split", type=float, default=0.2, help="Part de validation si pas de fichier split.")
    train.add_argument("--test-split", type=float, default=0.1, help="Part de test si pas de fichier split.")
    train.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Graine du découpage, de l'initialisation et de l'ordre des batches."
        ),
    )
    train.set_defaults(func=run_train)

    test = commands.add_parser("test", help="Évalue un checkpoint.")
    add_data_arguments(test, require_target=True)
    add_model_arguments(test)
    _add_runtime_arguments(test)
    test.add_argument("--checkpoint", required=True, help="Fichier .ckpt à évaluer.")
    test.add_argument("--val-split", type=float, default=0.2)
    test.add_argument("--test-split", type=float, default=0.1)
    test.add_argument("--seed", type=int, default=42)
    test.set_defaults(func=run_test)

    predict = commands.add_parser("predict", help="Prédit avec un checkpoint.")
    add_data_arguments(predict, require_target=False)
    add_model_arguments(predict)
    predict.add_argument("--checkpoint", required=True, help="Fichier .ckpt entraîné.")
    predict.add_argument("--output", required=True, help="CSV des prédictions.")
    predict.set_defaults(func=run_predict)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Lance la commande demandée. Renvoie 0, ou 1 si les données sont invalides."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (ValueError, FileNotFoundError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0
