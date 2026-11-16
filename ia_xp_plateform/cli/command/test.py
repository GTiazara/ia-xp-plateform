"""Commande `test` : évalue un checkpoint sur le jeu de test."""

import argparse

from ia_xp_plateform.cli.command.common import (
    apply_saved_preprocessing,
    build_datamodule,
    build_model,
    build_task,
    make_trainer,
)
from ia_xp_plateform.module import ClassificationModule


def run_test(args: argparse.Namespace) -> None:
    """Charge le checkpoint et calcule la perte et la précision de test."""
    datamodule = build_datamodule(args)
    apply_saved_preprocessing(datamodule.preprocessing, args.checkpoint, "test")
    datamodule.setup()
    if datamodule.test_dataset is None:
        raise ValueError(
            "Aucune ligne de test. Fournissez un fichier split qui en contient, "
            "ou gardez --test-split supérieur à 0."
        )
    model = build_model(args, len(datamodule.train_dataset.feature_columns))
    module = ClassificationModule(
        model,
        threshold=args.threshold,
        task=build_task(args),
    )
    trainer = make_trainer(args, output_dir=None, max_epochs=1)
    trainer.test(module, datamodule=datamodule, ckpt_path=args.checkpoint)
