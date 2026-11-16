"""Commande `train` : entraîne un modèle et écrit son checkpoint."""

import argparse

import pytorch_lightning as pl

from ia_xp_plateform.callback import CallbackConfig, create_callbacks
from ia_xp_plateform.cli.command.common import (
    build_datamodule,
    build_model,
    build_task,
    make_trainer,
    preprocessing_path,
    save_preprocessing,
)
from ia_xp_plateform.module import ClassificationModule


def run_train(args: argparse.Namespace) -> None:
    """Entraîne, puis affiche le chemin du meilleur checkpoint.

    La graine de l'expérience est posée avant toute construction : elle
    fixe l'initialisation des poids et l'ordre des batches. Le découpage
    garde son propre générateur.
    """
    pl.seed_everything(int(args.seed), workers=True, verbose=False)
    datamodule = build_datamodule(args)
    datamodule.setup()
    model = build_model(args, len(datamodule.train_dataset.feature_columns))
    module = ClassificationModule(
        model,
        learning_rate=args.learning_rate,
        threshold=args.threshold,
        task=build_task(args),
    )
    callbacks = create_callbacks(
        CallbackConfig(
            output_dir=args.output_dir,
            early_stopping_patience=args.patience,
        )
    )
    trainer = make_trainer(
        args,
        output_dir=args.output_dir,
        max_epochs=args.epochs,
        callbacks=callbacks,
    )
    trainer.fit(module, datamodule=datamodule)
    best = trainer.checkpoint_callback.best_model_path
    if best and datamodule.preprocessing is not None:
        save_preprocessing(
            preprocessing_path(best), datamodule.preprocessing.states()
        )
    print(best)
