"""Commande `train` : entraîne un modèle et écrit son checkpoint."""

import argparse

from ia_xp_plateform.callback import CallbackConfig, create_callbacks
from ia_xp_plateform.cli.command.common import build_datamodule, build_model, make_trainer
from ia_xp_plateform.module import ClassificationModule


def run_train(args: argparse.Namespace) -> None:
    """Entraîne, puis affiche le chemin du meilleur checkpoint."""
    datamodule = build_datamodule(args)
    datamodule.setup()
    model = build_model(args, len(datamodule.train_dataset.feature_columns))
    module = ClassificationModule(
        model,
        learning_rate=args.learning_rate,
        threshold=args.threshold,
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
    print(trainer.checkpoint_callback.best_model_path)
