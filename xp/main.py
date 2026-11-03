"""Point d'entrée principal de la plateforme ia-xp-plateform.

Usage (depuis la racine du projet) :

    poetry run python xp/main.py                 # lance l'XP par défaut (XP-01)
    poetry run python xp/main.py -cn <config>     # lance une autre configuration
    poetry run python xp/main.py trainer.max_epochs=5 datamodule.batch_size=64

Voir `run.sh` pour des exemples, et `docs/xp-01-perceptron-conduite.md` pour le
contexte de la première XP (un perceptron qui décide d'accélérer ou non).
"""
from pathlib import Path

import hydra
from dotenv import load_dotenv
from lightning_fabric import seed_everything
from omegaconf import DictConfig

from xp.cli.launch_experiment import launch_experiment


@hydra.main(config_path="../conf", config_name="train_xp01_perceptron", version_base="1.3")
def launch_train(config: DictConfig) -> None:
    """Fonction principale appelée par Hydra pour lancer une XP."""
    if "seed" in config.keys():
        seed_everything(config.seed, workers=True)
    return launch_experiment(config)


if __name__ == "__main__":
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        load_dotenv(env_file, override=True)
    launch_train()
