"""DataModule Lightning pour l'XP-01 (voiture virtuelle / perceptron).

Inspiré de `bdf.train.data.datamodule.DataModule`, très simplifié : il n'y a
qu'un seul dataset synthétique à générer puis à répartir en train/val/test
(pas de fichier HDF5, pas de `Splitter` externe, pas d'augmentations).
"""
import torch
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from xp.cli.utils import get_pylogger
from xp.train.data.datasets.car_driving import CarDrivingConfig
from xp.train.data.datasets.car_driving import CarDrivingDataset

log = get_pylogger(__name__)


class CarDrivingDataModule(LightningDataModule):
    """Génère et répartit (train/val/test) le jeu de données de l'XP-01.

    Parameters
    ----------
    n_samples : int, optional
        Nombre total d'échantillons synthétiques à générer, par défaut 2000.
    val_split : float, optional
        Proportion réservée à la validation, par défaut 0.2.
    test_split : float, optional
        Proportion réservée au test, par défaut 0.1.
    batch_size : int, optional
        Taille des batches, par défaut 32.
    num_workers : int, optional
        Nombre de workers pour le `DataLoader`, par défaut 0.
    seed : int, optional
        Graine aléatoire pour la génération des données et le split,
        par défaut 42.
    label_noise : float, optional
        Proportion d'étiquettes bruitées (voir `CarDrivingConfig`),
        par défaut 0.05.
    """

    def __init__(
        self,
        n_samples: int = 2000,
        val_split: float = 0.2,
        test_split: float = 0.1,
        batch_size: int = 32,
        num_workers: int = 0,
        seed: int = 42,
        label_noise: float = 0.05,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.val_split = val_split
        self.test_split = test_split
        self.dataset_config = CarDrivingConfig(
            n_samples=n_samples, label_noise=label_noise, seed=seed
        )

    def setup(self, stage: str) -> None:
        """Génère le dataset complet puis le découpe en train/val/test."""
        full_dataset = CarDrivingDataset(self.dataset_config)
        n_total = len(full_dataset)
        n_val = int(n_total * self.val_split)
        n_test = int(n_total * self.test_split)
        n_train = n_total - n_val - n_test

        generator = torch.Generator().manual_seed(self.seed)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, [n_train, n_val, n_test], generator=generator
        )
        log.info(
            f"Jeu de données XP-01 : {n_train} train / {n_val} val / {n_test} test"
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=False,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return self.test_dataloader()
