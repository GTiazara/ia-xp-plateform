"""Tests du dataset synthétique XP-01 (voiture virtuelle)."""
import torch

from xp.train.data.datasets.car_driving import CarDrivingConfig
from xp.train.data.datasets.car_driving import CarDrivingDataset
from xp.train.data.datasets.car_driving import FEATURES


def test_dataset_length_matches_config():
    config = CarDrivingConfig(n_samples=123, seed=0)
    dataset = CarDrivingDataset(config)

    assert len(dataset) == 123


def test_dataset_item_shapes_and_ranges():
    config = CarDrivingConfig(n_samples=50, seed=0)
    dataset = CarDrivingDataset(config)
    sample = dataset[0]

    assert sample["features"].shape == (len(FEATURES),)
    assert sample["label"].shape == ()
    assert torch.all(sample["features"] >= 0.0)
    assert torch.all(sample["features"] <= 1.0)
    assert sample["label"].item() in (0.0, 1.0)


def test_dataset_is_reproducible_with_same_seed():
    config = CarDrivingConfig(n_samples=50, seed=7)
    dataset_a = CarDrivingDataset(config)
    dataset_b = CarDrivingDataset(config)

    assert torch.equal(dataset_a.features, dataset_b.features)
    assert torch.equal(dataset_a.labels, dataset_b.labels)


def test_expert_rule_without_noise_is_mostly_sensible():
    """Sans obstacle, ou avec une grande distance, on doit pouvoir accélérer."""
    config = CarDrivingConfig(n_samples=500, label_noise=0.0, seed=1)
    dataset = CarDrivingDataset(config)

    no_obstacle_mask = dataset.features[:, 1] == 0.0
    assert torch.all(dataset.labels[no_obstacle_mask] == 1.0)
