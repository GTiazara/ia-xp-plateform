"""Tests des fichiers d'état écrits à côté des checkpoints."""

import argparse

import pytest

from ia_xp_plateform.cli.command.common import (
    apply_saved_preprocessing,
    build_preprocessing,
    load_preprocessing,
    preprocessing_path,
    save_preprocessing,
)
from ia_xp_plateform.factory.transform_factory import create_preprocessing


def test_the_state_file_follows_the_checkpoint_name():
    assert (
        preprocessing_path("logs/best-model.ckpt").name
        == "best-model.preprocessing.json"
    )
    assert (
        preprocessing_path("logs/best-model-v1.ckpt").name
        == "best-model-v1.preprocessing.json"
    )


def test_states_roundtrip_through_json(tmp_path):
    states = [
        {
            "name": "standardize",
            "state": {"name": "standardize", "mean": [1.0], "std": [2.0]},
        }
    ]
    path = tmp_path / "best-model.preprocessing.json"
    save_preprocessing(path, states)

    assert load_preprocessing(path) == states


def test_a_missing_state_file_names_itself(tmp_path):
    path = tmp_path / "best-model.preprocessing.json"

    with pytest.raises(FileNotFoundError, match="best-model.preprocessing.json"):
        load_preprocessing(path)


def test_a_missing_transform_name_is_reported(tmp_path):
    preprocessing = create_preprocessing({"predict": [{"name": "standardize"}]})
    checkpoint = tmp_path / "best-model.ckpt"
    save_preprocessing(preprocessing_path(checkpoint), [])

    with pytest.raises(ValueError, match="standardize"):
        apply_saved_preprocessing(preprocessing, checkpoint, "predict")


def test_a_pipeline_without_statistics_does_not_need_a_file(tmp_path):
    preprocessing = create_preprocessing(
        {"predict": [{"name": "clip", "min": 0, "max": 1}]}
    )

    apply_saved_preprocessing(
        preprocessing, tmp_path / "absent.ckpt", "predict"
    )


def test_an_argparse_namespace_builds_no_pipeline():
    args = argparse.Namespace(data="jeu.csv")

    assert build_preprocessing(args) is None
