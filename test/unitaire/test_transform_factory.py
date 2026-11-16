"""Tests de la factory de transformations."""

import pytest
import torch

from ia_xp_plateform.factory import TRANSFORMS, create_preprocessing, create_transform


def test_the_four_names_build_a_transform():
    assert set(TRANSFORMS) == {"clip", "gaussian_noise", "min_max", "standardize"}
    assert create_transform("clip", min=0).name == "clip"
    assert create_transform("gaussian_noise", std=0.2).name == "gaussian_noise"
    assert create_transform("min_max").name == "min_max"
    assert create_transform("standardize").name == "standardize"


def test_an_unknown_name_lists_the_available_transforms():
    with pytest.raises(ValueError, match="inconnue") as error:
        create_transform("inconnue")

    message = str(error.value)
    assert "inconnue" in message
    assert "standardize" in message
    assert "min_max" in message


def test_an_unknown_parameter_names_the_transform():
    with pytest.raises(ValueError, match="standardize"):
        create_transform("standardize", nope=1)


def test_a_step_without_a_name_is_rejected():
    with pytest.raises(ValueError, match="name"):
        create_preprocessing({"train": [{"std": 0.1}]})


def test_none_builds_empty_pipelines():
    preprocessing = create_preprocessing(None)
    x = torch.tensor([[1.0]])

    for split in ("train", "val", "test", "predict"):
        assert preprocessing.pipeline(split).transforms == []
        assert torch.equal(preprocessing.pipeline(split)(x), x)


def test_a_missing_split_is_an_empty_pipeline():
    preprocessing = create_preprocessing({"train": [{"name": "standardize"}]})

    assert preprocessing.pipeline("train").transforms[0].name == "standardize"
    assert preprocessing.pipeline("val").transforms == []
