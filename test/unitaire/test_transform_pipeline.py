"""Tests de `Pipeline` et `Preprocessing`."""

import pytest
import torch

from ia_xp_plateform.transform.elementwise import Clip
from ia_xp_plateform.transform.pipeline import Pipeline, Preprocessing
from ia_xp_plateform.transform.scaling import Standardize


def test_clip_then_standardize_learns_on_the_clipped_values():
    x = torch.tensor([[-10.0, 0.0], [5.0, 10.0], [100.0, 20.0]])
    clipped = x.clamp(min=0, max=10)
    preprocessing = Preprocessing(
        train=Pipeline([Clip(min=0, max=10), Standardize()]),
        val=Pipeline([Clip(min=0, max=10), Standardize()]),
    )

    preprocessing.fit(x)
    learned = preprocessing.pipeline("train").transforms[1]

    assert torch.allclose(learned.mean, clipped.mean(dim=0))
    assert torch.allclose(learned.std, clipped.std(dim=0, unbiased=False))


def test_val_reuses_the_state_learned_on_train():
    x = torch.tensor([[0.0], [10.0]])
    preprocessing = Preprocessing(
        train=Pipeline([Standardize()]),
        val=Pipeline([Standardize()]),
    )

    preprocessing.fit(x)
    train_mean = preprocessing.pipeline("train").transforms[0].mean
    val_mean = preprocessing.pipeline("val").transforms[0].mean

    assert torch.equal(val_mean, train_mean)


def test_the_nth_occurrence_reuses_the_state_of_the_same_rank():
    x = torch.tensor([[0.0], [10.0]])
    preprocessing = Preprocessing(
        train=Pipeline([Standardize(), Standardize()]),
        val=Pipeline([Standardize(), Standardize()]),
    )

    preprocessing.fit(x)
    train = preprocessing.pipeline("train").transforms
    val = preprocessing.pipeline("val").transforms

    assert not torch.allclose(train[0].mean, train[1].mean)
    assert torch.allclose(val[0].mean, train[0].mean)
    assert torch.allclose(val[1].mean, train[1].mean)


def test_a_stateful_transform_missing_from_train_names_the_split():
    preprocessing = Preprocessing(
        train=Pipeline([Clip(min=0, max=1)]),
        val=Pipeline([Standardize()]),
    )

    with pytest.raises(ValueError, match="standardize") as error:
        preprocessing.fit(torch.zeros(2, 1))

    assert "val" in str(error.value)


def test_load_states_marks_the_preprocessing_fitted_without_learning():
    source = Standardize().fit(torch.tensor([[0.0], [4.0]]))
    preprocessing = Preprocessing(
        train=Pipeline([Standardize()]),
        val=Pipeline([Standardize()]),
    )

    preprocessing.load_states([{"name": "standardize", "state": source.state_dict()}])

    assert preprocessing.fitted
    assert torch.allclose(
        preprocessing.pipeline("val").transforms[0].mean, source.mean
    )


def test_a_name_absent_from_the_states_is_rejected():
    preprocessing = Preprocessing(val=Pipeline([Standardize()]))

    with pytest.raises(ValueError, match="standardize"):
        preprocessing.load_states([])


def test_an_empty_pipeline_returns_its_input():
    x = torch.tensor([[1.0, 2.0]])

    assert torch.equal(Pipeline()(x), x)
