"""Tests de `standardize` et `min_max`."""

import pytest
import torch

from ia_xp_plateform.transform.scaling import MinMax, Standardize


def _scaled_columns() -> torch.Tensor:
    column_a = torch.linspace(0, 5000, 11)
    column_b = torch.linspace(0, 3, 11)
    return torch.stack([column_a, column_b], dim=1)


def test_standardize_centers_each_column_and_scales_it_to_unit_std():
    x = _scaled_columns()
    transform = Standardize().fit(x)
    transformed = transform(x)

    assert transformed.shape == x.shape
    assert torch.allclose(transformed.mean(dim=0), torch.zeros(2), atol=1e-5)
    assert torch.allclose(
        transformed.std(dim=0, unbiased=False), torch.ones(2), atol=1e-5
    )


def test_standardize_matches_a_single_row_with_the_same_row_of_the_batch():
    x = _scaled_columns()
    transform = Standardize().fit(x)

    assert torch.allclose(transform(x)[0], transform(x[0]))


def test_standardize_maps_a_constant_column_to_zero():
    x = torch.tensor([[5.0, 1.0], [5.0, 3.0], [5.0, 5.0]])
    transformed = Standardize().fit(x)(x)

    assert torch.equal(transformed[:, 0], torch.zeros(3))


def test_standardize_refuses_to_run_before_it_is_fitted():
    with pytest.raises(RuntimeError, match="standardize"):
        Standardize()(torch.zeros(2, 1))


def test_standardize_state_roundtrip_reproduces_the_same_values():
    x = _scaled_columns()
    learned = Standardize().fit(x)
    restored = Standardize()
    restored.load_state_dict(learned.state_dict())

    assert torch.allclose(restored(x), learned(x))


def test_min_max_maps_the_learned_range_to_the_unit_interval():
    x = _scaled_columns()
    transformed = MinMax().fit(x)(x)

    assert transformed.shape == x.shape
    assert torch.allclose(transformed.min(dim=0).values, torch.zeros(2), atol=1e-5)
    assert torch.allclose(transformed.max(dim=0).values, torch.ones(2), atol=1e-5)


def test_min_max_does_not_truncate_a_value_outside_the_learned_range():
    learned = MinMax().fit(torch.tensor([[0.0], [1.0]]))

    assert learned(torch.tensor([[2.0]])).item() > 1


def test_min_max_maps_a_constant_column_to_zero():
    x = torch.tensor([[4.0], [4.0], [4.0]])

    assert torch.equal(MinMax().fit(x)(x), torch.zeros(3, 1))


def test_min_max_state_is_refused_by_standardize():
    state = MinMax().fit(torch.tensor([[0.0], [1.0]])).state_dict()

    with pytest.raises(ValueError, match="min_max"):
        Standardize().load_state_dict(state)
