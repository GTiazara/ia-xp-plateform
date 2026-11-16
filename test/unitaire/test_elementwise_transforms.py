"""Tests de `clip` et `gaussian_noise`."""

import pytest
import torch

from ia_xp_plateform.transform.elementwise import Clip, GaussianNoise


def test_clip_limits_values_to_the_given_interval():
    transformed = Clip(min=0, max=10)(torch.tensor([-5.0, 3.0, 40.0]))

    assert torch.equal(transformed, torch.tensor([0.0, 3.0, 10.0]))


def test_clip_requires_at_least_one_bound():
    with pytest.raises(ValueError, match="clip"):
        Clip()


def test_clip_rejects_inverted_bounds():
    with pytest.raises(ValueError, match="clip"):
        Clip(min=2, max=1)


def test_gaussian_noise_rejects_a_non_positive_std():
    with pytest.raises(ValueError, match="gaussian_noise"):
        GaussianNoise(std=0)
    with pytest.raises(ValueError, match="gaussian_noise"):
        GaussianNoise(std=-1)


def test_gaussian_noise_changes_between_calls_unless_the_seed_is_reset():
    transform = GaussianNoise(std=0.5)
    x = torch.zeros(1000)

    assert not torch.equal(transform(x), transform(x))
    torch.manual_seed(0)
    first = transform(x)
    torch.manual_seed(0)
    second = transform(x)
    assert torch.equal(first, second)


def test_elementwise_transforms_have_an_empty_state():
    assert Clip(min=0).state_dict() == {}
    assert GaussianNoise(std=0.1).state_dict() == {}
