"""Tests unitaires de `BaseDataset`.

`BaseDataset` ne fait que deux choses : donner la taille du dataset et
renvoyer le couple `(x, y)` d'un index. On vérifie ce comportement, et la
vérification de cohérence entre `x` et `y`.
"""

import pytest
import torch

from ia_xp_plateform.data.dataset.base import BaseDataset


def test_len_returns_number_of_examples():
    x = torch.randn(5, 3)
    y = torch.randn(5, 1)

    dataset = BaseDataset(x, y)

    assert len(dataset) == 5


def test_getitem_returns_the_matching_pair():
    x = torch.arange(6).reshape(3, 2).float()
    y = torch.tensor([[0.0], [1.0], [0.0]])

    dataset = BaseDataset(x, y)
    item_x, item_y = dataset[1]

    assert torch.equal(item_x, x[1])
    assert torch.equal(item_y, y[1])


def test_transform_is_applied_to_x_only_and_the_stored_tensor_stays_raw():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y = torch.tensor([[0.0], [1.0]])

    dataset = BaseDataset(x, y, transform=lambda features: features * 2)
    item_x, item_y = dataset[0]

    assert torch.equal(item_x, x[0] * 2)
    assert torch.equal(item_y, y[0])
    assert torch.equal(dataset.x, x)


def test_without_a_transform_the_example_is_the_stored_tensor():
    x = torch.tensor([[1.0], [2.0]])
    y = torch.tensor([[0.0], [1.0]])
    dataset = BaseDataset(x, y)

    assert torch.equal(dataset[1][0], x[1])


def test_rejects_x_and_y_with_different_lengths():
    x = torch.randn(3, 2)
    y = torch.randn(2, 1)

    with pytest.raises(ValueError):
        BaseDataset(x, y)
