"""Tests unitaires de `BaseModel`.

`BaseModel` impose `forward` et reste un `nn.Module` normal. On vérifie
que l'instanciation est bloquée sans `forward`, et que l'API `nn.Module`
(paramètres, appel) fonctionne pour une sous-classe qui l'implémente.
"""

import pytest
import torch
from torch import nn

from ia_xp_plateform.models.base import BaseModel


def test_cannot_instantiate_without_forward():
    class Broken(BaseModel):
        pass

    with pytest.raises(TypeError):
        Broken()


def test_subclass_with_forward_behaves_like_a_nn_module():
    class Identity(BaseModel):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    model = Identity()

    assert isinstance(model, nn.Module)
    assert isinstance(model, BaseModel)

    x = torch.randn(4, 3)
    assert torch.equal(model(x), x)
