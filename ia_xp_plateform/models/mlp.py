"""

Un MLP (perceptron multicouche) empile des couches linéaires. La
profondeur est le nombre de couches cachées, la largeur leur nombre de
neurones. `forward` renvoie une probabilité entre 0 et 1 pour chaque
sortie, comme `Perceptron`.

"""

import torch
from torch import nn

from ia_xp_plateform.models.base import BaseModel


class MLP(BaseModel):
    """Classification par un réseau dont la profondeur est réglable.

    `hidden_layers` est le nombre de couches cachées. Chacune a la même
    largeur, `hidden_features`, et une ReLU. La dernière couche donne
    `output_features` probabilités. Avec une seule couche cachée et une
    seule sortie, `ClassificationModule` l'entraîne comme le perceptron.
    """

    def __init__(
        self,
        input_features: int = 2,
        output_features: int = 1,
        hidden_features: int = 16,
        hidden_layers: int = 1,
    ) -> None:
        """Initialise le MLP.

        Parameters
        ----------
        input_features : int, optional
            Nombre d'entrées, par défaut 2.
        output_features : int, optional
            Nombre de sorties, donc de probabilités, par défaut 1.
        hidden_features : int, optional
            Largeur de chaque couche cachée, par défaut 16.
        hidden_layers : int, optional
            Nombre de couches cachées, par défaut 1.
        """
        super().__init__()
        if input_features < 1:
            raise ValueError(
                f"input_features doit être au moins 1 ({input_features})."
            )
        if hidden_features < 1:
            raise ValueError(
                f"hidden_features doit être au moins 1 ({hidden_features})."
            )
        if hidden_layers < 1:
            raise ValueError(
                f"hidden_layers doit être au moins 1 ({hidden_layers})."
            )
        if output_features < 1:
            raise ValueError(
                f"output_features doit être au moins 1 ({output_features})."
            )

        self.output_features = output_features
        self.hidden_layers = hidden_layers
        blocks: list[nn.Module] = []
        width = input_features
        for _ in range(hidden_layers):
            blocks.append(nn.Linear(width, hidden_features))
            blocks.append(nn.ReLU())
            width = hidden_features
        blocks.append(nn.Linear(width, output_features))
        self.network = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Calcule une probabilité par sortie.

        Parameters
        ----------
        x : torch.Tensor
            Tenseur de forme `(batch, input_features)`.

        Returns
        -------
        torch.Tensor
            Tenseur de forme `(batch, output_features)`, chaque valeur
            entre 0 et 1.
        """
        return torch.sigmoid(self.network(x))
