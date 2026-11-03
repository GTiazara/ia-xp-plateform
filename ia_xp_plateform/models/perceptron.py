"""

Un perceptron est le plus simple des réseaux de neurones artificiels : il
combine ses entrées par une somme pondérée (+ un biais), puis une fonction
d'activation transforme cette somme en décision. Il respecte le contrat
`BaseModel` : un Trainer qui l'utilise n'a besoin de rien savoir de plus
que `forward(x) -> torch.Tensor`.

"""

import torch
from torch import nn

from ia_xp_plateform.models.base import BaseModel


class Perceptron(BaseModel):

    def __init__(
        self,
        input_features: int = 2,
        output_features: int = 1
    ) -> None :
        """Initialise le perceptron.

        Parameters
        ----------
        input_features : int, optional
            Nombre d'entrées du perceptron, par défaut 2.
        output_features : int, optional
            Nombre de sorties du perceptron, par défaut 1.
        """

        super().__init__()
        # Un perceptron = une seule couche linéaire (poids + biais).
        self.linear = nn.Linear(
            in_features = input_features,
            out_features = output_features
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Calcule le logit de décision pour un batch d'entrées.

        Parameters
        ----------
        x : torch.Tensor
            Tenseur de forme `(batch, in_features)`.

        Returns
        -------
        torch.Tensor
            Tenseur de forme `(batch,)` contenant le logit (avant activation)
            de la décision "accélérer".
        """
        logits = self.linear(x)
        # La fonction sigmoïde transforme le logit en une probabilité.
        return torch.sigmoid(logits)

