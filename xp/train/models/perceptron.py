"""XP-01 : un perceptron pour décider d'accélérer ou non en conduite.

Un perceptron est le plus simple des réseaux de neurones artificiels : il
combine ses entrées par une somme pondérée (+ un biais), puis une fonction
d'activation transforme cette somme en décision.

Pour cette première XP, les 4 entrées sont (voir
`xp.train.data.datasets.car_driving.FEATURES` pour le détail) :

    0. distance à l'obstacle (mètres)
    1. présence d'un obstacle (0 = non, 1 = oui)
    2. position de la voiture (0 = gauche, 1 = droite)
    3. vitesse (km/h)

et la sortie attendue est une décision binaire :

    0 -> ne pas accélérer
    1 -> accélérer

Voir `docs/xp-01-perceptron-conduite.md` pour le contexte pédagogique complet
de cette XP, et ses limites (un perceptron simple ne sait apprendre que des
relations linéairement séparables : c'est pour cela qu'on parlera plus tard
de MLP, de CNN, puis d'apprentissage par renforcement pour une vraie voiture
autonome).
"""
from pathlib import Path
from typing import Optional

import torch
from torch import nn

from xp.train.models.base import BaseModel

#: Nombre d'entrées par défaut du perceptron de l'XP-01.
DEFAULT_IN_FEATURES = 4


class Perceptron(BaseModel):
    """Perceptron à une seule couche (pas de couche cachée).

    Produit un unique logit ; la décision binaire (accélérer / ne pas
    accélérer) est obtenue en appliquant un seuil sur `sigmoid(logit)`. Ce
    seuillage est fait dans `xp.train.module.binary_classification.BinaryDecisionModule`
    plutôt que dans le modèle lui-même, afin de pouvoir entraîner avec
    `BCEWithLogitsLoss` (numériquement plus stable qu'un `sigmoid` + `BCELoss`).
    """

    def __init__(
        self,
        in_features: int = DEFAULT_IN_FEATURES,
        pretrained_weights: Optional[Path] = None,
    ) -> None:
        """Initialise le perceptron.

        Parameters
        ----------
        in_features : int, optional
            Nombre d'entrées du perceptron, par défaut 4.
        pretrained_weights : Path, optional
            Poids d'une XP précédente à recharger, par défaut `None`.
        """
        super().__init__(pretrained_weights=pretrained_weights)
        self.in_features = in_features
        # Un perceptron = une seule couche linéaire (poids + biais).
        self.linear = nn.Linear(in_features=in_features, out_features=1)
        self.load_pretrained_weights()

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
        return self.linear(x).squeeze(-1)
