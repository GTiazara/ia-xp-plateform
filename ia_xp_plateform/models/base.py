"""

Un `Trainer` ne doit jamais avoir besoin de connaître l'architecture d'un
modèle (nombre de couches, type d'activation, ...) : il doit seulement
savoir qu'il peut lui donner un batch d'entrées et récupérer une sortie.
`BaseModel` formalise ce contrat minimal : hériter de `nn.Module` (pour
garder `parameters()`, `.to(device)`, `state_dict()`, `.train()/.eval()`,
...) et imposer `forward`.

"""

from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseModel(nn.Module, ABC):
    """Contrat commun à tous les modèles de la plateforme.

    Toute classe qui hérite de `BaseModel` DOIT implémenter `forward`.
    Python refuse d'instancier une sous-classe qui ne le fait pas, ce qui
    évite de le découvrir seulement au moment de l'entraînement.

    Hériter de `BaseModel` (plutôt que de `nn.Module` directement) permet
    aussi d'écrire du code générique qui s'appuie sur ce contrat, par
    exemple :

    ```python
    def check_model(model: BaseModel) -> None:
        assert isinstance(model, BaseModel)
    ```
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Calcule la sortie du modèle pour un batch d'entrées.

        Parameters
        ----------
        x : torch.Tensor
            Tenseur de forme `(batch, ...)`, le format exact dépend du
            modèle (nombre d'entrées, image, séquence, ...).

        Returns
        -------
        torch.Tensor
            Sortie du modèle pour ce batch. Le Trainer n'a besoin de rien
            savoir de plus : ce que fait `forward` entre les deux reste un
            détail d'implémentation du modèle.
        """
        raise NotImplementedError
