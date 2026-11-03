"""

Le dataset de base regroupe les entrées (`x`) et les sorties attendues
(`y`) d'un jeu de données déjà prêt pour l'entraînement : il ne fait
qu'exposer ces deux tenseurs sous la forme attendue par PyTorch
(`torch.utils.data.Dataset`), sans se soucier de la façon dont `x` et `y`
ont été produits (lecture d'un fichier tabulaire, ...). Toutes les
autres classes de dataset (`TableDataset`, ...) héritent de celle-ci.

"""

import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """Dataset minimal : un couple de tenseurs (entrées, cibles).

    Chaque sous-classe n'a qu'une seule responsabilité : construire `x` et
    `y`, puis les transmettre à `BaseDataset`. Celui-ci se charge ensuite
    de la partie commune à tout `Dataset` PyTorch : donner la taille du
    jeu de données et renvoyer un exemple à partir de son index.
    """

    def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """Initialise le dataset à partir de tenseurs déjà construits.

        Parameters
        ----------
        x : torch.Tensor
            Entrées, de forme `(n_samples, n_features)`.
        y : torch.Tensor
            Cibles, de forme `(n_samples, n_targets)`.
        """
        if len(x) != len(y):
            # x et y avancent toujours ensemble, exemple par exemple :
            # un nombre d'exemples différent est forcément une erreur.
            raise ValueError(
                "x et y doivent avoir le même nombre d'exemples "
                f"({len(x)} != {len(y)})."
            )

        self.x = x
        self.y = y

    def __len__(self) -> int:
        """Nombre d'exemples du dataset."""
        return len(self.x)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Renvoie l'exemple `(entrée, cible)` à l'index donné."""
        return self.x[index], self.y[index]
