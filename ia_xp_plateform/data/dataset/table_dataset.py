"""

Un dataset tabulaire décrit un tableau (une ligne = un exemple, une colonne
= un attribut) déjà chargé et nettoyé par un datamodule : c'est le
datamodule qui sait lire un fichier CSV, Feather, HDF5, ... et produire un
tableau propre pour l'entraînement (pas de valeurs manquantes, colonnes
numériques). `TableDataset` n'a plus qu'un rôle, commun à tous les formats
de fichier : séparer les colonnes du tableau en entrées (`x`) et cible
(`y`), puis les convertir en tenseurs PyTorch.

"""

from __future__ import annotations

import pandas as pd
import torch

from ia_xp_plateform.data.dataset.base import BaseDataset


class TableDataset(BaseDataset):
    """Dataset construit à partir d'un tableau (`pandas.DataFrame`).

    Le tableau doit déjà être prêt pour l'entraînement : c'est le rôle du
    datamodule (pas de celui-ci) de lire le fichier source (CSV, Feather,
    HDF5, ...) et de le nettoyer avant de construire ce dataset.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        target_columns: str | list[str],
        feature_columns: list[str] | None = None,
    ) -> None:
        """Sépare le tableau en entrées (`x`) et cible (`y`).

        Parameters
        ----------
        dataframe : pandas.DataFrame
            Tableau déjà chargé et nettoyé par le datamodule (une ligne =
            un exemple).
        target_columns : str | list[str]
            Nom de la colonne cible (`y`), ou liste de colonnes si la
            cible a plusieurs dimensions.
        feature_columns : list[str] | None, optional
            Colonnes à utiliser comme entrées (`x`). Par défaut (`None`),
            toutes les colonnes du tableau sauf `target_columns`.
        """
        # On uniformise `target_columns` en liste : cela permet de la
        # retirer de la liste des colonnes et de traiter une ou plusieurs
        # cibles exactement de la même façon.
        if isinstance(target_columns, str):
            target_columns = [target_columns]

        if feature_columns is None:
            # Par défaut : toutes les colonnes qui ne sont pas la cible.
            feature_columns = [
                column
                for column in dataframe.columns
                if column not in target_columns
            ]

        self.target_columns = target_columns
        self.feature_columns = feature_columns

        # `to_numpy(dtype="float32", copy=True)` convertit les colonnes
        # sélectionnées en un tableau numpy de flottants 32 bits (le type
        # attendu par PyTorch), avec sa propre copie mémoire : pandas peut
        # sinon renvoyer un tableau non modifiable, que `torch.from_numpy`
        # accepte mal.
        x = torch.from_numpy(
            dataframe[feature_columns].to_numpy(dtype="float32", copy=True)
        )
        y = torch.from_numpy(
            dataframe[target_columns].to_numpy(dtype="float32", copy=True)
        )

        super().__init__(x, y)
