"""

Le datamodule tabulaire est le seul endroit de la plateforme qui sait lire
un fichier de données (CSV, Feather, HDF5) et le préparer pour
l'entraînement. Les données restent dans un seul fichier (`data.csv`,
`data.feather` ou `data.h5`). Le découpage train/val/test se fait de deux
façons :

- le datamodule découpe lui-même ce fichier (proportions configurables) ;
- un fichier compagnon, au même format et dans le même dossier, indique
  pour chaque ligne à quel split elle appartient : `split.csv` à côté de
  `data.csv`, `split.feather` à côté de `data.feather`, `split.h5` à côté
  de `data.h5`.

Chaque ligne du fichier de données porte un identifiant unique (colonne
`id_column`, `"id"` par défaut). Ce n'est pas une colonne d'entrée : il
sert à reconnaître l'exemple. Le fichier `split`, quand il est fourni,
n'est pas une copie des données et n'a pas besoin d'être dans le même
ordre : il associe cet identifiant à un split (`train`, `val` ou `test`).

Dans les deux cas, le datamodule construit ensuite un `TableDataset` par
split, en lui transmettant la colonne cible (`y`) et les colonnes
attributs (`x`). Le reste de la plateforme (modèle, Trainer) n'a jamais
besoin de savoir d'où viennent les données ni dans quel format elles
étaient stockées.

"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from ia_xp_plateform.data.dataset.table_dataset import TableDataset

# Valeurs attendues dans la colonne `split` du fichier compagnon.
_SPLIT_NAMES = ("train", "val", "test")


def _read_table(path: str | Path, hdf_key: str | None = None) -> pd.DataFrame:
    """Lit un fichier tabulaire, quel que soit son format.

    Le format est déduit de l'extension du fichier : `.csv`, `.feather`,
    ou `.h5` / `.hdf5`.

    Parameters
    ----------
    path : str | Path
        Chemin du fichier à lire.
    hdf_key : str | None, optional
        Clé de la table à lire, pour un fichier HDF5 qui en contient
        plusieurs. Ignoré pour les autres formats.

    Returns
    -------
    pandas.DataFrame
        Le tableau lu tel quel (aucun nettoyage n'est fait ici : le
        fichier est supposé déjà prêt pour l'entraînement).
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".feather":
        return pd.read_feather(path)
    if suffix in (".h5", ".hdf5"):
        return pd.read_hdf(path, key=hdf_key)

    raise ValueError(
        f"Format de fichier non supporté : '{suffix}' "
        "(attendu : .csv, .feather, .h5 ou .hdf5)."
    )


def _require_unique_id(dataframe: pd.DataFrame, id_column: str) -> None:
    """Vérifie que chaque ligne porte un identifiant unique.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Tableau de données en entrée.
    id_column : str
        Nom de la colonne identifiant. Elle est obligatoire.
    """
    if id_column not in dataframe.columns:
        raise ValueError(
            f"Le fichier de données doit contenir une colonne '{id_column}'."
        )
    if dataframe[id_column].duplicated().any():
        raise ValueError(
            f"La colonne '{id_column}' du fichier de données contient des "
            "doublons : elle doit identifier chaque ligne de façon unique."
        )


def _split_dataframe(
    dataframe: pd.DataFrame,
    val_split: float,
    test_split: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Découpe un tableau en trois parts : train, val, test.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Tableau complet à découper.
    val_split : float
        Proportion des lignes à mettre dans le jeu de validation.
    test_split : float
        Proportion des lignes à mettre dans le jeu de test.
    seed : int
        Graine du mélange aléatoire, pour un découpage reproductible d'un
        lancement à l'autre.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]
        Les tableaux `(train, val, test)`. Les lignes restantes, une fois
        `val_split` et `test_split` retirés, vont à l'entraînement.
    """
    n_rows = len(dataframe)

    # On mélange les indices (pas les lignes directement) pour pouvoir
    # ensuite les découper en trois tranches contiguës.
    generator = torch.Generator().manual_seed(seed)
    shuffled_index = torch.randperm(n_rows, generator=generator).tolist()

    n_test = int(n_rows * test_split)
    n_val = int(n_rows * val_split)

    train_index = shuffled_index[: n_rows - n_val - n_test]
    val_index = shuffled_index[n_rows - n_val - n_test : n_rows - n_test]
    test_index = shuffled_index[n_rows - n_test :]

    return (
        dataframe.iloc[train_index].reset_index(drop=True),
        dataframe.iloc[val_index].reset_index(drop=True),
        dataframe.iloc[test_index].reset_index(drop=True),
    )


def _split_from_file(
    dataframe: pd.DataFrame,
    split_frame: pd.DataFrame,
    id_column: str,
    split_column: str = "split",
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    """Découpe `dataframe` selon le fichier compagnon `split`.

    Le rapprochement se fait par `id_column`, présente dans les deux
    tableaux : `split_frame` n'a pas besoin d'être dans le même ordre que
    `dataframe`, ni de contenir les colonnes de données.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Tableau de données, avec une colonne `id_column` qui identifie
        chaque ligne de façon unique.
    split_frame : pandas.DataFrame
        Tableau compagnon avec une colonne `id_column` et une colonne
        `split_column` (`train`, `val` ou `test`).
    id_column : str
        Nom de la colonne identifiant commune aux deux tableaux.
    split_column : str, optional
        Nom de la colonne de `split_frame` qui contient le split.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame | None, pandas.DataFrame | None]
        Les tableaux `(train, val, test)`. `val` et `test` valent `None`
        si le fichier compagnon ne contient aucune ligne de ce split.
    """
    _require_unique_id(dataframe, id_column)
    if id_column not in split_frame.columns:
        raise ValueError(
            f"Le fichier split doit contenir une colonne '{id_column}'."
        )
    if split_column not in split_frame.columns:
        raise ValueError(
            f"Le fichier split doit contenir une colonne '{split_column}'."
        )
    if split_frame[id_column].duplicated().any():
        raise ValueError(
            f"La colonne '{id_column}' du fichier split contient des "
            "doublons : elle doit identifier chaque ligne de façon unique."
        )

    # On associe à chaque identifiant son split, puis on retrouve le split
    # de chaque ligne de `dataframe` par cet identifiant (et non par sa
    # position), avant de vérifier que le rapprochement est complet.
    split_by_id = split_frame.set_index(id_column)[split_column].astype(str)
    labels = dataframe[id_column].map(split_by_id)

    missing_ids = dataframe.loc[labels.isna(), id_column].tolist()
    if missing_ids:
        raise ValueError(
            "Des identifiants du fichier de données sont absents du "
            f"fichier split : {missing_ids[:5]}"
            + (" ..." if len(missing_ids) > 5 else "")
        )

    extra_ids = split_by_id.index[~split_by_id.index.isin(dataframe[id_column])]
    if len(extra_ids):
        raise ValueError(
            "Des identifiants du fichier split sont absents du fichier de "
            f"données : {extra_ids.tolist()[:5]}"
            + (" ..." if len(extra_ids) > 5 else "")
        )

    unknown = sorted(set(labels) - set(_SPLIT_NAMES))
    if unknown:
        raise ValueError(
            "Valeurs de split inconnues : "
            f"{unknown} (attendu : {', '.join(_SPLIT_NAMES)})."
        )

    parts: dict[str, pd.DataFrame | None] = {}
    for name in _SPLIT_NAMES:
        part = dataframe.loc[labels.to_numpy() == name].reset_index(drop=True)
        parts[name] = part if len(part) else None

    if parts["train"] is None:
        raise ValueError(
            "Le fichier split ne contient aucune ligne 'train'."
        )

    return parts["train"], parts["val"], parts["test"]


class TableDataModule(pl.LightningDataModule):
    """Datamodule Lightning pour un jeu de données tabulaire.

    Les données sont toujours dans un seul fichier. Deux façons de
    découper :

    1. Le datamodule s'en charge :
       `TableDataModule(data_path="data.csv", target_columns="y")`
    2. Un fichier compagnon `split.csv` (même dossier, même format),
       rapproché de `data.csv` par une colonne d'identifiant :
       `TableDataModule(data_path="data.csv", split_path="split.csv",
       target_columns="y")`
    """

    def __init__(
        self,
        data_path: str | Path,
        target_columns: str | list[str],
        feature_columns: list[str] | None = None,
        split_path: str | Path | None = None,
        id_column: str = "id",
        split_column: str = "split",
        val_split: float = 0.2,
        test_split: float = 0.1,
        split_seed: int = 42,
        hdf_key: str | None = None,
        split_hdf_key: str | None = None,
        batch_size: int = 32,
        num_workers: int = 0,
    ) -> None:
        """Configure le datamodule, sans encore rien lire sur le disque.

        Parameters
        ----------
        data_path : str | Path
            Fichier de données unique (`data.csv`, `data.feather` ou
            `data.h5`). Il n'est jamais dupliqué par split.
        target_columns : str | list[str]
            Colonne(s) cible(s) (`y`), transmise(s) telle quelle à chaque
            `TableDataset`.
        feature_columns : list[str] | None, optional
            Colonnes attributs (`x`). Par défaut (`None`), toutes les
            colonnes du jeu d'entraînement sauf `target_columns` et
            `id_column` ; les mêmes colonnes sont ensuite réutilisées pour
            val/test.
        split_path : str | Path | None, optional
            Fichier compagnon qui indique le split de chaque ligne,
            rapprochée de `data_path` par `id_column` (pas par l'ordre des
            lignes). S'il est absent, le datamodule découpe `data_path`
            lui-même.
        id_column : str, optional
            Nom de la colonne identifiant. Obligatoire dans `data_path`,
            et dans `split_path` quand il est fourni. Par défaut `"id"`.
            Exclue des colonnes attributs par défaut.
        split_column : str, optional
            Nom de la colonne du fichier compagnon qui contient `train`,
            `val` ou `test`. Par défaut `"split"`.
        val_split : float, optional
            Proportion du jeu de validation, utilisée seulement si
            `split_path` n'est pas fourni. Par défaut 0.2.
        test_split : float, optional
            Proportion du jeu de test, utilisée seulement si `split_path`
            n'est pas fourni. Par défaut 0.1.
        split_seed : int, optional
            Graine du découpage aléatoire, utilisée seulement si
            `split_path` n'est pas fourni. Par défaut 42.
        hdf_key : str | None, optional
            Clé à lire dans `data_path` quand c'est un fichier HDF5.
        split_hdf_key : str | None, optional
            Clé à lire dans `split_path` quand c'est un fichier HDF5. Par
            défaut, la même que `hdf_key`.
        batch_size : int, optional
            Taille des batches des `DataLoader`. Par défaut 32.
        num_workers : int, optional
            Nombre de processus pour charger les données en parallèle.
            Par défaut 0 (chargement dans le processus principal).
        """
        super().__init__()

        self.data_path = data_path
        self.target_columns = target_columns
        self.feature_columns = feature_columns
        self.split_path = split_path
        self.id_column = id_column
        self.split_column = split_column
        self.val_split = val_split
        self.test_split = test_split
        self.split_seed = split_seed
        self.hdf_key = hdf_key
        self.split_hdf_key = hdf_key if split_hdf_key is None else split_hdf_key
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset: TableDataset | None = None
        self.val_dataset: TableDataset | None = None
        self.test_dataset: TableDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        """Charge le fichier de données et construit un `TableDataset` par split.

        Si `split_path` est fourni, chaque ligne de `data` est rangée
        selon le split associé à son identifiant (`id_column`) dans le
        fichier compagnon. Sinon, le datamodule découpe le fichier
        lui-même.

        Lightning appelle `setup` avant l'entraînement (et de nouveau
        avant le test) : la relancer plusieurs fois reconstruit simplement
        les mêmes datasets, sans effet de bord.
        """
        full_dataframe = _read_table(self.data_path, hdf_key=self.hdf_key)
        _require_unique_id(full_dataframe, self.id_column)

        if self.split_path is not None:
            split_frame = _read_table(
                self.split_path, hdf_key=self.split_hdf_key
            )
            train_df, val_df, test_df = _split_from_file(
                full_dataframe,
                split_frame,
                id_column=self.id_column,
                split_column=self.split_column,
            )
        else:
            train_df, val_df, test_df = _split_dataframe(
                full_dataframe,
                val_split=self.val_split,
                test_split=self.test_split,
                seed=self.split_seed,
            )

        # Les colonnes attributs sont déduites une seule fois, à partir du
        # jeu d'entraînement, puis réutilisées pour val/test : cela
        # garantit que les trois splits utilisent exactement les mêmes
        # colonnes, dans le même ordre. `id_column` n'est jamais une
        # colonne attribut par défaut : c'est un identifiant, pas une
        # donnée d'entraînement.
        feature_columns = self.feature_columns
        if feature_columns is None:
            target_columns = (
                [self.target_columns]
                if isinstance(self.target_columns, str)
                else self.target_columns
            )
            excluded_columns = set(target_columns) | {self.id_column}
            feature_columns = [
                column
                for column in train_df.columns
                if column not in excluded_columns
            ]

        self.train_dataset = TableDataset(
            train_df, self.target_columns, feature_columns
        )
        self.val_dataset = (
            TableDataset(val_df, self.target_columns, feature_columns)
            if val_df is not None
            else None
        )
        self.test_dataset = (
            TableDataset(test_df, self.target_columns, feature_columns)
            if test_df is not None
            else None
        )

    def train_dataloader(self) -> DataLoader:
        """`DataLoader` d'entraînement (mélangé à chaque époque)."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader | None:
        """`DataLoader` de validation (non mélangé), ou `None` si absent."""
        if self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader | None:
        """`DataLoader` de test (non mélangé), ou `None` si absent."""
        if self.test_dataset is None:
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
