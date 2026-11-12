"""

La factory de datamodules construit un `LightningDataModule` à partir de
son nom. Le reste de la plateforme n'importe pas `TableDataModule` : il
donne un nom et les chemins de fichiers, et récupère un datamodule prêt à
être branché sur un entraîneur.

"""

import pytorch_lightning as pl

from ia_xp_plateform.data.datamodule.table_datamodule import TableDataModule

# Nom utilisé par la configuration -> classe concrète.
DATAMODULES: dict[str, type[pl.LightningDataModule]] = {
    "table": TableDataModule,
}


def create_datamodule(name: str, **kwargs) -> pl.LightningDataModule:
    """Construit le datamodule enregistré sous `name`.

    Parameters
    ----------
    name : str
        Nom du datamodule, une clé de `DATAMODULES` (par exemple
        `"table"`).
    **kwargs
        Arguments transmis tels quels au constructeur
        (`data_path`, `target_columns`, `split_path`, ...).

    Returns
    -------
    pytorch_lightning.LightningDataModule
        Une instance configurée, pas encore chargée : `setup()` lit les
        fichiers plus tard.
    """
    try:
        datamodule_cls = DATAMODULES[name]
    except KeyError as error:
        known = ", ".join(sorted(DATAMODULES))
        raise ValueError(
            f"Datamodule inconnu : '{name}'. "
            f"Datamodules disponibles : {known}."
        ) from error
    return datamodule_cls(**kwargs)
