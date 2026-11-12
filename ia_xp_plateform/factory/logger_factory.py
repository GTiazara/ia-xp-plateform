"""

La factory de loggers construit un logger Lightning à partir de son nom.
L'entraîneur ne sait pas si les courbes partent vers LitLogger, TensorBoard
ou ClearML : il reçoit un logger et y envoie les métriques déjà journalisées
par `ClassificationModule` (`train_loss`, `val_loss`, `val_acc`, ...).

Deux noms sont enregistrés : `"tensorboard"` (le défaut, local, pour revoir
les expériences en cours ou passées) et `"litlogger"` (lightning.ai). En
ajouter un autre, c'est une ligne dans `LOGGERS` et un fichier
`conf/logger/<nom>.yaml`.
"""

from pytorch_lightning.loggers import LitLogger, Logger, TensorBoardLogger

# Nom utilisé par la configuration -> classe concrète.
LOGGERS: dict[str, type[Logger]] = {
    "tensorboard": TensorBoardLogger,
    "litlogger": LitLogger,
}

# Les loggers ne nomment pas leur dossier de la même façon : TensorBoard
# attend `save_dir`, LitLogger attend `root_dir`.
LOGGER_DIR_KEYS: dict[str, str] = {
    "tensorboard": "save_dir",
    "litlogger": "root_dir",
}


def create_logger(logger_name: str, **kwargs) -> Logger:
    """Construit le logger enregistré sous `logger_name`.

    Parameters
    ----------
    logger_name : str
        Nom du logger, une clé de `LOGGERS` (`"tensorboard"` ou
        `"litlogger"`).
        Ce n'est pas le nom de l'expérience : celui-ci se passe dans
        `kwargs` sous la clef `name`, comme l'attendent les deux loggers.
    **kwargs
        Arguments transmis tels quels au constructeur du logger
        (`save_dir` ou `root_dir`, `name` de l'expérience, ...).

    Returns
    -------
    Logger
        Un logger prêt à être passé au `Trainer`.
    """
    try:
        logger_cls = LOGGERS[logger_name]
    except KeyError as error:
        known = ", ".join(sorted(LOGGERS))
        raise ValueError(
            f"Logger inconnu : '{logger_name}'. Loggers disponibles : {known}."
        ) from error
    return logger_cls(**kwargs)
