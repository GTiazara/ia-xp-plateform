"""

Les callbacks sont les observateurs de l'entraînement : ils ne calculent
pas le modèle, ils réagissent à ce qui se passe pendant le fit. Pour
l'instant, trois suffisent :

- `ModelCheckpoint` garde le meilleur modèle selon `val_loss` ;
- `EarlyStopping` arrête l'entraînement si `val_loss` ne baisse plus ;
- `LearningRateMonitor` note le taux d'apprentissage à chaque époque.

"""

from pathlib import Path

from pytorch_lightning.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

# Les deux callbacks qui regardent la validation surveillent la même
# quantité, pour que « le meilleur modèle » et « on arrête » veuillent
# dire la même chose.
_MONITOR = "val_loss"


class CallbackConfig:
    """Les deux réglages dont les callbacks ont besoin.

    Parameters
    ----------
    output_dir : str | Path
        Dossier de l'expérience. Les poids sont écrits dans
        `output_dir/checkpoints`.
    early_stopping_patience : int
        Nombre d'époques de validation sans amélioration de `val_loss`
        avant d'arrêter l'entraînement. Doit être au moins 1.
    """

    def __init__(
        self,
        output_dir: str | Path,
        early_stopping_patience: int,
    ) -> None:
        if early_stopping_patience < 1:
            raise ValueError(
                "early_stopping_patience doit être au moins 1 "
                f"({early_stopping_patience})."
            )
        self.output_dir = output_dir
        self.early_stopping_patience = early_stopping_patience


def create_callbacks(config: CallbackConfig) -> list[Callback]:
    """Construit les callbacks branchés sur un entraînement.

    Parameters
    ----------
    config : CallbackConfig
        Dossier de sortie et patience de l'arrêt anticipé.

    Returns
    -------
    list[Callback]
        Checkpoint, arrêt anticipé, puis suivi du taux d'apprentissage,
        dans cet ordre.
    """
    checkpoint = ModelCheckpoint(
        dirpath=Path(config.output_dir) / "checkpoints",
        filename="best-model",
        monitor=_MONITOR,
        mode="min",
        save_top_k=1,
    )
    early_stopping = EarlyStopping(
        monitor=_MONITOR,
        patience=config.early_stopping_patience,
        mode="min",
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    return [checkpoint, early_stopping, lr_monitor]
