"""Les callbacks construits sont bien acceptés par un entraînement Lightning.

Un tout petit module suffit : on vérifie que le checkpoint est écrit
quand `val_loss` est journalisé, pas la qualité du modèle.
"""

from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import CSVLogger
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ia_xp_plateform.callback import CallbackConfig, create_callbacks


class _Probe(pl.LightningModule):
    """Module minimal qui journalise `val_loss`, rien de plus."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        return nn.functional.mse_loss(self(x), y)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = nn.functional.mse_loss(self(x), y)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


def test_checkpoint_is_written_during_fit(tmp_path):
    x = torch.randn(8, 2)
    y = torch.randn(8, 1)
    loader = DataLoader(TensorDataset(x, y), batch_size=4)
    callbacks = create_callbacks(
        CallbackConfig(output_dir=tmp_path, early_stopping_patience=2)
    )

    trainer = pl.Trainer(
        max_epochs=1,
        callbacks=callbacks,
        logger=CSVLogger(save_dir=tmp_path, name="logs"),
        accelerator="cpu",
        devices=1,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(_Probe(), train_dataloaders=loader, val_dataloaders=loader)

    checkpoints = list((Path(tmp_path) / "checkpoints").glob("*.ckpt"))
    assert len(checkpoints) == 1
    assert checkpoints[0].name == "best-model.ckpt"
