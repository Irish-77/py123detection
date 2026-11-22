"""Callbacks the training loop adds beyond Lightning's own.

:class:`SaveEpochCheckpoints` exists because the stock ``ModelCheckpoint`` keeps the best three
by ``val_loss`` and overwrites ``last.ckpt`` every epoch. That is the right default for getting a
trained model out of a run, and useless for asking what the model looked like halfway through: by
the time a 24-epoch run finishes, epoch 12 is gone. Scoring an intermediate epoch is how you tell
a model that is still improving from one that peaked early and has been drifting since.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Set

import lightning as L

logger = logging.getLogger(__name__)


class SaveEpochCheckpoints(L.Callback):
    """Write a checkpoint of its own at the end of each named epoch.

    ``save_epochs=[6, 12, 18]`` leaves ``epoch_6.ckpt``, ``epoch_12.ckpt`` and ``epoch_18.ckpt``
    in ``dirpath``, next to whatever ``ModelCheckpoint`` is doing. Nothing overwrites them.

    Epochs are counted the way a training log reads them, from 1, so 12 means after twelve epochs
    of training. Lightning's own ``current_epoch`` is 0-indexed during the epoch, which is where
    the off-by-one lives; it is converted here so that a config file never has to think about it.

    :param dirpath: Directory to write into, created if it does not exist.
    :param save_epochs: 1-indexed epoch numbers. Epochs outside the run are simply never reached.
    """

    def __init__(self, dirpath: str | Path, save_epochs: Iterable[int]) -> None:
        super().__init__()
        self.dirpath = Path(dirpath)
        self._target_epochs: Set[int] = {int(epoch) - 1 for epoch in save_epochs}

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        # The sanity check runs validation before training has done anything, and would otherwise
        # write an "epoch 1" checkpoint of the initial weights.
        if trainer.sanity_checking or trainer.current_epoch not in self._target_epochs:
            return

        self.dirpath.mkdir(parents=True, exist_ok=True)
        path = self.dirpath / f"epoch_{trainer.current_epoch + 1}.ckpt"
        trainer.save_checkpoint(path)
        logger.info("Saved epoch %d to %s", trainer.current_epoch + 1, path)
