"""Checkpoints kept beyond the best three.

``ModelCheckpoint`` keeps the best three by ``val_loss`` and rewrites ``last.ckpt`` every epoch,
so a finished 24-epoch run cannot be asked what it looked like at epoch 12. These pin that
:class:`SaveEpochCheckpoints` writes exactly the epochs it was given, under the names a config
file names them by, and that nothing it writes is later overwritten.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pytest

pytest.importorskip("lightning", reason="the torch branch needs lightning")

#: Needs the torch extra; see the `torch` marker in pyproject.toml.
pytestmark = pytest.mark.torch

from py123detection.torch.callbacks import SaveEpochCheckpoints  # noqa: E402


class FakeTrainer:
    """Just enough trainer: the epoch counter, the sanity-check flag, and a recording save."""

    def __init__(self) -> None:
        self.current_epoch = 0
        self.sanity_checking = False
        self.saved: List[Path] = []

    def save_checkpoint(self, path) -> None:
        self.saved.append(Path(path))
        Path(path).write_text("checkpoint")


def run_epochs(callback: SaveEpochCheckpoints, epochs: int) -> FakeTrainer:
    trainer = FakeTrainer()
    for epoch in range(epochs):
        trainer.current_epoch = epoch
        callback.on_validation_epoch_end(trainer, None)
    return trainer


def test_it_saves_the_epochs_it_was_given_counted_from_one(tmp_path: Path) -> None:
    callback = SaveEpochCheckpoints(tmp_path, [6, 12, 18])

    trainer = run_epochs(callback, 24)

    assert [path.name for path in trainer.saved] == ["epoch_6.ckpt", "epoch_12.ckpt", "epoch_18.ckpt"]
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "epoch_12.ckpt",
        "epoch_18.ckpt",
        "epoch_6.ckpt",
    ]


def test_the_name_is_the_epoch_a_training_log_would_show(tmp_path: Path) -> None:
    """Lightning counts from 0 during an epoch, so epoch 1 is saved at ``current_epoch == 0``."""
    callback = SaveEpochCheckpoints(tmp_path, [1])

    trainer = run_epochs(callback, 3)

    assert [path.name for path in trainer.saved] == ["epoch_1.ckpt"]


def test_the_sanity_check_saves_nothing(tmp_path: Path) -> None:
    """It runs validation before any training, so it would otherwise write untrained weights."""
    callback = SaveEpochCheckpoints(tmp_path, [1])
    trainer = FakeTrainer()
    trainer.sanity_checking = True

    callback.on_validation_epoch_end(trainer, None)

    assert trainer.saved == []


def test_an_epoch_the_run_never_reaches_is_simply_not_saved(tmp_path: Path) -> None:
    callback = SaveEpochCheckpoints(tmp_path, [2, 40])

    trainer = run_epochs(callback, 3)

    assert [path.name for path in trainer.saved] == ["epoch_2.ckpt"]


def test_it_makes_its_directory(tmp_path: Path) -> None:
    """The callback can be built before the run directory exists, which is how train.py does it."""
    target = tmp_path / "work_dirs" / "run" / "checkpoints"
    callback = SaveEpochCheckpoints(target, [1])

    run_epochs(callback, 1)

    assert (target / "epoch_1.ckpt").exists()


def test_the_epochs_may_be_given_in_any_order(tmp_path: Path) -> None:
    callback = SaveEpochCheckpoints(tmp_path, [12, 6])

    trainer = run_epochs(callback, 13)

    assert [path.name for path in trainer.saved] == ["epoch_6.ckpt", "epoch_12.ckpt"]
