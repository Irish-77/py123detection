"""Feeding one or several 123D datasets to a Lightning trainer.

Training concatenates the datasets and shuffles across all of them, so a batch may mix nuScenes
and Argoverse frames. Validation keeps them apart — one loader each — because a single averaged
number over several datasets hides which one regressed.

The mix is set by how many frames each dataset serves rather than by a sampler: a dataset with
10000 frames next to one with 1000 is drawn from ten times as often. That is the whole mechanism,
and it is worth saying plainly. Set ``num_frames`` per dataset to choose the mix directly, or let
``sample_rate_hz`` and the length of the logs decide it.

Each worker reseeds its copy of every dataset's augmentation generator from the worker's own seed
and the trainer's rank, so workers and ranks draw different augmentations, every epoch draws new
ones, and the same run repeated draws the same ones.
"""

from __future__ import annotations

import functools
import logging
import multiprocessing
from typing import Callable, Iterator, List, Optional, Sequence

import lightning as L
from torch.utils.data import ConcatDataset, DataLoader, Dataset, get_worker_info

from py123detection.torch.data.collate import collate_rigs
from py123detection.torch.data.dataset import Py123DDetectionDataset

logger = logging.getLogger(__name__)


def _leaf_datasets(dataset: Dataset) -> Iterator[Dataset]:
    if isinstance(dataset, ConcatDataset):
        for part in dataset.datasets:
            yield from _leaf_datasets(part)
    else:
        yield dataset


def _init_worker(_worker_id: int, rank: int = 0) -> None:
    """Keep each dataloader worker single-threaded, and give it its own augmentation stream.

    OpenCV and torch both default to using every core, so N workers each spawning N threads
    oversubscribes the machine badly enough to be slower than one worker.

    The worker's seed comes from the main process's generator, which ``seed_everything`` seeds,
    and differs per worker; the rank separates the ranks, whose main processes are seeded alike.
    """
    import cv2
    import torch

    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
    torch.set_num_threads(1)

    info = get_worker_info()
    if info is not None:
        for dataset in _leaf_datasets(info.dataset):
            if hasattr(dataset, "reseed"):
                dataset.reseed(info.seed, rank)


class Py123DDataModule(L.LightningDataModule):
    """Train on one or more 123D datasets, validate on each separately.

    :param train: Datasets to concatenate and shuffle for training.
    :param val: Datasets to validate on, each getting its own dataloader in the order given.
    :param collate_fn: Defaults to :func:`~py123detection.torch.data.collate.collate_rigs`, which
        handles rigs of different sizes.
    """

    def __init__(
        self,
        train: Sequence[Py123DDetectionDataset] = (),
        val: Sequence[Py123DDetectionDataset] = (),
        batch_size: int = 1,
        num_workers: int = 0,
        prefetch_factor: int = 4,
        collate_fn: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.train = list(train)
        self.val = list(val)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.collate_fn = collate_fn or collate_rigs

        self._check_taxonomies()
        if self.train:
            logger.info(
                "Training on %d dataset(s), %s frames in total: %s",
                len(self.train),
                sum(len(dataset) for dataset in self.train),
                ", ".join(f"{dataset.name} ({len(dataset)})" for dataset in self.train),
            )

    def _check_taxonomies(self) -> None:
        """Every dataset has to agree on what the classes are.

        Class ids are positions, so two datasets with different class lists would train the same
        output channel on different objects. The legacy code checked only the first dataset,
        which is exactly the one that cannot be wrong.
        """
        datasets = self.train + self.val
        if not datasets:
            return

        reference = datasets[0]
        for dataset in datasets[1:]:
            if dataset.taxonomy.class_names != reference.taxonomy.class_names:
                raise ValueError(
                    f"Dataset '{dataset.name}' uses taxonomy '{dataset.taxonomy.name}' with classes "
                    f"{list(dataset.taxonomy.class_names)}, but '{reference.name}' uses "
                    f"'{reference.taxonomy.name}' with {list(reference.taxonomy.class_names)}. "
                    f"Mixed training needs one class list."
                )

    @property
    def rank(self) -> int:
        trainer = getattr(self, "trainer", None)
        return int(trainer.global_rank) if trainer is not None else 0

    def setup(self, stage: Optional[str] = None) -> None:
        """Separate the ranks' augmentation streams when the main process loads the data itself."""
        for dataset in self.train:
            if hasattr(dataset, "reseed") and getattr(dataset, "seed", None) is not None:
                dataset.reseed(dataset.seed, self.rank)

    @property
    def class_names(self) -> List[str]:
        datasets = self.train + self.val
        return list(datasets[0].taxonomy.class_names) if datasets else []

    def _loader(self, dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
            drop_last=shuffle,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            worker_init_fn=functools.partial(_init_worker, rank=self.rank),
            multiprocessing_context=multiprocessing.get_context("spawn") if self.num_workers > 0 else None,
        )

    def train_dataloader(self) -> DataLoader:
        if not self.train:
            raise ValueError("No training datasets were given.")
        combined = self.train[0] if len(self.train) == 1 else ConcatDataset(self.train)
        return self._loader(combined, shuffle=True)

    def val_dataloader(self) -> List[DataLoader]:
        return [self._loader(dataset, shuffle=False) for dataset in self.val]
