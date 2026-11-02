"""The training machinery every detector in this package shares.

:class:`Detector3D` owns the Lightning loop, the optimiser and the logging; a subclass supplies a
backbone and a head and says how one batch becomes predictions. PETR and BEVFormer are then
siblings rather than one inheriting from the other, which is what stops a change to PETR silently
changing BEVFormer — in the thesis they were the same class and it did.

The one piece of real machinery here is **per-dataset loss**. Training on nuScenes and Argoverse
together means a batch holds frames from both, and averaging their losses into one number hides
which dataset is failing. The loss is instead computed once per dataset in the batch and weighted
by that dataset's share, which gives the same gradient and a readable ``train_loss_<dataset>``
for each.

Distributed training needs one more thing. The loss normalises by the number of ground-truth boxes
across all ranks, which is an all-reduce. Doing that inside each per-dataset group would have rank
0 reducing over three groups while rank 1 reduces over two, and the job hangs. :meth:`sync_box_count`
does it once, before the split.
"""

from __future__ import annotations

import functools
import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import lightning as L
import torch
from torch import nn

logger = logging.getLogger(__name__)


def petr_schedule(
    step: int,
    steps_per_epoch: int,
    max_epochs: int,
    warmup_steps: int = 500,
    warmup_ratio: float = 1.0 / 3.0,
    min_lr_ratio: float = 1e-3,
) -> float:
    """Learning-rate factor at one optimiser step: PETR's and BEVFormer's published schedule.

    mmcv's ``CosineAnnealing`` with ``by_epoch=True``: the rate is constant within an epoch and
    follows a cosine from 1 to ``min_lr_ratio`` across epochs. The first ``warmup_steps`` steps
    ramp it linearly up from ``warmup_ratio`` of that. The thesis trained with exactly this.
    """
    epoch = step // max(1, steps_per_epoch)
    progress = min(max(epoch / max(1, max_epochs), 0.0), 1.0)
    factor = min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))
    if step < warmup_steps:
        factor *= warmup_ratio + (1.0 - warmup_ratio) * step / max(1, warmup_steps)
    return factor


class Detector3D(L.LightningModule):
    """Base for camera-based 3D detectors.

    A subclass sets ``self.backbone`` and ``self.head`` and implements :meth:`forward`. The head
    must return ``{"all_cls_scores": (L, B, Q, C), "all_bbox_preds": (L, B, Q, 10)}`` and provide
    ``loss(boxes, labels, predictions, ...)`` and ``get_bboxes(predictions)``.

    :param learning_rate: Base rate for the head. The backbone gets a tenth of it, which is the
        usual treatment for a pretrained ImageNet trunk under a randomly initialised head.
    :param weight_decay: Applied to weights only, never to norms or biases.
    :param warmup_steps: Optimiser steps of linear warmup from a third of the rate, before the
        per-epoch cosine. See :func:`petr_schedule`.
    """

    def __init__(
        self,
        learning_rate: float = 2e-4,
        weight_decay: float = 0.01,
        max_epochs: int = 24,
        warmup_steps: int = 500,
        backbone_lr_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.warmup_steps = warmup_steps
        self.backbone_lr_scale = backbone_lr_scale
        self.backbone: Optional[nn.Module] = None
        self.head: Optional[nn.Module] = None
        self._validation_losses: List[Tuple[torch.Tensor, int]] = []

    # -- what a subclass provides ---------------------------------------------------------------

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    # -- the loop -------------------------------------------------------------------------------

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._step(batch, stage="train", sync=True)

    def validation_step(self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        loss = self._step(batch, stage="val", sync=False)
        self._validation_losses.append((loss.detach(), len(batch["gt_labels_3d"])))
        return loss

    def on_validation_epoch_start(self) -> None:
        self._validation_losses = []

    def on_validation_epoch_end(self) -> None:
        """One ``val_loss`` over every validation dataset.

        Validation runs one dataloader per dataset, and a key logged from several of them is a key
        Lightning refuses to reconcile. Accumulating here gives a single frame-weighted number for
        the checkpoint monitor to watch, while the per-dataset keys keep saying which dataset a
        number came from.
        """
        if not self._validation_losses:
            return

        frames = sum(count for _, count in self._validation_losses)
        total = sum(loss * count for loss, count in self._validation_losses) / max(1, frames)
        self.log("val_loss", total, prog_bar=True, sync_dist=True, batch_size=frames)
        self._validation_losses = []

    def _step(self, batch: Dict[str, Any], stage: str, sync: bool) -> torch.Tensor:
        predictions = self(batch)

        # The official PETR computes the loss in fp32 even under mixed precision: the Hungarian
        # matcher's costs are small differences of similar numbers and fp16 loses them.
        with torch.autocast(device_type=self.device.type, enabled=False):
            predictions = {
                key: value.float() if torch.is_tensor(value) else value
                for key, value in predictions.items()
            }
            return self._loss(batch, predictions, stage=stage, sync=sync)

    def _loss(self, batch: Dict[str, Any], predictions: Dict[str, torch.Tensor], stage: str, sync: bool) -> torch.Tensor:
        boxes, labels = batch["gt_bboxes_3d"], batch["gt_labels_3d"]
        batch_size = len(labels)
        box_count = self.sync_box_count(labels, self.device)

        total = torch.zeros((), device=self.device)
        for name, indices in self.group_by_dataset(batch).items():
            grouped = {
                key: value[:, indices] if torch.is_tensor(value) and value.dim() > 1 else value
                for key, value in predictions.items()
            }
            terms = self.head.loss(
                [boxes[index] for index in indices],
                [labels[index] for index in indices],
                grouped,
                sync=sync,
                num_boxes_synced=box_count,
            )
            grouped_loss = sum(terms.values())
            total = total + grouped_loss * (len(indices) / batch_size)

            # Named groups get their own keys; an unnamed one would collide with the aggregate
            # below, and would say nothing the aggregate does not.
            #
            # add_dataloader_idx=False matters for validation, which runs one loader per dataset:
            # Lightning would otherwise suffix every key with the loader index, and no plain
            # "val_loss" would exist for the checkpoint monitor to watch.
            if name is not None:
                self.log(
                    f"{stage}_loss_{name}", grouped_loss,
                    batch_size=len(indices), add_dataloader_idx=False,
                )
                for term, value in terms.items():
                    self.log(
                        f"{stage}_{term}_{name}", value,
                        batch_size=len(indices), add_dataloader_idx=False,
                    )

        if stage == "train":
            self.log("train_loss", total, prog_bar=True, sync_dist=True, batch_size=batch_size)
        return total

    @staticmethod
    def group_by_dataset(batch: Dict[str, Any]) -> Dict[Optional[str], List[int]]:
        """Indices of the batch, grouped by which dataset each sample came from.

        A batch from a single dataset produces one group keyed ``None``, which costs nothing and
        keeps the code path the same either way.
        """
        metas = batch.get("img_metas")
        if not isinstance(metas, list) or not metas or not isinstance(metas[0], dict):
            return {None: list(range(len(batch["gt_labels_3d"])))}

        names = [meta.get("dataset_name") for meta in metas]
        if len(set(names)) <= 1:
            return {names[0] if names else None: list(range(len(names)))}

        groups: Dict[Optional[str], List[int]] = {}
        for index, name in enumerate(names):
            groups.setdefault(name, []).append(index)
        return groups

    @staticmethod
    def sync_box_count(labels: Sequence[torch.Tensor], device: torch.device) -> torch.Tensor:
        """Ground-truth box count across all ranks, reduced exactly once.

        The loss divides by this, so every rank must agree. Reducing inside the per-dataset loop
        instead would have ranks calling all_reduce a different number of times whenever their
        batches hold different datasets, and the job would hang rather than fail.
        """
        count = torch.as_tensor(
            float(sum(len(item) for item in labels)), dtype=torch.float, device=device
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(count)
            count = count / torch.distributed.get_world_size()
        return torch.clamp(count, min=1.0)

    # -- optimisation ---------------------------------------------------------------------------

    def parameter_groups(self) -> List[Dict[str, Any]]:
        """Four groups: backbone and head, each split into decayed and not.

        Norm and bias parameters are excluded from weight decay because shrinking them toward zero
        changes what the layer computes rather than regularising it.
        """
        groups: Dict[str, List[nn.Parameter]] = {
            "backbone_decay": [], "backbone_no_decay": [], "head_decay": [], "head_no_decay": [],
        }
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            where = "backbone" if name.startswith("backbone.") else "head"
            decay = "no_decay" if parameter.ndim <= 1 else "decay"
            groups[f"{where}_{decay}"].append(parameter)

        return [
            {
                "params": parameters,
                "lr": self.learning_rate * (self.backbone_lr_scale if key.startswith("backbone") else 1.0),
                "weight_decay": 0.0 if key.endswith("no_decay") else self.weight_decay,
                "name": key,
            }
            for key, parameters in groups.items()
            if parameters
        ]

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.AdamW(self.parameter_groups(), lr=self.learning_rate)
        steps_per_epoch = max(1, self.trainer.estimated_stepping_batches // max(1, self.max_epochs))
        schedule = functools.partial(
            petr_schedule,
            steps_per_epoch=steps_per_epoch,
            max_epochs=self.max_epochs,
            warmup_steps=self.warmup_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, schedule),
                "interval": "step",
            },
        }

    # -- inference ------------------------------------------------------------------------------

    def predict_step(self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0) -> List[Dict[str, Any]]:
        """Decoded detections for one batch, ready for
        :mod:`py123detection.core.evaluation`."""
        predictions = self(batch)
        decoded = self.head.get_bboxes(predictions)

        results = []
        for index, (boxes, scores) in enumerate(decoded):
            confidence, labels = scores.max(dim=-1)
            metas = batch["img_metas"][index] if isinstance(batch.get("img_metas"), list) else {}
            results.append(
                {
                    "boxes": boxes.detach().cpu(),
                    "scores": confidence.detach().cpu(),
                    "labels": labels.detach().cpu(),
                    "scene_uuid": metas.get("scene_uuid"),
                    "frame_idx": metas.get("frame_idx"),
                    "dataset_name": metas.get("dataset_name"),
                    "ego_pose": batch["ego_pose"][index].detach().cpu(),
                }
            )
        return results
