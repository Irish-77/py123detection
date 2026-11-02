"""Train a camera-based 3D detector.

    py123det-train experiment=nuscenes
    py123det-train experiment=nuscenes_av2 model=bevformer taxonomy=mixed2_3cls
    py123det-train experiment=nuscenes trainer=smoke paths.data_root=/data/123d

Every option comes from ``conf/``; see ``conf/config.yaml`` for what the groups are.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import hydra
import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from omegaconf import DictConfig, OmegaConf

from py123detection.core.taxonomy import get_taxonomy
from py123detection.torch.build import build_datasets, build_model
from py123detection.torch.data import Py123DDataModule

logger = logging.getLogger(__name__)


def build_data(cfg: DictConfig) -> Py123DDataModule:
    """Training data concatenated and shuffled, validation data kept one loader per dataset.

    Keeping validation apart is what makes a mixed run diagnosable: one averaged number over three
    datasets cannot tell you which of them regressed.
    """
    return Py123DDataModule(
        train=build_datasets(cfg, "train"),
        val=build_datasets(cfg, "val"),
        batch_size=cfg.dataloader.batch_size,
        num_workers=cfg.dataloader.num_workers,
    )


def build_callbacks(cfg: DictConfig) -> List[L.Callback]:
    checkpoints = Path(cfg.paths.work_dir) / cfg.run.name / "checkpoints"
    return [
        ModelCheckpoint(
            dirpath=checkpoints,
            filename="{epoch:02d}-{val_loss:.3f}",
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]


def build_logger(cfg: DictConfig):
    if not cfg.run.get("wandb"):
        from lightning.pytorch.loggers import CSVLogger

        return CSVLogger(save_dir=cfg.paths.work_dir, name=cfg.run.name)

    from lightning.pytorch.loggers import WandbLogger

    return WandbLogger(project=cfg.run.project, name=cfg.run.name, save_dir=cfg.paths.work_dir)


def resume_from(cfg: DictConfig) -> str | None:
    """The checkpoint to continue from, or None to start fresh.

    ``run.resume="last"`` resolves to this run's own ``last.ckpt`` *if it exists*: a requeued job
    reruns the same command, and the first attempt has no checkpoint to find.
    """
    resume = cfg.run.get("resume")
    if not resume:
        return None
    if resume != "last":
        return str(resume)

    last = Path(cfg.paths.work_dir) / cfg.run.name / "checkpoints" / "last.ckpt"
    if last.exists():
        logger.info("Resuming from %s", last)
        return str(last)
    logger.info("No checkpoint at %s; starting from scratch", last)
    return None


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    L.seed_everything(cfg.run.seed, workers=True)
    torch.set_float32_matmul_precision("high")

    taxonomy = get_taxonomy(cfg.taxonomy.name)
    model = build_model(
        cfg,
        num_classes=len(taxonomy.class_names),
        image_size=tuple(cfg.images.size),
        max_epochs=cfg.trainer.max_epochs,
        warmup_steps=cfg.trainer.warmup_steps,
    )
    logger.info(
        "%s with %s: %.1fM parameters",
        cfg.model.name,
        cfg.taxonomy.name,
        sum(p.numel() for p in model.parameters()) / 1e6,
    )

    trainer = L.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        precision=cfg.trainer.precision,
        devices=cfg.trainer.devices,
        strategy=cfg.trainer.get("strategy", "auto"),
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        limit_train_batches=cfg.trainer.limit_train_batches,
        limit_val_batches=cfg.trainer.limit_val_batches,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        default_root_dir=cfg.paths.work_dir,
        callbacks=build_callbacks(cfg),
        logger=build_logger(cfg),
    )
    trainer.fit(model, datamodule=build_data(cfg), ckpt_path=resume_from(cfg))


if __name__ == "__main__":
    main()
