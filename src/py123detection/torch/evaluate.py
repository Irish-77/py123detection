"""Score a trained checkpoint with the nuScenes detection metric.

    py123det-eval experiment=nuscenes evaluation.checkpoint=work_dirs/petr_nuscenes/last.ckpt

The model predicts in the ego frame; the metric measures in the global frame. The conversion
happens once here, applied identically to predictions and ground truth, so an error in it cannot
flatter the score.

**One score per dataset.** A mixed nuScenes + Argoverse 2 + Waymo run used to collapse into a
single number, which cannot tell you which dataset regressed — the training loop has always kept
its validation losses apart for exactly that reason, and this now matches it. Set
``evaluation.combined=true`` to additionally score every frame together, which is the number older
runs reported.

**Inference holds one frame.** Each dataset's predictions stream to
``<output_dir>/<dataset>/predictions.jsonl`` as they are made, and scoring reads them back. Peak
memory is therefore one dataset's boxes rather than every dataset's at once, which is what an
earlier three-split pass died of at 67 GB. It also means a run can be scored again — different
class range, different true-positive set, different box cap — without the model:
``evaluation.rescore_from=<output_dir>`` skips inference entirely.
"""

from __future__ import annotations

import gc
import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

import hydra
import lightning as L
import numpy as np
from lightning.pytorch.callbacks import BasePredictionWriter
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from py123detection.core.evaluation import EvalBox, evaluate, format_summary
from py123detection.core.geometry import heading_yaw, transform_points
from py123detection.core.taxonomy import get_taxonomy
from py123detection.torch.build import build_datasets, build_filters
from py123detection.torch.data.collate import collate_rigs
from py123detection.torch.predictions import (
    FramePrediction,
    PredictionWriter,
    frames_for,
    merge_rank_files,
    rank_files,
)

logger = logging.getLogger(__name__)

#: Columns of a decoded box: ``[x, y, z, w, l, h, yaw, vx, vy]`` in the ego frame.
YAW = 6


def frame_token(metas: Dict[str, Any]) -> str:
    """A token identifying one frame across datasets."""
    return f"{metas['scene_uuid']}:{metas['frame_idx']}"


def to_global(boxes: np.ndarray, ego_pose: np.ndarray) -> np.ndarray:
    """Move ego-frame boxes into the global frame.

    Centres are transformed by the full pose; yaw picks up only the pose's rotation about z, and
    velocity is rotated but not translated, since it is a direction rather than a position.
    """
    if len(boxes) == 0:
        return boxes

    moved = boxes.copy()
    rotation = ego_pose[:3, :3]
    moved[:, :3] = transform_points(ego_pose, boxes[:, :3])
    moved[:, YAW] = boxes[:, YAW] + heading_yaw(rotation)
    if boxes.shape[1] > 7:
        moved[:, 7:9] = boxes[:, 7:9] @ rotation[:2, :2].T
    return moved


def as_eval_boxes(
    boxes: np.ndarray,
    names: Sequence[str],
    token: str,
    scores: np.ndarray = None,
) -> List[EvalBox]:
    """Global-frame boxes as the metric's box type.

    The per-field conversion is done for the whole array at once and handed to the loop as native
    Python floats. Boxing nine numpy scalars per box cost more than everything else in scoring put
    together once a split ran to millions of boxes; the arithmetic is unchanged.
    """
    count = len(boxes)
    if count == 0:
        return []

    half_yaw = np.asarray(boxes[:, YAW], dtype=np.float64) / 2.0
    translations = boxes[:, 0:3].tolist()
    sizes = boxes[:, 3:6].tolist()
    cosines = np.cos(half_yaw).tolist()
    sines = np.sin(half_yaw).tolist()
    velocities = boxes[:, 7:9].tolist() if boxes.shape[1] > 8 else None
    confidences = np.asarray(scores, dtype=np.float64).tolist() if scores is not None else None

    result = []
    for index in range(count):
        translation = translations[index]
        size = sizes[index]
        result.append(
            EvalBox(
                sample_token=token,
                translation=(translation[0], translation[1], translation[2]),
                size=(size[0], size[1], size[2]),
                rotation=(cosines[index], 0.0, 0.0, sines[index]),
                velocity=tuple(velocities[index]) if velocities is not None else (0.0, 0.0),
                detection_name=names[index],
                detection_score=confidences[index] if confidences is not None else -1.0,
            )
        )
    return result


def top_scoring(scores: np.ndarray, limit: Optional[int]) -> Optional[np.ndarray]:
    """Indices of the ``limit`` highest-scoring boxes, or None to keep every one.

    The devkit caps a submission at 500 boxes a sample; this branch does not, so the cap is opt-in
    via ``evaluation.max_boxes_per_sample`` and off by default — turning it on changes scores, and
    numbers already published were produced without it. Applied before the metric's box objects are
    built, so the cap saves the memory rather than spending it first.
    """
    if limit is None or len(scores) <= limit:
        return None
    return np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")[:limit]


def frame_from_prediction(
    item: Dict[str, Any],
    batch: Dict[str, Any],
    index: int,
    max_boxes_per_sample: Optional[int] = None,
) -> FramePrediction:
    """One sample of :meth:`Detector3D.predict_step`'s output, as the record written to disk.

    The model predicts in the ego frame; this is where predictions and ground truth are moved into
    the global frame together, by the same call, so an error in the conversion cannot flatter the
    score. Ground truth is taken from the batch rather than returned by ``predict_step``, which
    leaves the model's inference contract about predictions alone.
    """
    pose = item["ego_pose"].detach().cpu().numpy().astype(np.float64)
    boxes = item["boxes"].detach().cpu().numpy().astype(np.float64)
    scores = item["scores"].detach().cpu().numpy().astype(np.float64)
    labels = item["labels"].detach().cpu().numpy().astype(np.int64)

    keep = top_scoring(scores, max_boxes_per_sample)
    if keep is not None:
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

    return FramePrediction(
        token=frame_token(item),
        dataset=item.get("dataset_name") or "unknown",
        ego_translation=pose[:3, 3],
        boxes=to_global(boxes, pose),
        scores=scores,
        labels=labels,
        gt_boxes=to_global(batch["gt_bboxes_3d"][index].detach().cpu().numpy().astype(np.float64), pose),
        gt_labels=batch["gt_labels_3d"][index].detach().cpu().numpy().astype(np.int64),
    )


class FramePredictionWriter(BasePredictionWriter):
    """Write each rank's predictions to its own file as they are made.

    Lightning's own prediction return path collects every batch on every rank, which is the thing
    this stage exists to avoid: ``Trainer.predict(return_predictions=False)`` with a per-batch
    writer keeps the peak at one batch. One file per rank rather than a cross-rank gather, so a
    rank that dies leaves its share on disk instead of taking the whole run's output with it.

    Files are routed by each frame's own ``dataset`` field, not by ``dataloader_idx``, so the
    layout follows the data whether the datasets arrive as separate loaders or one concatenation.
    """

    def __init__(self, output_dir: Path, max_boxes_per_sample: Optional[int] = None) -> None:
        super().__init__(write_interval="batch")
        self.output_dir = Path(output_dir)
        self.max_boxes_per_sample = max_boxes_per_sample
        self.rank = 0
        self.frames = 0
        self._writers: Dict[str, PredictionWriter] = {}

    def on_predict_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self.rank = trainer.global_rank
        self.frames = 0
        self._writers = {}

    def _writer_for(self, dataset: str) -> PredictionWriter:
        writer = self._writers.get(dataset)
        if writer is None:
            path = self.output_dir / dataset / f"predictions_rank{self.rank}.jsonl"
            writer = PredictionWriter(path).__enter__()
            self._writers[dataset] = writer
        return writer

    def write_on_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        prediction: Any,
        batch_indices: Optional[Sequence[int]],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        for index, item in enumerate(prediction or []):
            frame = frame_from_prediction(item, batch, index, self.max_boxes_per_sample)
            self._writer_for(frame.dataset).write(frame)
            self.frames += 1

    def on_predict_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        for writer in self._writers.values():
            writer.__exit__()
        self._writers = {}


def score_frames(
    frames: Iterator[FramePrediction],
    class_names: Sequence[str],
    output_dir: Optional[Path] = None,
    class_range: Optional[Dict[str, float]] = None,
    tp_metrics: str = "nuscenes",
) -> Dict[str, Any]:
    """Score a stream of frames.

    The metric needs every box of the set at once, so this is where memory peaks — at one dataset's
    worth when called per dataset, which is the point of calling it that way.
    """
    ground_truth: Dict[str, List[EvalBox]] = {}
    predictions: Dict[str, List[EvalBox]] = {}
    ego_translations: Dict[str, np.ndarray] = {}

    for frame in frames:
        token = frame.token
        ego_translations[token] = frame.ego_translation
        predictions[token] = as_eval_boxes(
            frame.boxes,
            [class_names[label] for label in frame.labels],
            token,
            frame.scores,
        )
        ground_truth[token] = as_eval_boxes(
            frame.gt_boxes,
            [class_names[label] for label in frame.gt_labels],
            token,
        )

    logger.info("Scoring %d frames", len(ground_truth))
    return evaluate(
        ground_truth,
        predictions,
        class_names=class_names,
        ego_translations=ego_translations,
        output_dir=str(output_dir) if output_dir is not None else None,
        class_range=class_range,
        tp_metrics=tp_metrics,
        verbose=False,  # printed once by the caller
    )


def score_predictions(
    path: Path,
    class_names: Sequence[str],
    output_dir: Optional[Path] = None,
    class_range: Optional[Dict[str, float]] = None,
    tp_metrics: str = "nuscenes",
    dataset: Optional[str] = None,
) -> Dict[str, Any]:
    """Score a predictions file written by a previous run, without the model."""
    return score_frames(
        frames_for(path, dataset),
        class_names,
        output_dir=output_dir,
        class_range=class_range,
        tp_metrics=tp_metrics,
    )


def _settings(cfg: DictConfig) -> Dict[str, Any]:
    """How these numbers were produced, recorded beside them.

    ``score_empty_frames`` defaults to false, which flatters the score: a frame whose ground truth
    is empty is dropped upstream, so the false positives a model puts on an empty scene are never
    counted. That default is deliberate — it keeps already-published numbers reproducible — which
    makes it all the more important that every metrics file says which setting produced it.
    """
    return {
        "score_empty_frames": bool(cfg.evaluation.get("score_empty_frames", False)),
        "max_boxes_per_sample": cfg.evaluation.get("max_boxes_per_sample"),
        "tp_metrics": cfg.evaluation.tp_metrics,
    }


def _write(path: Path, content: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2, default=float))
    logger.info("Wrote %s", path)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    L.seed_everything(cfg.run.seed, workers=True)
    taxonomy = get_taxonomy(cfg.taxonomy.name)
    class_names = list(taxonomy.class_names)
    class_range = (
        OmegaConf.to_container(cfg.evaluation.class_range, resolve=True)
        if cfg.evaluation.get("class_range")
        else None
    )
    tp_metrics = cfg.evaluation.tp_metrics
    max_boxes = cfg.evaluation.get("max_boxes_per_sample")
    output_dir = Path(cfg.evaluation.output_dir)
    settings = _settings(cfg)

    rescore_from = cfg.evaluation.get("rescore_from")
    if rescore_from:
        names = _rescore(Path(rescore_from), output_dir, class_names, class_range, tp_metrics, settings)
        logger.info("Re-scored %d dataset(s) from %s", len(names), rescore_from)
        return

    checkpoint = cfg.evaluation.get("checkpoint")
    if not checkpoint:
        raise ValueError("Set evaluation.checkpoint=<path to a .ckpt>, or evaluation.rescore_from=<dir>.")

    model_class = hydra.utils.get_class(cfg.model._target_)
    model = model_class.load_from_checkpoint(checkpoint, map_location="cpu")
    logger.info("Loaded %s from %s", cfg.model.name, checkpoint)

    # Validation keeps the training filters, except that scoring empty frames means not dropping
    # them. Only this flag differs, so a relaxed eval still measures the same box population.
    filters = build_filters(cfg)
    if settings["score_empty_frames"]:
        filters = replace(filters, drop_frames_without_boxes=False)
        logger.info("Scoring frames with no ground truth, so empty-scene false positives count")

    datasets = build_datasets(cfg, "val", filters=filters)
    loaders = [
        DataLoader(
            dataset,
            batch_size=cfg.dataloader.batch_size,
            shuffle=False,
            num_workers=cfg.dataloader.num_workers,
            collate_fn=collate_rigs,
        )
        for dataset in datasets
    ]

    # Inference runs under a Trainer so that `evaluation.devices` shards it across GPUs, which also
    # makes Detector3D.predict_step the one inference path rather than a second copy of it here.
    #
    # Precision is evaluation's own setting and defaults to fp32. It deliberately does NOT inherit
    # cfg.trainer.precision: every shipped trainer config is 16-mixed, evaluation has always run in
    # fp32, and quietly adopting the training precision would move every score.
    writer = FramePredictionWriter(output_dir, max_boxes_per_sample=max_boxes)
    trainer = L.Trainer(
        accelerator="auto",
        devices=cfg.evaluation.get("devices", 1),
        strategy=cfg.evaluation.get("strategy", "auto"),
        precision=cfg.evaluation.get("precision", "32-true"),
        logger=False,
        enable_checkpointing=False,
        callbacks=[writer],
    )
    trainer.predict(model, dataloaders=loaders, return_predictions=False)
    # Every rank has to have finished writing before rank zero reads any of it.
    trainer.strategy.barrier("predictions-written")
    logger.info("rank %d wrote %d frames", trainer.global_rank, writer.frames)

    if not trainer.is_global_zero:
        return

    results: Dict[str, Any] = {}
    for dataset in datasets:
        name = dataset.name
        target = output_dir / name
        predictions_path = target / "predictions.jsonl"

        shards = rank_files(target)
        if not shards:
            logger.warning("%s: no predictions written, skipping", name)
            continue
        merged = merge_rank_files(shards, predictions_path)
        logger.info(
            "%s: merged %d shard(s) into %d frames (%d duplicate(s) from sampler padding)",
            name, len(shards), merged["frames"], merged["duplicates"],
        )

        summary = score_predictions(
            predictions_path, class_names, output_dir=target, class_range=class_range, tp_metrics=tp_metrics
        )
        summary["evaluation"].update(settings)
        summary["evaluation"]["dataset"] = name
        summary["evaluation"]["shards"] = len(shards)
        summary["evaluation"]["duplicate_frames_dropped"] = merged["duplicates"]
        _write(target / "metrics.json", summary)
        print(f"\n######## {name} ########")
        print(format_summary(summary))

        results[name] = _headline(summary)
        # The whole reason scoring happens per dataset: let this one go before the next begins.
        del summary
        gc.collect()

    index: Dict[str, Any] = {"evaluation": settings, "datasets": results}

    if cfg.evaluation.get("combined") and len(results) > 1:
        logger.info("Scoring every dataset together as well; this holds all of them in memory at once")
        combined = score_frames(
            _chain(output_dir / name / "predictions.jsonl" for name in results),
            class_names,
            output_dir=output_dir / "combined",
            class_range=class_range,
            tp_metrics=tp_metrics,
        )
        combined["evaluation"].update(settings)
        _write(output_dir / "combined" / "metrics.json", combined)
        print("\n######## combined ########")
        print(format_summary(combined))
        index["combined"] = _headline(combined)

    _write(output_dir / "metrics.json", index)
    print("\n" + _table(results))


def _rescore(
    source: Path,
    output_dir: Path,
    class_names: Sequence[str],
    class_range: Optional[Dict[str, float]],
    tp_metrics: str,
    settings: Dict[str, Any],
) -> List[str]:
    """Score predictions a previous run left on disk, with no model and no data."""
    found = sorted(source.glob("*/predictions.jsonl"))
    if not found:
        raise ValueError(f"No */predictions.jsonl under {source}; nothing to re-score.")

    results: Dict[str, Any] = {}
    for predictions_path in found:
        name = predictions_path.parent.name
        target = output_dir / name
        summary = score_predictions(
            predictions_path, class_names, output_dir=target, class_range=class_range, tp_metrics=tp_metrics
        )
        summary["evaluation"].update(settings)
        summary["evaluation"]["dataset"] = name
        summary["evaluation"]["rescored_from"] = str(predictions_path)
        _write(target / "metrics.json", summary)
        print(f"\n######## {name} ########")
        print(format_summary(summary))
        results[name] = _headline(summary)
        del summary
        gc.collect()

    _write(output_dir / "metrics.json", {"evaluation": settings, "datasets": results})
    print("\n" + _table(results))
    return list(results)


def _chain(paths) -> Iterator[FramePrediction]:
    for path in paths:
        yield from frames_for(path)


def _headline(summary: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "mAP": float(summary["mean_ap"]),
        "NDS": float(summary["nd_score"]),
        "frames": summary["evaluation"].get("num_frames"),
        **{name: float(value) for name, value in summary["tp_errors"].items()},
    }


def _table(results: Dict[str, Any]) -> str:
    """One row per dataset, which is the view a mixed run needs and a single number cannot give."""
    if not results:
        return "no datasets scored"
    columns = ["mAP", "NDS", "frames"]
    header = f"{'dataset':12s}" + "".join(f"{column:>10s}" for column in columns)
    rows = [header, "-" * len(header)]
    for name, values in results.items():
        cells = "".join(
            f"{values[column]:>10.4f}" if isinstance(values[column], float) else f"{values[column]:>10}"
            for column in columns
        )
        rows.append(f"{name:12s}{cells}")
    return "\n".join(rows)


if __name__ == "__main__":
    main()
