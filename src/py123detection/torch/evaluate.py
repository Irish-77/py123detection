"""Score a trained checkpoint with the nuScenes detection metric.

    py123det-eval experiment=nuscenes evaluation.checkpoint=work_dirs/petr_nuscenes/last.ckpt

The model predicts in the ego frame; the metric measures in the global frame. The conversion
happens once here, applied identically to predictions and ground truth, so an error in it cannot
flatter the score.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence

import hydra
import lightning as L
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import ConcatDataset, DataLoader

from py123detection.core.evaluation import EvalBox, evaluate, format_summary
from py123detection.core.taxonomy import get_taxonomy
from py123detection.torch.build import build_datasets
from py123detection.torch.data.collate import collate_rigs

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
    rotation, translation = ego_pose[:3, :3], ego_pose[:3, 3]
    moved[:, :3] = boxes[:, :3] @ rotation.T + translation
    moved[:, YAW] = boxes[:, YAW] + np.arctan2(rotation[1, 0], rotation[0, 0])
    if boxes.shape[1] > 7:
        moved[:, 7:9] = boxes[:, 7:9] @ rotation[:2, :2].T
    return moved


def as_eval_boxes(
    boxes: np.ndarray,
    names: Sequence[str],
    token: str,
    scores: np.ndarray = None,
) -> List[EvalBox]:
    """Global-frame boxes as the metric's box type."""
    result = []
    for index, row in enumerate(boxes):
        half_yaw = float(row[YAW]) / 2.0
        result.append(
            EvalBox(
                sample_token=token,
                translation=(float(row[0]), float(row[1]), float(row[2])),
                size=(float(row[3]), float(row[4]), float(row[5])),
                rotation=(float(np.cos(half_yaw)), 0.0, 0.0, float(np.sin(half_yaw))),
                velocity=(float(row[7]), float(row[8])) if len(row) > 8 else (0.0, 0.0),
                detection_name=names[index],
                detection_score=float(scores[index]) if scores is not None else -1.0,
            )
        )
    return result


def collect(
    model: L.LightningModule,
    loader: DataLoader,
    class_names: Sequence[str],
    device: torch.device,
) -> Dict[str, Dict[str, Any]]:
    """Run the model over the loader, gathering predictions and ground truth per frame.

    Ground truth is recorded for every frame, including frames with no boxes: a frame left out is
    a frame the model is not penalised for missing.
    """
    model = model.to(device).eval()
    ground_truth: Dict[str, List[EvalBox]] = {}
    predictions: Dict[str, List[EvalBox]] = {}
    ego_translations: Dict[str, np.ndarray] = {}

    with torch.no_grad():
        for batch in loader:
            on_device = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            decoded = model.head.get_bboxes(model(on_device))

            for index, (boxes, scores) in enumerate(decoded):
                metas = batch["img_metas"][index]
                token = frame_token(metas)
                pose = batch["ego_pose"][index].cpu().numpy().astype(np.float64)
                ego_translations[token] = pose[:3, 3]

                confidence, labels = scores.max(dim=-1)
                predicted = to_global(boxes.cpu().numpy().astype(np.float64), pose)
                predictions[token] = as_eval_boxes(
                    predicted,
                    [class_names[label] for label in labels.cpu().numpy()],
                    token,
                    confidence.cpu().numpy(),
                )

                truth = batch["gt_bboxes_3d"][index].cpu().numpy().astype(np.float64)
                truth_labels = batch["gt_labels_3d"][index].cpu().numpy()
                ground_truth[token] = as_eval_boxes(
                    to_global(truth, pose),
                    [class_names[label] for label in truth_labels],
                    token,
                )

    logger.info("Collected %d frames", len(ground_truth))
    return {
        "ground_truth": ground_truth,
        "predictions": predictions,
        "ego_translations": ego_translations,
    }


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    checkpoint = cfg.evaluation.get("checkpoint")
    if not checkpoint:
        raise ValueError("Set evaluation.checkpoint=<path to a .ckpt>.")

    L.seed_everything(cfg.run.seed, workers=True)
    taxonomy = get_taxonomy(cfg.taxonomy.name)

    model_class = hydra.utils.get_class(cfg.model._target_)
    model = model_class.load_from_checkpoint(checkpoint, map_location="cpu")
    logger.info("Loaded %s from %s", cfg.model.name, checkpoint)

    datasets = build_datasets(cfg, "val")
    loader = DataLoader(
        datasets[0] if len(datasets) == 1 else ConcatDataset(datasets),
        batch_size=cfg.dataloader.batch_size,
        shuffle=False,
        num_workers=cfg.dataloader.num_workers,
        collate_fn=collate_rigs,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gathered = collect(model, loader, taxonomy.class_names, device)

    output_dir = Path(cfg.evaluation.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = evaluate(
        gathered["ground_truth"],
        gathered["predictions"],
        class_names=taxonomy.class_names,
        ego_translations=gathered["ego_translations"],
        output_dir=str(output_dir),
        class_range=OmegaConf.to_container(cfg.evaluation.class_range, resolve=True)
        if cfg.evaluation.get("class_range")
        else None,
        tp_metrics=cfg.evaluation.tp_metrics,
        verbose=False,  # the summary is printed once, below
    )

    print(format_summary(summary))
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2, default=float))
    logger.info("Wrote %s", output_dir / "metrics.json")


if __name__ == "__main__":
    main()
