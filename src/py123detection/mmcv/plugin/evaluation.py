"""Evaluating an mmdetection3d run against the ground truth stored in an exported pickle.

mmdetection3d's ``NuScenesDataset.evaluate`` hands predictions to the devkit's ``NuScenesEval``,
which reads ground truth from a nuScenes database and requires an official split. Neither exists
for a 123D export, so this module builds the ground truth from ``data_infos`` instead and hands it
to :mod:`py123detection.core.evaluation`, which runs the devkit's own metric code.

Ground truth goes through the same conversion as predictions
(``LiDARInstance3DBoxes`` -> ``output_to_nusc_box`` -> ``lidar_nusc_box_to_global``), so a box
layout mismatch cannot skew the metric in either direction.

The deviations from the official protocol are listed in :mod:`py123detection.core.evaluation`.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from py123detection.core.evaluation import (
    EVALUATION_MODE,
    EvalBox,
    build_detection_config,
    evaluate,
    format_summary,
    load_predictions,
)

__all__ = [
    "EVALUATION_MODE",
    "build_detection_config",
    "evaluate_from_infos",
    "format_summary",
    "ground_truth_from_infos",
]


def ground_truth_from_infos(
    data_infos: Sequence[Dict[str, Any]],
    class_names: Sequence[str],
    cfg,
    eval_version: str = "detection_cvpr_2019",
) -> Dict[str, list]:
    """Global-frame ground-truth boxes per frame token, built from mmdet3d infos.

    Every frame gets an entry, empty when it holds no box of an evaluated class: a frame dropped
    here is a frame the model is never penalised for missing.
    """
    import torch
    from mmdet3d.core.bbox import LiDARInstance3DBoxes
    from mmdet3d.datasets.nuscenes_dataset import lidar_nusc_box_to_global, output_to_nusc_box

    class_names = list(class_names)
    class_index = {name: index for index, name in enumerate(class_names)}
    ground_truth: Dict[str, list] = {}

    for info in data_infos:
        token = info["token"]
        names = np.asarray(info["gt_names"]).reshape(-1)
        keep = np.array([name in class_index for name in names], dtype=bool)
        boxes = np.asarray(info["gt_boxes"], dtype=np.float64).reshape(-1, 7)[keep]
        if boxes.shape[0] == 0:
            ground_truth[token] = []
            continue

        # mmdet3d replaces nuScenes' nan velocities by 0 before building boxes.
        velocity = np.nan_to_num(np.asarray(info["gt_velocity"], dtype=np.float64).reshape(-1, 2)[keep])
        num_lidar = np.asarray(info["num_lidar_pts"]).reshape(-1)[keep].astype(np.int64)
        num_radar = np.asarray(info.get("num_radar_pts", np.zeros(len(names)))).reshape(-1)[keep].astype(np.int64)
        num_pts = num_lidar + num_radar
        labels = np.array([class_index[name] for name in names[keep]], dtype=np.int64)

        detection = dict(
            boxes_3d=LiDARInstance3DBoxes(
                torch.from_numpy(np.concatenate([boxes, velocity], axis=1)),
                box_dim=9,
                origin=(0.5, 0.5, 0.5),
            ),
            scores_3d=torch.ones(boxes.shape[0]),
            labels_3d=torch.from_numpy(labels),
        )
        nusc_boxes = output_to_nusc_box(detection)
        for source_index, box in enumerate(nusc_boxes):
            box.token = str(source_index)  # to find num_pts again after the radius filter
        nusc_boxes = lidar_nusc_box_to_global(info, nusc_boxes, class_names, cfg, eval_version)

        ground_truth[token] = [
            EvalBox(
                sample_token=token,
                translation=tuple(np.asarray(box.center, dtype=np.float64).tolist()),
                size=tuple(np.asarray(box.wlh, dtype=np.float64).tolist()),
                rotation=tuple(np.asarray(box.orientation.elements, dtype=np.float64).tolist()),
                velocity=tuple(np.asarray(box.velocity[:2], dtype=np.float64).tolist()),
                num_pts=int(num_pts[int(box.token)]),
                detection_name=class_names[int(box.label)],
                detection_score=-1.0,
                attribute_name="",
            )
            for box in nusc_boxes
        ]

    return ground_truth


def evaluate_from_infos(
    result_path: str,
    data_infos: Sequence[Dict[str, Any]],
    class_names: Sequence[str],
    output_dir: str,
    class_range: Optional[Mapping[str, float]] = None,
    eval_version: str = "detection_cvpr_2019",
    tp_metrics: str = "nuscenes",
    verbose: bool = True,
) -> Dict[str, Any]:
    """Evaluate a nuScenes result JSON against the ground truth in the infos.

    :param result_path: ``results_nusc.json`` from ``NuScenesDataset._format_bbox``.
    :param data_infos: The dataset's infos, in dataset order.
    :param class_range: Per-class radius overrides, see
        :func:`~py123detection.core.evaluation.build_detection_config`.
    :param tp_metrics: Which true-positive errors enter NDS; see
        :data:`~py123detection.core.evaluation.TP_METRIC_SETS`.
    :returns: The serialized metrics plus ``meta`` and ``evaluation`` blocks.
    """
    class_names = list(class_names)
    cfg = build_detection_config(class_names, class_range, base=eval_version)

    predictions, meta = load_predictions(result_path, cfg.max_boxes_per_sample, verbose=verbose)
    ground_truth = ground_truth_from_infos(data_infos, class_names, cfg, eval_version)
    ego_translations = {
        info["token"]: np.asarray(info["ego2global_translation"], dtype=np.float64) for info in data_infos
    }

    summary = evaluate(
        ground_truth=ground_truth,
        predictions=predictions,
        class_names=class_names,
        ego_translations=ego_translations,
        output_dir=output_dir,
        class_range=class_range,
        eval_version=eval_version,
        tp_metrics=tp_metrics,
        verbose=verbose,
    )
    summary["meta"] = meta
    return summary
