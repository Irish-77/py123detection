"""nuScenes-style detection evaluation against the ground truth stored in an exported pickle.

mmdetection3d's ``NuScenesDataset.evaluate`` hands predictions to the devkit's ``NuScenesEval``,
which reads ground truth from the nuScenes database and requires an official split. Here the
devkit's ``EvalBoxes`` are built from ``data_infos`` instead and its metric code
(``accumulate`` / ``calc_ap`` / ``calc_tp``) runs on them, so any dataset can be evaluated as long
as its class names are nuScenes detection names (``DetectionBox`` asserts that).

Ground truth goes through the same path as predictions (``LiDARInstance3DBoxes`` ->
``output_to_nusc_box`` -> ``lidar_nusc_box_to_global``), so a box layout mismatch cannot skew the
metric.

Differences from the official protocol:

* ``add_center_dist`` uses the frame's ``ego2global_translation`` instead of the ``LIDAR_TOP``
  ego pose record, the same quantity read from the pickle;
* no bike-rack filter, since there is no ``static_object.bicycle_rack`` to look up;
* exports carry no attributes, so ``attr_err`` is ``nan`` per box and the devkit scores it as 1.0,
  as in CoIn3D's Lyft / Waymo tables. NDS includes that fixed penalty.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

EVALUATION_MODE = "py123detection.self_contained"


def build_detection_config(
    class_names: Sequence[str],
    class_range: Optional[Mapping[str, float]] = None,
    base: str = "detection_cvpr_2019",
):
    """A devkit ``DetectionConfig`` restricted to ``class_names``.

    ``DetectionConfig.__init__`` insists on the ten nuScenes classes, so the named base config is
    created first and its plain attributes are narrowed afterwards.

    :param class_range: Radius in metres per class, on top of the base config's radii.
    """
    from nuscenes.eval.detection.config import config_factory

    cfg = config_factory(base)
    ranges: Dict[str, float] = dict(cfg.class_range)
    ranges.update(class_range or {})
    missing = [name for name in class_names if name not in ranges]
    if missing:
        raise ValueError(f"No evaluation range for classes {missing}; pass class_range={{name: metres}}.")
    cfg.class_range = {name: float(ranges[name]) for name in class_names}
    cfg.class_names = list(cfg.class_range.keys())
    return cfg


def ground_truth_from_infos(
    data_infos: Sequence[Dict[str, Any]],
    class_names: Sequence[str],
    cfg,
    eval_version: str = "detection_cvpr_2019",
):
    """The devkit's ground-truth ``EvalBoxes``, built from mmdet3d infos.

    Boxes are converted exactly like mmdet3d converts predictions. Every frame gets an entry,
    empty when it has no box of an evaluated class.
    """
    import torch
    from mmdet3d.core.bbox import LiDARInstance3DBoxes
    from mmdet3d.datasets.nuscenes_dataset import lidar_nusc_box_to_global, output_to_nusc_box
    from nuscenes.eval.common.data_classes import EvalBoxes
    from nuscenes.eval.detection.data_classes import DetectionBox

    class_names = list(class_names)
    class_index = {name: index for index, name in enumerate(class_names)}
    gt_boxes = EvalBoxes()

    for info in data_infos:
        token = info["token"]
        names = np.asarray(info["gt_names"]).reshape(-1)
        keep = np.array([name in class_index for name in names], dtype=bool)
        boxes = np.asarray(info["gt_boxes"], dtype=np.float64).reshape(-1, 7)[keep]
        if boxes.shape[0] == 0:
            gt_boxes.add_boxes(token, [])
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

        ego_translation = np.asarray(info["ego2global_translation"], dtype=np.float64)
        eval_boxes = []
        for box in nusc_boxes:
            center = np.asarray(box.center, dtype=np.float64)
            eval_boxes.append(
                DetectionBox(
                    sample_token=token,
                    translation=tuple(center.tolist()),
                    size=tuple(np.asarray(box.wlh, dtype=np.float64).tolist()),
                    rotation=tuple(np.asarray(box.orientation.elements, dtype=np.float64).tolist()),
                    velocity=tuple(np.asarray(box.velocity[:2], dtype=np.float64).tolist()),
                    ego_translation=tuple((center - ego_translation).tolist()),
                    num_pts=int(num_pts[int(box.token)]),
                    detection_name=class_names[int(box.label)],
                    detection_score=-1.0,
                    attribute_name="",
                )
            )
        gt_boxes.add_boxes(token, eval_boxes)

    return gt_boxes


def _filter_boxes(eval_boxes, max_dist: Mapping[str, float]) -> Tuple[int, int]:
    """The devkit's ``filter_eval_boxes`` without the bike-rack step. Returns box counts before and after."""
    before = after = 0
    for token in eval_boxes.sample_tokens:
        boxes = eval_boxes[token]
        before += len(boxes)
        boxes = [box for box in boxes if box.ego_dist < max_dist[box.detection_name]]
        boxes = [box for box in boxes if box.num_pts != 0]
        eval_boxes.boxes[token] = boxes
        after += len(boxes)
    return before, after


def evaluate_from_infos(
    result_path: str,
    data_infos: Sequence[Dict[str, Any]],
    class_names: Sequence[str],
    output_dir: str,
    class_range: Optional[Mapping[str, float]] = None,
    eval_version: str = "detection_cvpr_2019",
    verbose: bool = True,
) -> Dict[str, Any]:
    """Evaluate a nuScenes result JSON against the ground truth in the infos.

    Follows ``nuscenes.eval.detection.evaluate.DetectionEval`` and writes the same
    ``metrics_summary.json`` / ``metrics_details.json`` into ``output_dir``.

    :param result_path: ``results_nusc.json`` from ``NuScenesDataset._format_bbox``.
    :param data_infos: The dataset's infos, in dataset order.
    :param class_range: Per-class radius overrides, see :func:`build_detection_config`.
    :return: The serialized ``DetectionMetrics`` plus ``meta`` and ``evaluation`` blocks.
    """
    from nuscenes.eval.common.loaders import load_prediction
    from nuscenes.eval.detection.algo import accumulate, calc_ap, calc_tp
    from nuscenes.eval.detection.constants import TP_METRICS
    from nuscenes.eval.detection.data_classes import DetectionBox, DetectionMetricDataList, DetectionMetrics

    class_names = list(class_names)
    cfg = build_detection_config(class_names, class_range, base=eval_version)
    os.makedirs(output_dir, exist_ok=True)
    start = time.time()

    pred_boxes, meta = load_prediction(result_path, cfg.max_boxes_per_sample, DetectionBox, verbose=verbose)
    gt_boxes = ground_truth_from_infos(data_infos, class_names, cfg, eval_version)

    pred_tokens, gt_tokens = set(pred_boxes.sample_tokens), set(gt_boxes.sample_tokens)
    if pred_tokens != gt_tokens:
        raise AssertionError(
            f"Prediction tokens ({len(pred_tokens)}) do not match the pickle's frames ({len(gt_tokens)}): "
            f"{len(pred_tokens - gt_tokens)} unknown, {len(gt_tokens - pred_tokens)} missing."
        )

    # The devkit's add_center_dist for predictions, with the ego position from the pickle. The GT
    # boxes already carry it.
    ego_by_token = {
        info["token"]: np.asarray(info["ego2global_translation"], dtype=np.float64) for info in data_infos
    }
    for token in pred_boxes.sample_tokens:
        ego = ego_by_token[token]
        for box in pred_boxes[token]:
            box.ego_translation = tuple((np.asarray(box.translation, dtype=np.float64) - ego).tolist())
    counts = {"pred": _filter_boxes(pred_boxes, cfg.class_range), "gt": _filter_boxes(gt_boxes, cfg.class_range)}

    # From here on the same as DetectionEval.evaluate().
    metric_data_list = DetectionMetricDataList()
    for class_name in cfg.class_names:
        for dist_th in cfg.dist_ths:
            md = accumulate(gt_boxes, pred_boxes, class_name, cfg.dist_fcn_callable, dist_th)
            metric_data_list.set(class_name, dist_th, md)

    metrics = DetectionMetrics(cfg)
    for class_name in cfg.class_names:
        for dist_th in cfg.dist_ths:
            ap = calc_ap(metric_data_list[(class_name, dist_th)], cfg.min_recall, cfg.min_precision)
            metrics.add_label_ap(class_name, dist_th, ap)
        for metric_name in TP_METRICS:
            metric_data = metric_data_list[(class_name, cfg.dist_th_tp)]
            if class_name in ["traffic_cone"] and metric_name in ["attr_err", "vel_err", "orient_err"]:
                tp = np.nan
            elif class_name in ["barrier"] and metric_name in ["attr_err", "vel_err"]:
                tp = np.nan
            else:
                tp = calc_tp(metric_data, cfg.min_recall, metric_name)
            metrics.add_label_tp(class_name, metric_name, tp)
    metrics.add_runtime(time.time() - start)

    summary = metrics.serialize()
    summary["meta"] = meta
    summary["evaluation"] = {
        "mode": EVALUATION_MODE,
        "num_frames": len(gt_tokens),
        "boxes_before_after_filter": counts,
        "class_range": dict(cfg.class_range),
    }
    with open(os.path.join(output_dir, "metrics_summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    with open(os.path.join(output_dir, "metrics_details.json"), "w") as handle:
        json.dump(metric_data_list.serialize(), handle, indent=2)

    if verbose:
        print(format_summary(summary))
    return summary


def format_summary(summary: Dict[str, Any]) -> str:
    """The metrics table as the devkit's ``main()`` prints it."""
    err_names = {
        "trans_err": "mATE",
        "scale_err": "mASE",
        "orient_err": "mAOE",
        "vel_err": "mAVE",
        "attr_err": "mAAE",
    }
    lines = [f"mAP: {summary['mean_ap']:.4f}"]
    for key, label in err_names.items():
        lines.append(f"{label}: {summary['tp_errors'][key]:.4f}")
    lines.append(f"NDS: {summary['nd_score']:.4f}")
    lines.append(f"Eval time: {summary.get('eval_time', float('nan')):.1f}s")
    lines.append(f"Evaluation mode: {summary.get('evaluation', {}).get('mode', '?')}")
    lines.append("")
    lines.append("Per-class results:")
    lines.append(f"{'Object Class':<22}{'AP':>8}{'ATE':>8}{'ASE':>8}{'AOE':>8}{'AVE':>8}{'AAE':>8}")
    for class_name, ap in summary["mean_dist_aps"].items():
        errors = summary["label_tp_errors"][class_name]
        lines.append(
            f"{class_name:<22}{ap:>8.3f}"
            + "".join(f"{errors[key]:>8.3f}" for key in ("trans_err", "scale_err", "orient_err", "vel_err", "attr_err"))
        )
    return "\n".join(lines)
