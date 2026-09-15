"""Self-contained nuScenes-style detection evaluation from an exported pickle.

mmdetection3d's ``NuScenesDataset.evaluate`` hands predictions to the nuScenes devkit's
``NuScenesEval``, which reads ground truth from the nuScenes JSON database and asserts that the
prediction tokens equal an official split. That ties evaluation to one dataset. CoIn3D worked
around it for Lyft and Waymo by converting those datasets *into* nuScenes databases and patching
the devkit.

This module takes the other route. The pickle already holds every ground-truth box, so the
devkit's ``EvalBoxes`` are built from ``data_infos`` and the devkit's own metric code
(``accumulate`` / ``calc_ap`` / ``calc_tp``, ``DetectionMetrics``) runs on them. No database, no
split registry, any dataset — provided the class names are nuScenes detection names, which the
devkit's ``DetectionBox`` asserts (CoIn3D's ``car / pedestrian / motorcycle`` spelling exists for
exactly this reason).

Ground truth goes through the **same code path as predictions**: ``LiDARInstance3DBoxes`` ->
``output_to_nusc_box`` -> ``lidar_nusc_box_to_global``. Whatever box layout the pickle uses, GT and
predictions are interpreted identically, so the metric cannot be skewed by a convention mismatch
between the two.

What differs from the official protocol, and why:

* ``add_center_dist`` uses the frame's ``ego2global_translation`` instead of the ``LIDAR_TOP``
  ``ego_pose`` record — the same quantity, read from the pickle;
* the nuScenes-specific bike-rack filter is not applied (there is no
  ``static_object.bicycle_rack`` to look up);
* GT boxes with ``num_lidar_pts + num_radar_pts == 0`` are dropped, as in the devkit;
* attributes: an export carries none, so ``attr_err`` is ``nan`` per box and the devkit's
  ``cummean`` scores the attribute error as 1.0 — exactly what CoIn3D's Lyft / Waymo tables show.
  Read NDS with that fixed penalty in mind, or look at mAP and the other four TP errors.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

EVALUATION_MODE = "py123detection.self_contained"


def build_detection_config(
    class_names: Sequence[str],
    class_range: Optional[Mapping[str, float]] = None,
    base: str = "detection_cvpr_2019",
):
    """Build a devkit ``DetectionConfig`` restricted to the classes actually evaluated.

    ``DetectionConfig.__init__`` asserts the full ten-class nuScenes list, so the config is
    created from a named base and restricted afterwards — the attributes are plain and nothing in
    the metric code re-validates them.

    :param class_names: Classes to evaluate, in order.
    :param class_range: Evaluation radius in metres per class. Classes named like nuScenes
        classes default to the base config's radius (car 50, pedestrian 40, motorcycle 40, ...).
    :param base: Named devkit config to start from.
    :return: The restricted config.
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
    """Build the devkit's GT ``EvalBoxes`` from mmdet3d infos.

    Boxes are converted exactly like mmdet3d converts *predictions* — ``LiDARInstance3DBoxes``
    with ``origin=(0.5, 0.5, 0.5)`` as in ``get_ann_info``, then ``output_to_nusc_box`` and
    ``lidar_nusc_box_to_global`` — so GT and predictions share one interpretation of the box
    layout. Every frame gets an entry, empty when it holds no box of an evaluated class.

    :param data_infos: The dataset's infos, in dataset order.
    :param class_names: Evaluated class names; boxes of other classes are ignored.
    :param cfg: The ``DetectionConfig`` (its ``class_range`` filters by radius, as for predictions).
    :param eval_version: Passed through to ``lidar_nusc_box_to_global``.
    :return: An ``EvalBoxes`` of ``DetectionBox`` ground truth.
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

        # mmdet3d substitutes 0 for nuScenes' nan velocities before building boxes; do the same.
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
            box.token = str(source_index)  # survives the radius filter below
        nusc_boxes = lidar_nusc_box_to_global(info, nusc_boxes, class_names, cfg, eval_version)

        ego_translation = np.asarray(info["ego2global_translation"], dtype=np.float64)
        eval_boxes = []
        for box in nusc_boxes:
            source_index = int(box.token)
            center = np.asarray(box.center, dtype=np.float64)
            eval_boxes.append(
                DetectionBox(
                    sample_token=token,
                    translation=tuple(center.tolist()),
                    size=tuple(np.asarray(box.wlh, dtype=np.float64).tolist()),
                    rotation=tuple(np.asarray(box.orientation.elements, dtype=np.float64).tolist()),
                    velocity=tuple(np.asarray(box.velocity[:2], dtype=np.float64).tolist()),
                    ego_translation=tuple((center - ego_translation).tolist()),
                    num_pts=int(num_pts[source_index]),
                    detection_name=class_names[int(box.label)],
                    detection_score=-1.0,
                    attribute_name="",
                )
            )
        gt_boxes.add_boxes(token, eval_boxes)

    return gt_boxes


def _add_ego_translation(eval_boxes, ego_by_token: Mapping[str, np.ndarray]) -> None:
    """The devkit's ``add_center_dist``, with the ego position read from the pickle."""
    for token in eval_boxes.sample_tokens:
        ego = ego_by_token[token]
        for box in eval_boxes[token]:
            box.ego_translation = tuple((np.asarray(box.translation, dtype=np.float64) - ego).tolist())


def _filter_boxes(eval_boxes, max_dist: Mapping[str, float]) -> Tuple[int, int]:
    """The devkit's ``filter_eval_boxes`` minus the nuScenes-only bike-rack step.

    :return: ``(boxes before, boxes after)``.
    """
    before = after = 0
    for token in eval_boxes.sample_tokens:
        boxes = eval_boxes[token]
        before += len(boxes)
        boxes = [box for box in boxes if box.ego_dist < max_dist[box.detection_name]]
        boxes = [box for box in boxes if not box.num_pts == 0]
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
    """Evaluate a nuScenes-format result JSON against the ground truth held in the infos.

    Mirrors ``nuscenes.eval.detection.evaluate.DetectionEval`` step for step, writing the same
    ``metrics_summary.json`` / ``metrics_details.json`` into ``output_dir`` so downstream readers
    (mmdet3d's ``_evaluate_single`` included) see the familiar layout.

    :param result_path: The ``results_nusc.json`` written by ``NuScenesDataset._format_bbox``.
    :param data_infos: The dataset's infos, in dataset order (``self.data_infos``).
    :param class_names: Evaluated classes, in the order the model was trained with.
    :param output_dir: Where to write the metric files.
    :param class_range: Per-class evaluation radius overrides; see :func:`build_detection_config`.
    :param eval_version: Named devkit config used as the base.
    :param verbose: Print the summary table.
    :return: The serialized ``DetectionMetrics``, plus ``meta`` and an ``evaluation`` block.
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

    ego_by_token = {
        info["token"]: np.asarray(info["ego2global_translation"], dtype=np.float64) for info in data_infos
    }
    _add_ego_translation(pred_boxes, ego_by_token)  # GT already carries ego_translation
    counts = {"pred": _filter_boxes(pred_boxes, cfg.class_range), "gt": _filter_boxes(gt_boxes, cfg.class_range)}

    # --- the body of DetectionEval.evaluate(), verbatim in structure --------------------------
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
    """Render a metrics summary the way the devkit's ``main()`` prints it."""
    err_names = {
        "trans_err": "mATE",
        "scale_err": "mASE",
        "orient_err": "mAOE",
        "vel_err": "mAVE",
        "attr_err": "mAAE",
    }
    lines: List[str] = [f"mAP: {summary['mean_ap']:.4f}"]
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
