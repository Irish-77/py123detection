"""nuScenes-style detection metrics, for either branch and any set of class names.

The nuScenes devkit computes the metric this field reports, so this module runs the devkit's own
``accumulate`` / ``calc_ap`` / ``calc_tp`` rather than reimplementing them. What it does not use is
``DetectionBox``, whose constructor accepts only the ten nuScenes detection names. That single
assertion is the reason a cross-dataset taxonomy such as ``vehicle / pedestrian / cyclist`` cannot
be scored by the devkit at all, and the reason the thesis grew an 882-line reimplementation beside
it. :class:`EvalBox` carries the same fields without the whitelist, so one engine serves both
branches and every taxonomy.

Ground truth and predictions arrive as :class:`EvalBox` in the **global** frame. Each branch has
its own adapter: :mod:`py123detection.mmcv.plugin.evaluation` builds them from an exported info
pickle, the torch branch from 123D scenes.

Differences from the official protocol, all deliberate:

* the ego position comes from the frame's ``ego2global_translation`` rather than a ``LIDAR_TOP``
  ego-pose record, because that is the quantity 123D actually stores;
* there is no bike-rack filter, since there is no ``static_object.bicycle_rack`` to look up;
* 123D carries no object attributes, so ``attr_err`` is ``nan`` per box and the devkit scores it
  as a flat 1.0. :data:`TP_METRIC_SETS` lets you drop it instead — see :func:`evaluate`.
"""

from __future__ import annotations

import json
import logging
import os
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

EVALUATION_MODE = "py123detection.self_contained"

#: Which true-positive errors enter NDS.
#:
#: ``nuscenes`` is the official five. ``no_attributes`` drops ``attr_err``, which is what you want
#: whenever the data carries no attributes: the devkit scores a missing attribute as the worst
#: possible value, so keeping it subtracts a flat 0.1 from NDS that says nothing about the model.
#: NDS is then ``(k * mAP + sum of k scores) / 2k`` either way, so the two are not comparable and
#: the choice is recorded in the summary.
TP_METRIC_SETS: Dict[str, Tuple[str, ...]] = {
    "nuscenes": ("trans_err", "scale_err", "orient_err", "vel_err", "attr_err"),
    "no_attributes": ("trans_err", "scale_err", "orient_err", "vel_err"),
}

#: Errors that are meaningless for a class and are excluded from its average, as the devkit does.
UNDEFINED_TP_ERRORS: Dict[str, Tuple[str, ...]] = {
    "traffic_cone": ("attr_err", "vel_err", "orient_err"),  # rotationally symmetric and static
    "barrier": ("attr_err", "vel_err"),  # static
}


@dataclass
class EvalBox:
    """One box for the metric, in the global frame.

    Field-compatible with the devkit's ``DetectionBox`` — the metric code reads these names — but
    without its assertion that ``detection_name`` is one of the ten nuScenes classes.
    """

    sample_token: str
    translation: Tuple[float, float, float]
    size: Tuple[float, float, float]
    """``(width, length, height)``, the devkit's ``wlh`` order."""
    rotation: Tuple[float, float, float, float]
    """Quaternion ``(w, x, y, z)``."""
    detection_name: str
    velocity: Tuple[float, float] = (0.0, 0.0)
    ego_translation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Box centre relative to the ego position. :func:`add_center_distances` fills it in."""
    num_pts: int = -1
    detection_score: float = -1.0
    attribute_name: str = ""

    @property
    def ego_dist(self) -> float:
        return float(np.linalg.norm(np.asarray(self.ego_translation[:2], dtype=np.float64)))

    def serialize(self) -> Dict[str, Any]:
        return {
            "sample_token": self.sample_token,
            "translation": list(self.translation),
            "size": list(self.size),
            "rotation": list(self.rotation),
            "velocity": list(self.velocity),
            "ego_translation": list(self.ego_translation),
            "num_pts": int(self.num_pts),
            "detection_name": self.detection_name,
            "detection_score": float(self.detection_score),
            "attribute_name": self.attribute_name,
        }

    @classmethod
    def deserialize(cls, content: Mapping[str, Any]) -> "EvalBox":
        """Read one box of a nuScenes-format result JSON."""
        return cls(
            sample_token=content["sample_token"],
            translation=tuple(content["translation"]),
            size=tuple(content["size"]),
            rotation=tuple(content["rotation"]),
            velocity=tuple(content.get("velocity", (0.0, 0.0))),
            ego_translation=tuple(content.get("ego_translation", (0.0, 0.0, 0.0))),
            num_pts=int(content.get("num_pts", -1)),
            detection_name=content["detection_name"],
            detection_score=float(content.get("detection_score", -1.0)),
            attribute_name=content.get("attribute_name", ""),
        )


@dataclass
class EvaluationInput:
    """Everything the metric needs about one evaluation run."""

    ground_truth: Dict[str, list] = field(default_factory=dict)
    """Ground-truth boxes per frame token. Every evaluated frame needs an entry, empty or not —
    a frame left out is a frame the model is not penalised for missing."""

    predictions: Dict[str, list] = field(default_factory=dict)
    ego_translations: Dict[str, np.ndarray] = field(default_factory=dict)
    """Ego position per frame token, used to compute each box's distance from the ego vehicle."""


def build_detection_config(
    class_names: Sequence[str],
    class_range: Optional[Mapping[str, float]] = None,
    base: str = "detection_cvpr_2019",
):
    """A devkit ``DetectionConfig`` restricted to ``class_names``.

    ``DetectionConfig.__init__`` insists on the ten nuScenes classes, so the named base config is
    created first and its plain attributes are narrowed afterwards.

    :param class_range: Radius in metres per class, on top of the base config's radii.
    :raises ValueError: if a class has no radius. Guessing one silently decides how much of the
        scene counts, so it has to be stated.
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


def to_eval_boxes(boxes_by_token: Mapping[str, Iterable[EvalBox]]):
    """Wrap per-frame boxes in the devkit's ``EvalBoxes`` container."""
    from nuscenes.eval.common.data_classes import EvalBoxes

    container = EvalBoxes()
    for token, boxes in boxes_by_token.items():
        container.add_boxes(token, list(boxes))
    return container


def load_predictions(result_path: str, max_boxes_per_sample: int, verbose: bool = False):
    """Read a nuScenes-format result JSON into ``EvalBoxes`` of :class:`EvalBox`."""
    from nuscenes.eval.common.loaders import load_prediction

    return load_prediction(result_path, max_boxes_per_sample, EvalBox, verbose=verbose)


def add_center_distances(eval_boxes, ego_translations: Mapping[str, np.ndarray]) -> None:
    """Fill in each box's offset from its frame's ego position, in place.

    The devkit's ``add_center_dist`` reads the ego pose from a nuScenes database; here it comes
    from the frame itself.
    """
    for token in eval_boxes.sample_tokens:
        ego = np.asarray(ego_translations[token], dtype=np.float64)
        for box in eval_boxes[token]:
            box.ego_translation = tuple((np.asarray(box.translation, dtype=np.float64) - ego).tolist())


def filter_eval_boxes(eval_boxes, max_dist: Mapping[str, float]) -> Tuple[int, int]:
    """The devkit's ``filter_eval_boxes`` without the bike-rack step.

    Drops boxes beyond their class radius, and boxes explicitly recorded as having zero lidar
    points. A count of ``-1`` means the dataset does not record one, and is kept.

    :returns: box counts before and after.
    """
    before = after = 0
    for token in eval_boxes.sample_tokens:
        boxes = eval_boxes[token]
        before += len(boxes)
        boxes = [box for box in boxes if box.ego_dist < max_dist[box.detection_name]]
        boxes = [box for box in boxes if box.num_pts != 0]
        eval_boxes.boxes[token] = boxes
        after += len(boxes)
    return before, after


def _nd_score(mean_ap: float, tp_errors: Mapping[str, float], tp_metrics: Sequence[str]) -> float:
    """NDS over a chosen set of true-positive errors.

    The devkit weights mAP by the number of TP metrics, so with ``k`` of them the score is
    ``(k * mAP + sum of k capped scores) / 2k`` and mAP is always half the total.
    """
    total = float(len(tp_metrics)) * float(mean_ap)
    for metric_name in tp_metrics:
        error = tp_errors[metric_name]
        total += 1.0 - min(1.0, 1.0 if np.isnan(error) else float(error))
    return total / (2.0 * len(tp_metrics))


def evaluate(
    ground_truth: Mapping[str, Iterable[EvalBox]],
    predictions,
    class_names: Sequence[str],
    ego_translations: Mapping[str, np.ndarray],
    output_dir: Optional[str] = None,
    class_range: Optional[Mapping[str, float]] = None,
    eval_version: str = "detection_cvpr_2019",
    tp_metrics: str = "nuscenes",
    verbose: bool = True,
) -> Dict[str, Any]:
    """Score predictions against ground truth, following ``DetectionEval.evaluate``.

    :param ground_truth: Boxes per frame token, in the global frame. Every evaluated frame needs
        an entry even when it has no box, or the frames a model failed on vanish from recall.
    :param predictions: Either the same mapping shape, or an ``EvalBoxes`` from
        :func:`load_predictions`.
    :param ego_translations: Ego position per frame token.
    :param tp_metrics: A key of :data:`TP_METRIC_SETS`.
    :raises AssertionError: if the prediction and ground-truth frames differ. Scoring a subset
        silently inflates recall, which is the one mistake this metric cannot show you.
    :returns: The devkit's serialized metrics plus ``evaluation`` describing how they were made.
    """
    from nuscenes.eval.detection.algo import accumulate, calc_ap, calc_tp
    from nuscenes.eval.detection.data_classes import DetectionMetricDataList, DetectionMetrics

    if tp_metrics not in TP_METRIC_SETS:
        raise ValueError(f"Unknown tp_metrics '{tp_metrics}'. Available: {sorted(TP_METRIC_SETS)}")
    selected_metrics = TP_METRIC_SETS[tp_metrics]

    class_names = list(class_names)
    cfg = build_detection_config(class_names, class_range, base=eval_version)
    start = time.time()

    gt_boxes = to_eval_boxes(ground_truth)
    pred_boxes = predictions if hasattr(predictions, "sample_tokens") else to_eval_boxes(predictions)

    pred_tokens, gt_tokens = set(pred_boxes.sample_tokens), set(gt_boxes.sample_tokens)
    if pred_tokens != gt_tokens:
        raise AssertionError(
            f"Prediction frames ({len(pred_tokens)}) do not match the ground truth ({len(gt_tokens)}): "
            f"{len(pred_tokens - gt_tokens)} unknown, {len(gt_tokens - pred_tokens)} missing."
        )

    add_center_distances(pred_boxes, ego_translations)
    add_center_distances(gt_boxes, ego_translations)
    counts = {
        "pred": filter_eval_boxes(pred_boxes, cfg.class_range),
        "gt": filter_eval_boxes(gt_boxes, cfg.class_range),
    }

    metric_data_list = DetectionMetricDataList()
    for class_name in cfg.class_names:
        for dist_th in cfg.dist_ths:
            metric_data_list.set(
                class_name, dist_th, accumulate(gt_boxes, pred_boxes, class_name, cfg.dist_fcn_callable, dist_th)
            )

    metrics = DetectionMetrics(cfg)
    for class_name in cfg.class_names:
        for dist_th in cfg.dist_ths:
            metrics.add_label_ap(
                class_name, dist_th, calc_ap(metric_data_list[(class_name, dist_th)], cfg.min_recall, cfg.min_precision)
            )
        for metric_name in TP_METRIC_SETS["nuscenes"]:
            if metric_name in UNDEFINED_TP_ERRORS.get(class_name, ()):
                value = np.nan
            elif metric_name not in selected_metrics:
                value = np.nan
            else:
                value = calc_tp(metric_data_list[(class_name, cfg.dist_th_tp)], cfg.min_recall, metric_name)
            metrics.add_label_tp(class_name, metric_name, value)
    metrics.add_runtime(time.time() - start)

    # Averaging an all-NaN column is exactly what excluding a metric means, so numpy's warning
    # about it is noise rather than news.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
        summary = metrics.serialize()
    summary["nd_score"] = _nd_score(summary["mean_ap"], summary["tp_errors"], selected_metrics)
    summary["evaluation"] = {
        "mode": EVALUATION_MODE,
        "num_frames": len(gt_tokens),
        "tp_metrics": list(selected_metrics),
        "boxes_before_after_filter": counts,
        "class_range": dict(cfg.class_range),
    }

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "metrics_summary.json"), "w") as handle:
            json.dump(summary, handle, indent=2)
        with open(os.path.join(output_dir, "metrics_details.json"), "w") as handle:
            json.dump(metric_data_list.serialize(), handle, indent=2)

    if verbose:
        print(format_summary(summary))
    return summary


def format_summary(summary: Mapping[str, Any]) -> str:
    """The metrics table as the devkit's ``main()`` prints it, plus what was configured."""
    err_names = {
        "trans_err": "mATE",
        "scale_err": "mASE",
        "orient_err": "mAOE",
        "vel_err": "mAVE",
        "attr_err": "mAAE",
    }
    evaluation = summary.get("evaluation", {})
    selected = set(evaluation.get("tp_metrics", TP_METRIC_SETS["nuscenes"]))

    lines = [f"mAP: {summary['mean_ap']:.4f}"]
    for key, label in err_names.items():
        value = summary["tp_errors"][key]
        suffix = "" if key in selected else "   (excluded from NDS)"
        lines.append(f"{label}: {value:.4f}{suffix}")
    lines.append(f"NDS: {summary['nd_score']:.4f}")
    lines.append(f"Eval time: {summary.get('eval_time', float('nan')):.1f}s")
    lines.append(f"Evaluation mode: {evaluation.get('mode', '?')} | TP metrics: {len(selected)}")
    lines.append("")
    lines.append("Per-class results:")
    lines.append(f"{'Object Class':<22}{'AP':>8}{'ATE':>8}{'ASE':>8}{'AOE':>8}{'AVE':>8}{'AAE':>8}")
    for class_name, ap in summary["mean_dist_aps"].items():
        errors = summary["label_tp_errors"][class_name]
        lines.append(
            f"{class_name:<22}{ap:>8.3f}"
            + "".join(f"{errors[key]:>8.3f}" for key in err_names)
        )
    return "\n".join(lines)
