"""Freeze the evaluation path's numbers before it is reworked.

Run this *before* changing anything under ``py123detection.torch.evaluate``::

    ../envs/modern/bin/python tools/freeze_evaluation_golden.py --out tests/golden/evaluation_golden.json

The rework changes two things that must not move a metric: boxes are gathered per dataset and
streamed to disk instead of accumulated over every dataset at once, and ``as_eval_boxes`` builds the
metric's box type with array maths instead of a Python loop. "The tests still pass" is too weak a
claim for that, so the numbers are pinned directly.

The fixture is synthetic on purpose — no checkpoint, no 123D logs, no GPU — so the guarantee is
reproducible anywhere. Two frame sets are frozen because the rework also adds
``evaluation.score_empty_frames``: ``all_frames`` keeps the frames whose ground truth is empty,
``nonempty_only`` drops them, and the two scores differ exactly because a model is penalised for
predicting boxes where there are none.

Per-dataset metrics are frozen too, computed here by scoring each dataset's frames with the
*current* code. That is the reference the new per-dataset path has to reproduce: without it, a
refactor could produce plausible per-dataset numbers that are wrong in the same way for every
dataset and nothing would notice.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO = Path(__file__).resolve().parents[1]

SEED = 20261108

#: Small enough to score in a second, large enough that every class has true and false positives at
#: several distances, which is what makes the AP and the TP errors non-degenerate.
SHAPE = {"scenes": 4, "frames_per_scene": 8, "gt_per_frame": 7, "queries": 40}

CLASS_NAMES = ["vehicle", "pedestrian", "bicycle"]
CLASS_RANGE = {"vehicle": 50.0, "pedestrian": 40.0, "bicycle": 40.0}
DATASETS = ["nuscenes", "av2", "wod"]

#: Columns of a decoded box, matching ``py123detection.torch.evaluate``.
BOX_COLUMNS = 9
YAW = 6

#: Every fourth frame has no ground truth, so the empty-frame question is exercised.
EMPTY_EVERY = 4


def _ego_pose(rng: np.random.Generator) -> np.ndarray:
    """A non-identity pose, so a mistake in the ego-to-global conversion cannot hide."""
    angle = rng.uniform(-np.pi, np.pi)
    pose = np.eye(4, dtype=np.float64)
    pose[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    pose[:3, 3] = rng.uniform(-400.0, 400.0, size=3)
    return pose


def _boxes(rng: np.random.Generator, count: int) -> np.ndarray:
    boxes = np.zeros((count, BOX_COLUMNS), dtype=np.float64)
    boxes[:, 0] = rng.uniform(-45.0, 45.0, size=count)
    boxes[:, 1] = rng.uniform(-45.0, 45.0, size=count)
    boxes[:, 2] = rng.uniform(-2.0, 1.0, size=count)
    boxes[:, 3] = rng.uniform(1.5, 2.5, size=count)
    boxes[:, 4] = rng.uniform(3.0, 5.5, size=count)
    boxes[:, 5] = rng.uniform(1.3, 2.2, size=count)
    boxes[:, YAW] = rng.uniform(-np.pi, np.pi, size=count)
    boxes[:, 7] = rng.uniform(-6.0, 6.0, size=count)
    boxes[:, 8] = rng.uniform(-6.0, 6.0, size=count)
    return boxes


def fixed_frames() -> List[Dict[str, Any]]:
    """The frames both the fixture and the test score, built identically from one seed.

    Predictions are the ground truth perturbed, plus false positives, plus duplicates of a few true
    boxes — so the metric sees matches, misses and double counts rather than a clean separation.
    """
    rng = np.random.default_rng(SEED)
    frames: List[Dict[str, Any]] = []

    for scene in range(SHAPE["scenes"]):
        for index in range(SHAPE["frames_per_scene"]):
            empty = (len(frames) % EMPTY_EVERY) == 0
            gt_count = 0 if empty else SHAPE["gt_per_frame"]
            gt_boxes = _boxes(rng, gt_count)
            gt_labels = rng.integers(0, len(CLASS_NAMES), size=gt_count)

            # Perturbed copies of the ground truth, then noise boxes to fill the query budget.
            matched = min(gt_count, SHAPE["queries"] // 2)
            perturbed = gt_boxes[:matched].copy()
            if matched:
                perturbed[:, :2] += rng.normal(0.0, 1.2, size=(matched, 2))
                perturbed[:, 3:6] *= rng.normal(1.0, 0.06, size=(matched, 3))
                perturbed[:, YAW] += rng.normal(0.0, 0.15, size=matched)
            noise = _boxes(rng, SHAPE["queries"] - matched)
            pred_boxes = np.concatenate([perturbed, noise], axis=0)
            pred_labels = np.concatenate(
                [gt_labels[:matched], rng.integers(0, len(CLASS_NAMES), size=len(noise))]
            ).astype(np.int64)
            # Matches score higher on average but overlap with the noise, so ranking matters.
            scores = np.concatenate(
                [rng.uniform(0.4, 1.0, size=matched), rng.uniform(0.0, 0.7, size=len(noise))]
            )
            order = np.argsort(-scores)

            frames.append(
                {
                    "token": f"scene{scene}:{index}",
                    "dataset": DATASETS[scene % len(DATASETS)],
                    "ego_pose": _ego_pose(rng),
                    "gt_boxes": gt_boxes,
                    "gt_labels": gt_labels.astype(np.int64),
                    "pred_boxes": pred_boxes[order],
                    "pred_labels": pred_labels[order],
                    "pred_scores": scores[order],
                }
            )
    return frames


def score(frames: List[Dict[str, Any]], output_dir: str = None) -> Dict[str, Any]:
    """Score one set of frames exactly as ``torch.evaluate`` does today."""
    from py123detection.core.evaluation import evaluate
    from py123detection.torch.evaluate import as_eval_boxes, to_global

    ground_truth: Dict[str, Any] = {}
    predictions: Dict[str, Any] = {}
    ego_translations: Dict[str, np.ndarray] = {}

    for frame in frames:
        token, pose = frame["token"], frame["ego_pose"]
        ego_translations[token] = pose[:3, 3]
        predictions[token] = as_eval_boxes(
            to_global(frame["pred_boxes"], pose),
            [CLASS_NAMES[label] for label in frame["pred_labels"]],
            token,
            frame["pred_scores"],
        )
        ground_truth[token] = as_eval_boxes(
            to_global(frame["gt_boxes"], pose),
            [CLASS_NAMES[label] for label in frame["gt_labels"]],
            token,
        )

    return evaluate(
        ground_truth,
        predictions,
        class_names=CLASS_NAMES,
        ego_translations=ego_translations,
        output_dir=output_dir,
        class_range=CLASS_RANGE,
        tp_metrics="no_attributes",
        verbose=False,
    )


def _keep(summary: Dict[str, Any]) -> Dict[str, Any]:
    """The parts of a metrics dict that must not move, rounded so a diff stays readable."""
    return {
        "mean_ap": round(float(summary["mean_ap"]), 8),
        "nd_score": round(float(summary["nd_score"]), 8),
        "tp_errors": {name: round(float(value), 8) for name, value in sorted(summary["tp_errors"].items())},
        "mean_dist_aps": {
            name: round(float(value), 8) for name, value in sorted(summary["mean_dist_aps"].items())
        },
        "evaluation": {
            key: summary["evaluation"][key]
            for key in sorted(summary["evaluation"])
            if key in {"mode", "num_frames", "tp_metrics", "boxes_before_filter", "boxes_after_filter"}
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO / "tests" / "golden" / "evaluation_golden.json")
    args = parser.parse_args()

    frames = fixed_frames()
    nonempty = [frame for frame in frames if len(frame["gt_boxes"]) > 0]

    fixture: Dict[str, Any] = {
        "seed": SEED,
        "shape": SHAPE,
        "class_names": CLASS_NAMES,
        "class_range": CLASS_RANGE,
        "datasets": DATASETS,
        "empty_every": EMPTY_EVERY,
        "note": (
            "Frozen from py123detection.torch.evaluate before the per-dataset and streaming rework. "
            "all_frames keeps empty-ground-truth frames (evaluation.score_empty_frames=true); "
            "nonempty_only drops them, which is today's default. per_dataset is the reference the "
            "new per-dataset path must reproduce."
        ),
        "frame_counts": {"all_frames": len(frames), "nonempty_only": len(nonempty)},
        "all_frames": _keep(score(frames)),
        "nonempty_only": _keep(score(nonempty)),
        "per_dataset": {
            dataset: _keep(score([frame for frame in nonempty if frame["dataset"] == dataset]))
            for dataset in DATASETS
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out}")
    print(f"  all_frames     mAP {fixture['all_frames']['mean_ap']:.6f}  NDS {fixture['all_frames']['nd_score']:.6f}")
    print(
        f"  nonempty_only  mAP {fixture['nonempty_only']['mean_ap']:.6f}  "
        f"NDS {fixture['nonempty_only']['nd_score']:.6f}"
    )
    for dataset in DATASETS:
        entry = fixture["per_dataset"][dataset]
        print(f"  {dataset:10s}     mAP {entry['mean_ap']:.6f}  NDS {entry['nd_score']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
