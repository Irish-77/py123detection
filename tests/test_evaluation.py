"""Unit tests for the shared detection metric.

Two things need proving. That the numbers are the devkit's, since the whole point of running its
``accumulate`` / ``calc_ap`` / ``calc_tp`` rather than a reimplementation is to report the metric
everyone else reports. And that dropping ``DetectionBox`` really does lift the ten-name limit,
because that limit is what stopped three of this package's own taxonomies being evaluable.
"""

from __future__ import annotations

import numpy as np
import pytest

from py123detection.core.evaluation import (
    TP_METRIC_SETS,
    EvalBox,
    build_detection_config,
    evaluate,
    filter_eval_boxes,
    to_eval_boxes,
)

CAR_SIZE = (1.9, 4.5, 1.6)
UPRIGHT = (1.0, 0.0, 0.0, 0.0)


def box(token, x, y, name="car", score=-1.0, size=CAR_SIZE, num_pts=10, velocity=(0.0, 0.0)):
    return EvalBox(
        sample_token=token,
        translation=(float(x), float(y), 0.0),
        size=size,
        rotation=UPRIGHT,
        velocity=velocity,
        num_pts=num_pts,
        detection_name=name,
        detection_score=float(score),
    )


def scene(num_frames=8, num_boxes=6, name="car", jitter=0.0, seed=0):
    """Ground truth, matching predictions offset by ``jitter`` metres, and ego positions."""
    rng = np.random.default_rng(seed)
    ground_truth, predictions, ego = {}, {}, {}
    for frame in range(num_frames):
        token = f"frame_{frame}"
        centers = rng.uniform(-30.0, 30.0, size=(num_boxes, 2))
        ground_truth[token] = [box(token, x, y, name=name) for x, y in centers]
        offsets = rng.normal(0.0, jitter, size=(num_boxes, 2)) if jitter else np.zeros((num_boxes, 2))
        predictions[token] = [
            box(token, x + dx, y + dy, name=name, score=float(rng.uniform(0.2, 1.0)))
            for (x, y), (dx, dy) in zip(centers, offsets)
        ]
        ego[token] = np.zeros(3)
    return ground_truth, predictions, ego


def test_perfect_predictions_score_a_perfect_map() -> None:
    ground_truth, predictions, ego = scene()
    summary = evaluate(ground_truth, predictions, ["car"], ego, verbose=False)

    assert summary["mean_ap"] == pytest.approx(1.0)
    assert summary["tp_errors"]["trans_err"] == pytest.approx(0.0, abs=1e-9)
    assert summary["tp_errors"]["scale_err"] == pytest.approx(0.0, abs=1e-9)


def test_missing_every_object_scores_zero() -> None:
    ground_truth, _, ego = scene()
    empty = {token: [] for token in ground_truth}
    summary = evaluate(ground_truth, empty, ["car"], ego, verbose=False)
    assert summary["mean_ap"] == pytest.approx(0.0)


# --- the point of the exercise: class names outside the nuScenes ten --------------------------


@pytest.mark.parametrize("name", ["vehicle", "cyclist", "two_wheeler", "generic_object"])
def test_taxonomy_class_names_the_devkit_rejects_are_evaluable(name: str) -> None:
    """``DetectionBox`` asserts its ``detection_name`` is one of ten. general_3cls, vehicle_1cls
    and default_123d all use names outside that list, so before this they could not be scored."""
    ground_truth, predictions, ego = scene(name=name)
    summary = evaluate(
        ground_truth, predictions, [name], ego, class_range={name: 50.0}, verbose=False
    )
    assert summary["mean_ap"] == pytest.approx(1.0)
    assert name in summary["mean_dist_aps"]


def test_devkit_detection_box_still_rejects_those_names() -> None:
    """The constraint this works around is real, not imagined."""
    from nuscenes.eval.detection.data_classes import DetectionBox

    with pytest.raises(AssertionError, match="Unknown detection_name"):
        DetectionBox(
            sample_token="t", translation=(0, 0, 0), size=CAR_SIZE, rotation=UPRIGHT, detection_name="cyclist"
        )


def test_matches_the_devkit_on_its_own_box_class() -> None:
    """Same inputs through ``DetectionBox`` and through ``EvalBox`` give the same AP, so swapping
    the container changed the reach of the metric and not the metric."""
    from nuscenes.eval.common.data_classes import EvalBoxes
    from nuscenes.eval.common.utils import center_distance
    from nuscenes.eval.detection.algo import accumulate, calc_ap
    from nuscenes.eval.detection.data_classes import DetectionBox

    ground_truth, predictions, ego = scene(jitter=0.7, seed=3)

    def as_detection_boxes(mapping):
        container = EvalBoxes()
        for token, boxes in mapping.items():
            container.add_boxes(token, [
                DetectionBox(
                    sample_token=b.sample_token, translation=b.translation, size=b.size,
                    rotation=b.rotation, velocity=b.velocity, ego_translation=b.translation,
                    num_pts=b.num_pts, detection_name=b.detection_name,
                    detection_score=b.detection_score, attribute_name="",
                )
                for b in boxes
            ])
        return container

    devkit_ap = calc_ap(
        accumulate(as_detection_boxes(ground_truth), as_detection_boxes(predictions), "car", center_distance, 2.0),
        0.1, 0.1,
    )
    summary = evaluate(ground_truth, predictions, ["car"], ego, verbose=False)
    assert summary["label_aps"]["car"][2.0] == pytest.approx(devkit_ap)


# --- NDS and the attribute question -----------------------------------------------------------


def test_dropping_attribute_error_changes_nds_by_the_documented_amount() -> None:
    """123D carries no attributes, so the devkit scores attr_err as a flat 1.0 and NDS eats a
    fixed penalty. Excluding it must change the score in exactly the way the formula says."""
    ground_truth, predictions, ego = scene(jitter=0.5, seed=11)

    with_attributes = evaluate(ground_truth, predictions, ["car"], ego, verbose=False)
    without = evaluate(ground_truth, predictions, ["car"], ego, tp_metrics="no_attributes", verbose=False)

    assert with_attributes["mean_ap"] == pytest.approx(without["mean_ap"])
    assert with_attributes["tp_errors"]["attr_err"] == pytest.approx(1.0)

    scores = [1.0 - min(1.0, with_attributes["tp_errors"][m]) for m in TP_METRIC_SETS["no_attributes"]]
    expected_five = (5 * with_attributes["mean_ap"] + sum(scores) + 0.0) / 10.0
    expected_four = (4 * without["mean_ap"] + sum(scores)) / 8.0
    assert with_attributes["nd_score"] == pytest.approx(expected_five)
    assert without["nd_score"] == pytest.approx(expected_four)
    assert without["nd_score"] > with_attributes["nd_score"]


def test_the_chosen_metric_set_is_recorded_in_the_summary() -> None:
    """Two NDS numbers from different metric sets are not comparable, so a summary has to say
    which it is."""
    ground_truth, predictions, ego = scene()
    summary = evaluate(ground_truth, predictions, ["car"], ego, tp_metrics="no_attributes", verbose=False)
    assert summary["evaluation"]["tp_metrics"] == list(TP_METRIC_SETS["no_attributes"])


def test_unknown_metric_set_is_rejected() -> None:
    ground_truth, predictions, ego = scene()
    with pytest.raises(ValueError, match="Unknown tp_metrics"):
        evaluate(ground_truth, predictions, ["car"], ego, tp_metrics="whatever", verbose=False)


# --- the mistakes the metric cannot show you ---------------------------------------------------


def test_frames_without_predictions_cannot_be_silently_skipped() -> None:
    """The thesis metric skipped frames a model predicted nothing for, which removes exactly the
    hardest frames from the recall denominator and inflates the score."""
    ground_truth, predictions, ego = scene()
    predictions.pop("frame_3")
    with pytest.raises(AssertionError, match="do not match the ground truth"):
        evaluate(ground_truth, predictions, ["car"], ego, verbose=False)


def test_a_class_without_a_range_is_rejected_not_guessed() -> None:
    with pytest.raises(ValueError, match="No evaluation range"):
        build_detection_config(["wombat"])


def test_boxes_beyond_the_class_radius_are_dropped() -> None:
    cfg = build_detection_config(["car"])
    boxes = to_eval_boxes({"t": [box("t", 10, 0), box("t", 200, 0)]})
    boxes["t"][0].ego_translation = (10.0, 0.0, 0.0)
    boxes["t"][1].ego_translation = (200.0, 0.0, 0.0)

    before, after = filter_eval_boxes(boxes, cfg.class_range)
    assert (before, after) == (2, 1)


def test_boxes_with_zero_lidar_points_are_dropped_but_unknown_counts_are_kept() -> None:
    cfg = build_detection_config(["car"])
    boxes = to_eval_boxes({"t": [box("t", 1, 0, num_pts=0), box("t", 2, 0, num_pts=-1)]})
    before, after = filter_eval_boxes(boxes, cfg.class_range)
    assert (before, after) == (2, 1)
    assert boxes["t"][0].num_pts == -1


def test_eval_box_round_trips_through_json() -> None:
    original = box("t", 1.0, 2.0, score=0.7, velocity=(1.5, -0.5))
    assert EvalBox.deserialize(original.serialize()) == original
