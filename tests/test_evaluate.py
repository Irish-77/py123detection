"""The evaluation path: that the rework moved no number, and that it still cannot lose a frame.

``tests/golden/evaluation_golden.json`` was frozen from the implementation that scored every dataset
together in one process, before per-dataset output, streamed predictions and the vectorised box
builder. Those three changes were made to bound memory and to make a mixed run diagnosable, and
none of them is allowed to change a score — so the scores are compared directly rather than
inferred from the tests passing.

Two of the checks here compare against the *previous* implementation inlined below rather than
against a stored number. That is deliberate: it records in executable form what "unchanged" meant,
and it is bit-exact, which a rounded fixture cannot be.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("nuscenes")

#: Needs the torch extra; see the `torch` marker in pyproject.toml.
pytestmark = pytest.mark.torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

from freeze_evaluation_golden import (  # noqa: E402
    CLASS_NAMES,
    CLASS_RANGE,
    DATASETS,
    fixed_frames,
    score,
)

from py123detection.core.evaluation import EvalBox  # noqa: E402
from py123detection.torch.evaluate import (  # noqa: E402
    YAW,
    as_eval_boxes,
    score_frames,
    score_predictions,
    to_global,
    top_scoring,
)
from py123detection.torch.predictions import (  # noqa: E402
    FramePrediction,
    PredictionWriter,
    frames_for,
    read_predictions,
)

GOLDEN = REPO / "tests" / "golden" / "evaluation_golden.json"
#: The fixture stores eight decimals; scoring is deterministic, so this is only guarding rounding.
TOLERANCE = 1e-8


@pytest.fixture(scope="module")
def golden():
    if not GOLDEN.exists():
        pytest.skip(f"{GOLDEN} missing; run tools/freeze_evaluation_golden.py")
    return json.loads(GOLDEN.read_text())


@pytest.fixture(scope="module")
def frames():
    return fixed_frames()


def _as_prediction(frame) -> FramePrediction:
    """One fixture frame as the streamed record, already in the global frame."""
    pose = frame["ego_pose"]
    return FramePrediction(
        token=frame["token"],
        dataset=frame["dataset"],
        ego_translation=pose[:3, 3],
        boxes=to_global(frame["pred_boxes"], pose),
        scores=frame["pred_scores"],
        labels=frame["pred_labels"],
        gt_boxes=to_global(frame["gt_boxes"], pose),
        gt_labels=frame["gt_labels"],
    )


def _close(actual, expected) -> bool:
    """``nan_ok`` because an excluded true-positive error is nan by design, not by accident.

    Under ``tp_metrics="no_attributes"`` the devkit leaves ``attr_err`` undefined, and a fixture
    that could not express "still undefined" would quietly stop checking it.
    """
    return actual == pytest.approx(expected, abs=TOLERANCE, nan_ok=True)


def _compare(actual, expected, label: str) -> None:
    assert _close(actual["mean_ap"], expected["mean_ap"]), label
    assert _close(actual["nd_score"], expected["nd_score"]), label
    for name, value in expected["tp_errors"].items():
        assert _close(actual["tp_errors"][name], value), f"{label}/{name}"
    for name, value in expected["mean_dist_aps"].items():
        assert _close(actual["mean_dist_aps"][name], value), f"{label}/AP/{name}"


# --------------------------------------------------------------------------------------------
# The previous implementations, kept as the reference for the parts that were optimised.
# --------------------------------------------------------------------------------------------


def reference_to_global(boxes: np.ndarray, ego_pose: np.ndarray) -> np.ndarray:
    """``to_global`` before it reused ``core.geometry``."""
    if len(boxes) == 0:
        return boxes
    moved = boxes.copy()
    rotation, translation = ego_pose[:3, :3], ego_pose[:3, 3]
    moved[:, :3] = boxes[:, :3] @ rotation.T + translation
    moved[:, YAW] = boxes[:, YAW] + np.arctan2(rotation[1, 0], rotation[0, 0])
    if boxes.shape[1] > 7:
        moved[:, 7:9] = boxes[:, 7:9] @ rotation[:2, :2].T
    return moved


def reference_as_eval_boxes(boxes, names, token, scores=None):
    """``as_eval_boxes`` before the per-field conversion was lifted out of the loop."""
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


# --------------------------------------------------------------------------------------------
# Bit-exactness of the optimised pieces
# --------------------------------------------------------------------------------------------


def test_the_vectorised_box_builder_is_bit_identical_to_the_loop_it_replaced(frames):
    """Not approximately equal: the same doubles.

    Worth asserting rather than assuming, because lifting ``np.cos`` out of the loop turns a scalar
    call into an array one and numpy is free to take a different code path for arrays.
    """
    checked = 0
    for frame in frames:
        boxes = to_global(frame["pred_boxes"], frame["ego_pose"])
        names = [CLASS_NAMES[label] for label in frame["pred_labels"]]
        actual = as_eval_boxes(boxes, names, frame["token"], frame["pred_scores"])
        expected = reference_as_eval_boxes(boxes, names, frame["token"], frame["pred_scores"])
        assert actual == expected, f"{frame['token']} differs"
        checked += len(actual)
    assert checked > 0


def test_reusing_core_geometry_leaves_the_global_conversion_unchanged(frames):
    for frame in frames:
        for key in ("pred_boxes", "gt_boxes"):
            actual = to_global(frame[key], frame["ego_pose"])
            expected = reference_to_global(frame[key], frame["ego_pose"])
            assert np.array_equal(actual, expected), f"{frame['token']}/{key}"


# --------------------------------------------------------------------------------------------
# The frozen scores
# --------------------------------------------------------------------------------------------


def test_scoring_every_frame_reproduces_the_fixture(frames, golden):
    _compare(score(frames), golden["all_frames"], "all_frames")


def test_scoring_only_frames_with_ground_truth_reproduces_the_fixture(frames, golden):
    nonempty = [frame for frame in frames if len(frame["gt_boxes"]) > 0]
    _compare(score(nonempty), golden["nonempty_only"], "nonempty_only")


@pytest.mark.parametrize("dataset", DATASETS)
def test_per_dataset_scoring_reproduces_the_fixture(frames, golden, dataset):
    """The new per-dataset path against per-dataset numbers computed by the old scoring code.

    Without this the refactor could produce per-dataset numbers that are plausible and wrong in the
    same way for every dataset, and nothing else here would notice.
    """
    subset = [
        _as_prediction(frame)
        for frame in frames
        if frame["dataset"] == dataset and len(frame["gt_boxes"]) > 0
    ]
    summary = score_frames(iter(subset), CLASS_NAMES, class_range=CLASS_RANGE, tp_metrics="no_attributes")
    _compare(summary, golden["per_dataset"][dataset], dataset)


def test_scoring_empty_frames_lowers_the_score(frames, golden):
    """The reason the flag exists, stated as a test.

    A frame with no ground truth that the model puts boxes on is pure false positive. Dropping such
    frames — which is the default, for reproducibility — therefore cannot help the score and in
    general flatters it.
    """
    assert golden["all_frames"]["mean_ap"] < golden["nonempty_only"]["mean_ap"]
    assert golden["frame_counts"]["all_frames"] > golden["frame_counts"]["nonempty_only"]


# --------------------------------------------------------------------------------------------
# Streaming predictions
# --------------------------------------------------------------------------------------------


def test_predictions_round_trip_without_losing_precision(frames, tmp_path):
    path = tmp_path / "predictions.jsonl"
    originals = [_as_prediction(frame) for frame in frames]
    with PredictionWriter(path) as writer:
        for frame in originals:
            writer.write(frame)
    assert writer.frames == len(originals)

    restored = list(read_predictions(path))
    assert len(restored) == len(originals)
    for before, after in zip(originals, restored):
        assert after.token == before.token
        assert after.dataset == before.dataset
        for field in ("ego_translation", "boxes", "scores", "gt_boxes"):
            assert np.array_equal(getattr(after, field), np.asarray(getattr(before, field), dtype=np.float64)), (
                f"{before.token}/{field}"
            )
        for field in ("labels", "gt_labels"):
            assert np.array_equal(getattr(after, field), getattr(before, field)), f"{before.token}/{field}"


def test_scoring_from_a_file_equals_scoring_in_line(frames, tmp_path, golden):
    """Re-scoring a saved run must give the first run's answer, or persisting it is pointless."""
    path = tmp_path / "predictions.jsonl"
    nonempty = [frame for frame in frames if len(frame["gt_boxes"]) > 0]
    with PredictionWriter(path) as writer:
        for frame in nonempty:
            writer.write(_as_prediction(frame))

    from_file = score_predictions(path, CLASS_NAMES, class_range=CLASS_RANGE, tp_metrics="no_attributes")
    _compare(from_file, golden["nonempty_only"], "rescored")


def test_splitting_by_dataset_loses_no_frame_and_duplicates_none(frames, tmp_path):
    """The failure mode of per-dataset scoring: a frame that lands in neither bucket, or in two.

    Scoring a subset silently inflates recall, which is the one mistake the metric cannot show you.
    """
    path = tmp_path / "predictions.jsonl"
    with PredictionWriter(path) as writer:
        for frame in frames:
            writer.write(_as_prediction(frame))

    everything = [frame.token for frame in read_predictions(path)]
    per_dataset = [frame.token for dataset in DATASETS for frame in frames_for(path, dataset)]

    assert sorted(per_dataset) == sorted(everything)
    assert len(set(per_dataset)) == len(per_dataset), "a frame was scored twice"


# --------------------------------------------------------------------------------------------
# The opt-in box cap
# --------------------------------------------------------------------------------------------


def test_the_box_cap_is_off_until_asked_for():
    scores = np.array([0.1, 0.9, 0.5])
    assert top_scoring(scores, None) is None
    assert top_scoring(scores, 3) is None, "a frame already under the cap is left alone"
    assert top_scoring(scores, 10) is None


def test_the_box_cap_keeps_the_highest_scoring_boxes_in_order():
    scores = np.array([0.1, 0.9, 0.5, 0.7])
    keep = top_scoring(scores, 2)
    assert keep.tolist() == [1, 3]
    assert np.array_equal(scores[keep], np.array([0.9, 0.7]))


def test_the_box_cap_breaks_ties_the_same_way_every_time():
    scores = np.array([0.5, 0.5, 0.5, 0.1])
    assert top_scoring(scores, 2).tolist() == top_scoring(scores, 2).tolist() == [0, 1]
