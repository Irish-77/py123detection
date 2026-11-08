"""Sharded evaluation: that splitting inference across ranks cannot change a score.

Inference now runs under ``Trainer.predict`` so ``evaluation.devices`` shards it, each rank writing
its own predictions file and rank zero merging them. Two things about that are quietly dangerous and
are pinned here.

The merge has to deduplicate. ``DistributedSampler`` pads the last batch so every rank sees the same
number of samples, which hands the first few frames of a split to two ranks; the dataset can repeat
a frame on its own account too, because a frame rejected by the box filters walks forward to the
next usable one. Scoring a frame twice inflates its hits and its misses together, and scoring a
subset inflates recall — neither announces itself.

And the conversion from ``predict_step``'s output has to match what the in-process loop did before
this change, bit for bit, or sharding would be "free" at the cost of a different number.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="the torch branch needs torch")
pytest.importorskip("nuscenes")

#: Needs the torch extra; see the `torch` marker in pyproject.toml.
pytestmark = pytest.mark.torch

import lightning as L  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

from freeze_evaluation_golden import CLASS_NAMES, CLASS_RANGE, fixed_frames  # noqa: E402

from py123detection.torch.evaluate import (  # noqa: E402
    FramePredictionWriter,
    frame_from_prediction,
    frame_token,
    score_predictions,
    to_global,
)
from py123detection.torch.models.base import Detector3D  # noqa: E402
from py123detection.torch.predictions import (  # noqa: E402
    FramePrediction,
    PredictionWriter,
    merge_rank_files,
    rank_files,
    read_predictions,
)

TOLERANCE = 1e-8


def _as_prediction(frame) -> FramePrediction:
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


def _write(path: Path, frames) -> None:
    with PredictionWriter(path) as writer:
        for frame in frames:
            writer.write(frame)


# --------------------------------------------------------------------------------------------
# The merge
# --------------------------------------------------------------------------------------------


def test_merging_shards_keeps_every_frame_exactly_once(tmp_path):
    frames = [_as_prediction(frame) for frame in fixed_frames()]
    _write(tmp_path / "predictions_rank0.jsonl", frames[0::2])
    _write(tmp_path / "predictions_rank1.jsonl", frames[1::2])

    counts = merge_rank_files(rank_files(tmp_path), tmp_path / "predictions.jsonl")

    assert counts == {"frames": len(frames), "duplicates": 0}
    merged = [frame.token for frame in read_predictions(tmp_path / "predictions.jsonl")]
    assert sorted(merged) == sorted(frame.token for frame in frames)


def test_merging_drops_the_frames_sampler_padding_duplicated(tmp_path):
    """What DistributedSampler padding actually looks like: the first frames repeated on rank 1."""
    frames = [_as_prediction(frame) for frame in fixed_frames()]
    _write(tmp_path / "predictions_rank0.jsonl", frames[0::2])
    _write(tmp_path / "predictions_rank1.jsonl", frames[1::2] + frames[:3])

    counts = merge_rank_files(rank_files(tmp_path), tmp_path / "predictions.jsonl")

    assert counts["duplicates"] == 3
    assert counts["frames"] == len(frames)
    tokens = [frame.token for frame in read_predictions(tmp_path / "predictions.jsonl")]
    assert len(set(tokens)) == len(tokens)


def test_merging_is_deterministic(tmp_path):
    frames = [_as_prediction(frame) for frame in fixed_frames()]
    for rank, shard in enumerate((frames[0::3], frames[1::3], frames[2::3])):
        _write(tmp_path / f"predictions_rank{rank}.jsonl", shard)

    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"
    assert merge_rank_files(rank_files(tmp_path), first) == merge_rank_files(rank_files(tmp_path), second)
    assert first.read_text() == second.read_text()


def test_rank_files_are_found_in_a_stable_order(tmp_path):
    for rank in (2, 0, 1):
        (tmp_path / f"predictions_rank{rank}.jsonl").write_text("")
    assert [path.name for path in rank_files(tmp_path)] == [
        "predictions_rank0.jsonl",
        "predictions_rank1.jsonl",
        "predictions_rank2.jsonl",
    ]
    assert rank_files(tmp_path / "nothing-here") == []


# --------------------------------------------------------------------------------------------
# Sharding cannot move a score
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("shards", [1, 2, 3, 5])
def test_scoring_a_merged_shard_set_equals_scoring_one_file(tmp_path, shards):
    """The claim sharding rests on: how the work was divided does not reach the metric."""
    frames = [_as_prediction(frame) for frame in fixed_frames() if len(frame["gt_boxes"]) > 0]

    single = tmp_path / "single.jsonl"
    _write(single, frames)
    expected = score_predictions(single, CLASS_NAMES, class_range=CLASS_RANGE, tp_metrics="no_attributes")

    sharded = tmp_path / f"sharded{shards}"
    sharded.mkdir()
    for rank in range(shards):
        _write(sharded / f"predictions_rank{rank}.jsonl", frames[rank::shards])
    counts = merge_rank_files(rank_files(sharded), sharded / "predictions.jsonl")
    assert counts["frames"] == len(frames)

    actual = score_predictions(
        sharded / "predictions.jsonl", CLASS_NAMES, class_range=CLASS_RANGE, tp_metrics="no_attributes"
    )
    assert actual["mean_ap"] == pytest.approx(expected["mean_ap"], abs=TOLERANCE)
    assert actual["nd_score"] == pytest.approx(expected["nd_score"], abs=TOLERANCE)
    for name, value in expected["tp_errors"].items():
        assert actual["tp_errors"][name] == pytest.approx(value, abs=TOLERANCE, nan_ok=True), name


def test_a_duplicated_frame_would_have_changed_the_score(tmp_path):
    """Proof the deduplication is load-bearing rather than decorative."""
    frames = [_as_prediction(frame) for frame in fixed_frames() if len(frame["gt_boxes"]) > 0]
    clean = tmp_path / "clean.jsonl"
    _write(clean, frames)
    expected = score_predictions(clean, CLASS_NAMES, class_range=CLASS_RANGE, tp_metrics="no_attributes")

    # The same frames, but with a few written twice under different tokens so they survive a merge.
    doubled = tmp_path / "doubled.jsonl"
    extra = []
    for frame in frames[:5]:
        copy = FramePrediction(**{**frame.__dict__, "token": frame.token + ":copy"})
        extra.append(copy)
    _write(doubled, frames + extra)
    actual = score_predictions(doubled, CLASS_NAMES, class_range=CLASS_RANGE, tp_metrics="no_attributes")

    assert actual["mean_ap"] != pytest.approx(expected["mean_ap"], abs=1e-6)


# --------------------------------------------------------------------------------------------
# The conversion from predict_step's output
# --------------------------------------------------------------------------------------------


def reference_frame(batch, index, boxes, scores):
    """The conversion as the in-process loop did it, before inference moved under a Trainer."""
    metas = batch["img_metas"][index]
    pose = batch["ego_pose"][index].cpu().numpy().astype(np.float64)
    confidence, labels = scores.max(dim=-1)
    return FramePrediction(
        token=frame_token(metas),
        dataset=metas.get("dataset_name", "unknown"),
        ego_translation=pose[:3, 3],
        boxes=to_global(boxes.cpu().numpy().astype(np.float64), pose),
        scores=confidence.cpu().numpy().astype(np.float64),
        labels=labels.cpu().numpy().astype(np.int64),
        gt_boxes=to_global(batch["gt_bboxes_3d"][index].cpu().numpy().astype(np.float64), pose),
        gt_labels=batch["gt_labels_3d"][index].cpu().numpy().astype(np.int64),
    )


def _fake_batch(frames):
    """A collated batch shaped the way ``collate_rigs`` produces it."""
    return {
        "img_metas": [
            {
                "scene_uuid": frame["token"].split(":")[0],
                "frame_idx": int(frame["token"].split(":")[1]),
                "dataset_name": frame["dataset"],
            }
            for frame in frames
        ],
        "ego_pose": torch.tensor(np.stack([frame["ego_pose"] for frame in frames]), dtype=torch.float64),
        "gt_bboxes_3d": [torch.tensor(frame["gt_boxes"], dtype=torch.float64) for frame in frames],
        "gt_labels_3d": [torch.tensor(frame["gt_labels"], dtype=torch.long) for frame in frames],
    }


def _decoded(frames):
    """``head.get_bboxes`` output: one ``(boxes, per-class scores)`` pair per sample."""
    pairs = []
    for frame in frames:
        boxes = torch.tensor(frame["pred_boxes"], dtype=torch.float64)
        scores = torch.zeros(len(boxes), len(CLASS_NAMES), dtype=torch.float64)
        scores[torch.arange(len(boxes)), torch.tensor(frame["pred_labels"])] = torch.tensor(
            frame["pred_scores"], dtype=torch.float64
        )
        pairs.append((boxes, scores))
    return pairs


def test_the_conversion_matches_the_in_process_loop_it_replaced():
    frames = fixed_frames()[:6]
    batch = _fake_batch(frames)
    for index, (boxes, scores) in enumerate(_decoded(frames)):
        confidence, labels = scores.max(dim=-1)
        item = {
            "boxes": boxes,
            "scores": confidence,
            "labels": labels,
            "scene_uuid": batch["img_metas"][index]["scene_uuid"],
            "frame_idx": batch["img_metas"][index]["frame_idx"],
            "dataset_name": batch["img_metas"][index]["dataset_name"],
            "ego_pose": batch["ego_pose"][index],
        }
        actual = frame_from_prediction(item, batch, index)
        expected = reference_frame(batch, index, boxes, scores)

        assert actual.token == expected.token
        assert actual.dataset == expected.dataset
        for field in ("ego_translation", "boxes", "scores", "gt_boxes"):
            assert np.array_equal(getattr(actual, field), getattr(expected, field)), field
        for field in ("labels", "gt_labels"):
            assert np.array_equal(getattr(actual, field), getattr(expected, field)), field


def test_the_box_cap_reaches_the_written_frame():
    frames = fixed_frames()[:1]
    batch = _fake_batch(frames)
    boxes, scores = _decoded(frames)[0]
    confidence, labels = scores.max(dim=-1)
    item = {
        "boxes": boxes,
        "scores": confidence,
        "labels": labels,
        "scene_uuid": batch["img_metas"][0]["scene_uuid"],
        "frame_idx": batch["img_metas"][0]["frame_idx"],
        "dataset_name": batch["img_metas"][0]["dataset_name"],
        "ego_pose": batch["ego_pose"][0],
    }
    capped = frame_from_prediction(item, batch, 0, max_boxes_per_sample=5)
    assert len(capped.boxes) == 5
    assert np.all(np.diff(capped.scores) <= 0), "the cap should keep the highest scores"
    assert capped.scores.max() == pytest.approx(confidence.numpy().max())


# --------------------------------------------------------------------------------------------
# End to end through a real Trainer
# --------------------------------------------------------------------------------------------


class StubPredictHead(torch.nn.Module):
    """A head that decodes whatever the stub detector was handed."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def get_bboxes(self, predictions):
        return predictions["decoded"]


class StubPredictDetector(Detector3D):
    def __init__(self) -> None:
        super().__init__()
        self.head = StubPredictHead()

    def forward(self, batch):
        return {"decoded": batch["decoded"]}


def test_a_real_predict_run_writes_one_file_per_rank_and_dataset(tmp_path):
    """The Lightning wiring: predict_step is reached, the writer routes by dataset, nothing is lost.

    ``return_predictions=False`` is the point — Lightning's own return path would collect every
    batch on every rank, which is what this stage exists to avoid.
    """
    frames = fixed_frames()[:9]
    batch = _fake_batch(frames)
    batch["decoded"] = _decoded(frames)

    writer = FramePredictionWriter(tmp_path)
    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        callbacks=[writer],
    )
    loader = torch.utils.data.DataLoader([batch], batch_size=None, collate_fn=lambda value: value)
    trainer.predict(StubPredictDetector(), dataloaders=loader, return_predictions=False)

    assert writer.frames == len(frames)

    expected_datasets = {frame["dataset"] for frame in frames}
    written = {}
    for dataset in expected_datasets:
        shards = rank_files(tmp_path / dataset)
        assert shards == [tmp_path / dataset / "predictions_rank0.jsonl"], dataset
        written[dataset] = list(read_predictions(shards[0]))

    assert sum(len(value) for value in written.values()) == len(frames)
    for dataset, group in written.items():
        assert all(frame.dataset == dataset for frame in group)

    # And the frames written match the direct conversion, so the callback adds no transformation.
    by_token = {frame.token: frame for group in written.values() for frame in group}
    for index in range(len(frames)):
        boxes, scores = batch["decoded"][index]
        expected = reference_frame(batch, index, boxes, scores)
        actual = by_token[expected.token]
        assert np.array_equal(actual.boxes, expected.boxes)
        assert np.array_equal(actual.scores, expected.scores)
        assert np.array_equal(actual.gt_boxes, expected.gt_boxes)


def test_the_writer_reports_nothing_when_a_rank_sees_no_batches(tmp_path):
    writer = FramePredictionWriter(tmp_path)
    trainer = L.Trainer(
        accelerator="cpu", devices=1, logger=False, enable_checkpointing=False, enable_progress_bar=False,
        callbacks=[writer],
    )
    loader = torch.utils.data.DataLoader([], batch_size=None, collate_fn=lambda value: value)
    trainer.predict(StubPredictDetector(), dataloaders=loader, return_predictions=False)

    assert writer.frames == 0
    assert rank_files(tmp_path) == []
