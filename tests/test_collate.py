"""Batching samples from rigs of different sizes.

Mixing datasets is the point of the exercise, and datasets disagree about how many cameras a
vehicle has. These tests pin what a batch does about that.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="the torch branch needs torch")

from py123detection.torch.data.collate import collate_rigs  # noqa: E402


def sample(num_cameras: int, num_boxes: int = 3, height: int = 8, width: int = 16):
    return {
        "img": torch.arange(num_cameras * 3 * height * width, dtype=torch.float32).reshape(
            num_cameras, 3, height, width
        ),
        "intrinsics": torch.eye(4).repeat(num_cameras, 1, 1) * 2.0,
        "extrinsics": torch.eye(4).repeat(num_cameras, 1, 1) * 3.0,
        "imu2img": torch.eye(4).repeat(num_cameras, 1, 1) * 4.0,
        "cam_valid_mask": torch.ones(num_cameras, dtype=torch.bool),
        "ego_pose": torch.eye(4),
        "gt_bboxes_3d": torch.zeros(num_boxes, 9),
        "gt_labels_3d": torch.zeros(num_boxes, dtype=torch.long),
        "timestamp": 1.0,
        "img_metas": {"ori_shape": [(height, width, 3)] * num_cameras, "dataset_name": "test"},
    }


def test_rigs_of_different_sizes_batch_together() -> None:
    batch = collate_rigs([sample(6), sample(7), sample(5)])

    assert batch["img"].shape[:2] == (3, 7)
    assert batch["intrinsics"].shape == (3, 7, 4, 4)
    assert batch["num_cams"].tolist() == [6, 7, 5]


def test_padded_cameras_are_masked_out() -> None:
    batch = collate_rigs([sample(6), sample(7)])
    assert batch["cam_mask"][0].tolist() == [True] * 6 + [False]
    assert batch["cam_mask"][1].tolist() == [True] * 7


def test_a_camera_without_an_image_is_masked_even_when_present() -> None:
    """The 'zero' missing-camera strategy keeps the slot and clears its flag; the batch mask has
    to combine that with the padding, or a model attends to a black image as if it were real."""
    one = sample(6)
    one["cam_valid_mask"][2] = False
    batch = collate_rigs([one, sample(7)])
    assert batch["cam_mask"][0].tolist() == [True, True, False, True, True, True, False]


def test_padded_projection_matrices_are_identity_not_zero() -> None:
    """A zero 4x4 is singular. A model that inverts or chains one gets NaN for a camera that does
    not exist, which is the hardest kind of bug to trace back to its cause."""
    batch = collate_rigs([sample(6), sample(7)])
    for key in ("intrinsics", "extrinsics", "imu2img"):
        padded = batch[key][0, 6]
        assert torch.equal(padded, torch.eye(4)), key
        assert torch.linalg.det(padded) != 0.0


def test_ground_truth_stays_a_list() -> None:
    """Padding boxes would invent objects, so the batch keeps one entry per sample."""
    batch = collate_rigs([sample(6, num_boxes=3), sample(6, num_boxes=11)])
    assert isinstance(batch["gt_bboxes_3d"], list)
    assert [boxes.shape[0] for boxes in batch["gt_bboxes_3d"]] == [3, 11]


def test_per_camera_metadata_is_extended_to_match_the_tensors() -> None:
    batch = collate_rigs([sample(5), sample(7)])
    assert len(batch["img_metas"][0]["ori_shape"]) == 7
    assert batch["img_metas"][0]["ori_shape"][-1] == batch["img_metas"][0]["ori_shape"][0]


def test_collating_does_not_mutate_the_samples() -> None:
    """The legacy collate extended img_metas in place, so a cached or reused sample grew a longer
    camera list every epoch."""
    first = sample(5)
    collate_rigs([first, sample(7)])
    assert len(first["img_metas"]["ori_shape"]) == 5


def test_same_sized_rigs_need_no_padding() -> None:
    batch = collate_rigs([sample(6), sample(6)])
    assert batch["img"].shape[:2] == (2, 6)
    assert batch["cam_mask"].all()


def test_empty_batch_is_empty() -> None:
    assert collate_rigs([]) == {}
