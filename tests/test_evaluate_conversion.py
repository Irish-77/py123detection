"""The model predicts in the ego frame; the metric measures in the global frame.

That conversion is applied to predictions and ground truth alike, so an error in it moves both
together and the score stays plausible. These tests pin it against values worked out by hand.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch", reason="the torch branch needs torch")

#: Needs the torch extra; see the `torch` marker in pyproject.toml.
pytestmark = pytest.mark.torch

from py123detection.torch.evaluate import as_eval_boxes, frame_token, to_global  # noqa: E402


def pose(yaw: float = 0.0, translation=(0.0, 0.0, 0.0)) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = np.array(
        [[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]]
    )
    matrix[:3, 3] = translation
    return matrix


def box(x=0.0, y=0.0, z=0.0, yaw=0.0, vx=0.0, vy=0.0) -> np.ndarray:
    return np.array([[x, y, z, 2.0, 4.0, 1.5, yaw, vx, vy]])


class TestToGlobal:
    def test_the_identity_pose_changes_nothing(self) -> None:
        original = box(x=10.0, y=5.0, yaw=0.3, vx=1.0)
        assert to_global(original, pose()) == pytest.approx(original)

    def test_translation_moves_the_centre_and_leaves_the_rest(self) -> None:
        moved = to_global(box(x=1.0, y=2.0, z=3.0, yaw=0.5), pose(translation=(10.0, 20.0, 30.0)))
        assert moved[0, :3] == pytest.approx([11.0, 22.0, 33.0])
        assert moved[0, 6] == pytest.approx(0.5), "translating must not rotate the box"
        assert moved[0, 3:6] == pytest.approx([2.0, 4.0, 1.5]), "size is frame independent"

    def test_rotation_turns_the_centre_and_the_heading_together(self) -> None:
        """A box ahead of an ego facing east ends up east of it, pointing east."""
        moved = to_global(box(x=10.0, y=0.0, yaw=0.0), pose(yaw=np.pi / 2))
        assert moved[0, :2] == pytest.approx([0.0, 10.0], abs=1e-9)
        assert moved[0, 6] == pytest.approx(np.pi / 2)

    def test_velocity_rotates_but_does_not_translate(self) -> None:
        """Velocity is a direction: moving the ego 100 m away must not change the speed."""
        far = pose(yaw=np.pi / 2, translation=(100.0, 100.0, 0.0))
        moved = to_global(box(vx=3.0, vy=0.0), far)

        assert moved[0, 7:9] == pytest.approx([0.0, 3.0], abs=1e-9)
        assert np.linalg.norm(moved[0, 7:9]) == pytest.approx(3.0)

    def test_the_conversion_is_invertible(self) -> None:
        original = box(x=7.0, y=-3.0, z=1.0, yaw=0.9, vx=2.0, vy=-1.0)
        there = pose(yaw=0.7, translation=(5.0, -2.0, 0.5))

        round_trip = to_global(to_global(original, there), np.linalg.inv(there))
        assert round_trip[:, :6] == pytest.approx(original[:, :6])
        assert round_trip[:, 7:] == pytest.approx(original[:, 7:])
        assert np.cos(round_trip[0, 6]) == pytest.approx(np.cos(original[0, 6]))

    def test_an_empty_frame_survives(self) -> None:
        """Frames with no boxes are kept, so the conversion has to accept them."""
        assert len(to_global(np.zeros((0, 9)), pose(yaw=1.0))) == 0


class TestEvalBoxes:
    def test_yaw_becomes_a_quaternion_about_the_vertical_axis(self) -> None:
        converted = as_eval_boxes(box(yaw=np.pi / 2), ["vehicle"], "token")[0]
        w, x, y, z = converted.rotation

        assert (x, y) == (0.0, 0.0), "a driving box only ever yaws"
        assert w == pytest.approx(np.cos(np.pi / 4))
        assert z == pytest.approx(np.sin(np.pi / 4))
        assert w**2 + x**2 + y**2 + z**2 == pytest.approx(1.0)

    def test_size_keeps_the_devkit_width_length_height_order(self) -> None:
        converted = as_eval_boxes(box(), ["vehicle"], "token")[0]
        assert converted.size == (2.0, 4.0, 1.5)

    def test_ground_truth_carries_no_score(self) -> None:
        """A score on ground truth would let it be ranked, which the metric must never do."""
        truth = as_eval_boxes(box(), ["vehicle"], "token")[0]
        predicted = as_eval_boxes(box(), ["vehicle"], "token", np.array([0.8]))[0]

        assert truth.detection_score == -1.0
        assert predicted.detection_score == pytest.approx(0.8)


class TestFrameToken:
    def test_frames_of_one_scene_differ(self) -> None:
        assert frame_token({"scene_uuid": "abc", "frame_idx": 0}) != frame_token(
            {"scene_uuid": "abc", "frame_idx": 1}
        )

    def test_the_same_index_in_different_scenes_differs(self) -> None:
        """Two datasets both numbering frames from zero must not collide."""
        assert frame_token({"scene_uuid": "abc", "frame_idx": 0}) != frame_token(
            {"scene_uuid": "xyz", "frame_idx": 0}
        )
