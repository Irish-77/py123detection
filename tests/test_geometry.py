"""Unit tests for the SE(3) helpers and the two yaw conventions."""

from __future__ import annotations

import numpy as np
import pytest
from py123d.geometry import PoseSE3
from pyquaternion import Quaternion

from py123detection.geometry import (
    heading_yaw,
    invert_rigid,
    matrix_to_translation_quaternion,
    mmdet3d_yaw,
    pose_to_matrix,
    rotate_vectors,
    transform_points,
)


def random_transform(seed: int) -> np.ndarray:
    """Build a reproducible random rigid transform."""
    rng = np.random.default_rng(seed)
    quaternion = Quaternion(np.asarray(rng.normal(size=4))).normalised
    matrix = np.eye(4)
    matrix[:3, :3] = quaternion.rotation_matrix
    matrix[:3, 3] = rng.normal(scale=10.0, size=3)
    return matrix


@pytest.mark.parametrize("seed", range(5))
def test_invert_rigid_is_the_inverse(seed: int) -> None:
    matrix = random_transform(seed)
    np.testing.assert_allclose(invert_rigid(matrix) @ matrix, np.eye(4), atol=1e-12)
    np.testing.assert_allclose(matrix @ invert_rigid(matrix), np.eye(4), atol=1e-12)


@pytest.mark.parametrize("seed", range(5))
def test_invert_rigid_matches_general_inverse(seed: int) -> None:
    matrix = random_transform(seed)
    np.testing.assert_allclose(invert_rigid(matrix), np.linalg.inv(matrix), atol=1e-10)


@pytest.mark.parametrize("seed", range(5))
def test_pose_matrix_round_trip(seed: int) -> None:
    matrix = random_transform(seed)
    translation, rotation = matrix_to_translation_quaternion(matrix)
    rebuilt = pose_to_matrix(
        PoseSE3.from_R_t(rotation=np.asarray(rotation), translation=np.asarray(translation))
    )
    np.testing.assert_allclose(rebuilt, matrix, atol=1e-12)


def test_matrix_to_translation_quaternion_is_scalar_first() -> None:
    """The exported quaternion must be [w, x, y, z], matching the nuScenes info convention."""
    quaternion = Quaternion(axis=[0, 0, 1], radians=0.5)
    matrix = np.eye(4)
    matrix[:3, :3] = quaternion.rotation_matrix
    _, rotation = matrix_to_translation_quaternion(matrix)
    np.testing.assert_allclose(np.abs(np.asarray(rotation)), np.abs(quaternion.elements), atol=1e-12)
    assert abs(rotation[0]) == pytest.approx(np.cos(0.25), abs=1e-12)


@pytest.mark.parametrize("seed", range(5))
def test_transform_points_and_rotate_vectors_differ_by_translation(seed: int) -> None:
    matrix = random_transform(seed)
    points = np.random.default_rng(seed).normal(size=(7, 3))
    np.testing.assert_allclose(
        transform_points(matrix, points) - rotate_vectors(matrix, points),
        np.tile(matrix[:3, 3], (7, 1)),
        atol=1e-12,
    )


@pytest.mark.parametrize("angle", [-3.0, -0.7, 0.0, 0.7, 3.0])
def test_yaw_conventions_agree_for_an_upright_rotation(angle: float) -> None:
    """With no roll or pitch the two conventions are the same angle."""
    rotation = Quaternion(axis=[0, 0, 1], radians=angle).rotation_matrix
    assert heading_yaw(rotation) == pytest.approx(angle, abs=1e-12)
    assert mmdet3d_yaw(rotation) == pytest.approx(angle, abs=1e-12)


@pytest.mark.parametrize("seed", range(8))
def test_mmdet3d_yaw_matches_pyquaternion(seed: int) -> None:
    """mmdet3d's converter reads yaw via pyquaternion, so we must reproduce it exactly."""
    rng = np.random.default_rng(seed)
    quaternion = Quaternion(np.asarray(rng.normal(size=4))).normalised
    assert mmdet3d_yaw(quaternion.rotation_matrix) == pytest.approx(quaternion.yaw_pitch_roll[0], abs=1e-12)


def test_yaw_conventions_diverge_when_tilted() -> None:
    """The conventions are genuinely different once roll/pitch is present — that is why it is an option."""
    rotation = (Quaternion(axis=[0, 0, 1], radians=0.9) * Quaternion(axis=[1, 0, 0], radians=1.2)).rotation_matrix
    assert abs(heading_yaw(rotation) - mmdet3d_yaw(rotation)) > 1e-3


@pytest.mark.parametrize("seed", range(5))
def test_yaw_is_sign_invariant_in_the_quaternion(seed: int) -> None:
    """q and -q are the same rotation and must give the same yaw."""
    rng = np.random.default_rng(seed)
    quaternion = Quaternion(np.asarray(rng.normal(size=4))).normalised
    negated = Quaternion(-quaternion.elements)
    assert mmdet3d_yaw(quaternion.rotation_matrix) == pytest.approx(mmdet3d_yaw(negated.rotation_matrix), abs=1e-12)
