"""SE(3) helpers shared by the export code.

123D poses and the mmdet3d info schema both use scalar-first quaternions, so the two map onto
each other directly. Everything in between is plain 4x4 matrices.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from py123d.geometry import PoseSE3
from py123d.geometry.utils.rotation_utils import get_quaternion_array_from_rotation_matrix


def pose_to_matrix(pose: PoseSE3) -> np.ndarray:
    return np.asarray(pose.transformation_matrix, dtype=np.float64)


def invert_rigid(matrix: np.ndarray) -> np.ndarray:
    """Invert a 4x4 rigid transform (no scale or shear) without a general matrix inverse."""
    rotation = matrix[:3, :3]
    inverted = np.eye(4, dtype=np.float64)
    inverted[:3, :3] = rotation.T
    inverted[:3, 3] = -rotation.T @ matrix[:3, 3]
    return inverted


def matrix_to_translation_quaternion(matrix: np.ndarray) -> Tuple[List[float], List[float]]:
    """Split a 4x4 transform into the ``([x, y, z], [qw, qx, qy, qz])`` pair used by mmdet3d infos."""
    pose = PoseSE3.from_transformation_matrix(np.asarray(matrix, dtype=np.float64))
    translation = [float(pose.x), float(pose.y), float(pose.z)]
    return translation, [float(pose.qw), float(pose.qx), float(pose.qy), float(pose.qz)]


def transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def rotate_vectors(matrix: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float64).reshape(-1, 3)
    return vectors @ matrix[:3, :3].T


def heading_yaw(rotation: np.ndarray) -> float:
    """Yaw as the heading of the frame's x axis, the convention of the nuScenes devkit's ``quaternion_yaw``."""
    return float(np.arctan2(rotation[1, 0], rotation[0, 0]))


def mmdet3d_yaw(rotation: np.ndarray) -> float:
    """Yaw as ``pyquaternion.Quaternion.yaw_pitch_roll[0]``, which mmdetection3d's converter uses.

    pyquaternion computes ``atan2(2(wz - xy), 1 - 2(y^2 + z^2))`` where the textbook z-y-x yaw has
    ``+xy``. Both agree for an upright box and drift apart with roll/pitch: a few milliradians in
    a lidar frame, but a large angle in a camera frame where the box lies on its side. Matching it
    keeps ``gt_boxes`` interchangeable with pickles from mmdetection3d's own converter.
    """
    qw, qx, qy, qz = get_quaternion_array_from_rotation_matrix(np.asarray(rotation, dtype=np.float64))
    return float(np.arctan2(2.0 * (qw * qz - qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))


#: Keyed by ExportConfig.yaw_convention.
YAW_CONVENTIONS = {"mmdet3d": mmdet3d_yaw, "heading": heading_yaw}
