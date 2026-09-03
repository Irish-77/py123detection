"""Small SE(3) helpers shared by the export code.

123D stores poses as :class:`~py123d.geometry.PoseSE3` (``[x, y, z, qw, qx, qy, qz]``,
scalar-first quaternion). The mmdetection3d nuScenes ``info`` schema stores poses as a
``(translation, rotation)`` pair with a scalar-first quaternion too, so the two representations
map onto each other directly. Everything in between is done with plain 4x4 homogeneous matrices.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import numpy.typing as npt
from py123d.geometry import PoseSE3

Array = npt.NDArray[np.float64]


def pose_to_matrix(pose: PoseSE3) -> Array:
    """Convert a :class:`PoseSE3` to a 4x4 homogeneous transformation matrix.

    :param pose: The pose to convert.
    :return: A ``(4, 4)`` float64 matrix mapping the pose's local frame into its parent frame.
    """
    return np.asarray(pose.transformation_matrix, dtype=np.float64)


def invert_rigid(matrix: Array) -> Array:
    """Invert a 4x4 rigid transform without a general matrix inverse.

    :param matrix: A ``(4, 4)`` rigid transform (rotation + translation, no scale/shear).
    :return: The inverted ``(4, 4)`` transform.
    """
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    inverted = np.eye(4, dtype=np.float64)
    inverted[:3, :3] = rotation.T
    inverted[:3, 3] = -rotation.T @ translation
    return inverted


def matrix_to_translation_quaternion(matrix: Array) -> Tuple[List[float], List[float]]:
    """Split a 4x4 transform into the ``(translation, quaternion)`` pair used by mmdet3d infos.

    :param matrix: A ``(4, 4)`` rigid transform.
    :return: ``(translation, rotation)`` where ``translation`` is ``[x, y, z]`` and ``rotation``
        is a scalar-first quaternion ``[qw, qx, qy, qz]`` — matching the nuScenes convention.
    """
    pose = PoseSE3.from_transformation_matrix(np.asarray(matrix, dtype=np.float64))
    return [float(pose.x), float(pose.y), float(pose.z)], [
        float(pose.qw),
        float(pose.qx),
        float(pose.qy),
        float(pose.qz),
    ]


def transform_points(matrix: Array, points: Array) -> Array:
    """Apply a 4x4 rigid transform to a set of 3D points.

    :param matrix: A ``(4, 4)`` rigid transform.
    :param points: An ``(N, 3)`` array of points in the source frame.
    :return: An ``(N, 3)`` array of points in the target frame.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def rotate_vectors(matrix: Array, vectors: Array) -> Array:
    """Apply only the rotation part of a 4x4 transform to a set of 3D vectors.

    Used for velocities, which are direction-like and must not pick up the translation.

    :param matrix: A ``(4, 4)`` rigid transform.
    :param vectors: An ``(N, 3)`` array of vectors in the source frame.
    :return: An ``(N, 3)`` array of vectors in the target frame.
    """
    vectors = np.asarray(vectors, dtype=np.float64).reshape(-1, 3)
    return vectors @ matrix[:3, :3].T


def yaw_from_rotation(matrix: Array, yaw: float) -> float:
    """Re-express a global-frame yaw angle in the frame defined by ``matrix``.

    The yaw is carried through as a heading *direction* rather than by decomposing Euler angles,
    which keeps the result well-defined when the transform has roll/pitch components.

    :param matrix: A ``(4, 4)`` transform from the source frame into the target frame.
    :param yaw: Heading in the source frame, in radians.
    :return: The heading in the target frame, in radians, wrapped to ``(-pi, pi]``.
    """
    direction = np.array([np.cos(yaw), np.sin(yaw), 0.0], dtype=np.float64)
    rotated = matrix[:3, :3] @ direction
    return float(np.arctan2(rotated[1], rotated[0]))


def heading_yaw(rotation: Array) -> float:
    """Yaw as the heading direction of the frame's x axis.

    This is the convention the official nuScenes devkit uses (``quaternion_yaw`` rotates
    ``[1, 0, 0]`` and takes ``atan2`` of the result).

    :param rotation: A ``(3, 3)`` rotation matrix.
    :return: The yaw in radians.
    """
    return float(np.arctan2(rotation[1, 0], rotation[0, 0]))


def mmdet3d_yaw(rotation: Array) -> float:
    """Yaw as ``pyquaternion.Quaternion.yaw_pitch_roll[0]``, which mmdetection3d's converter uses.

    pyquaternion's decomposition is *not* the textbook z-y-x yaw: it computes
    ``atan2(2(wz - xy), 1 - 2(y^2 + z^2))`` where the standard form has ``+xy``. The two agree
    exactly for an upright box and diverge as roll/pitch grows — a few milliradians in a lidar
    frame, but a large angle in a camera frame, where the box is tipped on its side.

    Reproducing it keeps exported ``gt_boxes`` numerically interchangeable with pickles from
    mmdetection3d's own converter. Use :func:`heading_yaw` instead when you want the geometric
    heading that the nuScenes evaluation assumes.

    :param rotation: A ``(3, 3)`` rotation matrix.
    :return: The yaw in radians.
    """
    from py123d.geometry.utils.rotation_utils import get_quaternion_array_from_rotation_matrix

    qw, qx, qy, qz = get_quaternion_array_from_rotation_matrix(np.asarray(rotation, dtype=np.float64))
    return float(np.arctan2(2.0 * (qw * qz - qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))


YAW_CONVENTIONS = {"mmdet3d": mmdet3d_yaw, "heading": heading_yaw}
"""Selectable yaw conventions, keyed by the value of ``ExportConfig.yaw_convention``."""
