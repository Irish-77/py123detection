"""Extracting per-frame 3D annotations from a 123D scene into mmdetection3d's layout.

mmdet3d's nuScenes info schema stores boxes in the **lidar frame** of the keyframe:
``gt_boxes`` is ``(N, 7)`` as ``[x, y, z, dx, dy, dz, yaw]`` with ``(dx, dy, dz)`` the box's
``(length, width, height)``, the position being the box *center* (mmdet3d re-anchors it to the
bottom face at load time via ``origin=(0.5, 0.5, 0.5)``), and ``gt_velocity`` the ``(N, 2)``
lidar-frame ground-plane velocity.

123D stores all of that in the **global frame**, so the work here is one rigid transform per
frame plus the taxonomy lookup.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import numpy.typing as npt
from py123d.datatypes import BoxDetectionsSE3

from py123detection.geometry import YAW_CONVENTIONS
from py123detection.taxonomy import Taxonomy

Array = npt.NDArray[np.float64]


@dataclass
class DetectionRecord:
    """One annotated object, kept in the global frame until it is projected into a target frame."""

    class_name: str
    """Taxonomy class name (an entry of :attr:`Taxonomy.class_names`)."""

    label_id: int
    """Index of :attr:`class_name` in the taxonomy."""

    center_global: Array
    """``(3,)`` box center in the global frame."""

    rotation_global: Array
    """``(3, 3)`` box orientation in the global frame."""

    size_lwh: Array
    """``(3,)`` box extent as ``(length, width, height)``, i.e. along the box's own x/y/z."""

    velocity_global: Array
    """``(3,)`` velocity in the global frame. Zero when the source log carries none."""

    num_lidar_points: int
    """Lidar points inside the box, or ``-1`` when the source log does not record it."""

    track_token: str
    """Instance identifier, consistent across frames of a log."""

    corners_global: Array
    """``(8, 3)`` box corners in the global frame, reused for 2D reprojection."""


@dataclass
class FrameAnnotations:
    """The 3D annotation block of one exported info, already in the lidar frame."""

    gt_boxes: Array
    """``(N, 7)`` ``[x, y, z, dx, dy, dz, yaw]`` in the lidar frame."""

    gt_names: npt.NDArray[np.str_]
    """``(N,)`` taxonomy class names."""

    gt_velocity: Array
    """``(N, 2)`` lidar-frame ``(vx, vy)``."""

    num_lidar_pts: npt.NDArray[np.int64]
    """``(N,)`` lidar point counts."""

    num_radar_pts: npt.NDArray[np.int64]
    """``(N,)`` radar point counts. Always zero — 123D does not carry this field."""

    valid_flag: npt.NDArray[np.bool_]
    """``(N,)`` mmdet3d's ``use_valid_flag`` mask."""

    track_tokens: List[str]
    """``(N,)`` instance identifiers, kept as an export extra for tracking work."""

    records: List[DetectionRecord]
    """The underlying global-frame records, reused by the 2D reprojection."""

    def __len__(self) -> int:
        return int(self.gt_boxes.shape[0])


def extract_detection_records(
    detections: Optional[BoxDetectionsSE3],
    taxonomy: Taxonomy,
) -> List[DetectionRecord]:
    """Map 123D box detections onto taxonomy classes, dropping anything unmapped.

    :param detections: The 123D detections at one iteration, or ``None``.
    :param taxonomy: Taxonomy deciding class names and label ids.
    :return: One record per kept detection.
    """
    records: List[DetectionRecord] = []
    if detections is None:
        return records

    for detection in detections:
        class_name = taxonomy.resolve(detection.attributes.label)
        if class_name is None:
            continue

        bounding_box = detection.bounding_box_se3
        center = bounding_box.center_se3
        velocity = detection.velocity_3d
        num_lidar_points = detection.attributes.num_lidar_points

        records.append(
            DetectionRecord(
                class_name=class_name,
                label_id=taxonomy.label_id(class_name),
                center_global=np.array([center.x, center.y, center.z], dtype=np.float64),
                rotation_global=np.asarray(center.rotation_matrix, dtype=np.float64),
                size_lwh=np.array(
                    [bounding_box.length, bounding_box.width, bounding_box.height],
                    dtype=np.float64,
                ),
                velocity_global=(
                    np.array([velocity.x, velocity.y, velocity.z], dtype=np.float64)
                    if velocity is not None
                    else np.zeros(3, dtype=np.float64)
                ),
                num_lidar_points=int(num_lidar_points) if num_lidar_points is not None else -1,
                track_token=str(detection.attributes.track_token),
                corners_global=np.asarray(bounding_box.corners_array, dtype=np.float64),
            )
        )
    return records


def build_frame_annotations(
    records: List[DetectionRecord],
    global_to_lidar: Array,
    yaw_convention: str = "mmdet3d",
    planar_velocity: bool = True,
) -> FrameAnnotations:
    """Project global-frame records into the lidar frame and assemble the mmdet3d arrays.

    :param records: Records from :func:`extract_detection_records`.
    :param global_to_lidar: ``(4, 4)`` transform from the global frame into the keyframe lidar frame.
    :param yaw_convention: ``"mmdet3d"`` reproduces the yaw mmdetection3d's own converter writes;
        ``"heading"`` uses the geometric heading the nuScenes devkit assumes. See
        :func:`py123detection.geometry.mmdet3d_yaw`.
    :param planar_velocity: Zero the global vertical velocity before rotating into the lidar
        frame, as mmdetection3d does. Leaving it set keeps exports interchangeable; clearing it
        keeps the true 3D velocity, which leaks the vertical component into the lidar-frame
        ``(vx, vy)`` whenever the lidar is tilted.
    :return: The assembled annotation block.
    """
    yaw_from_rotation = YAW_CONVENTIONS[yaw_convention]
    count = len(records)
    gt_boxes = np.zeros((count, 7), dtype=np.float64)
    gt_velocity = np.zeros((count, 2), dtype=np.float64)
    gt_names: List[str] = []
    num_lidar_pts = np.zeros((count,), dtype=np.int64)
    track_tokens: List[str] = []

    rotation = global_to_lidar[:3, :3]
    translation = global_to_lidar[:3, 3]

    for index, record in enumerate(records):
        center_lidar = rotation @ record.center_global + translation
        # Compose the full rotation rather than rotating the yaw alone: lidar-to-ego extrinsics
        # generally carry a little roll/pitch, which the yaw read-out below is sensitive to.
        box_rotation_lidar = rotation @ record.rotation_global
        yaw = yaw_from_rotation(box_rotation_lidar)

        velocity_global = record.velocity_global
        if planar_velocity:
            velocity_global = np.array([velocity_global[0], velocity_global[1], 0.0], dtype=np.float64)

        gt_boxes[index, :3] = center_lidar
        gt_boxes[index, 3:6] = record.size_lwh
        gt_boxes[index, 6] = yaw
        gt_velocity[index] = (rotation @ velocity_global)[:2]
        gt_names.append(record.class_name)
        num_lidar_pts[index] = record.num_lidar_points
        track_tokens.append(record.track_token)

    # 123D does not carry radar point counts, so mmdet3d's `valid_flag`
    # (num_lidar_pts + num_radar_pts > 0) reduces to the lidar term alone. Logs without point
    # counts report -1, which we treat as "unknown" and keep valid rather than silently drop.
    num_radar_pts = np.zeros((count,), dtype=np.int64)
    valid_flag = (num_lidar_pts > 0) | (num_lidar_pts < 0)

    return FrameAnnotations(
        gt_boxes=gt_boxes,
        gt_names=np.array(gt_names, dtype="<U32") if gt_names else np.array([], dtype="<U32"),
        gt_velocity=gt_velocity,
        num_lidar_pts=num_lidar_pts,
        num_radar_pts=num_radar_pts,
        valid_flag=valid_flag,
        track_tokens=track_tokens,
        records=records,
    )
