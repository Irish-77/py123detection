"""Per-frame 3D annotations in mmdetection3d's nuScenes layout.

123D stores boxes in the global frame, mmdet3d wants them in the keyframe's lidar frame:
``gt_boxes`` ``(N, 7)`` as ``[x, y, z, l, w, h, yaw]`` around the box center (mmdet3d moves the
origin to the bottom face at load time) and ``gt_velocity`` ``(N, 2)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping, Optional

import numpy as np
from py123d.datatypes import BoxDetectionsSE3

from py123detection.geometry import YAW_CONVENTIONS
from py123detection.taxonomy import Taxonomy

#: "streampetr": [x, y, z, l, w, h, yaw], written and read by StreamPETR and mmdetection3d >= 1.0.
#: "mmdet3d_0.17": [x, y, z, w, l, h, -yaw - pi/2], written by mmdetection3d 0.17's nuScenes
#: converter and fed verbatim into LiDARInstance3DBoxes by its dataset (PETR v1 pins 0.17).
BOX_LAYOUTS = ("streampetr", "mmdet3d_0.17")


@dataclass
class DetectionRecord:
    """One annotated object, in the global frame."""

    class_name: str
    label_id: int
    center_global: np.ndarray
    rotation_global: np.ndarray  # (3, 3)
    size_lwh: np.ndarray  # along the box's own x/y/z
    velocity_global: np.ndarray  # zero when the log carries none
    num_lidar_points: int  # -1 when the log does not record it
    track_token: str
    corners_global: np.ndarray  # (8, 3), reused for the 2D projection


@dataclass
class FrameAnnotations:
    """The 3D annotation block of one info, in the lidar frame."""

    gt_boxes: np.ndarray
    gt_names: np.ndarray
    gt_velocity: np.ndarray
    num_lidar_pts: np.ndarray
    num_radar_pts: np.ndarray
    valid_flag: np.ndarray
    track_tokens: List[str]
    records: List[DetectionRecord]

    def __len__(self) -> int:
        return int(self.gt_boxes.shape[0])


def extract_detection_records(
    detections: Optional[BoxDetectionsSE3],
    taxonomy: Taxonomy,
    velocity_overrides: Optional[Mapping[str, np.ndarray]] = None,
) -> List[DetectionRecord]:
    """Map 123D detections onto taxonomy classes, dropping unmapped ones.

    :param velocity_overrides: Global-frame velocities by track token (see
        :class:`py123detection.velocity.TrackVelocityTable`). Tracks not in it keep the stored velocity.
    """
    records: List[DetectionRecord] = []
    if detections is None:
        return records

    for detection in detections:
        class_name = taxonomy.resolve(detection.attributes.label)
        if class_name is None:
            continue

        box = detection.bounding_box_se3
        center = box.center_se3
        velocity = detection.velocity_3d
        num_lidar_points = detection.attributes.num_lidar_points
        track_token = str(detection.attributes.track_token)

        if velocity is not None:
            velocity_global = np.array([velocity.x, velocity.y, velocity.z], dtype=np.float64)
        else:
            velocity_global = np.zeros(3, dtype=np.float64)
        override = velocity_overrides.get(track_token) if velocity_overrides is not None else None
        if override is not None:
            velocity_global = np.asarray(override, dtype=np.float64).reshape(3)

        records.append(
            DetectionRecord(
                class_name=class_name,
                label_id=taxonomy.label_id(class_name),
                center_global=np.array([center.x, center.y, center.z], dtype=np.float64),
                rotation_global=np.asarray(center.rotation_matrix, dtype=np.float64),
                size_lwh=np.array([box.length, box.width, box.height], dtype=np.float64),
                velocity_global=velocity_global,
                num_lidar_points=int(num_lidar_points) if num_lidar_points is not None else -1,
                track_token=track_token,
                corners_global=np.asarray(box.corners_array, dtype=np.float64),
            )
        )
    return records


def build_frame_annotations(
    records: List[DetectionRecord],
    global_to_lidar: np.ndarray,
    yaw_convention: str = "mmdet3d",
    planar_velocity: bool = True,
    box_layout: str = "streampetr",
) -> FrameAnnotations:
    """Move global-frame records into the lidar frame and build the mmdet3d arrays.

    :param yaw_convention: See :data:`py123detection.geometry.YAW_CONVENTIONS`.
    :param planar_velocity: Zero the vertical velocity before rotating, as mmdetection3d does.
        Otherwise a tilted lidar leaks it into ``(vx, vy)``.
    :param box_layout: One of :data:`BOX_LAYOUTS`.
    """
    if box_layout not in BOX_LAYOUTS:
        raise ValueError(f"box_layout must be one of {BOX_LAYOUTS}, got {box_layout!r}.")
    compute_yaw = YAW_CONVENTIONS[yaw_convention]
    count = len(records)
    gt_boxes = np.zeros((count, 7), dtype=np.float64)
    gt_velocity = np.zeros((count, 2), dtype=np.float64)
    num_lidar_pts = np.zeros((count,), dtype=np.int64)

    rotation = global_to_lidar[:3, :3]
    translation = global_to_lidar[:3, 3]

    for index, record in enumerate(records):
        velocity = record.velocity_global
        if planar_velocity:
            velocity = np.array([velocity[0], velocity[1], 0.0], dtype=np.float64)

        gt_boxes[index, :3] = rotation @ record.center_global + translation
        gt_boxes[index, 3:6] = record.size_lwh
        # Rotate the full box orientation, not just the yaw: lidar extrinsics carry a little
        # roll/pitch, and the yaw read-out is sensitive to it.
        gt_boxes[index, 6] = compute_yaw(rotation @ record.rotation_global)
        gt_velocity[index] = (rotation @ velocity)[:2]
        num_lidar_pts[index] = record.num_lidar_points

    if box_layout == "mmdet3d_0.17":
        gt_boxes[:, 3:6] = gt_boxes[:, [4, 3, 5]]
        gt_boxes[:, 6] = -gt_boxes[:, 6] - np.pi / 2

    # mmdet3d's valid_flag is num_lidar_pts + num_radar_pts > 0. 123D has no radar counts, and
    # logs without lidar counts report -1, which stays valid instead of being dropped.
    return FrameAnnotations(
        gt_boxes=gt_boxes,
        gt_names=np.array([record.class_name for record in records], dtype="<U32"),
        gt_velocity=gt_velocity,
        num_lidar_pts=num_lidar_pts,
        num_radar_pts=np.zeros((count,), dtype=np.int64),
        valid_flag=num_lidar_pts != 0,
        track_tokens=[record.track_token for record in records],
        records=records,
    )
