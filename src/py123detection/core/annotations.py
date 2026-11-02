"""Per-frame 3D annotations in mmdetection3d's nuScenes layout.

123D stores boxes in the global frame, mmdet3d wants them in the keyframe's lidar frame:
``gt_boxes`` ``(N, 7)`` as ``[x, y, z, l, w, h, yaw]`` around the box center (mmdet3d moves the
origin to the bottom face at load time) and ``gt_velocity`` ``(N, 2)``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Mapping, Optional

import numpy as np
from py123d.datatypes import BoxDetectionsSE3

from py123detection.core.geometry import YAW_CONVENTIONS, heading_yaw
from py123detection.core.taxonomy import Taxonomy

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



def _bev_polygon(corners_global: np.ndarray):
    """Bird's-eye-view footprint of a box, as a shapely polygon."""
    from shapely.geometry import MultiPoint

    return MultiPoint(np.asarray(corners_global)[:, :2]).convex_hull


def _bev_iou(a: DetectionRecord, b: DetectionRecord) -> float:
    """Intersection over union of two boxes' bird's-eye-view footprints."""
    polygon_a, polygon_b = _bev_polygon(a.corners_global), _bev_polygon(b.corners_global)
    union = polygon_a.union(polygon_b).area
    if union <= 0.0:
        return 0.0
    return float(polygon_a.intersection(polygon_b).area / union)


def _merged_record(primary: DetectionRecord, secondary: DetectionRecord, target: str) -> DetectionRecord:
    """The box around both, measured in the primary's heading frame.

    Measuring in the primary's frame rather than along world axes keeps a diagonally parked pair
    from inflating into a much larger box. The frame is yaw-only, and so is the merged box's
    stored rotation: both branches emit ``[x, y, z, l, w, h, yaw]``, so a box measured with the
    primary's roll and pitch would no longer contain its own corners once those are dropped.
    """
    yaw = heading_yaw(primary.rotation_global)
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    rotation = np.array(
        [[cos_yaw, -sin_yaw, 0.0], [sin_yaw, cos_yaw, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )

    corners = np.concatenate([primary.corners_global, secondary.corners_global], axis=0)
    local = (corners - primary.center_global) @ rotation
    lower, upper = local.min(axis=0), local.max(axis=0)

    size_lwh = upper - lower
    center_global = rotation @ ((lower + upper) / 2.0) + primary.center_global

    num_points = primary.num_lidar_points
    if primary.num_lidar_points >= 0 and secondary.num_lidar_points >= 0:
        num_points = primary.num_lidar_points + secondary.num_lidar_points

    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=np.float64)
    corners_global = (signs * (size_lwh / 2.0)) @ rotation.T + center_global

    return DetectionRecord(
        class_name=target,
        label_id=-1,
        center_global=center_global,
        rotation_global=rotation,
        size_lwh=size_lwh,
        velocity_global=primary.velocity_global,
        num_lidar_points=int(num_points),
        track_token=primary.track_token,
        corners_global=corners_global,
    )


def apply_merge_rules(records: List[DetectionRecord], taxonomy: Taxonomy) -> List[DetectionRecord]:
    """Combine riders with their vehicles, and drop what is left merge-only.

    Each secondary takes the free primary of the same rule it overlaps most, so a scene with two
    cyclists side by side pairs each rider with their own bicycle rather than both with one.
    Records not named by any rule pass through first and in order; merged ones follow.
    """
    if not taxonomy.merge_rules:
        return records

    untouched: List[DetectionRecord] = []
    by_rule: Dict[int, Dict[str, List[DetectionRecord]]] = {
        index: {"primary": [], "secondary": []} for index in range(len(taxonomy.merge_rules))
    }

    for record in records:
        for index, rule in enumerate(taxonomy.merge_rules):
            if record.class_name == rule.primary:
                by_rule[index]["primary"].append(record)
                break
            if record.class_name in rule.secondaries:
                by_rule[index]["secondary"].append(record)
                break
        else:
            untouched.append(record)

    merged: List[DetectionRecord] = []
    for index, rule in enumerate(taxonomy.merge_rules):
        primaries = by_rule[index]["primary"]
        paired: set = set()

        for secondary in by_rule[index]["secondary"]:
            best_index, best_iou = -1, 0.0
            for primary_index, primary in enumerate(primaries):
                if primary_index in paired:
                    continue
                iou = _bev_iou(secondary, primary)
                if iou > best_iou:
                    best_index, best_iou = primary_index, iou
            if best_index >= 0:
                paired.add(best_index)
                merged.append(_merged_record(primaries[best_index], secondary, rule.target))
            # A secondary that paired with nothing is dropped: a rider with no vehicle.

        if not rule.discard_unmerged_base:
            for primary_index, primary in enumerate(primaries):
                if primary_index not in paired:
                    merged.append(replace(primary, class_name=rule.target))

    return untouched + merged


def extract_detection_records(
    detections: Optional[BoxDetectionsSE3],
    taxonomy: Taxonomy,
    velocity_overrides: Optional[Mapping[str, np.ndarray]] = None,
) -> List[DetectionRecord]:
    """Map 123D detections onto taxonomy classes, dropping unmapped ones.

    :param velocity_overrides: Global-frame velocities by track token (see
        :class:`py123detection.core.velocity.TrackVelocityTable`). Tracks not in it keep the stored velocity.
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
                label_id=-1,  # assigned below: a merge can still change which class this is
                center_global=np.array([center.x, center.y, center.z], dtype=np.float64),
                rotation_global=np.asarray(center.rotation_matrix, dtype=np.float64),
                size_lwh=np.array([box.length, box.width, box.height], dtype=np.float64),
                velocity_global=velocity_global,
                num_lidar_points=int(num_lidar_points) if num_lidar_points is not None else -1,
                track_token=track_token,
                corners_global=np.asarray(box.corners_array, dtype=np.float64),
            )
        )

    records = apply_merge_rules(records, taxonomy)
    return [replace(record, label_id=taxonomy.label_id(record.class_name))
            for record in records if taxonomy.is_final(record.class_name)]


def build_frame_annotations(
    records: List[DetectionRecord],
    global_to_lidar: np.ndarray,
    yaw_convention: str = "mmdet3d",
    planar_velocity: bool = True,
    box_layout: str = "streampetr",
) -> FrameAnnotations:
    """Move global-frame records into the lidar frame and build the mmdet3d arrays.

    :param yaw_convention: See :data:`py123detection.core.geometry.YAW_CONVENTIONS`.
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
