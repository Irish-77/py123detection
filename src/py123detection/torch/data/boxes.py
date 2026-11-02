"""A frame's ground truth: 123D box detections as filtered ego-frame boxes.

Boxes are read before any image, because a frame left without boxes is rejected and decoding six
or seven JPEGs to find that out is the expensive way round.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from py123detection.core.annotations import DetectionRecord, extract_detection_records
from py123detection.core.lidar_points import LidarPointCounts
from py123detection.core.sources import Source
from py123detection.core.taxonomy import Taxonomy
from py123detection.core.velocity import TrackVelocityTable
from py123detection.core.visibility import VISIBILITY_MODES, rig_visibility

#: Columns of ``gt_bboxes_3d``.
BOX_COLUMNS = ("x", "y", "z", "w", "l", "h", "yaw", "vx", "vy")


@dataclass(frozen=True)
class BoxFilters:
    """Which ground-truth boxes a frame contributes.

    Every one of these removes annotations a camera-only model cannot be expected to find, so
    leaving them off does not make a benchmark harder, it makes it meaningless.
    """

    drop_boxes_without_lidar_points: bool = True
    """Drop boxes the log records as having zero lidar returns, i.e. fully occluded objects.
    A count of ``-1`` means the dataset does not record one and is always kept."""

    lidar_point_cache: Optional[str] = None
    """Take point counts from this cache instead, where it has them — see
    :mod:`py123detection.core.lidar_points`. Waymo's 123D logs record no counts, so without it
    no Waymo box is ever dropped as unseen."""

    drop_frames_without_boxes: bool = True
    """Skip a frame left with no box at all, rather than training on an empty target."""

    pc_range: Optional[Tuple[float, float, float, float, float, float]] = None
    """``(x_min, y_min, z_min, x_max, y_max, z_max)`` in the ego frame. Boxes outside are dropped.
    The legacy implementation accepted the z bounds and applied only x and y."""

    visibility: Optional[str] = None
    """``"center"`` or ``"corners"``: drop boxes no camera of the rig could see. A rig covering the
    full circle makes this a no-op."""

    visibility_margin_deg: float = 0.0
    """Widen each camera's field of view before testing, so an object on the edge is kept."""

    def __post_init__(self) -> None:
        if self.visibility is not None and self.visibility not in VISIBILITY_MODES:
            raise ValueError(f"visibility must be one of {VISIBILITY_MODES}, got {self.visibility!r}.")


@dataclass
class Boxes:
    """One frame's boxes, in the ego frame and not yet augmented."""

    boxes: np.ndarray
    """``(M, 9)`` float64 rows of :data:`BOX_COLUMNS`."""

    labels: np.ndarray
    """``(M,)`` int64 indices into the taxonomy's class names."""

    class_names: List[str]

    def __len__(self) -> int:
        return int(self.boxes.shape[0])


class BoxLoader:
    """Reads, maps, filters and converts one frame's boxes.

    :param source: Where the logs came from; velocities are derived from the same logs read at
        their native frame rate.
    :param scenes: The resolved logs. The first one's rig geometry defines camera visibility.
    :param cameras: The rig's camera names, for the visibility filter.
    :param compute_velocity: Derive velocity from track positions over time, for datasets that
        store none, such as Argoverse 2 and Waymo.
    """

    def __init__(
        self,
        source: Source,
        scenes: Sequence[Any],
        taxonomy: Taxonomy,
        filters: BoxFilters,
        cameras: Sequence[str],
        compute_velocity: bool = False,
    ) -> None:
        self.source = source
        self.taxonomy = taxonomy
        self.filters = filters
        self.compute_velocity = compute_velocity

        self._visibility = None
        if filters.visibility is not None and scenes:
            self._visibility = rig_visibility(scenes[0], cameras, margin_deg=filters.visibility_margin_deg)

        self._velocity_cache: Dict[str, TrackVelocityTable] = {}
        self._native_scenes: Optional[Dict[str, Any]] = None

        self._point_counts = None
        if filters.lidar_point_cache is not None:
            data_root = source.data_root if source.data_root is not None else os.environ.get("PY123D_DATA_ROOT", "")
            self._point_counts = LidarPointCounts(filters.lidar_point_cache, Path(data_root) / "logs")

    def load(self, scene, iteration: int, world_to_ego: np.ndarray) -> Boxes:
        detections = scene.get_box_detections_se3_at_iteration(iteration)
        overrides = None
        if self.compute_velocity:
            timestamp_us = int(scene.get_timestamp_at_iteration(iteration).time_us)
            overrides = self._velocities_for(scene).at(timestamp_us)

        records = extract_detection_records(detections, self.taxonomy, velocity_overrides=overrides)
        if self._point_counts is not None and records:
            counts = self._point_counts.for_frame(scene, iteration)
            records = [
                replace(record, num_lidar_points=counts[record.track_token]) if record.track_token in counts else record
                for record in records
            ]
        records = [record for record in records if self._keep(record, world_to_ego)]
        if not records:
            return Boxes(np.zeros((0, len(BOX_COLUMNS))), np.zeros((0,), np.int64), [])

        boxes = np.stack([self._to_ego_box(record, world_to_ego) for record in records])
        finite = np.isfinite(boxes).all(axis=1)
        records = [record for record, keep in zip(records, finite) if keep]
        return Boxes(
            boxes=boxes[finite],
            labels=np.array([record.label_id for record in records], dtype=np.int64),
            class_names=[record.class_name for record in records],
        )

    def _velocities_for(self, scene) -> TrackVelocityTable:
        """Track velocities for one log, derived once.

        Built from the log at its native frame rate, not from this dataset's subsampled view: a
        difference taken across a half-second gap is not the velocity at either end, and the
        estimator refuses gaps that long rather than guessing.
        """
        table = self._velocity_cache.get(scene.scene_uuid)
        if table is not None:
            return table

        if self._native_scenes is None:
            self._native_scenes = {
                native.scene_uuid: native for native in self.source.load_scenes(native_rate=True)
            }
        native = self._native_scenes.get(scene.scene_uuid, scene)
        table = TrackVelocityTable.from_scene(native)
        self._velocity_cache[scene.scene_uuid] = table
        return table

    def _keep(self, record: DetectionRecord, world_to_ego: np.ndarray) -> bool:
        if self.filters.drop_boxes_without_lidar_points and record.num_lidar_points == 0:
            return False

        center_ego = world_to_ego[:3, :3] @ record.center_global + world_to_ego[:3, 3]

        pc_range = self.filters.pc_range
        if pc_range is not None:
            lower, upper = np.asarray(pc_range[:3]), np.asarray(pc_range[3:])
            if np.any(center_ego < lower) or np.any(center_ego > upper):
                return False

        if self._visibility is not None:
            corners_ego = record.corners_global @ world_to_ego[:3, :3].T + world_to_ego[:3, 3]
            if not self._visibility.sees(self.filters.visibility, center_ego, corners_ego):
                return False

        return True

    @staticmethod
    def _to_ego_box(record: DetectionRecord, world_to_ego: np.ndarray) -> np.ndarray:
        """One ``[x, y, z, w, l, h, yaw, vx, vy]`` row in the ego frame."""
        rotation = world_to_ego[:3, :3]
        center = rotation @ record.center_global + world_to_ego[:3, 3]

        heading = rotation @ np.array([record.rotation_global[0, 0], record.rotation_global[1, 0], 0.0])
        yaw = float(np.arctan2(heading[1], heading[0]))

        velocity = rotation @ record.velocity_global
        length, width, height = record.size_lwh

        return np.array(
            [center[0], center[1], center[2], width, length, height, yaw, velocity[0], velocity[1]],
            dtype=np.float64,
        )
