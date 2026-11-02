"""Everything both branches share.

``core`` sits underneath :mod:`py123detection.mmcv` (export a 123D dataset to an mmdetection3d
info pickle) and :mod:`py123detection.torch` (read 123D logs straight into a PyTorch pipeline).
It depends on py123d and numpy only — never on torch, mmcv or mmdetection3d — so it imports in
either environment.

What lives here is the part of "3D detection on 123D data" that has nothing to do with which
training stack you picked: what a class is (:mod:`~py123detection.core.taxonomy`), how a box moves
between frames (:mod:`~py123detection.core.geometry`), how a track becomes a velocity
(:mod:`~py123detection.core.velocity`), which boxes a frame contributes
(:mod:`~py123detection.core.annotations`), and which logs you meant
(:mod:`~py123detection.core.sources`).
"""

from py123detection.core.annotations import (
    BOX_LAYOUTS,
    DetectionRecord,
    FrameAnnotations,
    build_frame_annotations,
    extract_detection_records,
)
from py123detection.core.geometry import (
    YAW_CONVENTIONS,
    heading_yaw,
    invert_rigid,
    matrix_to_translation_quaternion,
    mmdet3d_yaw,
    pose_to_matrix,
    rotate_vectors,
    transform_points,
)
from py123detection.core.sources import Source
from py123detection.core.taxonomy import (
    COIN3D_3CLS,
    DEFAULT_123D,
    GENERAL_3CLS,
    NUSCENES_DETECTION,
    TAXONOMIES,
    VEHICLE_1CLS,
    VOID,
    Taxonomy,
    get_taxonomy,
)
from py123detection.core.velocity import TrackVelocityTable, velocities_from_track

__all__ = [
    "BOX_LAYOUTS",
    "COIN3D_3CLS",
    "DEFAULT_123D",
    "DetectionRecord",
    "FrameAnnotations",
    "GENERAL_3CLS",
    "NUSCENES_DETECTION",
    "Source",
    "TAXONOMIES",
    "TrackVelocityTable",
    "Taxonomy",
    "VEHICLE_1CLS",
    "VOID",
    "YAW_CONVENTIONS",
    "build_frame_annotations",
    "extract_detection_records",
    "get_taxonomy",
    "heading_yaw",
    "invert_rigid",
    "matrix_to_translation_quaternion",
    "mmdet3d_yaw",
    "pose_to_matrix",
    "rotate_vectors",
    "transform_points",
    "velocities_from_track",
]
