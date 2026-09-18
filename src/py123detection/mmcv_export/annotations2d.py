"""Per-camera 2D / mono-3D annotations, which StreamPETR's dataset reads during training.

This mirrors ``get_2d_boxes`` in mmdetection3d's nuScenes converter: move each box into the
camera frame with that camera's own pose, project the corners and intersect their hull with the
image. Two differences follow from what 123D stores: ``visibilities`` are empty strings (123D has
no nuScenes visibility bins), and distorted cameras project through 123D's camera model instead
of a bare pinhole matrix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
from py123d.datatypes import BaseCameraMetadata

from py123detection.annotations import DetectionRecord
from py123detection.geometry import YAW_CONVENTIONS, invert_rigid

# Same minimum box side as the mmdet3d converter.
MIN_BOX_SIDE_PX = 1.0


@dataclass
class Camera2DAnnotations:
    """2D / mono-3D annotation arrays of one camera at one frame."""

    bboxes2d: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), dtype=np.float32))  # [x1, y1, x2, y2]
    labels2d: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.int64))
    centers2d: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.float32))
    depths: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    # [x, y, z, l, h, w, -yaw], mmdet3d's mono-3D layout
    bboxes3d_cam: np.ndarray = field(default_factory=lambda: np.zeros((0, 7), dtype=np.float32))
    # always empty, 123D has no crowd flag
    bboxes_ignore: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), dtype=np.float32))
    visibilities: List[str] = field(default_factory=list)


def project_records_to_camera(
    records: Sequence[DetectionRecord],
    camera_metadata: BaseCameraMetadata,
    camera_to_global: np.ndarray,
    yaw_convention: str = "mmdet3d",
) -> Camera2DAnnotations:
    """Project a frame's global-frame records into one camera.

    :param camera_to_global: ``(4, 4)`` camera pose at the camera's own timestamp.
    :param yaw_convention: Yaw read-out for ``bboxes3d_cam``. The conventions differ a lot here,
        because a box seen from a camera frame carries a large roll/pitch.
    """
    compute_yaw = YAW_CONVENTIONS[yaw_convention]
    annotations = Camera2DAnnotations()
    if len(records) == 0:
        return annotations

    global_to_camera = invert_rigid(camera_to_global)
    rotation, translation = global_to_camera[:3, :3], global_to_camera[:3, 3]
    width, height = camera_metadata.width, camera_metadata.height

    bboxes2d, labels2d, centers2d, depths, bboxes3d_cam = [], [], [], [], []
    for record in records:
        corners_camera = record.corners_global @ rotation.T + translation
        in_front = corners_camera[:, 2] > 0
        if not in_front.any():
            continue

        pixels, _, _ = camera_metadata.project_to_image(corners_camera[in_front])
        box2d = _clip_to_image(pixels, width, height)
        if box2d is None:
            continue
        x1, y1, x2, y2 = box2d
        if (x2 - x1) < MIN_BOX_SIDE_PX or (y2 - y1) < MIN_BOX_SIDE_PX:
            continue

        center_camera = rotation @ record.center_global + translation
        center_pixels, _, center_depth = camera_metadata.project_to_image(center_camera[None, :])
        depth = float(center_depth[0])
        if depth <= 0:
            continue

        # Mono-3D convention: (l, w, h) becomes (l, h, w), and the yaw is negated because the
        # camera's y axis points down.
        length, box_width, box_height = record.size_lwh
        yaw = compute_yaw(rotation @ record.rotation_global)

        bboxes2d.append([x1, y1, x2, y2])
        labels2d.append(record.label_id)
        centers2d.append([float(center_pixels[0, 0]), float(center_pixels[0, 1])])
        depths.append(depth)
        bboxes3d_cam.append(
            [
                float(center_camera[0]),
                float(center_camera[1]),
                float(center_camera[2]),
                float(length),
                float(box_height),
                float(box_width),
                -yaw,
            ]
        )

    if bboxes2d:
        annotations.bboxes2d = np.array(bboxes2d, dtype=np.float32)
        annotations.labels2d = np.array(labels2d, dtype=np.int64)
        annotations.centers2d = np.array(centers2d, dtype=np.float32)
        annotations.depths = np.array(depths, dtype=np.float32)
        annotations.bboxes3d_cam = np.array(bboxes3d_cam, dtype=np.float32)
        annotations.visibilities = [""] * len(bboxes2d)
    return annotations


def _clip_to_image(pixels: np.ndarray, width: int, height: int) -> Optional[Tuple[float, float, float, float]]:
    """Bounds of the projected corners' convex hull intersected with the image, or ``None``.

    Clipping the hull rather than its bounding box matters for boxes grazing the image edge, and
    is what mmdetection3d's ``post_process_coords`` does.
    """
    from shapely.geometry import MultiPoint, box

    hull = MultiPoint([(float(x), float(y)) for x, y in pixels]).convex_hull
    canvas = box(0, 0, float(width), float(height))
    if not hull.intersects(canvas):
        return None
    min_x, min_y, max_x, max_y = hull.intersection(canvas).bounds
    if max_x <= min_x or max_y <= min_y:
        return None
    return float(min_x), float(min_y), float(max_x), float(max_y)
