"""Per-camera 2D / mono-3D annotations, as StreamPETR's dataset expects them.

``CustomNuScenesDataset.get_data_info`` reads ``info['bboxes2d']``, ``['labels2d']``,
``['centers2d']``, ``['depths']`` and ``['bboxes_ignore']`` unconditionally during training, so
an export that only carries 3D boxes cannot train StreamPETR. Those fields are lists with one
entry per camera, in the same order as ``info['cams']``.

They are pure geometry: take the 3D box, move it into the camera frame using that camera's own
pose (cameras fire at slightly different times than the lidar keyframe), project the corners,
and intersect the projected hull with the image rectangle. This mirrors ``get_2d_boxes`` in
mmdetection3d's nuScenes converter, with two differences that follow from what 123D stores:

* visibility bins are a nuScenes annotation attribute that 123D does not carry, so
  ``visibilities`` is exported as empty strings ("unknown") rather than ``'1'``–``'4'``;
* the projection uses 123D's camera model, so a distorted (non-pre-rectified) camera projects
  through its distortion coefficients instead of a bare pinhole matrix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
from py123d.datatypes import BaseCameraMetadata

from py123detection.annotations import DetectionRecord
from py123detection.geometry import YAW_CONVENTIONS, invert_rigid

Array = npt.NDArray[np.float64]

MIN_BOX_SIDE_PX = 1.0
"""Boxes narrower or shorter than this (in pixels) are dropped, matching the mmdet3d converter."""


@dataclass
class Camera2DAnnotations:
    """The 2D / mono-3D annotation arrays for a single camera at a single frame."""

    bboxes2d: Array = field(default_factory=lambda: np.zeros((0, 4), dtype=np.float32))
    """``(M, 4)`` ``[x1, y1, x2, y2]`` image-space boxes."""

    labels2d: npt.NDArray[np.int64] = field(default_factory=lambda: np.zeros((0,), dtype=np.int64))
    """``(M,)`` taxonomy label ids."""

    centers2d: Array = field(default_factory=lambda: np.zeros((0, 2), dtype=np.float32))
    """``(M, 2)`` projected 3D box centers, in pixels."""

    depths: Array = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    """``(M,)`` depth of the projected centers, in metres."""

    bboxes3d_cam: Array = field(default_factory=lambda: np.zeros((0, 7), dtype=np.float32))
    """``(M, 7)`` camera-frame boxes as ``[x, y, z, l, h, w, -yaw]`` (mmdet3d's mono-3D layout)."""

    bboxes_ignore: Array = field(default_factory=lambda: np.zeros((0, 4), dtype=np.float32))
    """``(K, 4)`` ignored boxes. Always empty: 123D has no crowd flag."""

    visibilities: List[str] = field(default_factory=list)
    """``(M,)`` visibility tokens. Always ``""`` — 123D does not carry nuScenes visibility bins."""


def project_records_to_camera(
    records: Sequence[DetectionRecord],
    camera_metadata: BaseCameraMetadata,
    camera_to_global: Array,
    yaw_convention: str = "mmdet3d",
) -> Camera2DAnnotations:
    """Project 3D records into one camera and build its 2D annotation arrays.

    :param records: Global-frame detection records for the frame.
    :param camera_metadata: 123D camera metadata (intrinsics, distortion, image size).
    :param camera_to_global: ``(4, 4)`` camera-to-global transform at the camera's own timestamp.
    :param yaw_convention: Yaw read-out for ``bboxes3d_cam``; see
        :func:`py123detection.geometry.mmdet3d_yaw`. The two conventions differ substantially
        here, because a box lying in a camera frame carries a large roll/pitch.
    :return: The camera's annotation arrays. Empty when nothing projects into the image.
    """
    yaw_from_rotation = YAW_CONVENTIONS[yaw_convention]
    annotations = Camera2DAnnotations()
    if len(records) == 0:
        return annotations

    global_to_camera = invert_rigid(camera_to_global)
    width, height = camera_metadata.width, camera_metadata.height

    bboxes2d: List[List[float]] = []
    labels2d: List[int] = []
    centers2d: List[List[float]] = []
    depths: List[float] = []
    bboxes3d_cam: List[List[float]] = []

    for record in records:
        corners_camera = record.corners_global @ global_to_camera[:3, :3].T + global_to_camera[:3, 3]
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

        center_camera = global_to_camera[:3, :3] @ record.center_global + global_to_camera[:3, 3]
        center_pixels, _, center_depth = camera_metadata.project_to_image(center_camera[None, :])
        depth = float(center_depth[0])
        if depth <= 0:
            continue

        rotation_camera = global_to_camera[:3, :3] @ record.rotation_global
        # mmdet3d's mono-3D convention: dimensions reordered from (l, w, h) to (l, h, w), and the
        # yaw negated because the camera frame's y axis points down.
        length, box_width, box_height = record.size_lwh
        yaw_camera = yaw_from_rotation(rotation_camera)

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
                -yaw_camera,
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


def _clip_to_image(pixels: Array, width: int, height: int) -> Optional[Tuple[float, float, float, float]]:
    """Intersect the convex hull of the projected corners with the image rectangle.

    Clipping the *hull* rather than its bounding rectangle matters: a box that grazes the frame
    edge has a hull whose intersection with the image is far smaller than the clipped bounding
    box of its corners. mmdetection3d's ``post_process_coords`` does the hull intersection, so
    doing anything cheaper here would produce visibly different 2D boxes at frame edges.

    :param pixels: ``(K, 2)`` projected corner coordinates.
    :param width: Image width in pixels.
    :param height: Image height in pixels.
    :return: ``(x1, y1, x2, y2)``, or ``None`` when the box misses the image entirely.
    """
    from shapely.geometry import MultiPoint, box

    hull = MultiPoint([(float(x), float(y)) for x, y in pixels]).convex_hull
    canvas = box(0, 0, float(width), float(height))
    if not hull.intersects(canvas):
        return None
    intersection = hull.intersection(canvas)
    min_x, min_y, max_x, max_y = intersection.bounds
    if max_x <= min_x or max_y <= min_y:
        return None
    return float(min_x), float(min_y), float(max_x), float(max_y)
