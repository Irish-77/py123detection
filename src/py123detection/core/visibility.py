"""Which ground-truth boxes a rig's cameras can actually see.

A camera-only detector is scored on every annotated object, including the ones behind it. On a
360-degree rig like nuScenes' that is harmless. On Waymo's five forward-facing cameras, which
between them cover about 230 degrees, roughly a third of the annotations are objects no camera
observed — and counting them makes recall look worse than the model is, for reasons that have
nothing to do with the model.

:class:`RigVisibility` reads the rig geometry once and then answers, per box, whether any camera
could have seen it. Two tests, in increasing cost:

* ``center`` compares the box centre's bearing against each camera's horizontal field of view. It
  ignores elevation and image bounds, and is a good default.
* ``corners`` projects all eight corners and asks whether any lands inside an image. Stricter, and
  the right choice when the vertical field of view is narrow.

A rig covering the full circle short-circuits: every box is visible, and the filter costs nothing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

#: Coverage at or above this counts as the full circle, in degrees. Slightly under 360 because
#: rigs are built to overlap, not to tile exactly.
FULL_COVERAGE_DEG = 355.0

#: Minimum depth for a corner to be in front of a camera, in metres.
MIN_DEPTH_M = 0.1

VISIBILITY_MODES = ("center", "corners")


@dataclass(frozen=True)
class CameraView:
    """One camera's placement and optics, in the ego frame."""

    name: str
    bearing: float
    """Direction the camera looks, as an angle in the ego frame's xy plane."""

    half_fov: float
    """Half the horizontal field of view, including any configured margin."""

    ego_to_camera: np.ndarray
    intrinsics: np.ndarray
    width: int
    height: int


@dataclass(frozen=True)
class RigVisibility:
    """Where a rig's cameras point, and whether they leave any gap."""

    views: Tuple[CameraView, ...]
    coverage_deg: float

    @property
    def covers_full_circle(self) -> bool:
        return self.coverage_deg >= FULL_COVERAGE_DEG

    def sees_center(self, center_ego: np.ndarray) -> bool:
        """Whether the box centre's bearing falls in some camera's field of view."""
        if self.covers_full_circle:
            return True
        bearing = float(np.arctan2(center_ego[1], center_ego[0]))
        for view in self.views:
            offset = bearing - view.bearing
            if abs(float(np.arctan2(np.sin(offset), np.cos(offset)))) <= view.half_fov:
                return True
        return False

    def sees_corners(self, corners_ego: np.ndarray) -> bool:
        """Whether any of the eight corners projects inside some camera's image."""
        if self.covers_full_circle:
            return True

        homogeneous = np.hstack([corners_ego, np.ones((corners_ego.shape[0], 1))])
        for view in self.views:
            in_camera = (view.ego_to_camera @ homogeneous.T).T
            in_front = in_camera[in_camera[:, 2] > MIN_DEPTH_M]
            if in_front.size == 0:
                continue
            projected = (view.intrinsics @ in_front[:, :3].T).T
            pixels = projected[:, :2] / projected[:, 2:3]
            inside = (
                (pixels[:, 0] >= 0)
                & (pixels[:, 0] < view.width)
                & (pixels[:, 1] >= 0)
                & (pixels[:, 1] < view.height)
            )
            if bool(np.any(inside)):
                return True
        return False

    def sees(self, mode: str, center_ego: np.ndarray, corners_ego: np.ndarray) -> bool:
        if mode == "center":
            return self.sees_center(center_ego)
        if mode == "corners":
            return self.sees_corners(corners_ego)
        raise ValueError(f"Unknown visibility mode '{mode}'. Available: {VISIBILITY_MODES}")


def coverage_degrees(views: Sequence[CameraView]) -> float:
    """Total angle the cameras cover between them, counting overlap once."""
    if not views:
        return 0.0

    two_pi = 2.0 * np.pi
    arcs: List[Tuple[float, float]] = []
    for view in views:
        low = (view.bearing - view.half_fov) % two_pi
        high = (view.bearing + view.half_fov) % two_pi
        if low <= high:
            arcs.append((low, high))
        else:  # wraps past zero, so split it
            arcs.append((low, two_pi))
            arcs.append((0.0, high))

    arcs.sort()
    merged: List[Tuple[float, float]] = []
    for low, high in arcs:
        if merged and low <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], high))
        else:
            merged.append((low, high))

    return float(np.rad2deg(sum(high - low for low, high in merged)))


def rig_visibility(scene, cameras: Sequence[str], margin_deg: float = 0.0, iteration: int = 0) -> RigVisibility:
    """Read a rig's geometry from one frame.

    Extrinsics and intrinsics are rig-static in 123D, so a single frame describes the whole log.

    :param margin_deg: Widen each camera's field of view by this much, so a box just past the edge
        is not dropped over a fraction of a degree.
    """
    margin = np.deg2rad(margin_deg)
    views: List[CameraView] = []

    for name in cameras:
        camera = scene.get_camera_at_iteration(iteration, name)
        if camera is None:
            continue

        camera_to_ego = np.asarray(
            camera.metadata.camera_to_imu_se3.transformation_matrix, dtype=np.float64
        )
        optical_axis = camera_to_ego[:3, :3] @ np.array([0.0, 0.0, 1.0])
        intrinsics = np.asarray(camera.metadata.intrinsics.camera_matrix, dtype=np.float64)

        views.append(
            CameraView(
                name=str(name),
                bearing=float(np.arctan2(optical_axis[1], optical_axis[0])),
                half_fov=float(np.arctan(camera.metadata.width / (2.0 * intrinsics[0, 0]))) + margin,
                ego_to_camera=np.linalg.inv(camera_to_ego),
                intrinsics=intrinsics,
                width=int(camera.metadata.width),
                height=int(camera.metadata.height),
            )
        )

    visibility = RigVisibility(views=tuple(views), coverage_deg=coverage_degrees(views))
    logger.debug(
        "Rig of %d camera(s) covers %.1f degrees%s",
        len(views),
        visibility.coverage_deg,
        " (full circle, so the visibility filter is a no-op)" if visibility.covers_full_circle else "",
    )
    return visibility
