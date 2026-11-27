"""A 123D scene small enough to reason about: synthetic cameras that image a known point.

Only what the torch dataset reads is implemented. Each camera sees a black image with one bright
Gaussian blob centred, to sub-pixel precision, on the projection of :data:`MARKER`, a fixed point
in the ego frame. The blob's centroid survives resampling, so a test can check to a tenth of a
pixel that wherever the pipeline moved the pixels, the matrices moved the same way.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np

#: A point in front of the vehicle, slightly left and above the IMU, in the ego frame.
MARKER = np.array([12.0, 1.5, 0.8])

#: Standard deviation of the blob drawn at the marker's projection, in source pixels.
SIGMA = 5.0


def look_along(yaw: float, position: Sequence[float] = (1.0, 0.0, 1.5)) -> np.ndarray:
    """Camera-to-ego transform of a camera looking along ego yaw ``yaw``, level with the ground.

    Camera axes follow OpenCV: x right, y down, z forward.
    """
    forward = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    right = np.array([np.sin(yaw), -np.cos(yaw), 0.0])
    down = np.array([0.0, 0.0, -1.0])
    matrix = np.eye(4)
    matrix[:3, :3] = np.stack([right, down, forward], axis=1)
    matrix[:3, 3] = position
    return matrix


class FakeCamera:
    def __init__(self, name: str, width: int, height: int, focal: float, camera_to_ego: np.ndarray) -> None:
        self.name = name
        self.width, self.height = width, height
        self.intrinsics = np.array([[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]])
        self.camera_to_ego = camera_to_ego

    def project(self, point_ego: np.ndarray) -> Optional[Tuple[float, float]]:
        camera = np.linalg.inv(self.camera_to_ego) @ np.append(point_ego, 1.0)
        if camera[2] <= 0:
            return None
        pixel = self.intrinsics @ camera[:3]
        return pixel[0] / pixel[2], pixel[1] / pixel[2]

    def image(self) -> np.ndarray:
        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        projected = self.project(MARKER)
        if projected is not None:
            rows, columns = np.indices((self.height, self.width), dtype=np.float64)
            u, v = projected
            blob = 255.0 * np.exp(-((columns - u) ** 2 + (rows - v) ** 2) / (2.0 * SIGMA**2))
            image[...] = np.round(blob).astype(np.uint8)[..., None]
        return image

    def metadata(self) -> SimpleNamespace:
        return SimpleNamespace(
            camera_id=SimpleNamespace(name=self.name),
            camera_to_imu_se3=SimpleNamespace(transformation_matrix=self.camera_to_ego),
            intrinsics=SimpleNamespace(camera_matrix=self.intrinsics),
            width=self.width,
            height=self.height,
        )


class FakeScene:
    """A log of ``iterations`` identical frames, the ego vehicle parked at ``ego_to_world``."""

    def __init__(
        self,
        cameras: Sequence[FakeCamera],
        iterations: int = 4,
        uuid: str = "fake",
        ego_to_world: Optional[np.ndarray] = None,
        missing: Sequence[Tuple[int, str]] = (),
    ) -> None:
        self.cameras: Dict[str, FakeCamera] = {camera.name: camera for camera in cameras}
        self.number_of_iterations = iterations
        self.scene_uuid = uuid
        self.split, self.log_name = "fake_split", uuid
        self.ego_to_world = np.eye(4) if ego_to_world is None else ego_to_world
        self.missing = set(missing)
        self.decode_scales = []

    def get_camera_metadatas(self):
        return {index: camera.metadata() for index, camera in enumerate(self.cameras.values())}

    def get_camera_at_iteration(self, iteration: int, name: str, scale: Optional[int] = None):
        if (iteration, name) in self.missing or name not in self.cameras:
            return None
        self.decode_scales.append(scale)
        image = self.cameras[name].image()
        if scale is not None and scale > 1:
            height, width = image.shape[:2]
            image = cv2.resize(image, (width // scale, height // scale), interpolation=cv2.INTER_AREA)
        return SimpleNamespace(image=image)

    def get_ego_state_se3_at_iteration(self, iteration: int):
        return SimpleNamespace(imu_se3=SimpleNamespace(transformation_matrix=self.ego_to_world))

    def get_box_detections_se3_at_iteration(self, iteration: int):
        return None

    def get_timestamp_at_iteration(self, iteration: int):
        return SimpleNamespace(time_s=0.5 * iteration, time_us=500_000 * iteration)


def marker_centroid(image_chw: np.ndarray) -> Optional[Tuple[float, float]]:
    """Intensity-weighted centre of the blob in a ``(3, H, W)`` uint8 image."""
    weights = image_chw.astype(np.float64).mean(axis=0)
    total = weights.sum()
    if total == 0:
        return None
    rows, columns = np.indices(weights.shape)
    return float((columns * weights).sum() / total), float((rows * weights).sum() / total)
