"""Reading a rig's images for one frame, already in network geometry.

Per camera: compose the rig's fixed crop with the sample's resize, crop and flip into one affine
map, decode the image, resample it once through the map, and move the intrinsics by the same map.
Calibration and image size come from the log's camera metadata, which is rig-static in 123D, so
the map is known before any pixel is decoded.

That lets the decoder do the first part of a large reduction. When the network input is at most
half the sensor resolution, the image is decoded at 1/2, 1/4 or 1/8 size — for JPEG in the DCT
domain, which is several times faster than a full decode and a proper low-pass filter, where a
bilinear warp over a 2x reduction samples single pixels and aliases.

Images stay ``uint8``. Normalisation is the model's job, on the device — see
:meth:`~py123detection.torch.models.multiview.MultiViewDetector.normalize_images`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from py123detection.core.geometry import invert_rigid
from py123detection.torch.data.images import ImageDraw, ImagePipeline, ImageTransform, warp
from py123detection.torch.data.rigs import CameraRig


@dataclass(frozen=True)
class Calibration:
    """One camera's rig-static calibration."""

    ego_to_camera: np.ndarray
    """``(4, 4)`` float32."""

    intrinsics: np.ndarray
    """``(3, 3)`` float64, for the full-resolution image."""

    height: int
    width: int


@dataclass
class Cameras:
    """A rig's images for one frame, with the matrices that go with them."""

    images: np.ndarray
    """``(N, 3, H_pad, W_pad)`` uint8, zero outside the image and for a missing camera."""

    intrinsics: np.ndarray
    """``(N, 4, 4)`` float32, including everything done to the image. Identity where missing."""

    extrinsics: np.ndarray
    """``(N, 4, 4)`` float32, ego to camera. Identity where missing."""

    valid: np.ndarray
    """``(N,)`` bool, False where a camera had no image this frame."""

    source_sizes: List[Tuple[int, int, int]]
    """``(height, width, 3)`` of each image as recorded."""

    size: Tuple[int, int]
    """``(height, width)`` of every image before padding."""


class CameraLoader:
    """Loads :class:`Cameras` for a rig at one network input size.

    :param missing_camera: ``"skip"`` rejects a frame with a camera missing, ``"zero"`` keeps it
        with that camera blank and marked invalid.
    """

    def __init__(self, rig: CameraRig, images: ImagePipeline, missing_camera: str = "skip") -> None:
        if missing_camera not in ("skip", "zero"):
            raise ValueError(f"missing_camera must be 'skip' or 'zero', got {missing_camera!r}.")
        self.rig = rig
        self.images = images
        self.missing_camera = missing_camera
        self._calibrations: Dict[str, Dict[str, Calibration]] = {}

    def load(self, scene, iteration: int, draw: Union[ImageDraw, ImageTransform]) -> Optional[Cameras]:
        """The rig's cameras at one iteration, or ``None`` if the frame cannot be used."""
        calibrations = self.calibration(scene)
        height, width = self.images.size
        padded_height, padded_width = self.images.padded_size(height, width)
        count = len(self.rig)

        images = np.zeros((count, 3, padded_height, padded_width), dtype=np.uint8)
        intrinsics = np.tile(np.eye(4, dtype=np.float32), (count, 1, 1))
        extrinsics = np.tile(np.eye(4, dtype=np.float32), (count, 1, 1))
        valid = np.zeros(count, dtype=bool)
        source_sizes: List[Tuple[int, int, int]] = []

        for position, name in enumerate(self.rig.cameras):
            calibration = calibrations.get(name)
            image, matrix = None, None
            if calibration is not None:
                rig_matrix, (rig_height, rig_width) = self.rig.camera_matrix(
                    name, calibration.height, calibration.width
                )
                matrix = self.images.resolve(draw, rig_height, rig_width).matrix() @ rig_matrix
                factor = decode_factor(matrix)
                camera = scene.get_camera_at_iteration(iteration, name, scale=factor if factor > 1 else None)
                image = None if camera is None else camera.image

            if image is None:
                if self.missing_camera == "skip":
                    return None
                source_sizes.append((height, width, 3))
                continue

            source_sizes.append((calibration.height, calibration.width, 3))
            decoded = decoded_to_source(image.shape, calibration.height, calibration.width)
            images[position, :, :height, :width] = warp(image, matrix @ decoded, (height, width)).transpose(2, 0, 1)
            intrinsics[position, :3, :3] = matrix @ calibration.intrinsics
            extrinsics[position] = calibration.ego_to_camera
            valid[position] = True

        if not valid.any():
            return None
        return Cameras(images, intrinsics, extrinsics, valid, source_sizes, (height, width))

    def calibration(self, scene) -> Dict[str, Calibration]:
        """Calibration of each rig camera the log has, read once per log from its metadata."""
        cached = self._calibrations.get(scene.scene_uuid)
        if cached is not None:
            return cached

        by_name = {metadata.camera_id.name: metadata for metadata in scene.get_camera_metadatas().values()}
        calibrations: Dict[str, Calibration] = {}
        for name in self.rig.cameras:
            metadata = by_name.get(name)
            if metadata is None:
                continue
            camera_to_ego = np.asarray(metadata.camera_to_imu_se3.transformation_matrix, dtype=np.float64)
            calibrations[name] = Calibration(
                ego_to_camera=invert_rigid(camera_to_ego).astype(np.float32),
                intrinsics=np.asarray(metadata.intrinsics.camera_matrix, dtype=np.float64),
                height=int(metadata.height),
                width=int(metadata.width),
            )

        self._calibrations[scene.scene_uuid] = calibrations
        return calibrations


def decode_factor(matrix: np.ndarray) -> int:
    """The largest reduction in 1, 2, 4, 8 that still leaves the warp at or below full resolution.

    A reduction is only taken where the result would still be at least as fine as the network
    input along both axes, so nothing the network could have seen is lost.
    """
    scale = max(abs(float(matrix[0, 0])), abs(float(matrix[1, 1])))
    factor = 1
    while factor < 8 and scale * factor * 2 <= 1.0:
        factor *= 2
    return factor


def decoded_to_source(shape: Tuple[int, ...], height: int, width: int) -> np.ndarray:
    """Map from a possibly reduced decode's pixel coordinates to the recorded image's.

    A factor-``f`` reduction averages ``f`` source pixels into one, so decoded pixel ``j`` sits at
    ``f * j + (f - 1) / 2`` in the recorded image. The identity when nothing was reduced.
    """
    factor_y, factor_x = height / shape[0], width / shape[1]
    return np.array(
        [[factor_x, 0.0, (factor_x - 1.0) / 2.0], [0.0, factor_y, (factor_y - 1.0) / 2.0], [0.0, 0.0, 1.0]]
    )
