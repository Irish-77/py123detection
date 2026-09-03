"""Resolving 123D sensor payloads to the file paths mmdetection3d expects.

mmdet3d reads images and point clouds from disk: ``info['cams'][cam]['data_path']`` and
``info['lidar_path']`` are filenames handed to ``mmcv.imread`` / ``np.fromfile``. 123D, by
contrast, can store a sensor stream either as a **reference** (a relative path into the original
dataset, ``camera_store_option="path"``) or as an **embedded payload** (JPEG/PNG bytes or an MP4
frame index; ``lidar_store_option="binary"``).

Both cases have to end up as a path, so this module offers two strategies:

``reference``
    Read the stored relative path without decoding anything and join it onto the dataset's
    sensor root. Zero copies, zero extra disk — the exported pkl points straight at the original
    dataset files. Only possible when the log was converted with the ``path`` store option.

``extract``
    Decode the embedded payload through the 123D scene API and write it out to an export
    directory, then point the info at the written file. Needed for binary/MP4-backed logs, and
    usable as a "materialize a self-contained copy" mode for path-backed logs too.

:data:`SensorMode.AUTO` picks ``reference`` per stream when a path is available and falls back to
``extract`` otherwise, which is the right default for mixed exports.
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
from py123d.api import SceneAPI
from py123d.common.runtime import get_dataset_paths
from py123d.datatypes import CameraID, LidarID, ModalityType

logger = logging.getLogger(__name__)


class SensorMode(str, enum.Enum):
    """How to turn a 123D sensor payload into a path on disk."""

    AUTO = "auto"
    """Reference when the log stores a path, extract otherwise."""

    REFERENCE = "reference"
    """Always reference the original dataset file; fail if the log embeds the payload."""

    EXTRACT = "extract"
    """Always decode and write the payload into the export directory."""


class PathStyle(str, enum.Enum):
    """How paths are spelled in the exported infos."""

    ABSOLUTE = "absolute"
    """Fully resolved paths. Required for merged multi-dataset exports, where a single mmdet3d
    ``data_root`` cannot cover every source."""

    RELATIVE = "relative"
    """Paths relative to :attr:`SensorResolverConfig.relative_to`, for configs that set a
    matching ``data_root``."""


@dataclass
class SensorResolverConfig:
    """Configuration for :class:`SensorResolver`."""

    camera_mode: SensorMode = SensorMode.AUTO
    """Strategy for camera images."""

    lidar_mode: SensorMode = SensorMode.AUTO
    """Strategy for lidar point clouds."""

    path_style: PathStyle = PathStyle.ABSOLUTE
    """Absolute or :attr:`relative_to`-relative paths in the exported infos."""

    relative_to: Optional[Path] = None
    """Root that :attr:`PathStyle.RELATIVE` paths are expressed against."""

    extract_root: Optional[Path] = None
    """Directory that extracted payloads are written under. Required when extraction happens."""

    sensor_roots: Dict[str, Path] = field(default_factory=dict)
    """Per-dataset sensor-root overrides, e.g. ``{"nuscenes": Path("/data/nuscenes")}``.

    Falls back to 123D's own :func:`~py123d.common.runtime.get_dataset_paths` resolution, which
    reads ``NUSCENES_DATA_ROOT`` and friends from the environment.
    """

    lidar_features: int = 5
    """Number of per-point float32 values written per point when extracting lidar.

    ``5`` reproduces the nuScenes ``(x, y, z, intensity, ring)`` layout that mmdet3d's
    ``LoadPointsFromFile(load_dim=5)`` expects. Extra slots are zero-filled when 123D does not
    carry the corresponding feature.
    """

    jpeg_quality: int = 95
    """JPEG quality used when extracting camera images."""


class SensorExtractionError(RuntimeError):
    """Raised when a sensor payload cannot be turned into a usable path."""


class SensorResolver:
    """Resolves camera and lidar payloads for one export run."""

    def __init__(self, config: SensorResolverConfig) -> None:
        """Initialize the resolver.

        :param config: The resolver configuration.
        """
        self._config = config
        self._sensor_root_cache: Dict[str, Optional[Path]] = {}
        self._extracted_count = 0
        self._referenced_count = 0

    # -- statistics -----------------------------------------------------------------------

    @property
    def num_referenced(self) -> int:
        """How many payloads were resolved by reference."""
        return self._referenced_count

    @property
    def num_extracted(self) -> int:
        """How many payloads were written out by extraction."""
        return self._extracted_count

    # -- public API -----------------------------------------------------------------------

    def resolve_camera(self, scene: SceneAPI, iteration: int, camera_id: CameraID, camera_name: str) -> str:
        """Resolve the image of one camera at one iteration to a path.

        :param scene: The scene to read from.
        :param iteration: Iteration within the scene.
        :param camera_id: Camera to resolve.
        :param camera_name: Dataset-native camera name, used to lay out extracted files.
        :return: The path, spelled according to :attr:`SensorResolverConfig.path_style`.
        :raises SensorExtractionError: If the payload cannot be resolved under the chosen mode.
        """
        raw = scene.get_modality_column_at_iteration(
            iteration=iteration,
            column="data",
            modality_type=ModalityType.CAMERA,
            modality_id=camera_id,
            deserialize=False,
        )
        mode = self._config.camera_mode

        if isinstance(raw, str) and mode in (SensorMode.AUTO, SensorMode.REFERENCE):
            self._referenced_count += 1
            return self._spell(self._sensor_root(scene.dataset, "camera") / raw)

        if mode is SensorMode.REFERENCE:
            raise SensorExtractionError(
                f"Camera '{camera_name}' in log '{scene.log_name}' is stored as {type(raw).__name__}, not a path. "
                "Re-run the 123D conversion with camera_store_option='path', or export with camera_mode='extract'."
            )

        return self._spell(self._extract_camera(scene, iteration, camera_id, camera_name, raw))

    def resolve_lidar(
        self,
        scene: SceneAPI,
        iteration: int,
        lidar_id: LidarID,
        ego_to_lidar: Optional[np.ndarray] = None,
    ) -> Optional[str]:
        """Resolve the point cloud at one iteration to a path.

        :param scene: The scene to read from.
        :param iteration: Iteration within the scene.
        :param lidar_id: Lidar to resolve.
        :param ego_to_lidar: ``(4, 4)`` transform applied to extracted points. 123D hands out point
            clouds in the **ego** frame (every parser reframes them on load), so writing them
            alongside sensor-frame boxes needs this transform. Pass ``None`` to keep the ego frame.
            Ignored when the payload is resolved by reference.
        :return: The path, or ``None`` if the log carries no lidar at this iteration.
        :raises SensorExtractionError: If the payload cannot be resolved under the chosen mode.
        """
        raw = scene.get_modality_column_at_iteration(
            iteration=iteration,
            column="data",
            modality_type=ModalityType.LIDAR,
            modality_id=lidar_id,
            deserialize=False,
        )
        if raw is None:
            return None

        mode = self._config.lidar_mode
        if isinstance(raw, str) and mode in (SensorMode.AUTO, SensorMode.REFERENCE):
            self._referenced_count += 1
            return self._spell(self._sensor_root(scene.dataset, "lidar") / raw)

        if mode is SensorMode.REFERENCE:
            raise SensorExtractionError(
                f"Lidar in log '{scene.log_name}' is stored as {type(raw).__name__}, not a path. "
                "Re-run the 123D conversion with lidar_store_option='path', or export with lidar_mode='extract'."
            )

        return self._spell(self._extract_lidar(scene, iteration, lidar_id, ego_to_lidar))

    # -- internals ------------------------------------------------------------------------

    def _sensor_root(self, dataset: str, kind: str) -> Path:
        """Resolve the on-disk root that a dataset's relative sensor paths hang off."""
        if dataset in self._config.sensor_roots:
            return Path(self._config.sensor_roots[dataset])
        if dataset not in self._sensor_root_cache:
            self._sensor_root_cache[dataset] = get_dataset_paths().get_sensor_root(dataset)
        root = self._sensor_root_cache[dataset]
        if root is None:
            raise SensorExtractionError(
                f"No sensor root configured for dataset '{dataset}' (needed to resolve {kind} references). "
                f"Set the dataset's environment variable (e.g. NUSCENES_DATA_ROOT) or pass "
                f"sensor_roots={{'{dataset}': '/path/to/dataset'}} to the export config."
            )
        return Path(root)

    def _extract_dir(self, scene: SceneAPI, *parts: str) -> Path:
        """Build (and create) the directory that extracted payloads for a scene go into."""
        if self._config.extract_root is None:
            raise SensorExtractionError(
                "extract_root must be set to extract embedded sensor payloads "
                "(pass --extract-root on the CLI, or extract_root=... in the config)."
            )
        directory = Path(self._config.extract_root).joinpath(scene.split, scene.log_name, *parts)
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def _extract_camera(
        self,
        scene: SceneAPI,
        iteration: int,
        camera_id: CameraID,
        camera_name: str,
        raw: Any = None,
    ) -> Path:
        """Write a camera image to the export directory.

        A log that stores whole JPEG or PNG payloads is copied out byte-for-byte: 123D keeps the
        dataset's original encoded bytes, so decoding and re-encoding would only add generation
        loss. Anything else (an MP4 frame index, or a path being materialized on purpose) goes
        through the scene API and is written as JPEG.
        """
        timestamp_us = scene.get_modality_column_at_iteration(
            iteration=iteration,
            column="timestamp_us",
            modality_type=ModalityType.CAMERA,
            modality_id=camera_id,
            deserialize=False,
        )

        if isinstance(raw, bytes):
            suffix = ".png" if raw[:8] == b"\x89PNG\r\n\x1a\n" else ".jpg"
            target = self._extract_dir(scene, camera_name) / f"{timestamp_us}{suffix}"
            if not target.exists():
                target.write_bytes(raw)
                self._extracted_count += 1
            return target

        try:
            import cv2
        except ImportError as error:  # pragma: no cover - depends on the install extras
            raise SensorExtractionError(
                "re-encoding a camera image needs opencv-python (install py123detection[images])."
            ) from error

        camera = scene.get_camera_at_iteration(iteration=iteration, camera_id=camera_id)
        if camera is None:
            raise SensorExtractionError(
                f"Camera '{camera_name}' has no data at iteration {iteration} of log '{scene.log_name}'."
            )

        target = self._extract_dir(scene, camera_name) / f"{camera.timestamp.time_us}.jpg"
        if not target.exists():
            # 123D hands out RGB; cv2.imwrite expects BGR.
            image = camera.image
            if image.ndim == 3 and image.shape[2] == 3:
                image = image[:, :, ::-1]
            ok = cv2.imwrite(str(target), image, [int(cv2.IMWRITE_JPEG_QUALITY), self._config.jpeg_quality])
            if not ok:
                raise SensorExtractionError(f"cv2 failed to write extracted image to {target}.")
            self._extracted_count += 1
        return target

    def _extract_lidar(
        self,
        scene: SceneAPI,
        iteration: int,
        lidar_id: LidarID,
        ego_to_lidar: Optional[np.ndarray] = None,
    ) -> Path:
        """Decode a point cloud and write it as a flat float32 ``.bin``, nuScenes style.

        The points 123D returns are in the ego frame; ``ego_to_lidar`` moves them into whichever
        frame the export's boxes use, so a downstream ``LoadPointsFromFile`` and the annotations
        agree.
        """
        # Read the timestamp from the Arrow column rather than from the decoded modality: it names
        # the output file, and looking it up first lets an already-extracted frame skip the decode.
        timestamp_us = scene.get_modality_column_at_iteration(
            iteration=iteration,
            column="timestamp_us",
            modality_type=ModalityType.LIDAR,
            modality_id=lidar_id,
            deserialize=False,
        )
        if timestamp_us is None:
            raise SensorExtractionError(f"No lidar at iteration {iteration} of log '{scene.log_name}'.")

        target = self._extract_dir(scene, "lidar") / f"{timestamp_us}.bin"
        if not target.exists():
            lidar = scene.get_lidar_at_iteration(iteration=iteration, lidar_id=lidar_id)
            if lidar is None:
                raise SensorExtractionError(f"No lidar at iteration {iteration} of log '{scene.log_name}'.")
            points = np.asarray(lidar.point_cloud_3d, dtype=np.float64).reshape(-1, 3)
            if ego_to_lidar is not None:
                points = points @ ego_to_lidar[:3, :3].T + ego_to_lidar[:3, 3]
            points = points.astype(np.float32)
            num_features = max(3, self._config.lidar_features)
            padded = np.zeros((points.shape[0], num_features), dtype=np.float32)
            padded[:, :3] = points
            self._fill_extra_lidar_features(lidar, padded)
            padded.tofile(str(target))
            self._extracted_count += 1
        return target

    @staticmethod
    def _fill_extra_lidar_features(lidar, padded: np.ndarray) -> None:
        """Copy intensity (slot 3) and ring/channel (slot 4) into the nuScenes-style layout.

        Both features are optional in 123D; slots stay zero when the source log has no such
        per-point feature.
        """
        for slot, values in ((3, lidar.intensity), (4, lidar.channel)):
            if slot >= padded.shape[1] or values is None:
                continue
            values = np.asarray(values, dtype=np.float32).reshape(-1)
            if values.shape[0] == padded.shape[0]:
                padded[:, slot] = values

    def _spell(self, path: Union[str, Path]) -> str:
        """Render a resolved path according to the configured path style."""
        path = Path(path)
        if self._config.path_style is PathStyle.RELATIVE:
            if self._config.relative_to is None:
                raise SensorExtractionError("path_style='relative' requires relative_to to be set.")
            try:
                return str(path.resolve().relative_to(Path(self._config.relative_to).resolve()))
            except ValueError as error:
                raise SensorExtractionError(
                    f"Cannot express {path} relative to {self._config.relative_to}. "
                    "Use path_style='absolute' for exports that span several dataset roots."
                ) from error
        return str(path)
