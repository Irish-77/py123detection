"""Turning 123D sensor payloads into the file paths mmdetection3d reads.

123D stores a sensor stream either as a relative path into the original dataset
(``camera_store_option="path"``) or as embedded data (JPEG/PNG bytes, MP4 frames, binary point
clouds). mmdet3d needs a file on disk either way:

``reference``
    Join the stored relative path onto the dataset's sensor root. No copies; only works for
    path-backed logs.
``extract``
    Decode the payload through the scene API and write it under ``extract_root``. Needed for
    embedded payloads, and usable to materialize a self-contained copy of path-backed logs.

``auto`` references where a path is stored and extracts otherwise.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
from py123d.api import SceneAPI
from py123d.common.runtime import get_dataset_paths
from py123d.datatypes import CameraID, LidarID, ModalityType


class SensorMode(str, enum.Enum):
    AUTO = "auto"
    REFERENCE = "reference"  # fail if the log embeds the payload
    EXTRACT = "extract"


class PathStyle(str, enum.Enum):
    ABSOLUTE = "absolute"  # required for merged exports, which no single data_root covers
    RELATIVE = "relative"  # relative to SensorResolverConfig.relative_to


@dataclass
class SensorResolverConfig:
    camera_mode: SensorMode = SensorMode.AUTO
    lidar_mode: SensorMode = SensorMode.AUTO
    path_style: PathStyle = PathStyle.ABSOLUTE
    relative_to: Optional[Path] = None
    extract_root: Optional[Path] = None

    sensor_roots: Dict[str, Path] = field(default_factory=dict)
    """Per-dataset sensor roots, e.g. ``{"nuscenes": Path("/data/nuscenes")}``.

    Datasets not listed fall back to py123d's ``get_dataset_paths()``, which reads
    ``NUSCENES_DATA_ROOT`` and friends from the environment.
    """

    lidar_features: int = 5
    """Float32 values per point in extracted lidar files. 5 is nuScenes' ``(x, y, z, intensity, ring)``,
    what ``LoadPointsFromFile(load_dim=5)`` expects; features 123D lacks are zero-filled."""

    jpeg_quality: int = 95


class SensorExtractionError(RuntimeError):
    pass


class SensorResolver:
    """Resolves camera and lidar payloads for one export run."""

    def __init__(self, config: SensorResolverConfig) -> None:
        self._config = config
        self._sensor_root_cache: Dict[str, Optional[Path]] = {}
        self.num_referenced = 0
        self.num_extracted = 0

    def resolve_camera(self, scene: SceneAPI, iteration: int, camera_id: CameraID, camera_name: str) -> str:
        raw = scene.get_modality_column_at_iteration(
            iteration=iteration,
            column="data",
            modality_type=ModalityType.CAMERA,
            modality_id=camera_id,
            deserialize=False,
        )
        mode = self._config.camera_mode

        if isinstance(raw, str) and mode in (SensorMode.AUTO, SensorMode.REFERENCE):
            self.num_referenced += 1
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
        """Path of the point cloud at one iteration, or ``None`` if there is none.

        :param ego_to_lidar: Applied to extracted points, which 123D hands out in the ego frame.
            ``None`` keeps the ego frame. Ignored for referenced files.
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
            self.num_referenced += 1
            return self._spell(self._sensor_root(scene.dataset, "lidar") / raw)

        if mode is SensorMode.REFERENCE:
            raise SensorExtractionError(
                f"Lidar in log '{scene.log_name}' is stored as {type(raw).__name__}, not a path. "
                "Re-run the 123D conversion with lidar_store_option='path', or export with lidar_mode='extract'."
            )

        return self._spell(self._extract_lidar(scene, iteration, lidar_id, ego_to_lidar))

    def _sensor_root(self, dataset: str, kind: str) -> Path:
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
        if self._config.extract_root is None:
            raise SensorExtractionError(
                "extract_root must be set to extract embedded sensor payloads "
                "(pass --extract-root on the CLI, or extract_root=... in the config)."
            )
        directory = Path(self._config.extract_root).joinpath(scene.split, scene.log_name, *parts)
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def _extract_camera(self, scene: SceneAPI, iteration: int, camera_id: CameraID, camera_name: str, raw: Any) -> Path:
        # Embedded JPEG/PNG bytes are the dataset's original encoding, so copy them as they are
        # instead of re-encoding. Everything else (MP4 frames, paths being materialized) is
        # decoded and written as JPEG.
        if isinstance(raw, bytes):
            timestamp_us = scene.get_modality_column_at_iteration(
                iteration=iteration,
                column="timestamp_us",
                modality_type=ModalityType.CAMERA,
                modality_id=camera_id,
                deserialize=False,
            )
            suffix = ".png" if raw[:8] == b"\x89PNG\r\n\x1a\n" else ".jpg"
            target = self._extract_dir(scene, camera_name) / f"{timestamp_us}{suffix}"
            if not target.exists():
                target.write_bytes(raw)
                self.num_extracted += 1
            return target

        try:
            import cv2
        except ImportError as error:
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
            image = camera.image
            if image.ndim == 3 and image.shape[2] == 3:
                image = image[:, :, ::-1]  # RGB -> BGR for cv2
            ok = cv2.imwrite(str(target), image, [int(cv2.IMWRITE_JPEG_QUALITY), self._config.jpeg_quality])
            if not ok:
                raise SensorExtractionError(f"cv2 failed to write extracted image to {target}.")
            self.num_extracted += 1
        return target

    def _extract_lidar(
        self, scene: SceneAPI, iteration: int, lidar_id: LidarID, ego_to_lidar: Optional[np.ndarray]
    ) -> Path:
        """Write the point cloud as a flat float32 ``.bin``, nuScenes style."""
        # The timestamp names the file, so reading it from the Arrow column first lets frames that
        # were already extracted skip the decode.
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
        if target.exists():
            return target

        lidar = scene.get_lidar_at_iteration(iteration=iteration, lidar_id=lidar_id)
        if lidar is None:
            raise SensorExtractionError(f"No lidar at iteration {iteration} of log '{scene.log_name}'.")
        points = np.asarray(lidar.point_cloud_3d, dtype=np.float64).reshape(-1, 3)
        if ego_to_lidar is not None:
            points = points @ ego_to_lidar[:3, :3].T + ego_to_lidar[:3, 3]

        num_features = max(3, self._config.lidar_features)
        padded = np.zeros((points.shape[0], num_features), dtype=np.float32)
        padded[:, :3] = points.astype(np.float32)
        # Intensity and ring go into slots 3 and 4 when the log has them.
        for slot, values in ((3, lidar.intensity), (4, lidar.channel)):
            if slot >= num_features or values is None:
                continue
            values = np.asarray(values, dtype=np.float32).reshape(-1)
            if values.shape[0] == padded.shape[0]:
                padded[:, slot] = values
        padded.tofile(str(target))
        self.num_extracted += 1
        return target

    def _spell(self, path: Union[str, Path]) -> str:
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
