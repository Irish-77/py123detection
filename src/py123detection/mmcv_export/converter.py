"""The 123D -> mmdetection3d ``info`` converter.

One exported ``info`` dict describes one synchronized frame, in the schema mmdetection3d's
``NuScenesDataset`` (and its derivatives — PETR, StreamPETR, BEVDet/CoIn3D) reads. See
:mod:`py123detection.mmcv_export.schema` for the field-by-field description.

The reference frame throughout is the **lidar frame of the keyframe**, matching mmdet3d. 123D
stores everything in the global frame, so each frame reduces to composing

    ``lidar -> ego -> global``

from the scene's ego state and rig extrinsics, then inverting it to bring boxes and camera poses
back into the lidar frame.
"""

from __future__ import annotations

import logging
from bisect import bisect_left
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from py123d.api import SceneAPI
from py123d.datatypes import BaseCameraMetadata, CameraID, LidarID, ModalityType
from py123d.geometry import PoseSE3

from py123detection.annotations import (
    DetectionRecord,
    FrameAnnotations,
    build_frame_annotations,
    extract_detection_records,
)
from py123detection.geometry import invert_rigid, matrix_to_translation_quaternion, pose_to_matrix
from py123detection.mmcv_export.annotations2d import project_records_to_camera
from py123detection.mmcv_export.sensors import SensorResolver, SensorResolverConfig
from py123detection.mmcv_export.tokens import TokenResolver
from py123detection.taxonomy import NUSCENES_DETECTION, Taxonomy

logger = logging.getLogger(__name__)

DEFAULT_CAMERA_ORDER: Tuple[str, ...] = ("PCAM_F0", "PCAM_R0", "PCAM_L0", "PCAM_B0", "PCAM_L1", "PCAM_R1")
"""Default camera order, given as 123D :class:`CameraID` names.

On a nuScenes rig this resolves to ``CAM_FRONT, CAM_FRONT_RIGHT, CAM_FRONT_LEFT, CAM_BACK,
CAM_BACK_LEFT, CAM_BACK_RIGHT`` — the exact order mmdetection3d's nuScenes converter uses, so
pretrained multi-view models see cameras in the position they expect. Because it is expressed in
123D ids rather than dataset-native names, the same order applies to any 123D dataset.
"""


@dataclass
class ExportConfig:
    """Everything that shapes an export, independent of *which* scenes are exported."""

    taxonomy: Taxonomy = NUSCENES_DETECTION
    """Rule mapping 123D labels onto the mmdet3d ``class_names`` list."""

    camera_order: Tuple[str, ...] = DEFAULT_CAMERA_ORDER
    """Cameras to export, as 123D :class:`CameraID` names, in model input order."""

    camera_key: str = "native"
    """Key used for each camera inside ``info['cams']``.

    ``"native"`` uses the dataset's own channel name (``CAM_FRONT``), which is what stock
    mmdet3d configs expect. ``"camera_id"`` uses the 123D id (``PCAM_F0``); prefer it for merged
    multi-dataset exports, where native names differ between datasets and would otherwise produce
    inconsistent camera dictionaries.
    """

    reference_lidar: Optional[str] = None
    """:class:`LidarID` name whose sensor frame is the export's reference frame.

    ``None`` picks ``LIDAR_TOP`` when present, otherwise the first lidar in the log.
    """

    max_sweeps: int = 10
    """Maximum number of preceding lidar frames recorded in ``info['sweeps']``.

    Sweeps are taken at the **log's own frame rate**, not at the export rate, so subsampling the
    export to 2 Hz still yields sweeps at whatever rate the 123D log holds. What a 123D log cannot
    offer is data it never stored: a keyframe-only nuScenes conversion contains 2 Hz lidar, so its
    sweeps are 2 Hz rather than nuScenes' native 20 Hz. Convert with the interpolated (10 Hz)
    nuScenes profile if denser sweeps matter — camera-only models do not read them.

    The first frame of every log always gets an empty list, which is what StreamPETR's
    ``_set_sequence_group_flag`` reads as a sequence start. Set to ``0`` to always write an empty
    list.
    """

    with_2d_annotations: bool = True
    """Export per-camera 2D / mono-3D annotations. Required to train StreamPETR."""

    on_missing_camera: str = "skip_frame"
    """What to do when a configured camera has no data at a frame.

    ``"skip_frame"`` drops the frame (multi-view models need a fixed camera count),
    ``"drop_camera"`` exports the frame with fewer cameras, ``"error"`` raises.
    """

    sensors: SensorResolverConfig = field(default_factory=SensorResolverConfig)
    """How sensor payloads become paths — see :mod:`py123detection.mmcv_export.sensors`."""

    token_resolver: TokenResolver = field(default_factory=TokenResolver)
    """What goes into ``info['token']`` — see :mod:`py123detection.mmcv_export.tokens`."""

    include_track_tokens: bool = True
    """Also export per-box ``instance_tokens``. Ignored by mmdet3d, useful for tracking."""

    yaw_convention: str = "mmdet3d"
    """How box yaw is read out of a rotation matrix.

    ``"mmdet3d"`` (default) reproduces ``pyquaternion.Quaternion.yaw_pitch_roll[0]``, the
    read-out mmdetection3d's own nuScenes converter uses, so exported ``gt_boxes`` are numerically
    interchangeable with pickles produced by that converter. ``"heading"`` uses the geometric
    heading that the nuScenes evaluation devkit assumes; the two differ by a few milliradians in
    a lidar frame. See :func:`py123detection.geometry.mmdet3d_yaw`.
    """

    lidar2ego_mode: str = "static"
    """Whether ``lidar2ego`` is the rig calibration or is re-derived per frame.

    ``"static"`` (default) uses the rig extrinsic 123D stores in the lidar's modality metadata.
    This is what mmdetection3d's own converter writes, and it is the only thing 123D can offer
    directly: unlike cameras, the lidar Arrow table carries no per-row pose, just timestamps.

    ``"dynamic"`` folds the offset between the frame's ego timestamp and the lidar's own timestamp
    into the extrinsic, by looking up the ego pose at the lidar's timestamp:
    ``inv(ego2global_at_frame) @ ego2global_at_lidar_time @ lidar2ego_static``. That places the
    point cloud where the sensor actually was when it fired rather than where the frame's ego pose
    says it was. It only differs from ``"static"`` when the log's ego stream is denser than its
    frames — a keyframe-only nuScenes conversion stores one ego state per frame, so the nearest
    ego pose to the lidar timestamp *is* the frame's, and the two modes coincide.
    """

    camera2ego_mode: str = "static"
    """How a camera's timing offset is split between ``sensor2ego`` and ``ego2global``.

    Cameras fire a few milliseconds off the lidar keyframe, and 123D stores a per-frame,
    motion-compensated ``camera_to_global_se3``, so the offset is always accounted for — the
    question is only where it is booked.

    ``"static"`` (default) keeps ``sensor2ego`` at the rig calibration and moves ``ego2global`` to
    the camera's own timestamp, exactly as mmdetection3d's converter does. ``"dynamic"`` instead
    reports the frame's shared ``ego2global`` and books the offset into ``sensor2ego``, which is
    the convention the master-thesis pipeline used.

    Either way the composed ``sensor2lidar`` — the pair PETR and StreamPETR turn into
    ``lidar2img`` — is identical. The choice matters for BEVDet-family configs, which consume
    ``sensor2ego`` and ``ego2global`` separately.
    """

    lidar_frame: str = "sensor"
    """Which frame the export's geometry lives in.

    ``"sensor"`` (default) uses the reference lidar's own frame, matching mmdetection3d's nuScenes
    convention, and is what a *referenced* original point-cloud file is expressed in. Extracted
    point clouds are transformed into it, since 123D hands them out in the ego frame.

    ``"ego"`` uses the ego/IMU frame instead — ``lidar2ego`` becomes the identity and camera
    ``sensor2lidar`` becomes camera-to-ego. Prefer it when referencing files that a 123D parser
    already reframed to ego, or for cross-dataset exports where a single common frame matters more
    than matching one dataset's sensor convention.
    """

    planar_velocity: bool = True
    """Drop the global vertical velocity before rotating boxes into the lidar frame.

    mmdetection3d does this (it keeps only ``box_velocity()[:2]`` and pads a zero ``vz`` before
    rotating), and its evaluation path inverts the same assumption, so the default keeps the round
    trip consistent. Clearing it preserves the true 3D velocity, at the cost of letting a tilted
    lidar mix the vertical component into ``(vx, vy)``.
    """

    def __post_init__(self) -> None:
        for field_name in ("lidar2ego_mode", "camera2ego_mode"):
            value = getattr(self, field_name)
            if value not in ("static", "dynamic"):
                raise ValueError(f"{field_name} must be 'static' or 'dynamic', got {value!r}.")
        if self.lidar_frame not in ("sensor", "ego"):
            raise ValueError(f"lidar_frame must be 'sensor' or 'ego', got {self.lidar_frame!r}.")
        if self.yaw_convention not in ("mmdet3d", "heading"):
            raise ValueError(f"yaw_convention must be 'mmdet3d' or 'heading', got {self.yaw_convention!r}.")
        if self.camera_key not in ("native", "camera_id"):
            raise ValueError(f"camera_key must be 'native' or 'camera_id', got {self.camera_key!r}.")
        if self.on_missing_camera not in ("skip_frame", "drop_camera", "error"):
            raise ValueError(
                f"on_missing_camera must be 'skip_frame', 'drop_camera' or 'error', got {self.on_missing_camera!r}."
            )
        if self.max_sweeps < 0:
            raise ValueError(f"max_sweeps must be >= 0, got {self.max_sweeps}.")


@dataclass
class ExportStats:
    """Counters describing what one export run actually produced."""

    num_logs: int = 0
    num_frames: int = 0
    num_boxes: int = 0
    frames_skipped_missing_camera: int = 0
    frames_skipped_missing_ego: int = 0
    frames_skipped_missing_lidar: int = 0
    logs_skipped: int = 0
    dynamic_extrinsic_fallbacks: int = 0
    dropped_cameras: Counter = field(default_factory=Counter)
    class_counts: Counter = field(default_factory=Counter)
    num_referenced_payloads: int = 0
    num_extracted_payloads: int = 0

    def as_dict(self) -> Dict[str, Any]:
        """Return the counters as a plain dictionary, suitable for pickling or logging."""
        return {
            "num_logs": self.num_logs,
            "num_frames": self.num_frames,
            "num_boxes": self.num_boxes,
            "frames_skipped_missing_camera": self.frames_skipped_missing_camera,
            "frames_skipped_missing_ego": self.frames_skipped_missing_ego,
            "frames_skipped_missing_lidar": self.frames_skipped_missing_lidar,
            "logs_skipped": self.logs_skipped,
            "dynamic_extrinsic_fallbacks": self.dynamic_extrinsic_fallbacks,
            "dropped_cameras": dict(self.dropped_cameras),
            "class_counts": dict(self.class_counts),
            "num_referenced_payloads": self.num_referenced_payloads,
            "num_extracted_payloads": self.num_extracted_payloads,
        }


class MMDet3DConverter:
    """Converts 123D scenes into mmdetection3d ``info`` dictionaries."""

    def __init__(self, config: Optional[ExportConfig] = None) -> None:
        """Initialize the converter.

        :param config: Export configuration. Defaults to :class:`ExportConfig`.
        """
        self._config = config or ExportConfig()
        self._sensors = SensorResolver(self._config.sensors)
        self._stats = ExportStats()

    @property
    def config(self) -> ExportConfig:
        """The export configuration in use."""
        return self._config

    @property
    def stats(self) -> ExportStats:
        """Counters accumulated so far."""
        self._stats.num_referenced_payloads = self._sensors.num_referenced
        self._stats.num_extracted_payloads = self._sensors.num_extracted
        return self._stats

    # -------------------------------------------------------------------------------------

    def convert_scenes(
        self,
        scenes: Sequence[SceneAPI],
        timestamp_offset_us: int = 0,
        progress: bool = True,
        sweep_scenes: Optional[Dict[Tuple[str, str], SceneAPI]] = None,
    ) -> List[Dict[str, Any]]:
        """Convert several scenes (one per log) into a flat list of infos.

        :param scenes: Scenes to convert, each spanning one full log.
        :param timestamp_offset_us: Constant added to every exported timestamp. See
            :attr:`py123detection.sources.Source.timestamp_offset_us`.
        :param progress: Show a progress bar.
        :param sweep_scenes: Optional native-rate views of the same logs, keyed by
            ``(split, log_name)``, used to build ``info['sweeps']`` at the log's own frame rate
            when the export itself is subsampled.
        :return: Infos in log order, and within a log in frame order.
        """
        iterator: Sequence[SceneAPI] = scenes
        if progress:
            from tqdm import tqdm

            iterator = tqdm(scenes, desc="Exporting logs", unit="log")

        infos: List[Dict[str, Any]] = []
        for scene in iterator:
            sweep_scene = (sweep_scenes or {}).get((scene.split, scene.log_name))
            infos.extend(
                self.convert_scene(scene, timestamp_offset_us=timestamp_offset_us, sweep_scene=sweep_scene)
            )
        return infos

    def convert_scene(
        self,
        scene: SceneAPI,
        timestamp_offset_us: int = 0,
        sweep_scene: Optional[SceneAPI] = None,
    ) -> List[Dict[str, Any]]:
        """Convert one scene (one full log) into a list of infos, one per frame.

        :param scene: The scene to convert.
        :param timestamp_offset_us: Constant added to every exported timestamp.
        :param sweep_scene: Optional native-rate view of the same log, used for ``sweeps``.
        :return: The infos for this log, in frame order.
        """
        rig = self._resolve_rig(scene)
        if rig is None:
            self._stats.logs_skipped += 1
            return []
        camera_metadatas, reference_lidar_id, lidar_data_id = rig

        scene_token = f"{scene.split}/{scene.log_name}"
        frames: List[Dict[str, Any]] = []

        for iteration in range(scene.number_of_iterations):
            info = self._convert_frame(
                scene=scene,
                iteration=iteration,
                camera_metadatas=camera_metadatas,
                reference_lidar_id=reference_lidar_id,
                lidar_data_id=lidar_data_id,
                scene_token=scene_token,
                timestamp_offset_us=timestamp_offset_us,
            )
            if info is not None:
                frames.append(info)

        self._link_and_index(frames)
        # Without a native-rate view the export already visited every frame of the log, so its own
        # infos are the sweep candidates — re-walking the scene would resolve each payload twice.
        candidates = (
            self._build_sweep_table(sweep_scene, reference_lidar_id, lidar_data_id, timestamp_offset_us)
            if sweep_scene is not None
            else self._sweep_table_from_frames(frames)
        )
        self._attach_sweeps(frames, candidates)

        self._stats.num_logs += 1
        self._stats.num_frames += len(frames)
        return frames

    # -------------------------------------------------------------------------------------
    # Rig resolution
    # -------------------------------------------------------------------------------------

    def _resolve_rig(
        self, scene: SceneAPI
    ) -> Optional[Tuple[List[Tuple[CameraID, str, BaseCameraMetadata]], LidarID, LidarID]]:
        """Resolve the cameras and lidars this log contributes, or ``None`` to skip the log."""
        available_cameras = scene.get_camera_metadatas()
        cameras: List[Tuple[CameraID, str, BaseCameraMetadata]] = []
        for camera_name in self._config.camera_order:
            camera_id = CameraID.from_arbitrary(camera_name.lower())
            metadata = available_cameras.get(camera_id)
            if metadata is None:
                logger.warning(
                    "Log '%s' has no camera %s; skipping the log (available: %s).",
                    scene.log_name,
                    camera_name,
                    sorted(camera.name for camera in available_cameras),
                )
                return None
            cameras.append((camera_id, metadata.camera_name, metadata))

        lidar_metadatas = scene.get_lidar_metadatas()
        if not lidar_metadatas:
            logger.warning("Log '%s' has no lidar; skipping the log.", scene.log_name)
            return None

        if self._config.reference_lidar is not None:
            reference_lidar_id = LidarID.from_arbitrary(self._config.reference_lidar.lower())
            if reference_lidar_id not in lidar_metadatas:
                logger.warning(
                    "Log '%s' has no lidar %s; skipping the log (available: %s).",
                    scene.log_name,
                    self._config.reference_lidar,
                    sorted(lidar.name for lidar in lidar_metadatas),
                )
                return None
        elif LidarID.LIDAR_TOP in lidar_metadatas:
            reference_lidar_id = LidarID.LIDAR_TOP
        else:
            reference_lidar_id = sorted(lidar_metadatas, key=int)[0]

        # Point-cloud payloads usually live under the merged stream; the reference *frame* still
        # comes from a physical sensor, since that is the frame the stored file is expressed in.
        lidar_data_id = reference_lidar_id
        if scene.get_modality_metadata(ModalityType.LIDAR, LidarID.LIDAR_MERGED) is not None:
            lidar_data_id = LidarID.LIDAR_MERGED
            if len(lidar_metadatas) > 1:
                logger.warning(
                    "Log '%s' merges %d lidars; exported point clouds are referenced as-is and may not be "
                    "expressed in the '%s' frame that boxes use.",
                    scene.log_name,
                    len(lidar_metadatas),
                    reference_lidar_id.name,
                )

        return cameras, reference_lidar_id, lidar_data_id

    # -------------------------------------------------------------------------------------
    # Frame conversion
    # -------------------------------------------------------------------------------------

    def _convert_frame(
        self,
        scene: SceneAPI,
        iteration: int,
        camera_metadatas: List[Tuple[CameraID, str, BaseCameraMetadata]],
        reference_lidar_id: LidarID,
        lidar_data_id: LidarID,
        scene_token: str,
        timestamp_offset_us: int,
    ) -> Optional[Dict[str, Any]]:
        """Convert a single frame, or return ``None`` when the frame must be skipped."""
        ego_state = scene.get_ego_state_se3_at_iteration(iteration)
        if ego_state is None:
            self._stats.frames_skipped_missing_ego += 1
            return None

        ego_to_global = pose_to_matrix(ego_state.imu_se3)
        lidar_to_ego = self._lidar_to_ego(
            scene,
            reference_lidar_id,
            iteration=iteration,
            lidar_data_id=lidar_data_id,
            ego_to_global=ego_to_global,
        )
        lidar_to_global = ego_to_global @ lidar_to_ego
        global_to_lidar = invert_rigid(lidar_to_global)

        lidar_path = self._sensors.resolve_lidar(
            scene, iteration, lidar_data_id, ego_to_lidar=invert_rigid(lidar_to_ego)
        )
        if lidar_path is None:
            self._stats.frames_skipped_missing_lidar += 1
            return None

        cameras = self._convert_cameras(scene, iteration, camera_metadatas, global_to_lidar, ego_to_global)
        if cameras is None:
            self._stats.frames_skipped_missing_camera += 1
            return None

        raw_timestamp_us = scene.get_timestamp_at_iteration(iteration).time_us
        uuid = self._frame_uuid(scene, raw_timestamp_us)
        token = self._config.token_resolver.resolve(
            dataset=scene.dataset,
            split=scene.split,
            log_name=scene.log_name,
            timestamp_us=raw_timestamp_us,
            uuid=uuid,
        )

        lidar_translation, lidar_rotation = matrix_to_translation_quaternion(lidar_to_ego)
        ego_translation, ego_rotation = matrix_to_translation_quaternion(ego_to_global)

        info: Dict[str, Any] = {
            "lidar_path": lidar_path,
            "token": token,
            "prev": "",
            "next": "",
            "sweeps": [],
            "frame_idx": 0,
            "cams": {key: payload for key, _, payload in cameras},
            "scene_token": scene_token,
            "lidar2ego_translation": lidar_translation,
            "lidar2ego_rotation": lidar_rotation,
            "ego2global_translation": ego_translation,
            "ego2global_rotation": ego_rotation,
            "timestamp": int(raw_timestamp_us + timestamp_offset_us),
            # --- 123D provenance: ignored by mmdet3d, invaluable when debugging an export ---
            "py123d_uuid": uuid,
            "py123d_dataset": scene.dataset,
            "py123d_split": scene.split,
            "py123d_log_name": scene.log_name,
            "py123d_location": scene.location,
            "py123d_iteration": iteration,
        }

        annotations = self._convert_annotations(scene, iteration, global_to_lidar)
        info.update(
            {
                "gt_boxes": annotations.gt_boxes,
                "gt_names": annotations.gt_names,
                "gt_velocity": annotations.gt_velocity,
                "num_lidar_pts": annotations.num_lidar_pts,
                "num_radar_pts": annotations.num_radar_pts,
                "valid_flag": annotations.valid_flag,
            }
        )
        if self._config.include_track_tokens:
            info["instance_tokens"] = annotations.track_tokens

        if self._config.with_2d_annotations:
            info.update(self._convert_2d_annotations(annotations.records, cameras))

        self._stats.num_boxes += len(annotations)
        self._stats.class_counts.update(annotations.gt_names.tolist())
        return info

    def _convert_cameras(
        self,
        scene: SceneAPI,
        iteration: int,
        camera_metadatas: List[Tuple[CameraID, str, BaseCameraMetadata]],
        global_to_lidar: np.ndarray,
        frame_ego_to_global: np.ndarray,
    ) -> Optional[List[Tuple[str, BaseCameraMetadata, Dict[str, Any]]]]:
        """Build the ``info['cams']`` entries, or return ``None`` to skip the frame.

        :param scene: The scene to read from.
        :param iteration: Frame index.
        :param camera_metadatas: Cameras to export, in model input order.
        :param global_to_lidar: Transform into the export's reference frame.
        :param frame_ego_to_global: The frame's ``ego2global``, used by
            :attr:`ExportConfig.camera2ego_mode` ``"dynamic"``.
        """
        cameras: List[Tuple[str, BaseCameraMetadata, Dict[str, Any]]] = []

        for camera_id, camera_name, metadata in camera_metadatas:
            camera_to_global_pose = scene.get_modality_column_at_iteration(
                iteration=iteration,
                column="camera_to_global_se3",
                modality_type=ModalityType.CAMERA,
                modality_id=camera_id,
                deserialize=True,
            )
            camera_timestamp_us = scene.get_modality_column_at_iteration(
                iteration=iteration,
                column="timestamp_us",
                modality_type=ModalityType.CAMERA,
                modality_id=camera_id,
                deserialize=False,
            )
            if camera_to_global_pose is None or camera_timestamp_us is None:
                if self._config.on_missing_camera == "error":
                    raise RuntimeError(
                        f"Camera {camera_name} missing at iteration {iteration} of log '{scene.log_name}'."
                    )
                self._stats.dropped_cameras[camera_name] += 1
                if self._config.on_missing_camera == "skip_frame":
                    return None
                continue

            data_path = self._sensors.resolve_camera(scene, iteration, camera_id, camera_name)

            camera_to_global = pose_to_matrix(camera_to_global_pose)
            camera_to_lidar = global_to_lidar @ camera_to_global

            # The camera fires slightly off the frame's reference instant. 123D's per-frame
            # camera_to_global already accounts for that; camera2ego_mode only decides whether the
            # offset is reported in ego2global (static, mmdet3d's convention) or folded into
            # sensor2ego (dynamic). Both reconstruct the same camera_to_global, hence the same
            # sensor2lidar.
            if self._config.camera2ego_mode == "dynamic":
                camera_to_ego = invert_rigid(frame_ego_to_global) @ camera_to_global
                ego_to_global_for_camera = frame_ego_to_global
            else:
                camera_to_ego = pose_to_matrix(metadata.camera_to_imu_se3)
                ego_to_global_for_camera = camera_to_global @ invert_rigid(camera_to_ego)

            sensor_translation, sensor_rotation = matrix_to_translation_quaternion(camera_to_ego)
            ego_translation, ego_rotation = matrix_to_translation_quaternion(ego_to_global_for_camera)

            intrinsics = metadata.intrinsics
            if intrinsics is None:
                raise RuntimeError(f"Camera {camera_name} of log '{scene.log_name}' has no intrinsics.")

            key = camera_name if self._config.camera_key == "native" else camera_id.name
            payload = {
                "data_path": data_path,
                "type": key,
                "sample_data_token": f"{scene.split}/{scene.log_name}/{camera_name}/{camera_timestamp_us}",
                "sensor2ego_translation": sensor_translation,
                "sensor2ego_rotation": sensor_rotation,
                "ego2global_translation": ego_translation,
                "ego2global_rotation": ego_rotation,
                "timestamp": int(camera_timestamp_us),
                "sensor2lidar_rotation": np.ascontiguousarray(camera_to_lidar[:3, :3]),
                "sensor2lidar_translation": np.ascontiguousarray(camera_to_lidar[:3, 3]),
                "cam_intrinsic": np.ascontiguousarray(intrinsics.camera_matrix),
                "width": int(metadata.width),
                "height": int(metadata.height),
            }
            cameras.append((key, metadata, payload))

        if not cameras:
            return None
        return cameras

    def _lidar_to_ego(
        self,
        scene: SceneAPI,
        reference_lidar_id: LidarID,
        iteration: Optional[int] = None,
        lidar_data_id: Optional[LidarID] = None,
        ego_to_global: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """The lidar-to-ego transform defining the export's reference frame.

        Identity under ``lidar_frame="ego"``, where the ego frame *is* the reference frame.
        Otherwise the rig extrinsic, optionally motion-compensated — see
        :attr:`ExportConfig.lidar2ego_mode`.

        :param scene: The scene to read rig calibration from.
        :param reference_lidar_id: The reference lidar.
        :param iteration: Frame index, required for ``lidar2ego_mode="dynamic"``.
        :param lidar_data_id: Lidar stream whose timestamp is used, for the dynamic mode.
        :param ego_to_global: The frame's ``ego2global``, for the dynamic mode.
        :return: A ``(4, 4)`` transform.
        """
        if self._config.lidar_frame == "ego":
            return np.eye(4, dtype=np.float64)

        static = pose_to_matrix(scene.get_lidar_metadatas()[reference_lidar_id].lidar_to_imu_se3)
        if self._config.lidar2ego_mode == "static" or iteration is None or ego_to_global is None:
            return static

        ego_at_lidar = self._ego_to_global_at_lidar_time(scene, iteration, lidar_data_id or reference_lidar_id)
        if ego_at_lidar is None:
            return static
        # Re-express the sensor's true pose relative to the ego pose this frame reports, so that
        # `ego2global @ lidar2ego` reconstructs where the lidar actually was when it fired.
        return invert_rigid(ego_to_global) @ ego_at_lidar @ static

    def _ego_to_global_at_lidar_time(
        self, scene: SceneAPI, iteration: int, lidar_data_id: LidarID
    ) -> Optional[np.ndarray]:
        """Ego pose at the lidar's own capture time, or ``None`` when it cannot be resolved.

        Uses the nearest stored ego state rather than interpolating: a log's ego stream is the
        only evidence available, and inventing intermediate poses would hide how coarse it is.
        """
        lidar_timestamp_us = scene.get_modality_column_at_iteration(
            iteration=iteration,
            column="timestamp_us",
            modality_type=ModalityType.LIDAR,
            modality_id=lidar_data_id,
            deserialize=False,
        )
        if lidar_timestamp_us is None:
            return None
        ego_state = scene.get_ego_state_se3_at_timestamp(int(lidar_timestamp_us), criteria="nearest")
        if ego_state is None:
            self._stats.dynamic_extrinsic_fallbacks += 1
            return None
        return pose_to_matrix(ego_state.imu_se3)

    def _convert_annotations(self, scene: SceneAPI, iteration: int, global_to_lidar: np.ndarray) -> FrameAnnotations:
        """Extract and transform the 3D annotations of one frame."""
        detections = scene.get_box_detections_se3_at_iteration(iteration)
        records = extract_detection_records(detections, self._config.taxonomy)
        return build_frame_annotations(
            records,
            global_to_lidar,
            yaw_convention=self._config.yaw_convention,
            planar_velocity=self._config.planar_velocity,
        )

    def _convert_2d_annotations(
        self,
        records: List[DetectionRecord],
        cameras: List[Tuple[str, BaseCameraMetadata, Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """Project the frame's records into every camera and collect StreamPETR's 2D fields."""
        bboxes2d, labels2d, centers2d, depths = [], [], [], []
        bboxes3d_cams, bboxes_ignore, visibilities = [], [], []

        for _, metadata, payload in cameras:
            # Records are global-frame, so they project through the camera's global pose, which is
            # rebuilt from the pose pair already stored in the payload.
            camera_to_global = _camera_to_global_from_payload(payload)
            annotations = project_records_to_camera(
                records, metadata, camera_to_global, yaw_convention=self._config.yaw_convention
            )
            bboxes2d.append(annotations.bboxes2d)
            labels2d.append(annotations.labels2d)
            centers2d.append(annotations.centers2d)
            depths.append(annotations.depths)
            bboxes3d_cams.append(annotations.bboxes3d_cam)
            bboxes_ignore.append(annotations.bboxes_ignore)
            visibilities.append(annotations.visibilities)

        return {
            "bboxes2d": bboxes2d,
            "labels2d": labels2d,
            "centers2d": centers2d,
            "depths": depths,
            "bboxes3d_cams": bboxes3d_cams,
            "bboxes_ignore": bboxes_ignore,
            "visibilities": visibilities,
        }

    # -------------------------------------------------------------------------------------
    # Per-log post-processing
    # -------------------------------------------------------------------------------------

    @staticmethod
    def _link_and_index(frames: List[Dict[str, Any]]) -> None:
        """Fill ``prev`` / ``next`` / ``frame_idx`` for one log's frames."""
        for index, info in enumerate(frames):
            info["frame_idx"] = index
            info["prev"] = frames[index - 1]["token"] if index > 0 else ""
            info["next"] = frames[index + 1]["token"] if index + 1 < len(frames) else ""

    def _build_sweep_table(
        self,
        scene: SceneAPI,
        reference_lidar_id: LidarID,
        lidar_data_id: LidarID,
        timestamp_offset_us: int,
    ) -> List[Dict[str, Any]]:
        """Collect every lidar frame of a log, at the log's own rate, as sweep candidates.

        ``scene`` is normally a native-rate view of the log, so subsampling the export does not
        thin out the sweeps. Only paths and poses are read — point clouds are never decoded.

        :param scene: Scene to walk (native-rate view when the export is subsampled).
        :param reference_lidar_id: Lidar whose extrinsic defines the frame.
        :param lidar_data_id: Lidar stream to read paths from.
        :param timestamp_offset_us: Offset applied to exported timestamps.
        :return: Sweep candidates in chronological order.
        """
        if self._config.max_sweeps == 0:
            return []

        if reference_lidar_id not in scene.get_lidar_metadatas():
            return []

        candidates: List[Dict[str, Any]] = []
        for iteration in range(scene.number_of_iterations):
            ego_state = scene.get_ego_state_se3_at_iteration(iteration)
            if ego_state is None:
                continue

            # Resolve the extrinsic per sweep, so a dynamic lidar2ego is applied to sweeps too.
            ego_to_global = pose_to_matrix(ego_state.imu_se3)
            lidar_to_ego = self._lidar_to_ego(
                scene,
                reference_lidar_id,
                iteration=iteration,
                lidar_data_id=lidar_data_id,
                ego_to_global=ego_to_global,
            )
            lidar_translation, lidar_rotation = matrix_to_translation_quaternion(lidar_to_ego)

            try:
                lidar_path = self._sensors.resolve_lidar(
                    scene, iteration, lidar_data_id, ego_to_lidar=invert_rigid(lidar_to_ego)
                )
            except Exception as error:  # a sweep is optional context; never fail an export for one
                logger.debug("Skipping sweep at iteration %d of '%s': %s", iteration, scene.log_name, error)
                continue
            if lidar_path is None:
                continue

            raw_timestamp_us = scene.get_timestamp_at_iteration(iteration).time_us
            ego_translation, ego_rotation = matrix_to_translation_quaternion(ego_to_global)
            candidates.append(
                {
                    "raw_timestamp_us": raw_timestamp_us,
                    "lidar_to_global": ego_to_global @ lidar_to_ego,
                    "payload": {
                        "data_path": lidar_path,
                        "type": "lidar",
                        "sample_data_token": self._frame_uuid(scene, raw_timestamp_us),
                        "sensor2ego_translation": lidar_translation,
                        "sensor2ego_rotation": lidar_rotation,
                        "ego2global_translation": ego_translation,
                        "ego2global_rotation": ego_rotation,
                        "timestamp": int(raw_timestamp_us + timestamp_offset_us),
                    },
                }
            )
        return candidates

    @staticmethod
    def _sweep_table_from_frames(frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Reuse a log's own exported infos as its sweep candidates.

        Valid only when the export was not subsampled, i.e. the exported frames *are* every frame
        the log holds.

        :param frames: The log's exported infos, in frame order.
        :return: Sweep candidates in chronological order.
        """
        return [
            {
                "raw_timestamp_us": info["timestamp"],
                "lidar_to_global": _lidar_to_global(info),
                "payload": {
                    "data_path": info["lidar_path"],
                    "type": "lidar",
                    "sample_data_token": info["token"],
                    "sensor2ego_translation": info["lidar2ego_translation"],
                    "sensor2ego_rotation": info["lidar2ego_rotation"],
                    "ego2global_translation": info["ego2global_translation"],
                    "ego2global_rotation": info["ego2global_rotation"],
                    "timestamp": info["timestamp"],
                },
            }
            for info in frames
        ]

    def _attach_sweeps(self, frames: List[Dict[str, Any]], candidates: List[Dict[str, Any]]) -> None:
        """Fill ``info['sweeps']`` from the lidar frames preceding each exported frame.

        Sweeps are ordered newest first, matching mmdetection3d. The first frame of a log has no
        predecessor and keeps an empty list, which is what StreamPETR's
        ``_set_sequence_group_flag`` reads as "a new sequence starts here".

        :param frames: The log's exported infos, in frame order.
        :param candidates: Output of :meth:`_build_sweep_table`, in chronological order.
        """
        if self._config.max_sweeps == 0 or not candidates:
            return

        candidate_timestamps = [candidate["payload"]["timestamp"] for candidate in candidates]
        for info in frames:
            end = bisect_left(candidate_timestamps, info["timestamp"])
            start = max(0, end - self._config.max_sweeps)
            global_to_reference_lidar = invert_rigid(_lidar_to_global(info))

            sweeps: List[Dict[str, Any]] = []
            for candidate in reversed(candidates[start:end]):
                sensor_to_lidar = global_to_reference_lidar @ candidate["lidar_to_global"]
                sweep = dict(candidate["payload"])
                sweep["sensor2lidar_rotation"] = np.ascontiguousarray(sensor_to_lidar[:3, :3])
                sweep["sensor2lidar_translation"] = np.ascontiguousarray(sensor_to_lidar[:3, 3])
                sweeps.append(sweep)
            info["sweeps"] = sweeps

    @staticmethod
    def _frame_uuid(scene: SceneAPI, timestamp_us: int) -> str:
        """Return the deterministic 123D UUID of a frame."""
        from py123d.common.utils.uuid_utils import create_deterministic_uuid

        return str(
            create_deterministic_uuid(split=scene.split, log_name=scene.log_name, timestamp_us=timestamp_us)
        )


def _pose_pair_to_matrix(translation: Sequence[float], rotation: Sequence[float]) -> np.ndarray:
    """Rebuild a 4x4 transform from an exported ``(translation, quaternion)`` pair."""
    pose = PoseSE3.from_R_t(
        rotation=np.asarray(rotation, dtype=np.float64),
        translation=np.asarray(translation, dtype=np.float64),
    )
    return np.asarray(pose.transformation_matrix, dtype=np.float64)


def _lidar_to_global(info: Dict[str, Any]) -> np.ndarray:
    """Rebuild the ``lidar -> global`` transform of an info from its stored pose pairs."""
    lidar_to_ego = _pose_pair_to_matrix(info["lidar2ego_translation"], info["lidar2ego_rotation"])
    ego_to_global = _pose_pair_to_matrix(info["ego2global_translation"], info["ego2global_rotation"])
    return ego_to_global @ lidar_to_ego


def _camera_to_global_from_payload(payload: Dict[str, Any]) -> np.ndarray:
    """Rebuild the ``camera -> global`` transform from an exported camera payload."""
    camera_to_ego = _pose_pair_to_matrix(payload["sensor2ego_translation"], payload["sensor2ego_rotation"])
    ego_to_global = _pose_pair_to_matrix(payload["ego2global_translation"], payload["ego2global_rotation"])
    return ego_to_global @ camera_to_ego
