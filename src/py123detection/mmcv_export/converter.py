"""123D scenes -> mmdetection3d ``info`` dicts, one per frame.

The reference frame is the keyframe's lidar frame, as in mmdet3d. 123D stores everything in the
global frame, so each frame composes ``lidar -> ego -> global`` from the ego state and rig
extrinsics and inverts it. See :mod:`py123detection.mmcv_export.schema` for the fields.
"""

from __future__ import annotations

import logging
from bisect import bisect_left
from collections import Counter
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from py123d.api import SceneAPI
from py123d.common.utils.uuid_utils import create_deterministic_uuid
from py123d.datatypes import BaseCameraMetadata, CameraID, LidarID, ModalityType
from py123d.geometry import PoseSE3

from py123detection.annotations import (
    BOX_LAYOUTS,
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
from py123detection.velocity import TrackVelocityTable

VELOCITY_SOURCES = ("stored", "tracks")
EGO_POSE_SOURCES = ("sync", "nearest")

logger = logging.getLogger(__name__)

#: 123D camera ids. On a nuScenes rig this is CAM_FRONT, CAM_FRONT_RIGHT, CAM_FRONT_LEFT, CAM_BACK,
#: CAM_BACK_LEFT, CAM_BACK_RIGHT, the order mmdetection3d's converter uses and pretrained
#: multi-view models expect.
DEFAULT_CAMERA_ORDER: Tuple[str, ...] = ("PCAM_F0", "PCAM_R0", "PCAM_L0", "PCAM_B0", "PCAM_L1", "PCAM_R1")

# A camera as resolved for one log: (id, dataset-native name, metadata).
Camera = Tuple[CameraID, str, BaseCameraMetadata]


@dataclass
class ExportConfig:
    """Everything that shapes an export, independent of which scenes are exported."""

    taxonomy: Taxonomy = NUSCENES_DETECTION

    camera_order: Tuple[str, ...] = DEFAULT_CAMERA_ORDER
    """123D :class:`~py123d.datatypes.CameraID` names, in model input order."""

    camera_key: str = "native"
    """Key of each camera in ``info['cams']``: ``"native"`` (``CAM_FRONT``, what stock configs
    expect) or ``"camera_id"`` (``PCAM_F0``, consistent across datasets in merged exports)."""

    reference_lidar: Optional[str] = None
    """:class:`~py123d.datatypes.LidarID` name of the reference frame. ``None`` picks ``LIDAR_TOP``, else the first lidar."""

    max_sweeps: int = 10
    """Preceding lidar frames per info, taken at the log's own rate rather than the export rate.

    A keyframe-only nuScenes conversion only holds 2 Hz lidar, so its sweeps are 2 Hz. The first
    frame of a log always gets an empty list, which StreamPETR reads as a sequence start.
    """

    with_2d_annotations: bool = True
    """Per-camera 2D / mono-3D annotations, required by StreamPETR."""

    on_missing_camera: str = "skip_frame"
    """``"skip_frame"`` (multi-view models need a fixed camera count), ``"drop_camera"`` or ``"error"``."""

    sensors: SensorResolverConfig = field(default_factory=SensorResolverConfig)
    token_resolver: TokenResolver = field(default_factory=TokenResolver)

    include_track_tokens: bool = True
    """Also write per-box ``instance_tokens``."""

    yaw_convention: str = "mmdet3d"
    """``"mmdet3d"`` matches pickles from mmdetection3d's converter, ``"heading"`` the nuScenes
    devkit. See :func:`py123detection.geometry.mmdet3d_yaw`."""

    lidar2ego_mode: str = "static"
    """``"static"`` writes the rig extrinsic, like mmdetection3d's converter.

    ``"dynamic"`` also folds in the ego motion between the frame timestamp and the lidar's own
    timestamp: ``inv(ego2global_at_frame) @ ego2global_at_lidar_time @ lidar2ego_static``. 123D
    stores no per-row lidar pose, so this only differs from static when the ego stream is denser
    than the frames (not for keyframe-only nuScenes).
    """

    camera2ego_mode: str = "static"
    """Where a camera's timing offset to the frame is booked.

    ``"static"`` keeps ``sensor2ego`` at the rig calibration and moves ``ego2global`` to the
    camera timestamp, like mmdetection3d. ``"dynamic"`` keeps the frame's ``ego2global`` and folds
    the offset into ``sensor2ego``. ``sensor2lidar``, which PETR and StreamPETR use, is the same
    either way; BEVDet-style configs read the two parts separately.
    """

    lidar_frame: str = "sensor"
    """``"sensor"`` uses the reference lidar's frame (mmdet3d's convention, and the frame of a
    referenced point cloud file). ``"ego"`` uses the ego/IMU frame, so ``lidar2ego`` is the identity."""

    planar_velocity: bool = True
    """Drop the global vertical velocity before rotating into the lidar frame. mmdetection3d does
    this and its evaluation assumes it."""

    box_layout: str = "streampetr"
    """``gt_boxes`` column layout, see :data:`py123detection.annotations.BOX_LAYOUTS`. A wrong
    choice does not crash, it trains on transposed, mis-rotated boxes."""

    velocity_source: str = "stored"
    """``"stored"`` uses the log's velocity (zero if the dataset has none). ``"tracks"`` derives it
    from neighbouring annotations of each track, see :mod:`py123detection.velocity`."""

    velocity_max_dt_s: float = 0.25

    ego_pose_source: str = "sync"
    """Which ego state gives a frame's ``ego2global``: the sync table row (``"sync"``) or the
    state nearest to the frame timestamp (``"nearest"``).

    The sync table takes the first ego row at or after the frame timestamp, so an ego timestamp
    that is 1 us early skips the right row. py123d 0.6.0's Argoverse 2 parser round-trips the
    nanosecond stamps through float64: 5% of ego rows come back 1 us early and ~40% of frames get
    an ego pose 2-10 ms late (up to 9 cm). ``"nearest"`` avoids that.
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
        if self.box_layout not in BOX_LAYOUTS:
            raise ValueError(f"box_layout must be one of {BOX_LAYOUTS}, got {self.box_layout!r}.")
        if self.velocity_source not in VELOCITY_SOURCES:
            raise ValueError(f"velocity_source must be one of {VELOCITY_SOURCES}, got {self.velocity_source!r}.")
        if self.velocity_max_dt_s <= 0:
            raise ValueError(f"velocity_max_dt_s must be > 0, got {self.velocity_max_dt_s}.")
        if self.ego_pose_source not in EGO_POSE_SOURCES:
            raise ValueError(f"ego_pose_source must be one of {EGO_POSE_SOURCES}, got {self.ego_pose_source!r}.")


@dataclass
class ExportStats:
    num_logs: int = 0
    num_frames: int = 0
    num_boxes: int = 0
    frames_skipped_missing_camera: int = 0
    frames_skipped_missing_ego: int = 0
    frames_skipped_missing_lidar: int = 0
    logs_skipped: int = 0
    dynamic_extrinsic_fallbacks: int = 0
    boxes_without_track_velocity: int = 0
    ego_pose_fallbacks: int = 0
    ego_pose_max_offset_us: int = 0
    dropped_cameras: Counter = field(default_factory=Counter)
    class_counts: Counter = field(default_factory=Counter)
    num_referenced_payloads: int = 0
    num_extracted_payloads: int = 0

    def as_dict(self) -> Dict[str, Any]:
        stats = {f.name: getattr(self, f.name) for f in fields(self)}
        # Plain dicts, so the pickle does not reference collections.Counter.
        stats["dropped_cameras"] = dict(self.dropped_cameras)
        stats["class_counts"] = dict(self.class_counts)
        return stats


class MMDet3DConverter:
    def __init__(self, config: Optional[ExportConfig] = None) -> None:
        self.config = config or ExportConfig()
        self._sensors = SensorResolver(self.config.sensors)
        self._stats = ExportStats()

    @property
    def stats(self) -> ExportStats:
        self._stats.num_referenced_payloads = self._sensors.num_referenced
        self._stats.num_extracted_payloads = self._sensors.num_extracted
        return self._stats

    def convert_scenes(
        self,
        scenes: Sequence[SceneAPI],
        timestamp_offset_us: int = 0,
        progress: bool = True,
        native_scenes: Optional[Dict[Tuple[str, str], SceneAPI]] = None,
    ) -> List[Dict[str, Any]]:
        """Convert scenes (one per log) into a flat list of infos.

        :param native_scenes: Native-rate views of the same logs by ``(split, log_name)``, needed
            for sweeps and track velocities when ``scenes`` are subsampled.
        """
        if progress:
            from tqdm import tqdm

            scenes = tqdm(scenes, desc="Exporting logs", unit="log")

        infos: List[Dict[str, Any]] = []
        for scene in scenes:
            native_scene = (native_scenes or {}).get((scene.split, scene.log_name))
            infos.extend(self.convert_scene(scene, timestamp_offset_us=timestamp_offset_us, native_scene=native_scene))
        return infos

    def convert_scene(
        self,
        scene: SceneAPI,
        timestamp_offset_us: int = 0,
        native_scene: Optional[SceneAPI] = None,
    ) -> List[Dict[str, Any]]:
        """Convert one full log into infos, in frame order.

        :param native_scene: Native-rate view of the log, for sweeps and ``velocity_source="tracks"``.
            Omit when ``scene`` is not subsampled.
        """
        rig = self._resolve_rig(scene)
        if rig is None:
            self._stats.logs_skipped += 1
            return []
        cameras, reference_lidar_id, lidar_data_id = rig

        velocity_table = None
        if self.config.velocity_source == "tracks":
            velocity_table = TrackVelocityTable.from_scene(
                native_scene if native_scene is not None else scene, max_dt_s=self.config.velocity_max_dt_s
            )

        scene_token = f"{scene.split}/{scene.log_name}"
        frames = []
        for iteration in range(scene.number_of_iterations):
            info = self._convert_frame(
                scene,
                iteration,
                cameras,
                reference_lidar_id,
                lidar_data_id,
                scene_token,
                timestamp_offset_us,
                velocity_table,
            )
            if info is not None:
                frames.append(info)

        for index, info in enumerate(frames):
            info["frame_idx"] = index
            info["prev"] = frames[index - 1]["token"] if index > 0 else ""
            info["next"] = frames[index + 1]["token"] if index + 1 < len(frames) else ""

        # Without a native-rate view the frames already are every frame of the log, so reuse them
        # as sweep candidates instead of resolving every payload a second time.
        if native_scene is not None:
            candidates = self._sweep_candidates(native_scene, reference_lidar_id, lidar_data_id, timestamp_offset_us)
        else:
            candidates = [_sweep_candidate_from_info(info) for info in frames]
        self._attach_sweeps(frames, candidates)

        self._stats.num_logs += 1
        self._stats.num_frames += len(frames)
        return frames

    def _resolve_rig(self, scene: SceneAPI) -> Optional[Tuple[List[Camera], LidarID, LidarID]]:
        """Cameras, reference lidar and lidar data stream of a log, or ``None`` to skip the log."""
        available_cameras = scene.get_camera_metadatas()
        cameras = []
        for camera_name in self.config.camera_order:
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

        if self.config.reference_lidar is not None:
            reference_lidar_id = LidarID.from_arbitrary(self.config.reference_lidar.lower())
            if reference_lidar_id not in lidar_metadatas:
                logger.warning(
                    "Log '%s' has no lidar %s; skipping the log (available: %s).",
                    scene.log_name,
                    self.config.reference_lidar,
                    sorted(lidar.name for lidar in lidar_metadatas),
                )
                return None
        elif LidarID.LIDAR_TOP in lidar_metadatas:
            reference_lidar_id = LidarID.LIDAR_TOP
        else:
            reference_lidar_id = sorted(lidar_metadatas, key=int)[0]

        # Point clouds usually live in the merged stream, but the reference frame still comes from
        # a physical sensor.
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

    def _convert_frame(
        self,
        scene: SceneAPI,
        iteration: int,
        cameras: List[Camera],
        reference_lidar_id: LidarID,
        lidar_data_id: LidarID,
        scene_token: str,
        timestamp_offset_us: int,
        velocity_table: Optional[TrackVelocityTable],
    ) -> Optional[Dict[str, Any]]:
        """One info, or ``None`` when the frame has to be skipped."""
        ego_state = self._frame_ego_state(scene, iteration)
        if ego_state is None:
            self._stats.frames_skipped_missing_ego += 1
            return None

        ego_to_global = pose_to_matrix(ego_state.imu_se3)
        lidar_to_ego = self._lidar_to_ego(scene, iteration, reference_lidar_id, lidar_data_id, ego_to_global)
        global_to_lidar = invert_rigid(ego_to_global @ lidar_to_ego)

        lidar_path = self._sensors.resolve_lidar(
            scene, iteration, lidar_data_id, ego_to_lidar=invert_rigid(lidar_to_ego)
        )
        if lidar_path is None:
            self._stats.frames_skipped_missing_lidar += 1
            return None

        camera_payloads = self._convert_cameras(scene, iteration, cameras, global_to_lidar, ego_to_global)
        if camera_payloads is None:
            self._stats.frames_skipped_missing_camera += 1
            return None

        timestamp_us = scene.get_timestamp_at_iteration(iteration).time_us
        uuid = _frame_uuid(scene, timestamp_us)
        token = self.config.token_resolver.resolve(
            dataset=scene.dataset, split=scene.split, log_name=scene.log_name, timestamp_us=timestamp_us, uuid=uuid
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
            "cams": {key: payload for key, _, payload in camera_payloads},
            "scene_token": scene_token,
            "lidar2ego_translation": lidar_translation,
            "lidar2ego_rotation": lidar_rotation,
            "ego2global_translation": ego_translation,
            "ego2global_rotation": ego_rotation,
            "timestamp": int(timestamp_us + timestamp_offset_us),
            # Provenance, ignored by mmdet3d.
            "py123d_uuid": uuid,
            "py123d_dataset": scene.dataset,
            "py123d_split": scene.split,
            "py123d_log_name": scene.log_name,
            "py123d_location": scene.location,
            "py123d_iteration": iteration,
        }

        annotations = self._convert_annotations(scene, iteration, global_to_lidar, velocity_table)
        info["gt_boxes"] = annotations.gt_boxes
        info["gt_names"] = annotations.gt_names
        info["gt_velocity"] = annotations.gt_velocity
        info["num_lidar_pts"] = annotations.num_lidar_pts
        info["num_radar_pts"] = annotations.num_radar_pts
        info["valid_flag"] = annotations.valid_flag
        if self.config.include_track_tokens:
            info["instance_tokens"] = annotations.track_tokens
        if self.config.with_2d_annotations:
            info.update(self._convert_2d_annotations(annotations.records, camera_payloads))

        self._stats.num_boxes += len(annotations)
        self._stats.class_counts.update(annotations.gt_names.tolist())
        return info

    def _convert_cameras(
        self,
        scene: SceneAPI,
        iteration: int,
        cameras: List[Camera],
        global_to_lidar: np.ndarray,
        frame_ego_to_global: np.ndarray,
    ) -> Optional[List[Tuple[str, BaseCameraMetadata, Dict[str, Any]]]]:
        """``(key, metadata, payload)`` per camera, or ``None`` to skip the frame."""
        payloads = []
        for camera_id, camera_name, metadata in cameras:
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
                if self.config.on_missing_camera == "error":
                    raise RuntimeError(
                        f"Camera {camera_name} missing at iteration {iteration} of log '{scene.log_name}'."
                    )
                self._stats.dropped_cameras[camera_name] += 1
                if self.config.on_missing_camera == "skip_frame":
                    return None
                continue

            data_path = self._sensors.resolve_camera(scene, iteration, camera_id, camera_name)

            camera_to_global = pose_to_matrix(camera_to_global_pose)
            camera_to_lidar = global_to_lidar @ camera_to_global

            # 123D's camera_to_global is already taken at the camera's own timestamp. The mode only
            # decides whether that offset shows up in ego2global (static) or sensor2ego (dynamic);
            # both compose to the same camera_to_global.
            if self.config.camera2ego_mode == "dynamic":
                camera_to_ego = invert_rigid(frame_ego_to_global) @ camera_to_global
                camera_ego_to_global = frame_ego_to_global
            else:
                camera_to_ego = pose_to_matrix(metadata.camera_to_imu_se3)
                camera_ego_to_global = camera_to_global @ invert_rigid(camera_to_ego)

            sensor_translation, sensor_rotation = matrix_to_translation_quaternion(camera_to_ego)
            ego_translation, ego_rotation = matrix_to_translation_quaternion(camera_ego_to_global)

            intrinsics = metadata.intrinsics
            if intrinsics is None:
                raise RuntimeError(f"Camera {camera_name} of log '{scene.log_name}' has no intrinsics.")

            key = camera_name if self.config.camera_key == "native" else camera_id.name
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
            payloads.append((key, metadata, payload))

        return payloads or None

    def _frame_ego_state(self, scene: SceneAPI, iteration: int):
        if self.config.ego_pose_source == "nearest":
            timestamp_us = int(scene.get_timestamp_at_iteration(iteration).time_us)
            ego_state = scene.get_ego_state_se3_at_timestamp(timestamp_us, criteria="nearest")
            if ego_state is not None:
                if ego_state.timestamp is not None:
                    offset = abs(int(ego_state.timestamp.time_us) - timestamp_us)
                    self._stats.ego_pose_max_offset_us = max(self._stats.ego_pose_max_offset_us, offset)
                return ego_state
            self._stats.ego_pose_fallbacks += 1
        return scene.get_ego_state_se3_at_iteration(iteration)

    def _lidar_to_ego(
        self,
        scene: SceneAPI,
        iteration: int,
        reference_lidar_id: LidarID,
        lidar_data_id: LidarID,
        ego_to_global: np.ndarray,
    ) -> np.ndarray:
        """The export's ``lidar2ego``, see :attr:`ExportConfig.lidar_frame` and :attr:`ExportConfig.lidar2ego_mode`."""
        if self.config.lidar_frame == "ego":
            return np.eye(4, dtype=np.float64)

        static = pose_to_matrix(scene.get_lidar_metadatas()[reference_lidar_id].lidar_to_imu_se3)
        if self.config.lidar2ego_mode == "static":
            return static

        lidar_timestamp_us = scene.get_modality_column_at_iteration(
            iteration=iteration,
            column="timestamp_us",
            modality_type=ModalityType.LIDAR,
            modality_id=lidar_data_id,
            deserialize=False,
        )
        if lidar_timestamp_us is None:
            return static
        # Nearest stored ego state, not an interpolated one: inventing poses would hide how coarse
        # the ego stream is.
        ego_state = scene.get_ego_state_se3_at_timestamp(int(lidar_timestamp_us), criteria="nearest")
        if ego_state is None:
            self._stats.dynamic_extrinsic_fallbacks += 1
            return static
        # Relative to the ego pose this frame reports, so that ego2global @ lidar2ego lands where
        # the lidar was when it fired.
        return invert_rigid(ego_to_global) @ pose_to_matrix(ego_state.imu_se3) @ static

    def _convert_annotations(
        self,
        scene: SceneAPI,
        iteration: int,
        global_to_lidar: np.ndarray,
        velocity_table: Optional[TrackVelocityTable],
    ) -> FrameAnnotations:
        detections = scene.get_box_detections_se3_at_iteration(iteration)
        velocity_overrides = None
        if velocity_table is not None:
            velocity_overrides = velocity_table.at(scene.get_timestamp_at_iteration(iteration).time_us)
        records = extract_detection_records(detections, self.config.taxonomy, velocity_overrides=velocity_overrides)
        if velocity_overrides is not None:
            self._stats.boxes_without_track_velocity += sum(
                1 for record in records if record.track_token not in velocity_overrides
            )
        return build_frame_annotations(
            records,
            global_to_lidar,
            yaw_convention=self.config.yaw_convention,
            planar_velocity=self.config.planar_velocity,
            box_layout=self.config.box_layout,
        )

    def _convert_2d_annotations(
        self, records: List[DetectionRecord], camera_payloads: List[Tuple[str, BaseCameraMetadata, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """StreamPETR's per-camera 2D fields, one list entry per camera."""
        fields_2d: Dict[str, list] = {
            "bboxes2d": [],
            "labels2d": [],
            "centers2d": [],
            "depths": [],
            "bboxes3d_cams": [],
            "bboxes_ignore": [],
            "visibilities": [],
        }
        for _, metadata, payload in camera_payloads:
            # Records are global-frame, so project them through the camera pose stored in the payload.
            annotations = project_records_to_camera(
                records, metadata, _sensor_to_global(payload, "sensor"), yaw_convention=self.config.yaw_convention
            )
            fields_2d["bboxes2d"].append(annotations.bboxes2d)
            fields_2d["labels2d"].append(annotations.labels2d)
            fields_2d["centers2d"].append(annotations.centers2d)
            fields_2d["depths"].append(annotations.depths)
            fields_2d["bboxes3d_cams"].append(annotations.bboxes3d_cam)
            fields_2d["bboxes_ignore"].append(annotations.bboxes_ignore)
            fields_2d["visibilities"].append(annotations.visibilities)
        return fields_2d

    def _sweep_candidates(
        self, scene: SceneAPI, reference_lidar_id: LidarID, lidar_data_id: LidarID, timestamp_offset_us: int
    ) -> List[Dict[str, Any]]:
        """Every lidar frame of a (native-rate) scene, in time order. Point clouds are not decoded."""
        if self.config.max_sweeps == 0 or reference_lidar_id not in scene.get_lidar_metadatas():
            return []

        candidates = []
        for iteration in range(scene.number_of_iterations):
            ego_state = self._frame_ego_state(scene, iteration)
            if ego_state is None:
                continue

            ego_to_global = pose_to_matrix(ego_state.imu_se3)
            lidar_to_ego = self._lidar_to_ego(scene, iteration, reference_lidar_id, lidar_data_id, ego_to_global)
            lidar_translation, lidar_rotation = matrix_to_translation_quaternion(lidar_to_ego)

            try:
                lidar_path = self._sensors.resolve_lidar(
                    scene, iteration, lidar_data_id, ego_to_lidar=invert_rigid(lidar_to_ego)
                )
            except Exception as error:  # sweeps are optional context, never fail the export for one
                logger.debug("Skipping sweep at iteration %d of '%s': %s", iteration, scene.log_name, error)
                continue
            if lidar_path is None:
                continue

            timestamp_us = scene.get_timestamp_at_iteration(iteration).time_us
            ego_translation, ego_rotation = matrix_to_translation_quaternion(ego_to_global)
            candidates.append(
                {
                    "lidar_to_global": ego_to_global @ lidar_to_ego,
                    "payload": {
                        "data_path": lidar_path,
                        "type": "lidar",
                        "sample_data_token": _frame_uuid(scene, timestamp_us),
                        "sensor2ego_translation": lidar_translation,
                        "sensor2ego_rotation": lidar_rotation,
                        "ego2global_translation": ego_translation,
                        "ego2global_rotation": ego_rotation,
                        "timestamp": int(timestamp_us + timestamp_offset_us),
                    },
                }
            )
        return candidates

    def _attach_sweeps(self, frames: List[Dict[str, Any]], candidates: List[Dict[str, Any]]) -> None:
        """Fill ``info['sweeps']`` with the candidates before each frame, newest first like mmdetection3d."""
        if self.config.max_sweeps == 0 or not candidates:
            return

        timestamps = [candidate["payload"]["timestamp"] for candidate in candidates]
        for info in frames:
            end = bisect_left(timestamps, info["timestamp"])
            start = max(0, end - self.config.max_sweeps)
            global_to_lidar = invert_rigid(_sensor_to_global(info, "lidar"))

            sweeps = []
            for candidate in reversed(candidates[start:end]):
                sweep_to_lidar = global_to_lidar @ candidate["lidar_to_global"]
                sweep = dict(candidate["payload"])
                sweep["sensor2lidar_rotation"] = np.ascontiguousarray(sweep_to_lidar[:3, :3])
                sweep["sensor2lidar_translation"] = np.ascontiguousarray(sweep_to_lidar[:3, 3])
                sweeps.append(sweep)
            info["sweeps"] = sweeps


def _frame_uuid(scene: SceneAPI, timestamp_us: int) -> str:
    return str(create_deterministic_uuid(split=scene.split, log_name=scene.log_name, timestamp_us=timestamp_us))


def _sweep_candidate_from_info(info: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "lidar_to_global": _sensor_to_global(info, "lidar"),
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


def _pose_pair_to_matrix(translation: Sequence[float], rotation: Sequence[float]) -> np.ndarray:
    pose = PoseSE3.from_R_t(
        rotation=np.asarray(rotation, dtype=np.float64), translation=np.asarray(translation, dtype=np.float64)
    )
    return np.asarray(pose.transformation_matrix, dtype=np.float64)


def _sensor_to_global(entry: Dict[str, Any], sensor: str) -> np.ndarray:
    """Rebuild ``sensor -> global`` from the pose pairs of an info (``"lidar"``) or camera payload (``"sensor"``)."""
    sensor_to_ego = _pose_pair_to_matrix(entry[f"{sensor}2ego_translation"], entry[f"{sensor}2ego_rotation"])
    ego_to_global = _pose_pair_to_matrix(entry["ego2global_translation"], entry["ego2global_rotation"])
    return ego_to_global @ sensor_to_ego
