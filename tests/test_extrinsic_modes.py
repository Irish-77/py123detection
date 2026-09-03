"""Unit tests for the static / dynamic extrinsic modes.

The two modes answer different questions:

* ``lidar2ego_mode`` — is ``lidar2ego`` the rig calibration, or is it re-derived from the ego pose
  at the lidar's own capture time? 123D stores no per-frame lidar pose, so "dynamic" is
  reconstructed from the ego stream, and it can only differ from "static" when that stream is
  denser than the exported frames.
* ``camera2ego_mode`` — a camera fires off the frame's reference instant, and 123D's per-frame
  ``camera_to_global_se3`` already accounts for it. The mode only decides whether that offset is
  reported in ``ego2global`` or folded into ``sensor2ego``. The composed ``sensor2lidar`` is
  invariant, which these tests pin down.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pytest
from py123d.datatypes import CameraID, LidarID, ModalityType
from py123d.datatypes.sensors.lidar import LidarMetadata
from py123d.datatypes.sensors.pinhole_camera import PinholeCameraMetadata, PinholeIntrinsics
from py123d.datatypes.time import Timestamp
from py123d.datatypes.vehicle_state.ego_state import EgoStateSE3
from py123d.geometry import PoseSE3
from pyquaternion import Quaternion

from py123detection.geometry import invert_rigid, pose_to_matrix
from py123detection.mmcv_export.converter import ExportConfig, MMDet3DConverter
from py123detection.mmcv_export.sensors import SensorResolverConfig

LIDAR_TO_EGO = PoseSE3(0.98, 0.0, 1.84, *Quaternion(axis=[0, 0, 1], radians=0.01).elements)
CAMERA_TO_EGO = PoseSE3(1.7, 0.0, 1.5, *Quaternion(axis=[0, 1, 0], radians=0.02).elements)

FRAME_TIMESTAMP_US = 1_000_000
LIDAR_TIMESTAMP_US = FRAME_TIMESTAMP_US - 50_000  # nuScenes stamps the sweep start
CAMERA_TIMESTAMP_US = FRAME_TIMESTAMP_US - 35_000


def ego_pose_at(timestamp_us: int) -> PoseSE3:
    """A straight-line ego trajectory at 10 m/s, so pose differences are easy to reason about."""
    return PoseSE3(10.0 * timestamp_us / 1e6, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)


class FakeScene:
    """A one-frame scene exposing only what the converter reads.

    ``ego_stream_us`` lists the timestamps at which ego states exist. With a single entry the ego
    stream is exactly as coarse as the frames (the keyframe-only case); adding the lidar timestamp
    makes it denser, which is the case ``lidar2ego_mode="dynamic"`` exists for.
    """

    def __init__(self, ego_stream_us=(FRAME_TIMESTAMP_US,)) -> None:
        self._ego_stream_us = sorted(ego_stream_us)
        self.split = "fake_val"
        self.log_name = "log-0"
        self.dataset = "nuscenes"
        self.location = "town"
        self.number_of_iterations = 1

    # -- rig ---------------------------------------------------------------------------------

    def get_lidar_metadatas(self):
        return {LidarID.LIDAR_TOP: LidarMetadata("LIDAR_TOP", LidarID.LIDAR_TOP, LIDAR_TO_EGO)}

    def get_camera_metadatas(self):
        return {
            CameraID.PCAM_F0: PinholeCameraMetadata(
                camera_name="CAM_FRONT",
                camera_id=CameraID.PCAM_F0,
                intrinsics=PinholeIntrinsics(fx=1000.0, fy=1000.0, cx=800.0, cy=450.0),
                distortion=None,
                width=1600,
                height=900,
                camera_to_imu_se3=CAMERA_TO_EGO,
                is_undistorted=True,
            )
        }

    def get_modality_metadata(self, modality_type, modality_id=None):
        return None  # no merged lidar stream

    # -- per-frame data ------------------------------------------------------------------------

    def get_timestamp_at_iteration(self, iteration: int) -> Timestamp:
        return Timestamp.from_us(FRAME_TIMESTAMP_US)

    def get_ego_state_se3_at_iteration(self, iteration: int) -> EgoStateSE3:
        return self._ego_state(FRAME_TIMESTAMP_US)

    def get_ego_state_se3_at_timestamp(self, timestamp, criteria="exact") -> Optional[EgoStateSE3]:
        nearest = min(self._ego_stream_us, key=lambda t: abs(t - int(timestamp)))
        return self._ego_state(nearest)

    def get_box_detections_se3_at_iteration(self, iteration: int):
        return None

    def get_modality_column_at_iteration(self, iteration, column, modality_type, modality_id=None, deserialize=False):
        if modality_type == ModalityType.LIDAR:
            return LIDAR_TIMESTAMP_US if column == "timestamp_us" else "samples/LIDAR_TOP/x.bin"
        if modality_type == ModalityType.CAMERA:
            if column == "timestamp_us":
                return CAMERA_TIMESTAMP_US
            if column == "camera_to_global_se3":
                # Where the camera really was: the ego pose at its own capture time, plus the rig.
                matrix = pose_to_matrix(ego_pose_at(CAMERA_TIMESTAMP_US)) @ pose_to_matrix(CAMERA_TO_EGO)
                return PoseSE3.from_transformation_matrix(matrix)
            return "samples/CAM_FRONT/x.jpg"
        return None

    @staticmethod
    def _ego_state(timestamp_us: int) -> EgoStateSE3:
        from py123d.datatypes.vehicle_state.ego_state_metadata import EgoStateSE3Metadata

        metadata = EgoStateSE3Metadata(
            vehicle_name="fake",
            width=1.7,
            length=4.0,
            height=1.5,
            wheel_base=2.5,
            center_to_imu_se3=PoseSE3.identity(),
            rear_axle_to_imu_se3=PoseSE3.identity(),
        )
        return EgoStateSE3.from_imu(
            imu_se3=ego_pose_at(timestamp_us), metadata=metadata, timestamp=Timestamp.from_us(timestamp_us)
        )


def export_with(scene: FakeScene, **config_kwargs) -> dict:
    """Convert a specific scene and return its single info.

    Sensor paths are referenced against a fixed fake root — nothing is read from disk, the
    resolver only joins the stored relative path onto it.
    """
    config = ExportConfig(
        camera_order=("PCAM_F0",),
        reference_lidar="LIDAR_TOP",
        max_sweeps=0,
        with_2d_annotations=False,
        sensors=SensorResolverConfig(sensor_roots={"nuscenes": Path("/fake/root")}),
        **config_kwargs,
    )
    return MMDet3DConverter(config).convert_scene(scene)[0]


def transform(translation, rotation) -> np.ndarray:
    """Rebuild a 4x4 from an exported (translation, quaternion) pair."""
    matrix = np.eye(4)
    matrix[:3, :3] = Quaternion(np.asarray(rotation)).rotation_matrix
    matrix[:3, 3] = np.asarray(translation)
    return matrix


# -- configuration ----------------------------------------------------------------------------


def test_modes_default_to_static() -> None:
    config = ExportConfig()
    assert config.lidar2ego_mode == "static"
    assert config.camera2ego_mode == "static"


@pytest.mark.parametrize("field", ["lidar2ego_mode", "camera2ego_mode"])
def test_invalid_mode_is_rejected(field: str) -> None:
    with pytest.raises(ValueError, match=f"{field} must be 'static' or 'dynamic'"):
        ExportConfig(**{field: "sometimes"})


# -- lidar2ego ---------------------------------------------------------------------------------


def test_static_lidar2ego_is_the_rig_calibration() -> None:
    info = export_with(FakeScene(), lidar2ego_mode="static")
    np.testing.assert_allclose(
        transform(info["lidar2ego_translation"], info["lidar2ego_rotation"]),
        pose_to_matrix(LIDAR_TO_EGO),
        atol=1e-12,
    )


def test_dynamic_lidar2ego_is_a_noop_when_the_ego_stream_is_frame_rate() -> None:
    """A keyframe-only conversion stores one ego state per frame — nothing closer to look up."""
    scene = FakeScene(ego_stream_us=(FRAME_TIMESTAMP_US,))
    static = export_with(scene, lidar2ego_mode="static")
    dynamic = export_with(scene, lidar2ego_mode="dynamic")
    np.testing.assert_allclose(
        transform(static["lidar2ego_translation"], static["lidar2ego_rotation"]),
        transform(dynamic["lidar2ego_translation"], dynamic["lidar2ego_rotation"]),
        atol=1e-12,
    )


def test_dynamic_lidar2ego_absorbs_the_offset_when_the_ego_stream_is_denser() -> None:
    """With an ego state at the lidar's own timestamp, the 50 ms of ego motion moves into the extrinsic."""
    scene = FakeScene(ego_stream_us=(LIDAR_TIMESTAMP_US, FRAME_TIMESTAMP_US))
    static = export_with(scene, lidar2ego_mode="static")
    dynamic = export_with(scene, lidar2ego_mode="dynamic")

    static_matrix = transform(static["lidar2ego_translation"], static["lidar2ego_rotation"])
    dynamic_matrix = transform(dynamic["lidar2ego_translation"], dynamic["lidar2ego_rotation"])

    # 50 ms at 10 m/s = 0.5 m, and the ego moves +x, so the sensor sits 0.5 m behind in ego frame.
    assert dynamic_matrix[0, 3] - static_matrix[0, 3] == pytest.approx(-0.5, abs=1e-9)


def test_dynamic_lidar2ego_reconstructs_the_true_sensor_pose() -> None:
    """ego2global @ lidar2ego must land where the lidar actually was when it fired."""
    scene = FakeScene(ego_stream_us=(LIDAR_TIMESTAMP_US, FRAME_TIMESTAMP_US))
    info = export_with(scene, lidar2ego_mode="dynamic")

    lidar_to_global = transform(info["ego2global_translation"], info["ego2global_rotation"]) @ transform(
        info["lidar2ego_translation"], info["lidar2ego_rotation"]
    )
    expected = pose_to_matrix(ego_pose_at(LIDAR_TIMESTAMP_US)) @ pose_to_matrix(LIDAR_TO_EGO)
    np.testing.assert_allclose(lidar_to_global, expected, atol=1e-9)


def test_ego2global_is_the_frame_pose_under_both_modes() -> None:
    """Dynamic mode moves the offset into the extrinsic; it must not rewrite ego2global."""
    scene = FakeScene(ego_stream_us=(LIDAR_TIMESTAMP_US, FRAME_TIMESTAMP_US))
    for mode in ("static", "dynamic"):
        info = export_with(scene, lidar2ego_mode=mode)
        np.testing.assert_allclose(
            info["ego2global_translation"], pose_to_matrix(ego_pose_at(FRAME_TIMESTAMP_US))[:3, 3], atol=1e-12
        )


def test_dynamic_lidar2ego_is_ignored_in_the_ego_frame() -> None:
    """`lidar_frame="ego"` makes the ego frame the reference, so lidar2ego is the identity."""
    scene = FakeScene(ego_stream_us=(LIDAR_TIMESTAMP_US, FRAME_TIMESTAMP_US))
    info = export_with(scene, lidar_frame="ego", lidar2ego_mode="dynamic")
    np.testing.assert_allclose(
        transform(info["lidar2ego_translation"], info["lidar2ego_rotation"]), np.eye(4), atol=1e-12
    )


# -- camera2ego --------------------------------------------------------------------------------


def test_static_camera2ego_is_the_rig_calibration() -> None:
    camera = export_with(FakeScene(), camera2ego_mode="static")["cams"]["CAM_FRONT"]
    np.testing.assert_allclose(
        transform(camera["sensor2ego_translation"], camera["sensor2ego_rotation"]),
        pose_to_matrix(CAMERA_TO_EGO),
        atol=1e-12,
    )


def test_dynamic_camera2ego_folds_the_timing_offset_into_the_extrinsic() -> None:
    camera = export_with(FakeScene(), camera2ego_mode="dynamic")["cams"]["CAM_FRONT"]
    sensor_to_ego = transform(camera["sensor2ego_translation"], camera["sensor2ego_rotation"])
    # 35 ms at 10 m/s = 0.35 m.
    assert sensor_to_ego[0, 3] - pose_to_matrix(CAMERA_TO_EGO)[0, 3] == pytest.approx(-0.35, abs=1e-9)


def test_dynamic_camera2ego_reports_the_frames_shared_ego_pose() -> None:
    info = export_with(FakeScene(), camera2ego_mode="dynamic")
    camera = info["cams"]["CAM_FRONT"]
    np.testing.assert_allclose(camera["ego2global_translation"], info["ego2global_translation"], atol=1e-12)


def test_static_camera2ego_reports_the_cameras_own_ego_pose() -> None:
    info = export_with(FakeScene(), camera2ego_mode="static")
    camera = info["cams"]["CAM_FRONT"]
    np.testing.assert_allclose(
        camera["ego2global_translation"], pose_to_matrix(ego_pose_at(CAMERA_TIMESTAMP_US))[:3, 3], atol=1e-9
    )
    assert not np.allclose(camera["ego2global_translation"], info["ego2global_translation"])


@pytest.mark.parametrize("lidar_mode", ["static", "dynamic"])
def test_sensor2lidar_is_invariant_to_camera2ego_mode(lidar_mode: str) -> None:
    """PETR and StreamPETR only see sensor2lidar, so the camera convention must not disturb it."""
    scene = FakeScene(ego_stream_us=(LIDAR_TIMESTAMP_US, FRAME_TIMESTAMP_US))
    static = export_with(scene, camera2ego_mode="static", lidar2ego_mode=lidar_mode)["cams"]["CAM_FRONT"]
    dynamic = export_with(scene, camera2ego_mode="dynamic", lidar2ego_mode=lidar_mode)["cams"]["CAM_FRONT"]
    np.testing.assert_allclose(
        static["sensor2lidar_rotation"], dynamic["sensor2lidar_rotation"], atol=1e-12
    )
    np.testing.assert_allclose(
        static["sensor2lidar_translation"], dynamic["sensor2lidar_translation"], atol=1e-12
    )


@pytest.mark.parametrize("camera_mode", ["static", "dynamic"])
def test_camera_pose_round_trips_under_both_modes(camera_mode: str) -> None:
    """Whatever the split, ego2global @ sensor2ego must rebuild the true camera pose."""
    camera = export_with(FakeScene(), camera2ego_mode=camera_mode)["cams"]["CAM_FRONT"]
    camera_to_global = transform(camera["ego2global_translation"], camera["ego2global_rotation"]) @ transform(
        camera["sensor2ego_translation"], camera["sensor2ego_rotation"]
    )
    expected = pose_to_matrix(ego_pose_at(CAMERA_TIMESTAMP_US)) @ pose_to_matrix(CAMERA_TO_EGO)
    np.testing.assert_allclose(camera_to_global, expected, atol=1e-9)


def test_sensor2lidar_composes_with_the_dynamic_lidar_frame() -> None:
    """sensor2lidar must be expressed in whichever lidar frame the export chose."""
    scene = FakeScene(ego_stream_us=(LIDAR_TIMESTAMP_US, FRAME_TIMESTAMP_US))
    info = export_with(scene, lidar2ego_mode="dynamic")
    camera = info["cams"]["CAM_FRONT"]

    lidar_to_global = transform(info["ego2global_translation"], info["ego2global_rotation"]) @ transform(
        info["lidar2ego_translation"], info["lidar2ego_rotation"]
    )
    camera_to_global = pose_to_matrix(ego_pose_at(CAMERA_TIMESTAMP_US)) @ pose_to_matrix(CAMERA_TO_EGO)
    expected = invert_rigid(lidar_to_global) @ camera_to_global

    np.testing.assert_allclose(camera["sensor2lidar_rotation"], expected[:3, :3], atol=1e-9)
    np.testing.assert_allclose(camera["sensor2lidar_translation"], expected[:3, 3], atol=1e-9)
