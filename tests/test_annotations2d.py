"""Unit tests for the per-camera 2D / mono-3D projection."""

from __future__ import annotations

import numpy as np
import pytest
from py123d.datatypes import CameraID
from py123d.datatypes.sensors.pinhole_camera import PinholeCameraMetadata, PinholeIntrinsics
from py123d.geometry import PoseSE3
from pyquaternion import Quaternion

from py123detection.annotations import extract_detection_records
from py123detection.mmcv_export.annotations2d import project_records_to_camera
from tests.test_annotations import make_detection, make_detections

WIDTH, HEIGHT = 1600, 900


def make_camera() -> PinholeCameraMetadata:
    """A simple undistorted pinhole camera, principal point at the image center."""
    return PinholeCameraMetadata(
        camera_name="CAM_FRONT",
        camera_id=CameraID.PCAM_F0,
        intrinsics=PinholeIntrinsics(fx=1000.0, fy=1000.0, cx=WIDTH / 2, cy=HEIGHT / 2),
        distortion=None,
        width=WIDTH,
        height=HEIGHT,
        camera_to_imu_se3=PoseSE3.identity(),
        is_undistorted=True,
    )


def camera_at(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> np.ndarray:
    """Camera-to-global transform for a camera looking down global +z (OpenCV convention)."""
    matrix = np.eye(4)
    matrix[:3, 3] = [x, y, z]
    return matrix


def records_for(**kwargs):
    """Build detection records from one synthetic box."""
    from py123detection.taxonomy import NUSCENES_DETECTION

    return extract_detection_records(make_detections(make_detection(**kwargs)), NUSCENES_DETECTION)


def test_box_in_front_projects_into_the_image() -> None:
    # 123D cameras look along +z, so a box at z=+20 is straight ahead.
    annotations = project_records_to_camera(records_for(center=(0.0, 0.0, 20.0)), make_camera(), camera_at())

    assert annotations.bboxes2d.shape == (1, 4)
    assert annotations.labels2d.tolist() == [0]
    assert annotations.depths[0] == pytest.approx(20.0, abs=1e-6)
    # A centered box projects to the principal point.
    np.testing.assert_allclose(annotations.centers2d[0], [WIDTH / 2, HEIGHT / 2], atol=1e-6)
    assert annotations.visibilities == [""]
    assert annotations.bboxes_ignore.shape == (0, 4)


def test_box_behind_the_camera_is_dropped() -> None:
    annotations = project_records_to_camera(records_for(center=(0.0, 0.0, -20.0)), make_camera(), camera_at())
    assert annotations.bboxes2d.shape == (0, 4)
    assert annotations.depths.shape == (0,)


def test_box_far_off_axis_is_dropped() -> None:
    annotations = project_records_to_camera(records_for(center=(500.0, 0.0, 20.0)), make_camera(), camera_at())
    assert annotations.bboxes2d.shape == (0, 4)


def test_no_records_gives_empty_arrays_with_correct_shapes() -> None:
    annotations = project_records_to_camera([], make_camera(), camera_at())
    assert annotations.bboxes2d.shape == (0, 4)
    assert annotations.centers2d.shape == (0, 2)
    assert annotations.bboxes3d_cam.shape == (0, 7)
    assert annotations.labels2d.shape == (0,)


def test_projected_box_is_clipped_to_the_image() -> None:
    """A box straddling the frame edge must not report coordinates outside the canvas."""
    annotations = project_records_to_camera(
        records_for(center=(15.0, 0.0, 20.0), size=(20.0, 20.0, 20.0)), make_camera(), camera_at()
    )
    x1, y1, x2, y2 = annotations.bboxes2d[0]
    assert 0.0 <= x1 < x2 <= WIDTH
    assert 0.0 <= y1 < y2 <= HEIGHT


def test_mono3d_box_uses_the_lhw_ordering() -> None:
    """mmdet3d's mono-3D layout is [x, y, z, l, h, w, -yaw] — note h before w."""
    annotations = project_records_to_camera(
        records_for(center=(0.0, 0.0, 20.0), size=(4.0, 2.0, 1.5)), make_camera(), camera_at()
    )
    box = annotations.bboxes3d_cam[0]
    np.testing.assert_allclose(box[:3], [0.0, 0.0, 20.0], atol=1e-5)
    np.testing.assert_allclose(box[3:6], [4.0, 1.5, 2.0], atol=1e-5)


def test_camera_translation_shifts_the_projected_depth() -> None:
    annotations = project_records_to_camera(
        records_for(center=(0.0, 0.0, 20.0)), make_camera(), camera_at(z=5.0)
    )
    assert annotations.depths[0] == pytest.approx(15.0, abs=1e-6)


def test_yaw_conventions_differ_in_a_camera_frame() -> None:
    """A box lying in a camera frame carries a large roll, where the conventions diverge sharply."""
    # Rolling the camera -90 degrees about x points its optical axis along global +y, so a box
    # at global (0, 20, 0) sits in front of it while carrying a large roll in the camera frame.
    camera_to_global = np.eye(4)
    camera_to_global[:3, :3] = Quaternion(axis=[1, 0, 0], radians=-np.pi / 2).rotation_matrix

    common = dict(center=(0.0, 20.0, 0.0), yaw=0.7)
    mm = project_records_to_camera(records_for(**common), make_camera(), camera_to_global, "mmdet3d")
    heading = project_records_to_camera(records_for(**common), make_camera(), camera_to_global, "heading")

    assert mm.bboxes3d_cam.shape[0] == 1
    assert abs(mm.bboxes3d_cam[0, 6] - heading.bboxes3d_cam[0, 6]) > 1e-3


def test_arrays_have_matching_lengths() -> None:
    from py123detection.taxonomy import NUSCENES_DETECTION

    detections = make_detections(
        make_detection(center=(0.0, 0.0, 20.0), track_token="a"),
        make_detection(center=(2.0, 0.0, 25.0), track_token="b"),
        make_detection(center=(0.0, 0.0, -20.0), track_token="c"),  # behind: dropped
    )
    annotations = project_records_to_camera(
        extract_detection_records(detections, NUSCENES_DETECTION), make_camera(), camera_at()
    )
    count = annotations.bboxes2d.shape[0]
    assert count == 2
    assert annotations.labels2d.shape[0] == count
    assert annotations.centers2d.shape[0] == count
    assert annotations.depths.shape[0] == count
    assert annotations.bboxes3d_cam.shape[0] == count
    assert len(annotations.visibilities) == count
