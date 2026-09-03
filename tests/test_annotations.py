"""Unit tests for the annotation transform, on synthetic geometry with known answers."""

from __future__ import annotations

import numpy as np
import pytest
from py123d.datatypes import BoxDetectionsSE3
from py123d.datatypes.detections.box_detections import BoxDetectionAttributes, BoxDetectionSE3
from py123d.datatypes.detections.box_detections_metadata import BoxDetectionsSE3Metadata
from py123d.datatypes.time import Timestamp
from py123d.geometry import BoundingBoxSE3, PoseSE3, Vector3D
from py123d.parser.registry import NuScenesBoxDetectionLabel
from pyquaternion import Quaternion

from py123detection.annotations import build_frame_annotations, extract_detection_records
from py123detection.geometry import invert_rigid
from py123detection.taxonomy import NUSCENES_DETECTION

METADATA = BoxDetectionsSE3Metadata(box_detection_label_class=NuScenesBoxDetectionLabel)


def make_detection(
    label=NuScenesBoxDetectionLabel.VEHICLE_CAR,
    center=(10.0, 5.0, 1.0),
    yaw: float = 0.3,
    size=(4.0, 2.0, 1.5),
    velocity=(1.0, 2.0, 3.0),
    num_lidar_points: int = 7,
    track_token: str = "track-0",
) -> BoxDetectionSE3:
    """Build one SE3 box detection in the global frame."""
    quaternion = Quaternion(axis=[0, 0, 1], radians=yaw)
    return BoxDetectionSE3(
        attributes=BoxDetectionAttributes(
            label=label, track_token=track_token, num_lidar_points=num_lidar_points
        ),
        bounding_box_se3=BoundingBoxSE3(
            center_se3=PoseSE3(center[0], center[1], center[2], *quaternion.elements),
            length=size[0],
            width=size[1],
            height=size[2],
        ),
        velocity_3d=Vector3D(*velocity),
    )


def make_detections(*detections: BoxDetectionSE3) -> BoxDetectionsSE3:
    """Wrap detections into a modality container."""
    return BoxDetectionsSE3(
        box_detections=list(detections), timestamp=Timestamp.from_us(1_000_000), metadata=METADATA
    )


# -- record extraction ------------------------------------------------------------------------


def test_extract_keeps_mapped_and_drops_unmapped_labels() -> None:
    detections = make_detections(
        make_detection(NuScenesBoxDetectionLabel.VEHICLE_CAR, track_token="a"),
        make_detection(NuScenesBoxDetectionLabel.ANIMAL, track_token="b"),
        make_detection(NuScenesBoxDetectionLabel.HUMAN_PEDESTRIAN_ADULT, track_token="c"),
    )
    records = extract_detection_records(detections, NUSCENES_DETECTION)
    assert [record.class_name for record in records] == ["car", "pedestrian"]
    assert [record.track_token for record in records] == ["a", "c"]
    assert [record.label_id for record in records] == [0, 7]


def test_extract_handles_no_detections() -> None:
    assert extract_detection_records(None, NUSCENES_DETECTION) == []
    assert extract_detection_records(make_detections(), NUSCENES_DETECTION) == []


def test_extract_reads_size_as_length_width_height() -> None:
    records = extract_detection_records(make_detections(make_detection(size=(4.0, 2.0, 1.5))), NUSCENES_DETECTION)
    np.testing.assert_allclose(records[0].size_lwh, [4.0, 2.0, 1.5])


# -- lidar-frame projection ---------------------------------------------------------------------


def test_identity_transform_is_a_passthrough() -> None:
    records = extract_detection_records(make_detections(make_detection()), NUSCENES_DETECTION)
    annotations = build_frame_annotations(records, np.eye(4))

    np.testing.assert_allclose(annotations.gt_boxes[0, :3], [10.0, 5.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(annotations.gt_boxes[0, 3:6], [4.0, 2.0, 1.5], atol=1e-12)
    assert annotations.gt_boxes[0, 6] == pytest.approx(0.3, abs=1e-12)
    np.testing.assert_allclose(annotations.gt_velocity[0], [1.0, 2.0], atol=1e-12)
    assert annotations.gt_names.tolist() == ["car"]


def test_translation_and_rotation_are_applied() -> None:
    """A yaw of +90 degrees plus a shift maps a known global box to a known lidar-frame box."""
    global_to_lidar = np.eye(4)
    global_to_lidar[:3, :3] = Quaternion(axis=[0, 0, 1], radians=np.pi / 2).rotation_matrix
    global_to_lidar[:3, 3] = [1.0, 2.0, 3.0]

    records = extract_detection_records(
        make_detections(make_detection(center=(10.0, 5.0, 1.0), yaw=0.3)), NUSCENES_DETECTION
    )
    annotations = build_frame_annotations(records, global_to_lidar)

    np.testing.assert_allclose(annotations.gt_boxes[0, :3], [-5.0 + 1.0, 10.0 + 2.0, 1.0 + 3.0], atol=1e-12)
    assert annotations.gt_boxes[0, 6] == pytest.approx(0.3 + np.pi / 2, abs=1e-12)
    np.testing.assert_allclose(annotations.gt_boxes[0, 3:6], [4.0, 2.0, 1.5], atol=1e-12)


def test_velocity_rotates_but_does_not_translate() -> None:
    global_to_lidar = np.eye(4)
    global_to_lidar[:3, :3] = Quaternion(axis=[0, 0, 1], radians=np.pi / 2).rotation_matrix
    global_to_lidar[:3, 3] = [100.0, 200.0, 300.0]

    records = extract_detection_records(make_detections(make_detection(velocity=(1.0, 0.0, 0.0))), NUSCENES_DETECTION)
    annotations = build_frame_annotations(records, global_to_lidar)
    np.testing.assert_allclose(annotations.gt_velocity[0], [0.0, 1.0], atol=1e-12)


def test_planar_velocity_drops_the_vertical_component_before_rotating() -> None:
    """mmdet3d zeroes vz first; keeping it lets a tilted lidar mix it into (vx, vy)."""
    tilt = np.eye(4)
    tilt[:3, :3] = Quaternion(axis=[1, 0, 0], radians=0.2).rotation_matrix
    records = extract_detection_records(make_detections(make_detection(velocity=(0.0, 0.0, 5.0))), NUSCENES_DETECTION)

    planar = build_frame_annotations(records, tilt, planar_velocity=True)
    full = build_frame_annotations(records, tilt, planar_velocity=False)

    np.testing.assert_allclose(planar.gt_velocity[0], [0.0, 0.0], atol=1e-12)
    assert abs(full.gt_velocity[0, 1]) > 0.9


def test_yaw_convention_is_selectable() -> None:
    tilt = np.eye(4)
    tilt[:3, :3] = Quaternion(axis=[1, 0, 0], radians=0.6).rotation_matrix
    records = extract_detection_records(make_detections(make_detection(yaw=1.1)), NUSCENES_DETECTION)

    mm = build_frame_annotations(records, tilt, yaw_convention="mmdet3d").gt_boxes[0, 6]
    heading = build_frame_annotations(records, tilt, yaw_convention="heading").gt_boxes[0, 6]
    assert abs(mm - heading) > 1e-6


# -- annotation bookkeeping ----------------------------------------------------------------------


def test_valid_flag_follows_lidar_point_counts() -> None:
    records = extract_detection_records(
        make_detections(
            make_detection(num_lidar_points=5, track_token="a"),
            make_detection(num_lidar_points=0, track_token="b"),
            make_detection(num_lidar_points=None, track_token="c"),
        ),
        NUSCENES_DETECTION,
    )
    annotations = build_frame_annotations(records, np.eye(4))

    # Unknown counts (-1) stay valid rather than being silently dropped.
    assert annotations.valid_flag.tolist() == [True, False, True]
    assert annotations.num_lidar_pts.tolist() == [5, 0, -1]
    assert annotations.num_radar_pts.tolist() == [0, 0, 0]


def test_empty_annotations_have_the_right_shapes() -> None:
    annotations = build_frame_annotations([], np.eye(4))
    assert annotations.gt_boxes.shape == (0, 7)
    assert annotations.gt_velocity.shape == (0, 2)
    assert len(annotations) == 0


def test_missing_velocity_becomes_zero() -> None:
    detection = BoxDetectionSE3(
        attributes=BoxDetectionAttributes(label=NuScenesBoxDetectionLabel.VEHICLE_CAR, track_token="a"),
        bounding_box_se3=BoundingBoxSE3(center_se3=PoseSE3.identity(), length=1.0, width=1.0, height=1.0),
        velocity_3d=None,
    )
    annotations = build_frame_annotations(
        extract_detection_records(make_detections(detection), NUSCENES_DETECTION), np.eye(4)
    )
    np.testing.assert_allclose(annotations.gt_velocity[0], [0.0, 0.0])


def test_round_trip_through_a_random_transform_recovers_the_global_center() -> None:
    rng = np.random.default_rng(0)
    lidar_to_global = np.eye(4)
    lidar_to_global[:3, :3] = Quaternion(np.asarray(rng.normal(size=4))).normalised.rotation_matrix
    lidar_to_global[:3, 3] = rng.normal(scale=50, size=3)

    records = extract_detection_records(make_detections(make_detection(center=(10.0, 5.0, 1.0))), NUSCENES_DETECTION)
    annotations = build_frame_annotations(records, invert_rigid(lidar_to_global))
    back = lidar_to_global[:3, :3] @ annotations.gt_boxes[0, :3] + lidar_to_global[:3, 3]
    np.testing.assert_allclose(back, [10.0, 5.0, 1.0], atol=1e-10)
