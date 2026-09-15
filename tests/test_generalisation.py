"""Unit tests for the dataset-generalisation features added for the Argoverse 2 arm.

Covers: track-derived velocity (rule and overrides), the mmdet3d-0.17 box layout, qualified
native taxonomy keys, the log/timestamp token resolver, and the CoIn3D taxonomy's semantics.
"""

from __future__ import annotations

import numpy as np
import pytest
from py123d.parser.registry import AV2SensorBoxDetectionLabel, NuScenesBoxDetectionLabel

from py123detection.annotations import BOX_LAYOUTS, build_frame_annotations, extract_detection_records
from py123detection.mmcv_export.tokens import LogTimestampTokenResolver
from py123detection.taxonomy import COIN3D_3CLS, TAXONOMIES, VOID, Taxonomy
from py123detection.velocity import velocities_from_track
from tests.test_annotations import make_detection, make_detections

# -- velocity rule ------------------------------------------------------------------------------


def test_velocity_central_one_sided_and_isolated() -> None:
    times = np.array([0, 100_000, 200_000, 300_000, 700_000])
    centers = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [10, 0, 0]], dtype=float)
    v = velocities_from_track(times, centers, max_dt_s=0.25)
    # start: one-sided forward; middle: central; index 3: next is 0.4 s away -> backward only;
    # last: no neighbour within 0.25 s -> zero.
    assert np.allclose(v[:, 0], [10.0, 10.0, 10.0, 10.0, 0.0])
    assert np.allclose(v[:, 1:], 0.0)


def test_velocity_single_observation_is_zero() -> None:
    v = velocities_from_track(np.array([5]), np.array([[1.0, 2.0, 3.0]]))
    assert v.shape == (1, 3) and np.all(v == 0)


def test_velocity_rejects_unsorted_times() -> None:
    with pytest.raises(ValueError):
        velocities_from_track(np.array([1, 0]), np.zeros((2, 3)))


def test_velocity_overrides_replace_stored_velocity_by_track() -> None:
    detections = make_detections(
        make_detection(track_token="a", velocity=(1.0, 2.0, 3.0)),
        make_detection(track_token="b", velocity=(4.0, 5.0, 6.0)),
    )
    from py123detection.taxonomy import NUSCENES_DETECTION

    records = extract_detection_records(
        detections, NUSCENES_DETECTION, velocity_overrides={"a": np.array([9.0, 8.0, 7.0])}
    )
    by_token = {record.track_token: record for record in records}
    assert np.allclose(by_token["a"].velocity_global, [9.0, 8.0, 7.0])
    assert np.allclose(by_token["b"].velocity_global, [4.0, 5.0, 6.0])  # untouched


# -- box layout ---------------------------------------------------------------------------------


def test_box_layouts_are_an_exact_column_map() -> None:
    from py123detection.taxonomy import NUSCENES_DETECTION

    records = extract_detection_records(
        make_detections(make_detection(size=(4.0, 2.0, 1.5), yaw=0.3)), NUSCENES_DETECTION
    )
    identity = np.eye(4)
    streampetr = build_frame_annotations(records, identity, box_layout="streampetr").gt_boxes[0]
    mmdet3d = build_frame_annotations(records, identity, box_layout="mmdet3d_0.17").gt_boxes[0]
    assert np.allclose(streampetr[:3], mmdet3d[:3])
    assert np.allclose(streampetr[3:6], [4.0, 2.0, 1.5])  # l, w, h
    assert np.allclose(mmdet3d[3:6], [2.0, 4.0, 1.5])  # w, l, h
    assert np.isclose(mmdet3d[6], -streampetr[6] - np.pi / 2)
    assert "streampetr" in BOX_LAYOUTS and "mmdet3d_0.17" in BOX_LAYOUTS


def test_unknown_box_layout_is_rejected() -> None:
    with pytest.raises(ValueError):
        build_frame_annotations([], np.eye(4), box_layout="bogus")


# -- taxonomy ------------------------------------------------------------------------------------


def test_qualified_native_key_beats_bare_key_beats_default() -> None:
    taxonomy = Taxonomy(
        name="t",
        class_names=("x", "y", "z"),
        native_map={"AV2SensorBoxDetectionLabel.PEDESTRIAN": "x", "PEDESTRIAN": "y"},
        default_map={"PERSON": "z"},
    )
    assert taxonomy.resolve(AV2SensorBoxDetectionLabel.PEDESTRIAN) == "x"  # qualified wins
    assert taxonomy.resolve(AV2SensorBoxDetectionLabel.OFFICIAL_SIGNALER) == "z"  # default level
    bare_only = Taxonomy(name="u", class_names=("y", "z"), native_map={"PEDESTRIAN": "y"}, default_map={"PERSON": "z"})
    assert bare_only.resolve(AV2SensorBoxDetectionLabel.PEDESTRIAN) == "y"


def test_coin3d_taxonomy_is_registered_with_nuscenes_spelling() -> None:
    assert TAXONOMIES["coin3d_3cls"] is COIN3D_3CLS
    assert COIN3D_3CLS.class_names == ("car", "pedestrian", "motorcycle")


@pytest.mark.parametrize(
    "label, expected",
    [
        (AV2SensorBoxDetectionLabel.REGULAR_VEHICLE, "car"),
        (AV2SensorBoxDetectionLabel.BUS, "car"),
        (AV2SensorBoxDetectionLabel.RAILED_VEHICLE, "car"),
        (AV2SensorBoxDetectionLabel.PEDESTRIAN, "pedestrian"),
        (AV2SensorBoxDetectionLabel.OFFICIAL_SIGNALER, "pedestrian"),
        (AV2SensorBoxDetectionLabel.BICYCLE, "motorcycle"),
        (AV2SensorBoxDetectionLabel.MOTORCYCLE, "motorcycle"),
        (AV2SensorBoxDetectionLabel.BICYCLIST, None),
        (AV2SensorBoxDetectionLabel.MOTORCYCLIST, None),
        (AV2SensorBoxDetectionLabel.WHEELED_RIDER, None),
        (AV2SensorBoxDetectionLabel.WHEELCHAIR, None),
        (AV2SensorBoxDetectionLabel.BOLLARD, None),
        (AV2SensorBoxDetectionLabel.DOG, None),
        (NuScenesBoxDetectionLabel.VEHICLE_TRUCK, "car"),
        (NuScenesBoxDetectionLabel.VEHICLE_BICYCLE, "motorcycle"),
        (NuScenesBoxDetectionLabel.HUMAN_PEDESTRIAN_CHILD, "pedestrian"),
        (NuScenesBoxDetectionLabel.HUMAN_PEDESTRIAN_STROLLER, None),
        (NuScenesBoxDetectionLabel.VEHICLE_EMERGENCY_POLICE, None),
        (NuScenesBoxDetectionLabel.MOVABLE_OBJECT_TRAFFICCONE, None),
    ],
)
def test_coin3d_mapping(label, expected) -> None:
    assert COIN3D_3CLS.resolve(label) == expected


def test_coin3d_void_entries_are_explicit() -> None:
    assert COIN3D_3CLS.native_map["AV2SensorBoxDetectionLabel.BICYCLIST"] == VOID


# -- tokens ----------------------------------------------------------------------------------------


def test_log_timestamp_token_is_reproducible_and_ignores_the_uuid() -> None:
    resolver = LogTimestampTokenResolver()
    token = resolver.resolve("av2-sensor", "av2-sensor_val", "log-1", 315969904359876, "uuid-xyz")
    assert token == "log-1/315969904359876"
    assert resolver.name == "log_timestamp"
