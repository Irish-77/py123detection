"""Unit tests for the label taxonomy layer."""

from __future__ import annotations

import pytest
from py123d.parser.registry import (
    AV2SensorBoxDetectionLabel,
    NuScenesBoxDetectionLabel,
    WODPerceptionBoxDetectionLabel,
)

from py123detection.taxonomy import (
    GENERAL_3CLS,
    NUSCENES_DETECTION,
    TAXONOMIES,
    VEHICLE_1CLS,
    VOID,
    Taxonomy,
    get_taxonomy,
)


def test_nuscenes_taxonomy_matches_the_mmdet3d_class_order() -> None:
    """Label ids are positional, so the order has to match mmdet3d's NuScenesDataset.CLASSES."""
    assert NUSCENES_DETECTION.class_names == (
        "car",
        "truck",
        "trailer",
        "bus",
        "construction_vehicle",
        "bicycle",
        "motorcycle",
        "pedestrian",
        "traffic_cone",
        "barrier",
    )


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        (NuScenesBoxDetectionLabel.VEHICLE_CAR, "car"),
        (NuScenesBoxDetectionLabel.VEHICLE_BUS_BENDY, "bus"),
        (NuScenesBoxDetectionLabel.VEHICLE_BUS_RIGID, "bus"),
        (NuScenesBoxDetectionLabel.HUMAN_PEDESTRIAN_CHILD, "pedestrian"),
        (NuScenesBoxDetectionLabel.MOVABLE_OBJECT_TRAFFICCONE, "traffic_cone"),
        # Outside the 10-class benchmark, exactly as mmdet3d's NameMapping has no entry for them.
        (NuScenesBoxDetectionLabel.ANIMAL, None),
        (NuScenesBoxDetectionLabel.VEHICLE_EMERGENCY_POLICE, None),
        (NuScenesBoxDetectionLabel.MOVABLE_OBJECT_DEBRIS, None),
    ],
)
def test_nuscenes_native_resolution(label, expected) -> None:
    assert NUSCENES_DETECTION.resolve(label) == expected


def test_nuscenes_taxonomy_drops_everything_from_another_dataset() -> None:
    """It only has native-level rules, so it cannot classify a non-nuScenes label."""
    assert NUSCENES_DETECTION.resolve(WODPerceptionBoxDetectionLabel.TYPE_VEHICLE) is None


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        (NuScenesBoxDetectionLabel.VEHICLE_CAR, "vehicle"),
        (NuScenesBoxDetectionLabel.HUMAN_PEDESTRIAN_ADULT, "pedestrian"),
        (NuScenesBoxDetectionLabel.VEHICLE_BICYCLE, "cyclist"),
        (WODPerceptionBoxDetectionLabel.TYPE_VEHICLE, "vehicle"),
        (WODPerceptionBoxDetectionLabel.TYPE_PEDESTRIAN, "pedestrian"),
        (WODPerceptionBoxDetectionLabel.TYPE_CYCLIST, "cyclist"),
        (AV2SensorBoxDetectionLabel.REGULAR_VEHICLE, "vehicle"),
        (AV2SensorBoxDetectionLabel.PEDESTRIAN, "pedestrian"),
        (AV2SensorBoxDetectionLabel.BICYCLE, "cyclist"),
    ],
)
def test_general_taxonomy_spans_datasets(label, expected) -> None:
    """The point of a default-label taxonomy: one rule set, every 123D dataset."""
    assert GENERAL_3CLS.resolve(label) == expected


def test_vehicle_taxonomy_drops_non_vehicles() -> None:
    assert VEHICLE_1CLS.resolve(NuScenesBoxDetectionLabel.VEHICLE_TRUCK) == "vehicle"
    assert VEHICLE_1CLS.resolve(NuScenesBoxDetectionLabel.HUMAN_PEDESTRIAN_ADULT) is None


def test_label_ids_are_positions() -> None:
    assert NUSCENES_DETECTION.label_id("car") == 0
    assert NUSCENES_DETECTION.label_id("barrier") == 9
    assert NUSCENES_DETECTION.class_ids["pedestrian"] == 7


def test_native_map_wins_over_default_map() -> None:
    taxonomy = Taxonomy(
        name="test",
        class_names=("a", "b"),
        native_map={"VEHICLE_CAR": "a"},
        default_map={"VEHICLE": "b"},
    )
    assert taxonomy.resolve(NuScenesBoxDetectionLabel.VEHICLE_CAR) == "a"
    assert taxonomy.resolve(NuScenesBoxDetectionLabel.VEHICLE_TRUCK) == "b"


def test_void_and_missing_both_drop_the_box() -> None:
    taxonomy = Taxonomy(name="test", class_names=("a",), native_map={"VEHICLE_CAR": VOID})
    assert taxonomy.resolve(NuScenesBoxDetectionLabel.VEHICLE_CAR) is None
    assert taxonomy.resolve(NuScenesBoxDetectionLabel.VEHICLE_TRUCK) is None


def test_taxonomy_rejects_a_target_outside_its_class_names() -> None:
    with pytest.raises(ValueError, match="unknown class names"):
        Taxonomy(name="broken", class_names=("a",), native_map={"VEHICLE_CAR": "typo"})


def test_taxonomy_rejects_duplicate_class_names() -> None:
    with pytest.raises(ValueError, match="duplicate class names"):
        Taxonomy(name="broken", class_names=("a", "a"))


def test_get_taxonomy_lookup() -> None:
    assert get_taxonomy("general_3cls") is GENERAL_3CLS
    assert set(TAXONOMIES) == {taxonomy.name for taxonomy in TAXONOMIES.values()}
    with pytest.raises(KeyError, match="Unknown taxonomy"):
        get_taxonomy("nope")
