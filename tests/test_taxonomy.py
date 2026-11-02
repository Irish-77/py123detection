"""Unit tests for the label taxonomy layer."""

from __future__ import annotations

import pytest
from py123d.parser.registry import (
    AV2SensorBoxDetectionLabel,
    NuScenesBoxDetectionLabel,
    WODPerceptionBoxDetectionLabel,
)

from py123detection.core.taxonomy import (
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


# --- class order and checkpoint remapping ----------------------------------------------------
#
# The thesis ordered the ten nuScenes classes as the devkit's DETECTION_NAMES; this package orders
# them as mmdetection3d's NuScenesDataset.CLASSES. Both are right for their own consumer, the names
# carry the meaning, and only a model head is positional — so the port adopts one order and
# permutes existing heads.

THESIS_CLASS_ORDER = (
    "car", "truck", "bus", "trailer", "construction_vehicle",
    "pedestrian", "motorcycle", "bicycle", "traffic_cone", "barrier",
)


def test_the_two_class_orders_are_the_same_ten_names() -> None:
    assert set(THESIS_CLASS_ORDER) == set(NUSCENES_DETECTION.class_names)
    assert THESIS_CLASS_ORDER != NUSCENES_DETECTION.class_names


def test_class_permutation_swaps_exactly_bus_trailer_and_pedestrian_bicycle() -> None:
    from py123detection.core.taxonomy import class_permutation

    permutation = class_permutation(THESIS_CLASS_ORDER, NUSCENES_DETECTION.class_names)
    assert permutation == [0, 1, 3, 2, 4, 7, 6, 5, 8, 9]


def test_class_permutation_rejects_a_different_class_set() -> None:
    from py123detection.core.taxonomy import class_permutation

    with pytest.raises(ValueError, match="only in the old order"):
        class_permutation(("car", "truck"), ("car", "bus"))


def test_remap_head_classes_moves_each_class_to_its_new_row() -> None:
    import numpy as np

    from py123detection.core.taxonomy import remap_head_classes

    state_dict = {
        "head.cls_branches.0.4.weight": np.arange(40, dtype=np.float64).reshape(10, 4),
        "head.cls_branches.0.4.bias": np.arange(10, dtype=np.float64),
        "head.reg_branches.0.4.weight": np.zeros((10, 4)),
    }
    remapped = remap_head_classes(state_dict, THESIS_CLASS_ORDER, NUSCENES_DETECTION.class_names)

    for new_index, name in enumerate(NUSCENES_DETECTION.class_names):
        assert remapped["head.cls_branches.0.4.bias"][new_index] == THESIS_CLASS_ORDER.index(name)
    # Only the classification head is positional in the class dimension.
    assert np.array_equal(remapped["head.reg_branches.0.4.weight"], state_dict["head.reg_branches.0.4.weight"])
    assert np.array_equal(state_dict["head.cls_branches.0.4.bias"], np.arange(10)), "input was modified"


def test_remap_head_classes_refuses_when_nothing_matches() -> None:
    import numpy as np

    from py123detection.core.taxonomy import remap_head_classes

    with pytest.raises(ValueError, match="No parameters to remap"):
        remap_head_classes({"backbone.conv.weight": np.zeros((10, 4))},
                           THESIS_CLASS_ORDER, NUSCENES_DETECTION.class_names)


# --- presets carried over from the thesis ----------------------------------------------------


def test_mixed2_puts_a_motorcycle_with_the_vehicles_but_not_a_bicycle() -> None:
    """The thesis mapped nuScenes VEHICLE_MOTORCYCLE to 'vehicle' and VEHICLE_BICYCLE to
    'bicycle'; that asymmetry is deliberate and survives the move to a default-level table."""
    from py123detection.core.taxonomy import MIXED2_3CLS

    assert MIXED2_3CLS.resolve(NuScenesBoxDetectionLabel.VEHICLE_MOTORCYCLE) == "vehicle"
    assert MIXED2_3CLS.resolve(NuScenesBoxDetectionLabel.VEHICLE_BICYCLE) == "bicycle"


def test_mixed2_routes_argoverse_two_wheelers_through_the_merge_rules() -> None:
    from py123detection.core.taxonomy import MIXED2_3CLS

    assert MIXED2_3CLS.resolve(AV2SensorBoxDetectionLabel.MOTORCYCLE) == "av2_motorcycle"
    assert MIXED2_3CLS.resolve(AV2SensorBoxDetectionLabel.BICYCLIST) == "av2_bicyclist"
    # Intermediate, so it is not a class a model ever predicts.
    assert not MIXED2_3CLS.is_final("av2_bicyclist")
    assert MIXED2_3CLS.intermediate_names == frozenset(
        {"av2_bicycle", "av2_bicyclist", "av2_motorcycle", "av2_motorcyclist"}
    )


def test_mixed2_covers_datasets_the_thesis_had_no_table_for() -> None:
    """The thesis needed one hand-written table per dataset, so Waymo Motion and Nureasoning had
    none at all. Resolving through to_default covers them without new tables."""
    from py123d.parser.registry import NureasoningBoxDetectionLabel, WODMotionBoxDetectionLabel

    from py123detection.core.taxonomy import MIXED2_3CLS

    assert MIXED2_3CLS.resolve(WODMotionBoxDetectionLabel.TYPE_VEHICLE) == "vehicle"
    assert MIXED2_3CLS.resolve(WODMotionBoxDetectionLabel.TYPE_PEDESTRIAN) == "pedestrian"
    assert MIXED2_3CLS.resolve(NureasoningBoxDetectionLabel.VEHICLE_CAR) == "vehicle"
    # A bare native key is deliberately dataset-agnostic here: whatever the dataset, MIXED2 counts
    # a motorcycle among the vehicles.
    assert MIXED2_3CLS.resolve(NureasoningBoxDetectionLabel.VEHICLE_MOTORCYCLE) == "vehicle"


def test_mixed3_is_mixed2_with_one_class_rather_than_a_shorter_list() -> None:
    """The thesis "mixed3" is "mixed2" minus pedestrian and bicycle, not a separate mapping: every
    label mixed2 calls a vehicle has to stay one, or a rig comparison is also a class comparison."""
    from py123d.parser.registry import BOX_DETECTION_LABEL_REGISTRY

    from py123detection.core.taxonomy import MIXED2_3CLS, MIXED3_1CLS

    for enum in BOX_DETECTION_LABEL_REGISTRY.values():
        for label in enum:
            if MIXED2_3CLS.resolve(label) == "vehicle":
                assert MIXED3_1CLS.resolve(label) == "vehicle", label
            if MIXED2_3CLS.resolve(label) in ("pedestrian", "bicycle"):
                assert MIXED3_1CLS.resolve(label) is None, label


def test_mixed3_merges_an_argoverse_rider_onto_the_machine() -> None:
    """Argoverse boxes a motorcyclist apart from the motorcycle, nuScenes and Waymo do not. Without
    the merge the same scene yields one object in one dataset and two in the other."""
    from py123detection.core.taxonomy import MIXED3_1CLS

    assert MIXED3_1CLS.resolve(AV2SensorBoxDetectionLabel.MOTORCYCLE) == "av2_motorcycle"
    assert MIXED3_1CLS.resolve(AV2SensorBoxDetectionLabel.MOTORCYCLIST) == "av2_motorcyclist"
    assert not MIXED3_1CLS.is_final("av2_motorcyclist")
    rule = MIXED3_1CLS.rule_for("av2_motorcyclist")
    assert rule is not None and rule.target == "vehicle"
    assert rule.sources == ("av2_motorcycle", "av2_motorcyclist")


def test_mixed3_is_not_vehicle_1cls() -> None:
    """Both have one class called vehicle and they disagree on what is in it, so a run that names
    the wrong one trains on different ground truth and still looks right in the logs."""
    from py123detection.core.taxonomy import MIXED3_1CLS, VEHICLE_1CLS

    # The thesis counted a motorcycle as a vehicle; vehicle_1cls reads the default label, where a
    # motorcycle is a TWO_WHEELER and drops out.
    assert MIXED3_1CLS.resolve(NuScenesBoxDetectionLabel.VEHICLE_MOTORCYCLE) == "vehicle"
    assert VEHICLE_1CLS.resolve(NuScenesBoxDetectionLabel.VEHICLE_MOTORCYCLE) is None
    # And a tram is traffic to vehicle_1cls but not to mixed3.
    assert MIXED3_1CLS.resolve(AV2SensorBoxDetectionLabel.RAILED_VEHICLE) is None
    assert VEHICLE_1CLS.resolve(AV2SensorBoxDetectionLabel.RAILED_VEHICLE) == "vehicle"


def test_every_preset_resolves_every_label_of_every_dataset() -> None:
    """A preset that silently drops a whole dataset is the failure mode the per-dataset tables
    had; this asserts each one either classifies or deliberately voids every known label."""
    from py123d.parser.registry import BOX_DETECTION_LABEL_REGISTRY

    from py123detection.core.taxonomy import TAXONOMIES

    default_level = [name for name, taxonomy in TAXONOMIES.items() if taxonomy.default_map]
    for name in default_level:
        taxonomy = TAXONOMIES[name]
        for label_enum in BOX_DETECTION_LABEL_REGISTRY.values():
            for label in label_enum:
                resolved = taxonomy.resolve(label)
                assert resolved is None or resolved in taxonomy.class_names or resolved in taxonomy.intermediate_names, (
                    f"{name} resolved {label_enum.__name__}.{label.name} to {resolved!r}"
                )
