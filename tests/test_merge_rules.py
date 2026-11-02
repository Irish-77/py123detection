"""Unit tests for combining a rider with the vehicle they are on.

Argoverse 2 and KITTI-360 annotate a cyclist as two boxes; nuScenes and Waymo annotate one. A
:class:`~py123detection.core.taxonomy.MergeRule` makes both look the same to a model, and these
tests pin the four outcomes a rule has to get right: a pair merges, a lone rider disappears, a
lone vehicle is kept or dropped as configured, and two cyclists side by side do not cross over.
"""

from __future__ import annotations

import numpy as np
import pytest

from py123detection.core.annotations import DetectionRecord, apply_merge_rules
from py123detection.core.taxonomy import MergeRule, Taxonomy

CYCLIST = Taxonomy(
    name="test_cyclist",
    class_names=("vehicle", "cyclist"),
    native_map={"BICYCLE": "the_bicycle", "BICYCLIST": "the_rider", "CAR": "vehicle"},
    merge_rules=(MergeRule(target="cyclist", sources=("the_bicycle", "the_rider")),),
)

DISCARDING = Taxonomy(
    name="test_discarding",
    class_names=("cyclist",),
    native_map={"BICYCLE": "the_bicycle", "BICYCLIST": "the_rider"},
    merge_rules=(
        MergeRule(target="cyclist", sources=("the_bicycle", "the_rider"), discard_unmerged_base=True),
    ),
)


def corners(center, size, yaw=0.0):
    """The eight corners of an axis-aligned-in-its-own-frame box."""
    center, size = np.asarray(center, float), np.asarray(size, float)
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[cos_yaw, -sin_yaw, 0.0], [sin_yaw, cos_yaw, 0.0], [0.0, 0.0, 1.0]])
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float)
    return (signs * (size / 2.0)) @ rotation.T + center


def record(class_name, center, size, yaw=0.0, velocity=(0.0, 0.0, 0.0), num_points=5, token="t"):
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    return DetectionRecord(
        class_name=class_name,
        label_id=-1,
        center_global=np.asarray(center, float),
        rotation_global=np.array([[cos_yaw, -sin_yaw, 0.0], [sin_yaw, cos_yaw, 0.0], [0.0, 0.0, 1.0]]),
        size_lwh=np.asarray(size, float),
        velocity_global=np.asarray(velocity, float),
        num_lidar_points=num_points,
        track_token=token,
        corners_global=corners(center, size, yaw),
    )


def test_overlapping_pair_becomes_one_box() -> None:
    """A rider sitting on a bicycle yields a single box spanning both."""
    bicycle = record("the_bicycle", (0, 0, 0.5), (1.8, 0.6, 1.0), velocity=(3.0, 0.0, 0.0), num_points=40)
    rider = record("the_rider", (0, 0, 1.4), (0.6, 0.6, 1.8), num_points=60)

    merged = apply_merge_rules([bicycle, rider], CYCLIST)

    assert len(merged) == 1
    box = merged[0]
    assert box.class_name == "cyclist"
    # Spans the bicycle's length and reaches from the wheels to the top of the rider's head.
    assert box.size_lwh[0] == pytest.approx(1.8)
    assert box.size_lwh[2] == pytest.approx(2.3)
    assert box.center_global[2] == pytest.approx(1.15)
    # The vehicle is the primary, so its motion and both point counts carry over.
    assert box.velocity_global == pytest.approx([3.0, 0.0, 0.0])
    assert box.num_lidar_points == 100


def test_merged_box_contains_both_sets_of_corners() -> None:
    """Whatever the yaw, the result has to actually enclose what it merged."""
    bicycle = record("the_bicycle", (10, -4, 0.5), (1.8, 0.6, 1.0), yaw=0.9)
    rider = record("the_rider", (10.2, -3.9, 1.4), (0.6, 0.6, 1.8), yaw=1.1)

    box = apply_merge_rules([bicycle, rider], CYCLIST)[0]

    local = (np.concatenate([bicycle.corners_global, rider.corners_global]) - box.center_global) @ box.rotation_global
    assert np.all(local <= box.size_lwh / 2.0 + 1e-9)
    assert np.all(local >= -box.size_lwh / 2.0 - 1e-9)


def test_lone_rider_is_dropped() -> None:
    """A rider with no vehicle is not an object the taxonomy has a class for."""
    assert apply_merge_rules([record("the_rider", (0, 0, 1.0), (0.6, 0.6, 1.8))], CYCLIST) == []


def test_lone_vehicle_is_kept_by_default() -> None:
    bicycle = record("the_bicycle", (0, 0, 0.5), (1.8, 0.6, 1.0))
    merged = apply_merge_rules([bicycle], CYCLIST)
    assert [box.class_name for box in merged] == ["cyclist"]
    assert merged[0].size_lwh == pytest.approx([1.8, 0.6, 1.0])


def test_lone_vehicle_is_dropped_when_configured() -> None:
    """A parked bicycle with nobody on it is not a cyclist."""
    assert apply_merge_rules([record("the_bicycle", (0, 0, 0.5), (1.8, 0.6, 1.0))], DISCARDING) == []


def test_each_rider_pairs_with_their_own_vehicle() -> None:
    """Two cyclists abreast must not be paired across."""
    left_bicycle = record("the_bicycle", (0, 0, 0.5), (1.8, 0.6, 1.0), token="left")
    right_bicycle = record("the_bicycle", (0, 5, 0.5), (1.8, 0.6, 1.0), token="right")
    left_rider = record("the_rider", (0, 0.1, 1.4), (0.6, 0.6, 1.8))
    right_rider = record("the_rider", (0, 4.9, 1.4), (0.6, 0.6, 1.8))

    merged = apply_merge_rules([left_bicycle, right_bicycle, left_rider, right_rider], CYCLIST)

    assert len(merged) == 2
    assert {box.track_token for box in merged} == {"left", "right"}
    for box in merged:
        expected_y = 0.0 if box.track_token == "left" else 5.0
        assert box.center_global[1] == pytest.approx(expected_y, abs=0.1)


def test_records_outside_any_rule_pass_through_unchanged() -> None:
    car = record("vehicle", (20, 0, 1.0), (4.5, 1.9, 1.6))
    merged = apply_merge_rules([car], CYCLIST)
    assert merged == [car]


def test_unknown_lidar_count_is_not_summed() -> None:
    """-1 means "not recorded"; adding it to a real count would invent a number."""
    bicycle = record("the_bicycle", (0, 0, 0.5), (1.8, 0.6, 1.0), num_points=40)
    rider = record("the_rider", (0, 0, 1.4), (0.6, 0.6, 1.8), num_points=-1)
    assert apply_merge_rules([bicycle, rider], CYCLIST)[0].num_lidar_points == 40


def test_rule_targeting_an_undeclared_class_is_rejected() -> None:
    with pytest.raises(ValueError, match="not in class_names"):
        Taxonomy(
            name="bad_target",
            class_names=("vehicle",),
            native_map={"BICYCLE": "a", "BICYCLIST": "b"},
            merge_rules=(MergeRule(target="cyclist", sources=("a", "b")),),
        )


def test_mapping_to_a_merge_source_with_no_rule_is_rejected() -> None:
    """The thesis tables routed labels to 'bicyclist' in taxonomies that had no such class, and
    every one of those boxes vanished without a word. That is now an error at import."""
    with pytest.raises(ValueError, match="unknown class names"):
        Taxonomy(name="typo", class_names=("cyclist",), native_map={"BICYCLIST": "bicyclist"})


def test_rule_with_no_secondary_is_rejected() -> None:
    with pytest.raises(ValueError, match="needs a primary and at least one secondary"):
        Taxonomy(
            name="lonely",
            class_names=("cyclist",),
            native_map={"BICYCLE": "a"},
            merge_rules=(MergeRule(target="cyclist", sources=("a",)),),
        )


def test_merge_source_no_label_reaches_is_rejected() -> None:
    with pytest.raises(ValueError, match="merge sources no label reaches"):
        Taxonomy(
            name="unreachable",
            class_names=("cyclist",),
            native_map={"BICYCLE": "a"},
            merge_rules=(MergeRule(target="cyclist", sources=("a", "never_mapped")),),
        )
