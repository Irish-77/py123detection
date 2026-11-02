"""Mapping 123D box labels onto the ``class_names`` tuple of an mmdet3d config.

A :class:`Taxonomy` turns a 123D label into one of its class names, or drops the box. Lookups go
from most to least specific:

1. ``native_map`` by qualified enum member (``"AV2SensorBoxDetectionLabel.BICYCLIST"``), for when
   the same member name exists in several datasets and only one should be overridden;
2. ``native_map`` by bare member name (``"VEHICLE_CAR"``);
3. ``default_map`` by :class:`~py123d.datatypes.DefaultBoxDetectionLabel` member (``"VEHICLE"``).
   Every 123D dataset implements ``to_default()``, so a default-level taxonomy works for all of
   them, which is what cross-dataset merges rely on.

Labels that resolve to :data:`VOID`, or to nothing, are dropped.

Some datasets box a rider separately from the vehicle they are on: Argoverse 2 and KITTI-360
annotate ``BICYCLIST`` next to ``BICYCLE``, while nuScenes and Waymo annotate one object.
:class:`MergeRule` reconciles them by pairing the two boxes and emitting their hull, so the same
scene yields the same objects whichever dataset it came from. A rule's sources are *intermediate*
class names — reachable from a label but not in :attr:`Taxonomy.class_names` — which is what keeps
a rule scoped to the datasets that need it. Anything still intermediate after merging is dropped.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple

from py123d.datatypes.detections.box_detection_label import BoxDetectionLabel

logger = logging.getLogger(__name__)

#: Target class name that drops a label.
VOID = "void"


@dataclass(frozen=True)
class MergeRule:
    """Combine a rider and its vehicle into one box.

    Secondaries are paired to primaries by bird's-eye-view IoU, greedily and highest first, and a
    pair becomes the tight box around both sets of corners, keeping the primary's orientation and
    velocity. A secondary that pairs with nothing is dropped — a rider with no vehicle is not an
    object this taxonomy has a class for.
    """

    target: str
    """Class name the merged box gets. Must be in :attr:`Taxonomy.class_names`."""

    sources: Tuple[str, ...]
    """Class names to combine. The first is the primary and keeps its yaw and velocity."""

    discard_unmerged_base: bool = False
    """Drop a primary that paired with nothing, instead of keeping it as :attr:`target`.

    Set it where a lone primary is not the object you mean: an Argoverse ``BICYCLE`` with no
    ``BICYCLIST`` is a parked bicycle, not a cyclist.
    """

    @property
    def primary(self) -> str:
        return self.sources[0]

    @property
    def secondaries(self) -> Tuple[str, ...]:
        return tuple(self.sources[1:])


@dataclass(frozen=True)
class Taxonomy:
    """An ordered class list plus the rules mapping 123D labels onto it."""

    name: str
    """Identifier stored in the exported metadata, e.g. ``"nuscenes_detection_10cls"``."""

    class_names: Tuple[str, ...]
    """Ordered class names. The index of a name is its label id in the mmdet3d config."""

    native_map: Mapping[str, str] = field(default_factory=dict)
    """Dataset-native (optionally qualified) enum member name -> class name or :data:`VOID`."""

    default_map: Mapping[str, str] = field(default_factory=dict)
    """:class:`DefaultBoxDetectionLabel` member name -> class name or :data:`VOID`."""

    merge_rules: Tuple[MergeRule, ...] = ()
    """Rules combining a rider with its vehicle. See :class:`MergeRule`."""

    def __post_init__(self) -> None:
        if len(set(self.class_names)) != len(self.class_names):
            raise ValueError(f"Taxonomy '{self.name}' has duplicate class names: {self.class_names}")

        for rule in self.merge_rules:
            if rule.target not in self.class_names:
                raise ValueError(
                    f"Taxonomy '{self.name}' merge rule targets '{rule.target}', which is not in "
                    f"class_names {self.class_names}."
                )
            if len(rule.sources) < 2:
                raise ValueError(
                    f"Taxonomy '{self.name}' merge rule for '{rule.target}' needs a primary and at "
                    f"least one secondary, got {rule.sources}."
                )

        # A mapping may target a real class, VOID, or a merge source. Anything else is a typo, and
        # the thesis tables show why this matters: they routed labels to 'bicyclist' in taxonomies
        # with no such class and no merge rule, and every one of those boxes vanished silently.
        reachable = set(self.class_names) | {VOID} | self.intermediate_names
        for source, mapping in (("native_map", self.native_map), ("default_map", self.default_map)):
            unknown = {v for v in mapping.values() if v not in reachable}
            if unknown:
                raise ValueError(
                    f"Taxonomy '{self.name}' {source} targets unknown class names {sorted(unknown)}; "
                    f"declared class_names are {self.class_names} and merge sources are "
                    f"{sorted(self.intermediate_names)}."
                )

        orphans = {name for name in self.intermediate_names if name not in set(self.mapped_names)}
        if orphans:
            raise ValueError(
                f"Taxonomy '{self.name}' has merge sources no label reaches: {sorted(orphans)}."
            )

    @property
    def intermediate_names(self) -> FrozenSet[str]:
        """Merge-source names that are not classes in their own right."""
        sources = {name for rule in self.merge_rules for name in rule.sources}
        return frozenset(sources - set(self.class_names))

    @property
    def mapped_names(self) -> Tuple[str, ...]:
        """Every class name some label maps to, VOID excluded."""
        values = list(self.native_map.values()) + list(self.default_map.values())
        return tuple(sorted({value for value in values if value != VOID}))

    def is_final(self, class_name: str) -> bool:
        """Whether ``class_name`` survives on its own, i.e. is not merge-only."""
        return class_name in self.class_names

    def resolve(self, label: BoxDetectionLabel) -> Optional[str]:
        """Class name for a 123D label, or ``None`` if the box should be dropped."""
        class_name = self.native_map.get(f"{type(label).__name__}.{label.name}")
        if class_name is None:
            class_name = self.native_map.get(label.name)
        if class_name is None:
            class_name = self.default_map.get(label.to_default().name)
        if class_name is None or class_name == VOID:
            return None
        return class_name

    def label_id(self, class_name: str) -> int:
        return self.class_names.index(class_name)

    @property
    def class_ids(self) -> Dict[str, int]:
        return {name: index for index, name in enumerate(self.class_names)}

    def rule_for(self, class_name: str) -> Optional[MergeRule]:
        """The merge rule ``class_name`` takes part in, if any."""
        for rule in self.merge_rules:
            if class_name in rule.sources:
                return rule
        return None


#: The standard nuScenes 10-class benchmark, in mmdet3d's class order. Native-level only, so any
#: other dataset exported with it drops every box.
NUSCENES_DETECTION = Taxonomy(
    name="nuscenes_detection_10cls",
    class_names=(
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
    ),
    native_map={
        "VEHICLE_CAR": "car",
        "VEHICLE_TRUCK": "truck",
        "VEHICLE_TRAILER": "trailer",
        "VEHICLE_BUS_BENDY": "bus",
        "VEHICLE_BUS_RIGID": "bus",
        "VEHICLE_CONSTRUCTION": "construction_vehicle",
        "VEHICLE_BICYCLE": "bicycle",
        "VEHICLE_MOTORCYCLE": "motorcycle",
        "HUMAN_PEDESTRIAN_ADULT": "pedestrian",
        "HUMAN_PEDESTRIAN_CHILD": "pedestrian",
        "HUMAN_PEDESTRIAN_CONSTRUCTION_WORKER": "pedestrian",
        "HUMAN_PEDESTRIAN_POLICE_OFFICER": "pedestrian",
        "MOVABLE_OBJECT_TRAFFICCONE": "traffic_cone",
        "MOVABLE_OBJECT_BARRIER": "barrier",
        # mmdet3d's NuScenesDataset.NameMapping has no entry for these, so they are not part of
        # the benchmark.
        "HUMAN_PEDESTRIAN_PERSONAL_MOBILITY": VOID,
        "HUMAN_PEDESTRIAN_STROLLER": VOID,
        "HUMAN_PEDESTRIAN_WHEELCHAIR": VOID,
        "VEHICLE_EMERGENCY_AMBULANCE": VOID,
        "VEHICLE_EMERGENCY_POLICE": VOID,
        "MOVABLE_OBJECT_PUSHABLE_PULLABLE": VOID,
        "MOVABLE_OBJECT_DEBRIS": VOID,
        "STATIC_OBJECT_BICYCLE_RACK": VOID,
        "ANIMAL": VOID,
    },
)

#: Cross-dataset vehicle / pedestrian / cyclist over the default labels.
GENERAL_3CLS = Taxonomy(
    name="general_3cls",
    class_names=("vehicle", "pedestrian", "cyclist"),
    default_map={
        "VEHICLE": "vehicle",
        "TRAIN": "vehicle",
        "PERSON": "pedestrian",
        "TWO_WHEELER": "cyclist",
        "EGO": VOID,
        "ANIMAL": VOID,
        "TRAFFIC_SIGN": VOID,
        "TRAFFIC_CONE": VOID,
        "TRAFFIC_LIGHT": VOID,
        "BARRIER": VOID,
        "GENERIC_OBJECT": VOID,
        "OTHER": VOID,
    },
)

#: CoIn3D's unified classes, spelled with nuScenes names on purpose: the devkit's DetectionBox
#: only accepts the ten nuScenes detection names, and reusing three of them lets any dataset be
#: scored with the unmodified metric code. CoIn3D's mapping (data_preprocess/README.md): nuScenes
#: car/truck/trailer/bus/construction_vehicle -> car, pedestrian -> pedestrian,
#: bicycle/motorcycle -> motorcycle; Lyft car/truck/bus/emergency_vehicle/other_vehicle -> car;
#: Waymo VEHICLE/PEDESTRIAN/CYCLIST.
COIN3D_3CLS = Taxonomy(
    name="coin3d_3cls",
    class_names=("car", "pedestrian", "motorcycle"),
    native_map={
        # AV2 boxes the rider of a two-wheeler separately from the vehicle. nuScenes (and hence
        # CoIn3D) has no rider class, so keep the vehicle and drop the rider rather than label
        # one object twice.
        "AV2SensorBoxDetectionLabel.BICYCLIST": VOID,
        "AV2SensorBoxDetectionLabel.MOTORCYCLIST": VOID,
        "AV2SensorBoxDetectionLabel.WHEELED_RIDER": VOID,
        # Not in mmdet3d's NameMapping, so not in its pickles either.
        "NuScenesBoxDetectionLabel.HUMAN_PEDESTRIAN_PERSONAL_MOBILITY": VOID,
        "NuScenesBoxDetectionLabel.HUMAN_PEDESTRIAN_STROLLER": VOID,
        "NuScenesBoxDetectionLabel.HUMAN_PEDESTRIAN_WHEELCHAIR": VOID,
        "NuScenesBoxDetectionLabel.VEHICLE_EMERGENCY_AMBULANCE": VOID,
        "NuScenesBoxDetectionLabel.VEHICLE_EMERGENCY_POLICE": VOID,
    },
    default_map={
        "VEHICLE": "car",
        "TRAIN": "car",
        "PERSON": "pedestrian",
        "TWO_WHEELER": "motorcycle",
        "EGO": VOID,
        "ANIMAL": VOID,
        "TRAFFIC_SIGN": VOID,
        "TRAFFIC_CONE": VOID,
        "TRAFFIC_LIGHT": VOID,
        "BARRIER": VOID,
        "GENERIC_OBJECT": VOID,
        "OTHER": VOID,
    },
)

# --- Presets carried over from the master thesis ---------------------------------------------
#
# The thesis wrote one native table per dataset per taxonomy: nine files, up to four tables each,
# and still no entry for several 123D datasets. These reach the same classes through default_map,
# which every 123D dataset implements, and fall back to native entries only where a dataset really
# does disagree.

#: Eight classes: the nuScenes ten with ``construction_vehicle`` folded into ``truck`` and
#: ``barrier`` dropped. The thesis called this "mixed".
MIXED_8CLS = Taxonomy(
    name="mixed_8cls",
    class_names=("car", "truck", "bus", "trailer", "traffic_cone", "pedestrian", "motorcycle", "bicycle"),
    native_map={
        # nuScenes splits vehicles finely enough to keep the eight apart by name.
        "VEHICLE_CAR": "car",
        "VEHICLE_TRUCK": "truck",
        "VEHICLE_CONSTRUCTION": "truck",
        "VEHICLE_EMERGENCY_AMBULANCE": "truck",
        "VEHICLE_EMERGENCY_POLICE": "truck",
        "VEHICLE_BUS_BENDY": "bus",
        "VEHICLE_BUS_RIGID": "bus",
        "VEHICLE_TRAILER": "trailer",
        "VEHICLE_BICYCLE": "bicycle",
        "VEHICLE_MOTORCYCLE": "motorcycle",
        "MOVABLE_OBJECT_TRAFFICCONE": "traffic_cone",
        "MOVABLE_OBJECT_BARRIER": VOID,
        # Argoverse 2: same eight, its own names, plus riders to merge.
        "AV2SensorBoxDetectionLabel.REGULAR_VEHICLE": "car",
        "AV2SensorBoxDetectionLabel.TRUCK": "truck",
        "AV2SensorBoxDetectionLabel.BOX_TRUCK": "truck",
        "AV2SensorBoxDetectionLabel.TRUCK_CAB": "truck",
        "AV2SensorBoxDetectionLabel.LARGE_VEHICLE": "truck",
        "AV2SensorBoxDetectionLabel.BUS": "bus",
        "AV2SensorBoxDetectionLabel.ARTICULATED_BUS": "bus",
        "AV2SensorBoxDetectionLabel.SCHOOL_BUS": "bus",
        "AV2SensorBoxDetectionLabel.VEHICULAR_TRAILER": "trailer",
        "AV2SensorBoxDetectionLabel.MESSAGE_BOARD_TRAILER": VOID,
        "AV2SensorBoxDetectionLabel.TRAFFIC_LIGHT_TRAILER": VOID,
        "AV2SensorBoxDetectionLabel.CONSTRUCTION_CONE": "traffic_cone",
        "AV2SensorBoxDetectionLabel.CONSTRUCTION_BARREL": "traffic_cone",
        "AV2SensorBoxDetectionLabel.BICYCLE": "av2_bicycle",
        "AV2SensorBoxDetectionLabel.BICYCLIST": "av2_bicyclist",
        "AV2SensorBoxDetectionLabel.MOTORCYCLE": "av2_motorcycle",
        "AV2SensorBoxDetectionLabel.MOTORCYCLIST": "av2_motorcyclist",
        "AV2SensorBoxDetectionLabel.WHEELED_RIDER": VOID,
        "AV2SensorBoxDetectionLabel.RAILED_VEHICLE": VOID,
    },
    default_map={
        # Any other 123D dataset: a vehicle is a car unless its native name said otherwise.
        "VEHICLE": "car",
        "TRAIN": VOID,
        "PERSON": "pedestrian",
        "TWO_WHEELER": "bicycle",
        "TRAFFIC_CONE": "traffic_cone",
        "EGO": VOID,
        "ANIMAL": VOID,
        "TRAFFIC_SIGN": VOID,
        "TRAFFIC_LIGHT": VOID,
        "BARRIER": VOID,
        "GENERIC_OBJECT": VOID,
        "OTHER": VOID,
    },
    merge_rules=(
        MergeRule(target="motorcycle", sources=("av2_motorcycle", "av2_motorcyclist")),
        MergeRule(target="bicycle", sources=("av2_bicycle", "av2_bicyclist")),
    ),
)

#: Three classes across datasets: everything that drives is ``vehicle``, a motorcycle included,
#: and only a bicycle stays separate. The thesis called this "mixed2".
MIXED2_3CLS = Taxonomy(
    name="mixed2_3cls",
    class_names=("vehicle", "pedestrian", "bicycle"),
    native_map={
        # nuScenes counts a motorcycle among the vehicles; only the bicycle stays apart.
        "VEHICLE_MOTORCYCLE": "vehicle",
        # Argoverse 2 boxes riders separately, so its two-wheelers go through the merge rules.
        "AV2SensorBoxDetectionLabel.MOTORCYCLE": "av2_motorcycle",
        "AV2SensorBoxDetectionLabel.MOTORCYCLIST": "av2_motorcyclist",
        "AV2SensorBoxDetectionLabel.BICYCLE": "av2_bicycle",
        "AV2SensorBoxDetectionLabel.BICYCLIST": "av2_bicyclist",
        "AV2SensorBoxDetectionLabel.WHEELED_RIDER": VOID,
        "AV2SensorBoxDetectionLabel.RAILED_VEHICLE": VOID,
    },
    default_map={
        "VEHICLE": "vehicle",
        "TRAIN": VOID,
        "PERSON": "pedestrian",
        "TWO_WHEELER": "bicycle",
        "EGO": VOID,
        "ANIMAL": VOID,
        "TRAFFIC_SIGN": VOID,
        "TRAFFIC_CONE": VOID,
        "TRAFFIC_LIGHT": VOID,
        "BARRIER": VOID,
        "GENERIC_OBJECT": VOID,
        "OTHER": VOID,
    },
    merge_rules=(
        MergeRule(target="vehicle", sources=("av2_motorcycle", "av2_motorcyclist")),
        # A parked Argoverse bicycle with nobody on it is not a cyclist, so it is dropped.
        MergeRule(target="bicycle", sources=("av2_bicycle", "av2_bicyclist"), discard_unmerged_base=True),
    ),
)

#: One class: everything that drives, a motorcycle and its rider included. The thesis called this
#: "mixed3" and used it to compare camera rigs, where a second class only adds a way for the
#: comparison to move for a reason that is not the rig.
#:
#: It is :data:`MIXED2_3CLS` with pedestrian and bicycle dropped rather than
#: :data:`VEHICLE_1CLS` with a shorter class list: a motorcycle counts as a vehicle here, an
#: Argoverse rider is merged onto the machine they ride, and a train does not count at all.
MIXED3_1CLS = Taxonomy(
    name="mixed3_1cls",
    class_names=("vehicle",),
    native_map={
        # nuScenes counts a motorcycle among the vehicles.
        "VEHICLE_MOTORCYCLE": "vehicle",
        # Argoverse 2 boxes the rider separately, so both go through the merge rule.
        "AV2SensorBoxDetectionLabel.MOTORCYCLE": "av2_motorcycle",
        "AV2SensorBoxDetectionLabel.MOTORCYCLIST": "av2_motorcyclist",
        # Towed and driven are the same obstacle to a detector with one class.
        "AV2SensorBoxDetectionLabel.MESSAGE_BOARD_TRAILER": "vehicle",
        "AV2SensorBoxDetectionLabel.TRAFFIC_LIGHT_TRAILER": "vehicle",
        "AV2SensorBoxDetectionLabel.WHEELED_RIDER": VOID,
        "AV2SensorBoxDetectionLabel.RAILED_VEHICLE": VOID,
    },
    default_map={
        "VEHICLE": "vehicle",
        # A tram is on rails and behaves nothing like traffic; nuScenes has none anyway.
        "TRAIN": VOID,
        "PERSON": VOID,
        "TWO_WHEELER": VOID,
        "EGO": VOID,
        "ANIMAL": VOID,
        "TRAFFIC_SIGN": VOID,
        "TRAFFIC_CONE": VOID,
        "TRAFFIC_LIGHT": VOID,
        "BARRIER": VOID,
        "GENERIC_OBJECT": VOID,
        "OTHER": VOID,
    },
    merge_rules=(MergeRule(target="vehicle", sources=("av2_motorcycle", "av2_motorcyclist")),),
)

#: A single ``vehicle`` class over the default labels (``VEHICLE`` and ``TRAIN``).
#: Simpler than :data:`MIXED3_1CLS`: no dataset-native overrides, so a two-wheeler never counts.
VEHICLE_1CLS = Taxonomy(
    name="vehicle_1cls",
    class_names=("vehicle",),
    default_map={"VEHICLE": "vehicle", "TRAIN": "vehicle"},
)

#: Identity over DefaultBoxDetectionLabel, minus EGO.
DEFAULT_123D = Taxonomy(
    name="default_123d",
    class_names=(
        "vehicle",
        "train",
        "two_wheeler",
        "person",
        "animal",
        "traffic_sign",
        "traffic_cone",
        "traffic_light",
        "barrier",
        "generic_object",
        "other",
    ),
    default_map={
        "VEHICLE": "vehicle",
        "TRAIN": "train",
        "TWO_WHEELER": "two_wheeler",
        "PERSON": "person",
        "ANIMAL": "animal",
        "TRAFFIC_SIGN": "traffic_sign",
        "TRAFFIC_CONE": "traffic_cone",
        "TRAFFIC_LIGHT": "traffic_light",
        "BARRIER": "barrier",
        "GENERIC_OBJECT": "generic_object",
        "OTHER": "other",
        "EGO": VOID,
    },
)

#: The values accepted by --taxonomy.
TAXONOMIES: Dict[str, Taxonomy] = {
    taxonomy.name: taxonomy
    for taxonomy in (
        NUSCENES_DETECTION,
        GENERAL_3CLS,
        COIN3D_3CLS,
        MIXED_8CLS,
        MIXED2_3CLS,
        MIXED3_1CLS,
        VEHICLE_1CLS,
        DEFAULT_123D,
    )
}


def get_taxonomy(name: str) -> Taxonomy:
    if name not in TAXONOMIES:
        raise KeyError(f"Unknown taxonomy '{name}'. Available: {sorted(TAXONOMIES)}")
    return TAXONOMIES[name]


def class_permutation(old_class_names: Sequence[str], new_class_names: Sequence[str]) -> List[int]:
    """Indices that reorder rows laid out as ``old_class_names`` into ``new_class_names``.

    ``result[i]`` is where the class now at position ``i`` used to sit, so ``rows[permutation]``
    is the reordered array.

    :raises ValueError: if the two lists are not the same set of names. A permutation is only
        meaningful when nothing is added or removed.
    """
    if sorted(old_class_names) != sorted(new_class_names):
        only_old = sorted(set(old_class_names) - set(new_class_names))
        only_new = sorted(set(new_class_names) - set(old_class_names))
        raise ValueError(
            f"Cannot permute between different class sets: {only_old} only in the old order, "
            f"{only_new} only in the new one."
        )
    positions = {name: index for index, name in enumerate(old_class_names)}
    return [positions[name] for name in new_class_names]


def remap_head_classes(
    state_dict: Mapping[str, Any],
    old_class_names: Sequence[str],
    new_class_names: Sequence[str],
    marker: str = "cls_branches",
) -> Dict[str, Any]:
    """Reorder a trained classification head from one class order to another.

    Class ids are positions, so the same ten nuScenes names in a different order mean a checkpoint
    predicts the right things under the wrong labels. This permutes the rows of the head's final
    weight and bias, which is the whole of what has to change.

    Works on any mapping of row-indexable arrays, so a ``torch`` state dict and a dict of numpy
    arrays both go through and ``core`` stays free of a torch dependency.

    :param marker: Only parameters whose name contains this take part. Matching on shape alone
        would catch anything else that happens to have one row per class.
    :returns: A new dict; the input is not modified.
    """
    permutation = class_permutation(old_class_names, new_class_names)
    count = len(new_class_names)

    remapped: Dict[str, Any] = {}
    touched: List[str] = []
    for key, value in state_dict.items():
        shape = getattr(value, "shape", None)
        if marker in key and shape is not None and len(shape) >= 1 and shape[0] == count:
            remapped[key] = value[permutation]
            touched.append(key)
        else:
            remapped[key] = value

    if not touched:
        raise ValueError(
            f"No parameters to remap: nothing matched '{marker}' with {count} rows. "
            f"Pass the marker your head uses."
        )
    logger.info("Remapped %d classification parameter(s) to the new class order: %s", len(touched), touched)
    return remapped
