"""Mapping 123D detection labels onto the flat class-name list an mmdet3d config expects.

An mmdet3d ``NuScenesDataset`` is configured with an ordered ``class_names`` tuple and reads
``info['gt_names']`` as strings, resolving them to label ids by position. A :class:`Taxonomy`
is the rule that turns a 123D :class:`~py123d.datatypes.BoxDetectionLabel` into one of those
strings (or drops the box).

Two mapping levels are available, checked in order:

1. ``native_map`` — keyed by the dataset-native enum member name (e.g. ``"VEHICLE_CAR"`` for
   :class:`~py123d.parser.registry.NuScenesBoxDetectionLabel`), or by the *qualified* name
   ``"<LabelEnum>.<MEMBER>"`` (``"AV2SensorBoxDetectionLabel.BICYCLIST"``) when the same member
   name exists in several datasets' enums and only one of them should be overridden. Use it when
   a taxonomy needs the full granularity of one dataset, or to carve out per-dataset exceptions
   from an otherwise default-level taxonomy.
2. ``default_map`` — keyed by the :class:`~py123d.datatypes.DefaultBoxDetectionLabel` member
   name (``"VEHICLE"``, ``"PERSON"``, ...). Every dataset enum in 123D implements
   ``to_default()``, so a taxonomy expressed at this level applies unchanged to *any* 123D
   dataset. This is what makes cross-dataset (CoIn3D-style) merges work without per-dataset code.

Anything that resolves to :data:`VOID` — or to no entry at all — is dropped from the exported
annotations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Tuple

from py123d.datatypes.detections.box_detection_label import BoxDetectionLabel

VOID = "void"
"""Sentinel class name marking a label that should be dropped from the export."""


@dataclass(frozen=True)
class Taxonomy:
    """An ordered class list plus the rules mapping 123D labels onto it."""

    name: str
    """Identifier stored in the exported metadata, e.g. ``"nuscenes_detection_10cls"``."""

    class_names: Tuple[str, ...]
    """Ordered class names. The index of a name is its label id in the mmdet3d config."""

    native_map: Mapping[str, str] = field(default_factory=dict)
    """Dataset-native enum member name -> class name (or :data:`VOID`)."""

    default_map: Mapping[str, str] = field(default_factory=dict)
    """:class:`DefaultBoxDetectionLabel` member name -> class name (or :data:`VOID`)."""

    def __post_init__(self) -> None:
        if len(set(self.class_names)) != len(self.class_names):
            raise ValueError(f"Taxonomy '{self.name}' has duplicate class names: {self.class_names}")
        for source, mapping in (("native_map", self.native_map), ("default_map", self.default_map)):
            unknown = {v for v in mapping.values() if v != VOID and v not in self.class_names}
            if unknown:
                raise ValueError(
                    f"Taxonomy '{self.name}' {source} targets unknown class names {sorted(unknown)}; "
                    f"declared class_names are {self.class_names}."
                )

    def resolve(self, label: BoxDetectionLabel) -> Optional[str]:
        """Resolve a 123D label to a class name.

        :param label: The dataset-native label carried by a
            :class:`~py123d.datatypes.BoxDetectionSE3`.
        :return: The class name, or ``None`` if the box should be dropped.
        """
        # Most specific first: "<LabelEnum>.<MEMBER>" overrides one dataset's member without
        # touching a same-named member of another dataset's enum.
        class_name = self.native_map.get(f"{type(label).__name__}.{label.name}")
        if class_name is None:
            class_name = self.native_map.get(label.name)
        if class_name is None:
            class_name = self.default_map.get(label.to_default().name)
        if class_name is None or class_name == VOID:
            return None
        return class_name

    def label_id(self, class_name: str) -> int:
        """Return the mmdet3d label id (position in :attr:`class_names`) of a class name.

        :param class_name: One of :attr:`class_names`.
        :return: The label id.
        """
        return self.class_names.index(class_name)

    @property
    def class_ids(self) -> Dict[str, int]:
        """Mapping from class name to label id."""
        return {name: index for index, name in enumerate(self.class_names)}


# ----------------------------------------------------------------------------------------------
# Presets
# ----------------------------------------------------------------------------------------------

_NUSCENES_10CLS_NATIVE: Dict[str, str] = {
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
    # Explicitly dropped, matching mmdet3d's NuScenesDataset.NameMapping, which has no entry for
    # these categories and therefore leaves them out of the 10-class detection benchmark.
    "HUMAN_PEDESTRIAN_PERSONAL_MOBILITY": VOID,
    "HUMAN_PEDESTRIAN_STROLLER": VOID,
    "HUMAN_PEDESTRIAN_WHEELCHAIR": VOID,
    "VEHICLE_EMERGENCY_AMBULANCE": VOID,
    "VEHICLE_EMERGENCY_POLICE": VOID,
    "MOVABLE_OBJECT_PUSHABLE_PULLABLE": VOID,
    "MOVABLE_OBJECT_DEBRIS": VOID,
    "STATIC_OBJECT_BICYCLE_RACK": VOID,
    "ANIMAL": VOID,
}

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
    native_map=_NUSCENES_10CLS_NATIVE,
)
"""The standard nuScenes 10-class detection taxonomy, in mmdet3d's class order.

Only defined at the native level, so it applies to nuScenes-derived 123D logs. Other datasets
exported under this taxonomy would drop every box — use :data:`GENERAL_3CLS` or a custom
taxonomy for cross-dataset work.
"""

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
"""Cross-dataset 3-class taxonomy expressed over :class:`DefaultBoxDetectionLabel`.

Because it only uses the default label level it works for every 123D dataset, which is what
mixed-dataset training (as in CoIn3D) needs.
"""

_COIN3D_3CLS_NATIVE: Dict[str, str] = {
    # Argoverse 2 annotates the *rider* of a bicycle / motorcycle / scooter as its own box, on
    # top of the vehicle's box. nuScenes has no rider category — the rider is part of the
    # two-wheeler box ("cycle.with_rider") — and CoIn3D's mapping inherits that. Keeping the
    # vehicle and dropping the rider preserves those semantics and avoids labelling one
    # physical object twice. Qualified keys, so a RIDER member of another dataset is untouched.
    "AV2SensorBoxDetectionLabel.BICYCLIST": VOID,
    "AV2SensorBoxDetectionLabel.MOTORCYCLIST": VOID,
    "AV2SensorBoxDetectionLabel.WHEELED_RIDER": VOID,
    # CoIn3D's nuScenes mapping lists exactly the mmdet3d benchmark categories; the ones
    # mmdet3d's NameMapping omits stay out here too, matching its pickles box for box.
    "NuScenesBoxDetectionLabel.HUMAN_PEDESTRIAN_PERSONAL_MOBILITY": VOID,
    "NuScenesBoxDetectionLabel.HUMAN_PEDESTRIAN_STROLLER": VOID,
    "NuScenesBoxDetectionLabel.HUMAN_PEDESTRIAN_WHEELCHAIR": VOID,
    "NuScenesBoxDetectionLabel.VEHICLE_EMERGENCY_AMBULANCE": VOID,
    "NuScenesBoxDetectionLabel.VEHICLE_EMERGENCY_POLICE": VOID,
}

COIN3D_3CLS = Taxonomy(
    name="coin3d_3cls",
    class_names=("car", "pedestrian", "motorcycle"),
    native_map=_COIN3D_3CLS_NATIVE,
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
"""CoIn3D's cross-dataset taxonomy: vehicle / pedestrian / two-wheeler, **spelled with nuScenes
names** (``car`` / ``pedestrian`` / ``motorcycle``).

The spelling is the point. The nuScenes devkit's ``DetectionBox`` asserts that a class name is one
of the ten nuScenes detection names, so CoIn3D reuses three of them for its unified classes and
can evaluate any dataset with the unmodified nuScenes metric code. Their published mapping
(``data_preprocess/README.md``) is nuScenes ``car, truck, trailer, bus, construction_vehicle ->
car``, ``pedestrian -> pedestrian``, ``bicycle, motorcycle -> motorcycle``, Lyft ``car, truck, bus,
emergency_vehicle, other_vehicle -> car``, Waymo ``VEHICLE / PEDESTRIAN / CYCLIST``. Expressed over
123D's default labels (plus the per-dataset exceptions above) it applies to every 123D dataset.
"""

VEHICLE_1CLS = Taxonomy(
    name="vehicle_1cls",
    class_names=("vehicle",),
    default_map={"VEHICLE": "vehicle", "TRAIN": "vehicle"},
)
"""Single-class vehicle taxonomy, useful for car-only cross-dataset ablations."""

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
"""Identity taxonomy over :class:`DefaultBoxDetectionLabel` (minus ``EGO``)."""

TAXONOMIES: Dict[str, Taxonomy] = {
    taxonomy.name: taxonomy
    for taxonomy in (NUSCENES_DETECTION, GENERAL_3CLS, COIN3D_3CLS, VEHICLE_1CLS, DEFAULT_123D)
}
"""All built-in taxonomies, keyed by name — the values accepted by ``--taxonomy`` on the CLI."""


def get_taxonomy(name: str) -> Taxonomy:
    """Look up a built-in taxonomy by name.

    :param name: One of the keys of :data:`TAXONOMIES`.
    :return: The taxonomy.
    :raises KeyError: If the name is not a built-in taxonomy.
    """
    if name not in TAXONOMIES:
        raise KeyError(f"Unknown taxonomy '{name}'. Available: {sorted(TAXONOMIES)}")
    return TAXONOMIES[name]
