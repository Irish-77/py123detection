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
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Tuple

from py123d.datatypes.detections.box_detection_label import BoxDetectionLabel

#: Target class name that drops a label.
VOID = "void"


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

#: A single ``vehicle`` class over the default labels (``VEHICLE`` and ``TRAIN``).
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
    for taxonomy in (NUSCENES_DETECTION, GENERAL_3CLS, COIN3D_3CLS, VEHICLE_1CLS, DEFAULT_123D)
}


def get_taxonomy(name: str) -> Taxonomy:
    if name not in TAXONOMIES:
        raise KeyError(f"Unknown taxonomy '{name}'. Available: {sorted(TAXONOMIES)}")
    return TAXONOMIES[name]
