"""Predictions as py123d's own detection type, so py123d's viewer can draw them.

py123d ships a Viser viewer and the element system around it
(:mod:`py123d.visualization.viser`). Writing a second one here would copy several thousand lines of
maintained code into a package whose point is to be dependable, so this package provides the one
piece that was genuinely missing: the conversion from a predictions file to the type that viewer
already knows how to draw.

Two levels are offered, because they fail differently. :func:`to_box_array` is pure geometry and
depends on nothing but numpy — if a box comes out in the wrong place, it is wrong here and a test
can say so. :func:`to_detections` builds py123d's object graph on top of it and needs py123d
imported; it is a translation of labels and bookkeeping, not of geometry.

Needs the ``viz`` extra for the drawing helpers; the conversion itself needs only py123d, which the
base install already requires.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Tuple

import numpy as np

from py123detection.torch.predictions import FramePrediction

logger = logging.getLogger(__name__)

#: Taxonomies already warned about, so stepping through a scene does not repeat it per frame.
_WARNED: set = set()

if TYPE_CHECKING:  # pragma: no cover - import cost, and py123d types are not needed to type-check
    from py123detection.core.taxonomy import Taxonomy

#: Our box layout, ``[x, y, z, w, l, h, yaw, vx, vy]``.
YAW = 6

#: py123d's ``BoundingBoxSE3Index``: ``[X, Y, Z, QW, QX, QY, QZ, LENGTH, WIDTH, HEIGHT]``.
BOX_ARRAY_COLUMNS = 10


def to_box_array(boxes: np.ndarray) -> np.ndarray:
    """``(N, 7+)`` boxes in this package's layout as ``(N, 10)`` in py123d's.

    The two differ in more than order: ours carries yaw as an angle and sizes as ``w, l, h``, while
    py123d carries a quaternion and ``length, width, height``. Getting ``w`` and ``l`` the wrong way
    round swaps every box's footprint and still renders a plausible scene, so it is pinned by a test
    against py123d's own corner maths.
    """
    boxes = np.asarray(boxes, dtype=np.float64)
    if boxes.size == 0:
        return np.zeros((0, BOX_ARRAY_COLUMNS), dtype=np.float64)

    half_yaw = boxes[:, YAW] / 2.0
    out = np.zeros((len(boxes), BOX_ARRAY_COLUMNS), dtype=np.float64)
    out[:, 0:3] = boxes[:, 0:3]
    out[:, 3] = np.cos(half_yaw)  # QW; yaw is about z, so QX and QY stay zero
    out[:, 6] = np.sin(half_yaw)  # QZ
    out[:, 7] = boxes[:, 4]  # LENGTH is our l
    out[:, 8] = boxes[:, 3]  # WIDTH is our w
    out[:, 9] = boxes[:, 5]  # HEIGHT
    return out


def label_lookup(taxonomy: "Taxonomy") -> Dict[str, Any]:
    """Our class names back to py123d labels, inverting the taxonomy's own map.

    A taxonomy records py123d label to class name, which is many-to-one — ``general_3cls`` sends
    both ``VEHICLE`` and ``TRAIN`` to ``vehicle``. Going back therefore loses information, and the
    first label that maps to a class wins, which is the representative one in every taxonomy
    shipped here. Inverting the taxonomy rather than writing a second table means the two cannot
    drift apart.
    """
    from py123d.datatypes.detections import DefaultBoxDetectionLabel

    lookup: Dict[str, Any] = {}
    for label_name, class_name in taxonomy.default_map.items():
        if class_name == "void" or class_name in lookup:
            continue
        member = getattr(DefaultBoxDetectionLabel, label_name, None)
        if member is not None:
            lookup[class_name] = member
    return lookup


def unmapped_classes(taxonomy: "Taxonomy") -> Sequence[str]:
    """Classes of this taxonomy that py123d has no label for.

    py123d's vocabulary is coarse on purpose — ``VEHICLE`` and ``TWO_WHEELER``, not car, truck, bus,
    trailer, bicycle, motorcycle — so a fine-grained taxonomy cannot round-trip through it.
    ``mixed_8cls`` loses four of its eight classes, and ``nuscenes_detection_10cls`` loses all ten
    because it is built from ``native_map`` rather than ``default_map``. Those boxes still draw,
    as ``GENERIC_OBJECT``; what they lose is their colour in the viewer's legend.
    """
    lookup = label_lookup(taxonomy)
    return [name for name in taxonomy.class_names if name != "void" and name not in lookup]


def _warn_once(taxonomy: "Taxonomy") -> None:
    missing = unmapped_classes(taxonomy)
    if missing and taxonomy.name not in _WARNED:
        _WARNED.add(taxonomy.name)
        logger.warning(
            "py123d has no label for %s of %r: %s. They will draw as GENERIC_OBJECT, so the viewer "
            "cannot tell them apart by colour.",
            len(missing), taxonomy.name, ", ".join(missing),
        )


def to_detections(
    frame: FramePrediction,
    taxonomy: "Taxonomy",
    ground_truth: bool = False,
    score_threshold: float = 0.0,
    timestamp_us: int = 0,
) -> Any:
    """One frame as a ``BoxDetectionsSE3``, ready for py123d's viewer or its matplotlib helpers.

    :param ground_truth: Convert the frame's ground truth instead of its predictions.
    :param score_threshold: Predictions below this are left out; ignored for ground truth, which
        carries no score.
    :param timestamp_us: The frame's time. A predictions file does not record one — the metric has
        no use for it — so the caller supplies it when the viewer needs frames ordered.
    """
    from py123d.datatypes.detections import (
        BoxDetectionAttributes,
        BoxDetectionSE3,
        BoxDetectionsSE3,
        BoxDetectionsSE3Metadata,
        DefaultBoxDetectionLabel,
    )
    from py123d.datatypes.time import Timestamp
    from py123d.geometry import BoundingBoxSE3

    boxes = frame.gt_boxes if ground_truth else frame.boxes
    labels = frame.gt_labels if ground_truth else frame.labels
    if not ground_truth and score_threshold > 0.0 and len(boxes):
        keep = np.asarray(frame.scores, dtype=np.float64) >= score_threshold
        boxes, labels = boxes[keep], np.asarray(labels)[keep]

    _warn_once(taxonomy)
    lookup = label_lookup(taxonomy)
    names = list(taxonomy.class_names)
    array = to_box_array(boxes)
    kind = "gt" if ground_truth else "pred"

    detections = []
    for index, row in enumerate(array):
        class_name = names[int(labels[index])] if index < len(labels) else names[0]
        detections.append(
            BoxDetectionSE3(
                attributes=BoxDetectionAttributes(
                    label=lookup.get(class_name, DefaultBoxDetectionLabel.GENERIC_OBJECT),
                    # The viewer keys handles by track token; a predictions file has no tracks, so
                    # one stable token per box per frame is enough to keep them distinct.
                    track_token=f"{frame.token}:{kind}:{index}",
                ),
                bounding_box_se3=BoundingBoxSE3.from_array(row),
            )
        )

    return BoxDetectionsSE3(
        box_detections=detections,
        timestamp=Timestamp.from_us(timestamp_us),
        metadata=BoxDetectionsSE3Metadata(box_detection_label_class=DefaultBoxDetectionLabel),
    )


def wireframe(boxes: np.ndarray) -> np.ndarray:
    """``(N, 12, 2, 3)`` line segments for a set of boxes, via py123d's own corner maths.

    This is what a viser overlay draws. Using py123d's helpers rather than
    :mod:`py123detection.torch.visualize.boxes` keeps the viewer's geometry identical to everything
    else py123d renders in the same scene — the two agree, which a test checks, but agreeing by
    construction is better than agreeing by coincidence.
    """
    from py123d.geometry.utils.bounding_box_utils import bbse3_array_to_corners_array
    from py123d.visualization.viser.elements.box_detections_se3_element import (
        corners_array_to_edge_lines,
    )

    array = to_box_array(boxes)
    if len(array) == 0:
        return np.zeros((0, 12, 2, 3), dtype=np.float64)
    return corners_array_to_edge_lines(bbse3_array_to_corners_array(array))


def add_to_viser(
    server: Any,
    frame: FramePrediction,
    name: str = "/predictions",
    score_threshold: float = 0.3,
    colour: Tuple[int, int, int] = (214, 39, 40),
    line_width: float = 3.0,
    ground_truth: bool = False,
) -> Any:
    """Draw one frame's boxes into a running viser server as wireframes.

    Wireframes rather than solid meshes so predictions read as an overlay on whatever the scene
    already shows, which is how the thesis distinguished them from ground truth.

    :returns: the viser handle, so a caller stepping through frames can update or hide it.
    """
    boxes = frame.gt_boxes if ground_truth else frame.boxes
    if not ground_truth and len(boxes):
        keep = np.asarray(frame.scores, dtype=np.float64) >= score_threshold
        boxes = boxes[keep]

    points = wireframe(boxes).reshape(-1, 2, 3)
    # One colour per endpoint of every segment: viser wants colors shaped exactly like points, and
    # a per-box array is twelve times too short.
    colours = np.broadcast_to(
        np.asarray(colour, dtype=np.uint8), points.shape
    ).copy()
    return server.scene.add_line_segments(
        name, points=points, colors=colours, line_width=line_width
    )


def frames_by_token(path, dataset: Optional[str] = None) -> Dict[str, FramePrediction]:
    """A predictions file indexed by frame token, for a viewer stepping through a scene."""
    from py123detection.torch.predictions import frames_for

    return {frame.token: frame for frame in frames_for(path, dataset)}


def tokens_in_order(frames: Dict[str, FramePrediction]) -> Sequence[str]:
    """Frame tokens sorted by their frame index, which is how a scene is walked.

    A token is ``<scene uuid>:<frame index>``, so sorting them as strings puts frame 10 before
    frame 2. Sorting on the parsed index is the difference between a scene that plays and one that
    jumps about.
    """

    def key(token: str):
        scene, _, index = token.rpartition(":")
        return scene, int(index) if index.isdigit() else 0

    return sorted(frames, key=key)
