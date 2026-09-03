"""py123detection — a 123D toolkit for 3D multi-view object detection.

Two branches, sharing the 123D data model:

``py123detection.mmcv_export``
    Convert any 123D dataset into the ``info`` pickle that the mmcv / mmdetection3d ecosystem
    consumes, so existing PETR / StreamPETR / BEVDet configs train on 123D data unchanged.

``py123detection.mmcv_plugin``
    A small mmdetection3d-side plugin: a dataset class that reads exported pickles, including
    the ``portable`` (numpy-free) variant written for cross-version transfer.
"""

from py123detection.annotations import DetectionRecord, FrameAnnotations
from py123detection.sources import Source
from py123detection.taxonomy import (
    DEFAULT_123D,
    GENERAL_3CLS,
    NUSCENES_DETECTION,
    TAXONOMIES,
    VEHICLE_1CLS,
    Taxonomy,
    get_taxonomy,
)

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_123D",
    "DetectionRecord",
    "FrameAnnotations",
    "GENERAL_3CLS",
    "NUSCENES_DETECTION",
    "Source",
    "TAXONOMIES",
    "Taxonomy",
    "VEHICLE_1CLS",
    "__version__",
    "get_taxonomy",
]
