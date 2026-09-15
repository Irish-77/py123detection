"""py123detection — a 123D toolkit for 3D multi-view object detection.

Two branches, sharing the 123D data model:

``py123detection.mmcv_export``
    Convert any 123D dataset into the ``info`` pickle that the mmcv / mmdetection3d ecosystem
    consumes, so existing PETR / StreamPETR / BEVDet configs train on 123D data unchanged.

``py123detection.mmcv_plugin``
    A small mmdetection3d-side plugin: a dataset class that reads exported pickles, including
    the ``portable`` (numpy-free) variant written for cross-version transfer, and evaluates them
    without a nuScenes database.

The two branches live on opposite sides of a Python-version boundary (see the README): the
export side needs py123d, the plugin side needs mmdetection3d. Importing this package must
therefore succeed in *either* environment, so the export-side names below are optional — in a
training environment without py123d they are simply absent and :data:`HAS_EXPORT_SIDE` is
``False``, while :mod:`py123detection.mmcv_plugin` keeps working.
"""

__version__ = "0.1.0"

try:
    from py123detection.annotations import BOX_LAYOUTS, DetectionRecord, FrameAnnotations
    from py123detection.sources import Source
    from py123detection.taxonomy import (
        COIN3D_3CLS,
        DEFAULT_123D,
        GENERAL_3CLS,
        NUSCENES_DETECTION,
        TAXONOMIES,
        VEHICLE_1CLS,
        Taxonomy,
        get_taxonomy,
    )
except ImportError as error:  # pragma: no cover - exercised only in a py123d-less training env
    HAS_EXPORT_SIDE = False
    """Whether the export side (py123d-dependent) of the package is importable here."""
    EXPORT_IMPORT_ERROR: "ImportError | None" = error
    """The import error that disabled the export side, for diagnostics."""
    __all__ = ["EXPORT_IMPORT_ERROR", "HAS_EXPORT_SIDE", "__version__"]
else:
    HAS_EXPORT_SIDE = True
    EXPORT_IMPORT_ERROR = None
    __all__ = [
        "BOX_LAYOUTS",
        "COIN3D_3CLS",
        "DEFAULT_123D",
        "DetectionRecord",
        "EXPORT_IMPORT_ERROR",
        "FrameAnnotations",
        "GENERAL_3CLS",
        "HAS_EXPORT_SIDE",
        "NUSCENES_DETECTION",
        "Source",
        "TAXONOMIES",
        "Taxonomy",
        "VEHICLE_1CLS",
        "__version__",
        "get_taxonomy",
    ]
