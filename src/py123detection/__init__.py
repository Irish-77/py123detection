"""py123detection: train mmdetection3d models on 123D data.

``py123detection.mmcv_export`` writes mmdetection3d info pickles from 123D logs and needs py123d.
``py123detection.mmcv_plugin`` is the training-side dataset and evaluation and needs mmdetection3d.

The two usually live in different Python environments, so importing this package has to work in
both. Without py123d the export names below are missing and ``HAS_EXPORT_SIDE`` is False.
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
except ImportError as error:  # training environment without py123d
    HAS_EXPORT_SIDE = False
    EXPORT_IMPORT_ERROR = error
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
