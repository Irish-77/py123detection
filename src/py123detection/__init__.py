"""py123detection: camera-based 3D detection on 123D data.

Two branches sit on one shared core, and you pick the one that matches the ecosystem you are in:

``py123detection.core``
    The 123D data model both branches share: :class:`~py123detection.core.sources.Source`,
    :class:`~py123detection.core.taxonomy.Taxonomy`, the geometry, and the nuScenes detection
    metric. Needs py123d and numpy, nothing else.

``py123detection.mmcv`` — the **export branch**
    ``mmcv.export`` writes mmdetection3d info pickles from 123D logs (needs py123d);
    ``mmcv.plugin`` is the training-side dataset and evaluation (needs mmdetection3d). Existing
    PETR / StreamPETR / BEVDet configs then train on 123D data unchanged.

``py123detection.torch`` — the **torch branch**
    123D logs straight into PyTorch: dataset, PETR and BEVFormer, training and evaluation, with
    no mmcv anywhere. Needs the ``torch`` extra.

The export branch spans two Python environments that cannot hold each other's dependencies, so
importing this package has to work in both. Without py123d the names below are missing and
``HAS_EXPORT_SIDE`` is False; the torch branch is imported explicitly and never by this module.
"""

__version__ = "0.1.0"

try:
    from py123detection.core.annotations import BOX_LAYOUTS, DetectionRecord, FrameAnnotations
    from py123detection.core.sources import Source
    from py123detection.core.taxonomy import (
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
