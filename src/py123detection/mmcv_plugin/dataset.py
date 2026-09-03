"""A ``NuScenesDataset`` subclass that reads py123detection exports.

Kept free of import-time mmdet3d dependencies: :class:`Py123DNuScenesDataset` is only defined
(and registered) when mmdet3d is importable, so this module can also be imported from a
conversion environment to reuse :func:`rebuild_arrays`.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# Fields whose values are numpy arrays in a "numpy"-format export, with the dtype to restore.
_ARRAY_FIELDS: Dict[str, Any] = {
    "gt_boxes": np.float64,
    "gt_velocity": np.float64,
    "num_lidar_pts": np.int64,
    "num_radar_pts": np.int64,
    "valid_flag": np.bool_,
}
_STRING_ARRAY_FIELDS = ("gt_names",)
_CAM_ARRAY_FIELDS = ("sensor2lidar_rotation", "sensor2lidar_translation", "cam_intrinsic")
_SWEEP_ARRAY_FIELDS = ("sensor2lidar_rotation", "sensor2lidar_translation")
# Per-camera lists of arrays (StreamPETR's 2D annotations).
_PER_CAMERA_ARRAY_FIELDS: Dict[str, Any] = {
    "bboxes2d": np.float32,
    "labels2d": np.int64,
    "centers2d": np.float32,
    "depths": np.float32,
    "bboxes3d_cams": np.float32,
    "bboxes_ignore": np.float32,
}
_EMPTY_SHAPES: Dict[str, Sequence[int]] = {
    "gt_boxes": (0, 7),
    "gt_velocity": (0, 2),
    "bboxes2d": (0, 4),
    "centers2d": (0, 2),
    "bboxes3d_cams": (0, 7),
    "bboxes_ignore": (0, 4),
}


def _as_array(value: Any, dtype: Any, key: str) -> np.ndarray:
    """Convert one exported value back into a numpy array with the right shape and dtype."""
    array = np.asarray(value, dtype=dtype)
    if array.size == 0 and key in _EMPTY_SHAPES:
        array = array.reshape(_EMPTY_SHAPES[key])
    return array


def rebuild_arrays(info: Dict[str, Any]) -> Dict[str, Any]:
    """Restore numpy arrays in an info loaded from a ``portable``-format export.

    Infos from a ``numpy``-format export already hold arrays; converting them again is a no-op
    apart from an ``np.asarray`` that returns the same buffer, so this is safe to run
    unconditionally.

    :param info: One exported info, modified in place.
    :return: The same info, for convenience.
    """
    for key, dtype in _ARRAY_FIELDS.items():
        if key in info:
            info[key] = _as_array(info[key], dtype, key)
    for key in _STRING_ARRAY_FIELDS:
        if key in info:
            info[key] = np.asarray(info[key], dtype="<U32")

    for key, dtype in _PER_CAMERA_ARRAY_FIELDS.items():
        if key in info and isinstance(info[key], list):
            info[key] = [_as_array(value, dtype, key) for value in info[key]]

    for payload in info.get("cams", {}).values():
        for key in _CAM_ARRAY_FIELDS:
            if key in payload:
                payload[key] = np.asarray(payload[key], dtype=np.float64)
    for sweep in info.get("sweeps", []):
        for key in _SWEEP_ARRAY_FIELDS:
            if key in sweep:
                sweep[key] = np.asarray(sweep[key], dtype=np.float64)

    return info


try:  # pragma: no cover - exercised only inside an mmdetection3d environment
    from mmdet.datasets import DATASETS
    from mmdet3d.datasets import NuScenesDataset

    @DATASETS.register_module()
    class Py123DNuScenesDataset(NuScenesDataset):
        """``NuScenesDataset`` that understands py123detection export pickles.

        Two behaviours on top of the stock dataset:

        * arrays are rebuilt after loading, so ``array_format="portable"`` exports work;
        * when ``classes`` is left unset, the taxonomy recorded in the pickle's metadata supplies
          ``CLASSES``, keeping the class list in one place instead of duplicated in the config.
        """

        def __init__(self, *args, use_taxonomy_classes: bool = True, **kwargs) -> None:
            """Initialize the dataset.

            :param use_taxonomy_classes: Adopt ``metadata['class_names']`` from the pickle when
                the config does not pass ``classes`` explicitly.
            """
            self._use_taxonomy_classes = use_taxonomy_classes
            self._py123d_metadata: Dict[str, Any] = {}
            super().__init__(*args, **kwargs)

        @property
        def py123d_metadata(self) -> Dict[str, Any]:
            """The ``metadata`` block of the loaded pickle."""
            return self._py123d_metadata

        def load_annotations(self, ann_file: str) -> List[Dict[str, Any]]:
            """Load infos and restore numpy arrays.

            :param ann_file: Path to a py123detection export pickle.
            :return: The infos, sorted by timestamp and subsampled by ``load_interval``,
                matching the stock ``NuScenesDataset`` contract.
            """
            data_infos = super().load_annotations(ann_file)
            for info in data_infos:
                rebuild_arrays(info)

            metadata = getattr(self, "metadata", None)
            if isinstance(metadata, dict):
                self._py123d_metadata = metadata
                class_names: Optional[Sequence[str]] = metadata.get("class_names")
                if self._use_taxonomy_classes and class_names:
                    if tuple(self.CLASSES) != tuple(class_names):
                        logger.info(
                            "Py123DNuScenesDataset: adopting class_names %s from the export's "
                            "'%s' taxonomy (config had %s).",
                            list(class_names),
                            metadata.get("taxonomy"),
                            list(self.CLASSES),
                        )
                    self.CLASSES = tuple(class_names)
            return data_infos

except ImportError:  # pragma: no cover - conversion environments have no mmdet3d
    Py123DNuScenesDataset = None  # type: ignore[assignment]
