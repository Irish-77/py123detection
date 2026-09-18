"""A ``NuScenesDataset`` subclass for py123detection exports.

:class:`Py123DNuScenesDataset` is only defined when mmdet3d is importable, so :func:`rebuild_arrays`
can also be used from a conversion environment.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from py123detection.mmcv_plugin.evaluation import evaluate_from_infos

logger = logging.getLogger(__name__)

_ARRAY_FIELDS: Dict[str, Any] = {
    "gt_boxes": np.float64,
    "gt_velocity": np.float64,
    "num_lidar_pts": np.int64,
    "num_radar_pts": np.int64,
    "valid_flag": np.bool_,
}
_CAM_ARRAY_FIELDS = ("sensor2lidar_rotation", "sensor2lidar_translation", "cam_intrinsic")
_SWEEP_ARRAY_FIELDS = ("sensor2lidar_rotation", "sensor2lidar_translation")
# StreamPETR's 2D annotations, one array per camera.
_PER_CAMERA_ARRAY_FIELDS: Dict[str, Any] = {
    "bboxes2d": np.float32,
    "labels2d": np.int64,
    "centers2d": np.float32,
    "depths": np.float32,
    "bboxes3d_cams": np.float32,
    "bboxes_ignore": np.float32,
}
# An empty list comes back as shape (0,); mmdet3d indexing needs the column count.
_EMPTY_SHAPES: Dict[str, Sequence[int]] = {
    "gt_boxes": (0, 7),
    "gt_velocity": (0, 2),
    "bboxes2d": (0, 4),
    "centers2d": (0, 2),
    "bboxes3d_cams": (0, 7),
    "bboxes_ignore": (0, 4),
}


def _as_array(value: Any, dtype: Any, key: str) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.size == 0 and key in _EMPTY_SHAPES:
        array = array.reshape(_EMPTY_SHAPES[key])
    return array


def rebuild_arrays(info: Dict[str, Any]) -> Dict[str, Any]:
    """Restore the numpy arrays of an info from a ``portable`` export, in place.

    Safe on ``numpy`` exports too: ``np.asarray`` returns the arrays unchanged.
    """
    for key, dtype in _ARRAY_FIELDS.items():
        if key in info:
            info[key] = _as_array(info[key], dtype, key)
    if "gt_names" in info:
        info["gt_names"] = np.asarray(info["gt_names"], dtype="<U32")

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


try:
    from mmdet.datasets import DATASETS
    from mmdet3d.datasets import NuScenesDataset

    @DATASETS.register_module()
    class Py123DNuScenesDataset(NuScenesDataset):
        """``NuScenesDataset`` for py123detection exports.

        On top of the stock dataset it rebuilds ``portable`` arrays, takes ``CLASSES`` from the
        pickle's taxonomy when the config sets none, and by default evaluates against the ground
        truth in the pickle instead of a nuScenes database (``eval_mode="nuscenes"`` restores the
        stock evaluation).

        :param eval_class_range: Per-class evaluation radius in metres. Classes with nuScenes names
            default to the ``detection_cvpr_2019`` radii.
        """

        def __init__(
            self,
            *args,
            use_taxonomy_classes: bool = True,
            eval_mode: str = "self_contained",
            eval_class_range: Optional[Dict[str, float]] = None,
            **kwargs,
        ) -> None:
            if eval_mode not in ("self_contained", "nuscenes"):
                raise ValueError(f"eval_mode must be 'self_contained' or 'nuscenes', got {eval_mode!r}.")
            self._use_taxonomy_classes = use_taxonomy_classes
            self._eval_mode = eval_mode
            self._eval_class_range: Dict[str, float] = dict(eval_class_range or {})
            self.py123d_metadata: Dict[str, Any] = {}
            super().__init__(*args, **kwargs)
            # _format_bbox filters predictions by this radius before the metric runs, so every
            # evaluated class needs an entry.
            self.eval_detection_configs.class_range.update(self._eval_class_range)

        def _evaluate_single(self, result_path, logger=None, metric="bbox", result_name="pts_bbox"):
            if self._eval_mode == "nuscenes":
                return super()._evaluate_single(result_path, logger=logger, metric=metric, result_name=result_name)

            metrics = evaluate_from_infos(
                result_path,
                self.data_infos,
                list(self.CLASSES),
                os.path.dirname(result_path),
                class_range=self._eval_class_range or None,
                eval_version=self.eval_version,
                verbose=True,
            )

            # Same keys as mmdet3d's _evaluate_single, so logs and hooks look the same.
            detail: Dict[str, float] = {}
            prefix = f"{result_name}_NuScenes"
            for name in self.CLASSES:
                for key, value in metrics["label_aps"][name].items():
                    detail[f"{prefix}/{name}_AP_dist_{key}"] = float(f"{value:.4f}")
                for key, value in metrics["label_tp_errors"][name].items():
                    detail[f"{prefix}/{name}_{key}"] = float(f"{value:.4f}")
            for key, value in metrics["tp_errors"].items():
                detail[f"{prefix}/{self.ErrNameMapping[key]}"] = float(f"{value:.4f}")
            detail[f"{prefix}/NDS"] = metrics["nd_score"]
            detail[f"{prefix}/mAP"] = metrics["mean_ap"]
            return detail

        def get_data_info(self, index: int) -> Dict[str, Any]:
            """Stock ``get_data_info`` plus ``intrinsics`` / ``extrinsics`` / ``img_timestamp``.

            Those are the only additions the PETR family's ``CustomNuScenesDataset`` makes, so
            their configs can use this dataset directly.
            """
            input_dict = super().get_data_info(index)
            if not self.modality["use_camera"]:
                return input_dict
            intrinsics, extrinsics, img_timestamp = [], [], []
            for cam_info in self.data_infos[index]["cams"].values():
                lidar2cam_r = np.linalg.inv(cam_info["sensor2lidar_rotation"])
                lidar2cam_t = cam_info["sensor2lidar_translation"] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = cam_info["cam_intrinsic"]
                viewpad = np.eye(4)
                viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
                intrinsics.append(viewpad)
                extrinsics.append(lidar2cam_rt)
                img_timestamp.append(cam_info["timestamp"] / 1e6)
            input_dict.update(dict(intrinsics=intrinsics, extrinsics=extrinsics, img_timestamp=img_timestamp))
            return input_dict

        def load_annotations(self, ann_file: str) -> List[Dict[str, Any]]:
            data_infos = super().load_annotations(ann_file)
            for info in data_infos:
                rebuild_arrays(info)

            metadata = getattr(self, "metadata", None)
            if isinstance(metadata, dict):
                self.py123d_metadata = metadata
                class_names = metadata.get("class_names")
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

except ImportError:  # no mmdet3d in conversion environments
    Py123DNuScenesDataset = None  # type: ignore[assignment]
