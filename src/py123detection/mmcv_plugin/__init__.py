"""mmdetection3d-side plugin for pickles written by :mod:`py123detection.mmcv_export`.

A ``numpy``-format export already works with the stock ``NuScenesDataset``. The plugin is needed
for ``portable`` exports (arrays stored as lists), to take ``CLASSES`` from the pickle's
taxonomy, and to evaluate without a nuScenes database. mmdet3d is only imported if available, so
this also imports in a conversion environment.

From an mmdet3d config::

    plugin = True
    plugin_dir = "py123detection/mmcv_plugin/"   # or add py123detection to custom_imports

    data = dict(
        train=dict(type="Py123DNuScenesDataset", ann_file="py123d_infos_train.pkl", ...),
    )
"""

from py123detection.mmcv_plugin.dataset import Py123DNuScenesDataset, rebuild_arrays

__all__ = ["Py123DNuScenesDataset", "rebuild_arrays"]
