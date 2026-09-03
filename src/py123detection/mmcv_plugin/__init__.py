"""mmdetection3d-side plugin for pickles produced by :mod:`py123detection.mmcv_export`.

This package lives on the *training* side of the version boundary. It imports mmdet3d lazily, so
importing :mod:`py123detection` in a conversion environment (where mmdet3d is absent) stays safe.

An export written with ``array_format="numpy"`` needs nothing from here — it is byte-for-byte the
schema ``mmdet3d.datasets.NuScenesDataset`` already reads. The plugin earns its keep in two cases:

* the export used ``array_format="portable"``, which stores arrays as nested lists to dodge
  numpy's cross-major-version pickle incompatibility, and something has to rebuild the arrays;
* you want the dataset's ``CLASSES`` to come from the pickle's own taxonomy metadata rather than
  being repeated in the config.

Use it from an mmdet3d config::

    plugin = True
    plugin_dir = "py123detection/mmcv_plugin/"   # or add py123detection to custom_imports

    data = dict(
        train=dict(type="Py123DNuScenesDataset", ann_file="py123d_infos_train.pkl", ...),
    )
"""

from py123detection.mmcv_plugin.dataset import Py123DNuScenesDataset, rebuild_arrays

__all__ = ["Py123DNuScenesDataset", "rebuild_arrays"]
