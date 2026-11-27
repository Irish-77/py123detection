"""Deprecated alias for :mod:`py123detection.mmcv.plugin`.

Kept as a real package directory so ``plugin_dir = "py123detection/mmcv_plugin/"`` in an existing
mmdet3d config still imports, and still triggers the ``@DATASETS.register_module()`` exactly once.
"""

from py123detection._compat import alias as _alias

_module = _alias("mmcv_plugin", "mmcv.plugin", ("dataset", "evaluation"))
globals().update({k: v for k, v in vars(_module).items() if not k.startswith("__")})
__all__ = list(getattr(_module, "__all__", []))
del _alias, _module
