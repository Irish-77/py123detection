"""Deprecated alias for :mod:`py123detection.mmcv.export`."""

from py123detection._compat import alias as _alias

_module = _alias(
    "mmcv_export",
    "mmcv.export",
    ("annotations2d", "cli", "converter", "export", "schema", "sensors", "tokens", "writer"),
)
globals().update({k: v for k, v in vars(_module).items() if not k.startswith("__")})
__all__ = list(getattr(_module, "__all__", []))
del _alias, _module
