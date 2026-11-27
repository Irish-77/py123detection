"""Deprecated alias for :mod:`py123detection.core.velocity`."""

from py123detection._compat import alias as _alias

_module = _alias("velocity", "core.velocity")
globals().update({k: v for k, v in vars(_module).items() if not k.startswith("__")})
__all__ = getattr(_module, "__all__", [k for k in vars(_module) if not k.startswith("_")])
del _alias, _module
