"""Import shims for the module layout before ``core`` / ``mmcv`` / ``torch``.

``py123detection.taxonomy`` and friends moved under :mod:`py123detection.core`, and
``py123detection.mmcv_export`` / ``mmcv_plugin`` under :mod:`py123detection.mmcv`. The old names
still resolve, so an mmdet3d config pinned to ``plugin_dir = "py123detection/mmcv_plugin/"`` keeps
working.

The alias points ``sys.modules`` at the *same* module object rather than re-importing the file.
Re-importing would run ``@DATASETS.register_module()`` a second time and give mmdet3d two classes
with one name, so submodules are aliased explicitly too.
"""

from __future__ import annotations

import sys
import warnings
from importlib import import_module
from types import ModuleType
from typing import Sequence


def alias(old_name: str, new_name: str, submodules: Sequence[str] = ()) -> ModuleType:
    """Make ``py123detection.<old_name>`` resolve to the module now called ``<new_name>``.

    :param submodules: names under the package to alias as well, so
        ``from py123detection.<old_name>.<sub> import x`` finds the cached module instead of
        executing the file again.
    """
    module = import_module(f"py123detection.{new_name}")
    sys.modules[f"py123detection.{old_name}"] = module
    for sub in submodules:
        sys.modules[f"py123detection.{old_name}.{sub}"] = import_module(f"py123detection.{new_name}.{sub}")
    warnings.warn(
        f"py123detection.{old_name} moved to py123detection.{new_name}; "
        f"the old name still works but will be removed.",
        DeprecationWarning,
        stacklevel=3,
    )
    return module
