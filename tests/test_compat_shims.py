"""The deprecated import paths, which nothing checked.

``py123detection.taxonomy`` and friends moved under ``core``, and ``mmcv_export`` / ``mmcv_plugin``
under ``mmcv``. The old names still resolve so an mmdet3d config pinned to
``plugin_dir = "py123detection/mmcv_plugin/"`` keeps working — and until now no test imported a
single old path, so nothing would have noticed them breaking.

The property that matters is not that the names resolve but that they resolve to the *same module
object*. ``alias`` points ``sys.modules`` at the already-imported module instead of executing the
file again; re-executing ``mmcv/plugin/dataset.py`` would run ``@DATASETS.register_module()`` a
second time and leave mmdet3d with two different classes under one name. That failure appears far
away from its cause, which is why it is worth a test here.

These shims import only ``core``, so they run without the torch extra and this module is not marked
``torch``. The two ``mmcv`` shims need py123d, which the base install provides.
"""

from __future__ import annotations

import importlib
import sys
import warnings

import pytest

#: ``old name`` -> ``new name``, as ``_compat.alias`` wires them.
MOVED = {
    "annotations": "core.annotations",
    "geometry": "core.geometry",
    "sources": "core.sources",
    "taxonomy": "core.taxonomy",
    "velocity": "core.velocity",
    "mmcv_export": "mmcv.export",
    "mmcv_plugin": "mmcv.plugin",
}

#: Submodules each package aliases, so ``from <old>.<sub> import x`` finds the cached module.
SUBMODULES = {
    "mmcv_export": ("annotations2d", "cli", "converter", "export", "schema", "sensors", "tokens", "writer"),
    "mmcv_plugin": ("dataset", "evaluation"),
}


def _import(name: str):
    """Import ``py123detection.<name>`` with the deprecation warning suppressed."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return importlib.import_module(f"py123detection.{name}")


@pytest.mark.parametrize("old", sorted(MOVED))
def test_the_old_name_still_imports(old):
    assert _import(old) is not None


@pytest.mark.parametrize("old,new", sorted(MOVED.items()))
def test_the_old_name_is_the_same_module_object_as_the_new_one(old, new):
    """Not merely equivalent: identical.

    A copy would mean module-level side effects running twice, and for the mmcv plugin that means
    two classes registered under one name.
    """
    assert _import(old) is _import(new)
    assert sys.modules[f"py123detection.{old}"] is sys.modules[f"py123detection.{new}"]


@pytest.mark.parametrize(
    "old,sub",
    [(old, sub) for old, subs in sorted(SUBMODULES.items()) for sub in subs],
)
def test_an_aliased_submodule_resolves_to_the_cached_module(old, sub):
    """``from py123detection.mmcv_export.writer import ...`` must not re-execute the file."""
    _import(old)  # aliasing the package is what registers its submodules
    new = MOVED[old]
    assert sys.modules[f"py123detection.{old}.{sub}"] is _import(f"{new}.{sub}")


@pytest.mark.parametrize("old,new", sorted(MOVED.items()))
def test_aliasing_warns_and_names_both_paths(old, new):
    """The warning is the only thing telling anyone to migrate, so it has to name both paths.

    This calls ``alias`` rather than reloading the shim module. Reloading would re-execute the
    real module and replace its module-level singletons -- which broke an unrelated identity
    assertion in ``test_taxonomy.py`` when this test first did it, and is the very hazard the
    aliasing exists to avoid. ``alias`` is safe to call again because ``import_module`` returns the
    cached module rather than running the file.
    """
    from py123detection._compat import alias

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        module = alias(old, new, SUBMODULES.get(old, ()))

    assert module is sys.modules[f"py123detection.{new}"], "alias re-executed the module"
    messages = [str(entry.message) for entry in caught if entry.category is DeprecationWarning]
    assert messages, f"py123detection.{old} moved without saying so"
    assert any(old in message and new in message for message in messages), messages


def test_a_name_reached_through_the_old_path_is_the_same_object():
    """What a caller actually depends on: the class, not the module."""
    from py123detection.core.taxonomy import Taxonomy

    assert _import("taxonomy").Taxonomy is Taxonomy


def test_the_shim_republishes_the_public_names():
    """``from py123detection.mmcv_export import export_to_mmdet3d`` is the documented spelling."""
    shim = _import("mmcv_export")
    new = _import("mmcv.export")
    for name in getattr(new, "__all__", []):
        assert hasattr(shim, name), f"mmcv_export lost {name}"
