"""Writing the exported infos to a pickle that a *different* Python can read.

Conversion needs py123d (Python >= 3.9), while the mmcv/mmdetection3d stack it feeds is usually
pinned to Python 3.8. The pickle therefore has to survive a version hop, and there are exactly
three things that can break it:

1. **Pickle protocol.** Protocol 5 is unreadable before Python 3.8. We default to protocol 4,
   which every Python from 3.4 onwards reads, and never write anything higher by default.
2. **Class identity.** A pickle that references ``pathlib.Path``, a 123D enum, or any custom
   class only loads where that class is importable — and unpickling would then drag py123d into
   the training environment. :func:`validate_portable` walks the payload before writing and
   rejects anything outside a small allowlist of built-ins and numpy arrays.
3. **numpy's own pickle format.** This is the one that actually bites people. numpy 2 renamed
   ``numpy.core`` to ``numpy._core``, and arrays pickled under numpy 2 fail to load under
   numpy 1 with ``ModuleNotFoundError: No module named 'numpy._core'``. The reverse direction is
   fine. Since mmdet3d environments are essentially always on numpy 1.x, exporting from a numpy 2
   environment produces a pickle their trainer cannot open.

For (3) there are two ways out, and :func:`dump_infos` supports both: install ``numpy<2`` in the
conversion environment (recommended — nothing else changes), or pass
``array_format="portable"`` to store every array as nested Python lists, which sidesteps numpy's
pickle format entirely. Portable pickles are larger and need
:mod:`py123detection.mmcv_plugin` on the training side to rebuild the arrays.
"""

from __future__ import annotations

import logging
import pickle
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_PICKLE_PROTOCOL = 4
"""Highest protocol every supported reader (Python >= 3.4) understands."""

PAYLOAD_FORMAT = "py123detection.mmdet3d_infos.v1"
"""Value of ``metadata['format']``, so readers can recognise and version-check the payload."""

_PORTABLE_SCALARS = (str, int, float, bool, type(None))


class PortabilityError(TypeError):
    """Raised when a payload contains something that would not unpickle in a clean environment."""


@dataclass
class WriteResult:
    """What :func:`dump_infos` actually wrote."""

    path: Path
    """Destination of the pickle."""

    num_infos: int
    """Number of frames written."""

    size_bytes: int
    """Size of the written file."""

    protocol: int
    """Pickle protocol used."""

    array_format: str
    """``"numpy"`` or ``"portable"``."""

    numpy_version: str
    """numpy version that produced the file."""

    warnings: List[str]
    """Compatibility warnings raised while writing, also echoed through the logger."""


def validate_portable(obj: Any, path: str = "payload") -> None:
    """Check that ``obj`` only contains types that unpickle without extra imports.

    :param obj: The object to check, walked recursively.
    :param path: Dotted path used in error messages.
    :raises PortabilityError: On the first offending value found.
    """
    if isinstance(obj, _PORTABLE_SCALARS):
        return
    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            raise PortabilityError(f"{path}: object-dtype arrays are not portable.")
        return
    if isinstance(obj, np.generic):
        return
    if isinstance(obj, dict):
        for key, value in obj.items():
            if not isinstance(key, str):
                raise PortabilityError(f"{path}: dict keys must be strings, found {type(key).__name__}.")
            validate_portable(value, f"{path}[{key!r}]")
        return
    if isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj):
            validate_portable(value, f"{path}[{index}]")
        return
    raise PortabilityError(
        f"{path}: {type(obj).__module__}.{type(obj).__name__} is not portable across environments. "
        "Exported infos may only contain str/int/float/bool/None, numpy arrays, lists, tuples and "
        "string-keyed dicts."
    )


def to_portable_arrays(obj: Any) -> Any:
    """Recursively replace numpy arrays and scalars with plain Python lists and numbers.

    :param obj: The object to convert.
    :return: An equivalent structure containing no numpy types.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {key: to_portable_arrays(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [to_portable_arrays(value) for value in obj]
    if isinstance(obj, tuple):
        return tuple(to_portable_arrays(value) for value in obj)
    return obj


def numpy_compatibility_warnings(array_format: str) -> List[str]:
    """Return the compatibility warnings that apply to the current numpy version.

    :param array_format: ``"numpy"`` or ``"portable"``.
    :return: Human-readable warnings; empty when nothing is at risk.
    """
    warnings: List[str] = []
    if array_format == "portable":
        return warnings

    major = int(np.__version__.split(".")[0])
    if major >= 2:
        warnings.append(
            f"Exporting with numpy {np.__version__}. Arrays pickled by numpy 2.x cannot be loaded by "
            "numpy 1.x ('No module named numpy._core'), and mmdetection3d environments are almost "
            "always on numpy 1.x. Either install 'numpy<2' in this conversion environment, or "
            "re-run with array_format='portable' and load the pickle through py123detection.mmcv_plugin."
        )
    return warnings


def build_metadata(
    taxonomy_name: str,
    class_names: Sequence[str],
    version: str,
    sources: Sequence[str],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble the ``metadata`` block stored next to the infos.

    mmdetection3d reads ``metadata['version']`` and nothing else, so the remaining keys are ours:
    they record how the export was produced, which is what makes a pickle auditable months later.

    :param taxonomy_name: Name of the taxonomy used.
    :param class_names: Ordered class names — copy this into the mmdet3d config's ``class_names``.
    :param version: Dataset version string surfaced as ``metadata['version']``.
    :param sources: Labels of the sources that contributed frames.
    :param extra: Extra keys to merge in (statistics, config echoes, ...).
    :return: The metadata dictionary.
    """
    metadata: Dict[str, Any] = {
        "version": version,
        "format": PAYLOAD_FORMAT,
        "producer": "py123detection",
        "taxonomy": taxonomy_name,
        "class_names": list(class_names),
        "sources": list(sources),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
    }
    if extra:
        metadata.update(extra)
    return metadata


def dump_infos(
    infos: List[Dict[str, Any]],
    output_path: Union[str, Path],
    metadata: Dict[str, Any],
    protocol: int = DEFAULT_PICKLE_PROTOCOL,
    array_format: str = "numpy",
    validate: bool = True,
) -> WriteResult:
    """Write ``{'infos': ..., 'metadata': ...}`` to a pickle mmdetection3d can load.

    :param infos: The exported frames.
    :param output_path: Destination path (``.pkl``).
    :param metadata: Metadata block, typically from :func:`build_metadata`.
    :param protocol: Pickle protocol. Values above 4 are refused unless the caller is explicit
        about targeting Python >= 3.8 readers only.
    :param array_format: ``"numpy"`` keeps numpy arrays (what mmdet3d expects natively);
        ``"portable"`` converts them to nested lists, sidestepping numpy's pickle format at the
        cost of size and needing :mod:`py123detection.mmcv_plugin` on the reading side.
    :param validate: Walk the payload with :func:`validate_portable` before writing.
    :return: A :class:`WriteResult` describing the written file.
    :raises PortabilityError: If ``validate`` is set and the payload is not portable.
    """
    if array_format not in ("numpy", "portable"):
        raise ValueError(f"array_format must be 'numpy' or 'portable', got {array_format!r}.")
    if protocol > 5 or protocol < 2:
        raise ValueError(f"protocol must be between 2 and 5, got {protocol}.")

    payload: Dict[str, Any] = {"infos": infos, "metadata": dict(metadata)}
    payload["metadata"]["array_format"] = array_format
    payload["metadata"]["pickle_protocol"] = protocol

    if array_format == "portable":
        payload = to_portable_arrays(payload)

    if validate:
        validate_portable(payload)

    warnings = numpy_compatibility_warnings(array_format)
    if protocol > DEFAULT_PICKLE_PROTOCOL:
        warnings.append(
            f"Pickle protocol {protocol} requires Python >= 3.8 to read; protocol "
            f"{DEFAULT_PICKLE_PROTOCOL} is portable back to Python 3.4."
        )
    for warning in warnings:
        logger.warning("%s", warning)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as handle:
        pickle.dump(payload, handle, protocol=protocol)

    return WriteResult(
        path=output_path,
        num_infos=len(infos),
        size_bytes=output_path.stat().st_size,
        protocol=protocol,
        array_format=array_format,
        numpy_version=np.__version__,
        warnings=warnings,
    )


def inspect_pickle(path: Union[str, Path]) -> Tuple[Dict[str, Any], int]:
    """Load an exported pickle and return its metadata and frame count.

    Useful as a smoke test from the *training* environment: if this succeeds there, the exported
    file is readable by mmdetection3d.

    :param path: Path to the pickle.
    :return: ``(metadata, num_infos)``.
    """
    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    return payload.get("metadata", {}), len(payload.get("infos", []))
