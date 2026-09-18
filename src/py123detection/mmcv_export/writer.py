"""Writing the infos to a pickle that the training environment can read.

Conversion needs py123d (Python >= 3.9), while mmdetection3d setups are usually on Python 3.8
and numpy 1.x. Three things break a pickle across that gap:

1. Pickle protocol 5 is unreadable before Python 3.8, so the default is 4.
2. Any custom class (``pathlib.Path``, a 123D enum, ...) only unpickles where it is importable.
   :func:`validate_portable` rejects everything but built-ins and numpy arrays before writing.
3. Arrays pickled under numpy 2 fail to load under numpy 1 (``No module named 'numpy._core'``).
   Either install ``numpy<2`` for the conversion, or write ``array_format="portable"``, which
   stores arrays as nested lists and needs :mod:`py123detection.mmcv_plugin` to rebuild them.
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

DEFAULT_PICKLE_PROTOCOL = 4  # readable from Python 3.4 on

PAYLOAD_FORMAT = "py123detection.mmdet3d_infos.v1"

_PORTABLE_SCALARS = (str, int, float, bool, type(None))


class PortabilityError(TypeError):
    """The payload holds something that would not unpickle in a clean environment."""


@dataclass
class WriteResult:
    path: Path
    num_infos: int
    size_bytes: int
    protocol: int
    array_format: str
    numpy_version: str
    warnings: List[str]


def validate_portable(obj: Any, path: str = "payload") -> None:
    """Raise :class:`PortabilityError` on the first value that needs more than built-ins and numpy to unpickle."""
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
    """Recursively replace numpy arrays and scalars with lists and Python numbers."""
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
    if array_format == "portable" or int(np.__version__.split(".")[0]) < 2:
        return []
    return [
        f"Exporting with numpy {np.__version__}. Arrays pickled by numpy 2.x cannot be loaded by "
        "numpy 1.x ('No module named numpy._core'), and mmdetection3d environments are almost "
        "always on numpy 1.x. Either install 'numpy<2' in this conversion environment, or "
        "re-run with array_format='portable' and load the pickle through py123detection.mmcv_plugin."
    ]


def build_metadata(
    taxonomy_name: str,
    class_names: Sequence[str],
    version: str,
    sources: Sequence[str],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """The ``metadata`` block stored next to the infos.

    mmdetection3d only reads ``version``; the rest records how the export was produced.
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
    """Write ``{'infos': ..., 'metadata': ...}`` for mmdetection3d.

    :param protocol: 2 to 5. Anything above 4 warns, since it needs Python >= 3.8 to read.
    :param array_format: ``"numpy"``, or ``"portable"`` to store arrays as nested lists.
    :param validate: Check the payload with :func:`validate_portable` before writing.
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
    """``(metadata, number of infos)`` of an exported pickle. A cheap readability check for the training env."""
    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    return payload.get("metadata", {}), len(payload.get("infos", []))
