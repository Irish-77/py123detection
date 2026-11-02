"""Freezing and comparing dataset output, so a refactor has to prove it changed nothing.

The rewrite of the 123D torch dataset is the one step in the port that can silently change
results. :func:`summarise_sample` reduces one ``__getitem__`` return value to something small
enough to commit and exact enough that any change to a pixel, a box or a matrix moves it: small
tensors are stored in full, large ones by a digest of their bytes.

``tools/freeze_dataset_golden.py`` writes the fixture from the legacy implementation and
``tests/test_dataset_equivalence.py`` asserts the new one reproduces it, so both sides run this
same code.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Mapping

import numpy as np

#: Tensors with at most this many elements are stored elementwise; bigger ones by digest.
MAX_INLINE_ELEMENTS = 4096

#: Decimals kept when a tensor is stored elementwise. Enough to catch a real change, loose
#: enough to survive a different summation order on the same maths.
INLINE_DECIMALS = 5


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def digest(array: np.ndarray) -> str:
    """Stable digest of an array's exact bytes, independent of how it was produced."""
    contiguous = np.ascontiguousarray(array)
    hasher = hashlib.blake2b(digest_size=16)
    hasher.update(str(contiguous.dtype).encode())
    hasher.update(str(contiguous.shape).encode())
    hasher.update(contiguous.tobytes())
    return hasher.hexdigest()


def summarise_array(array: Any) -> Dict[str, Any]:
    """Shape, dtype and contents of one array, inline when small and hashed when not.

    Floating-point stats come back rounded, so a summary is readable in a diff and a failure says
    which way a value moved rather than only that it moved.
    """
    array = _to_numpy(array)
    summary: Dict[str, Any] = {"shape": list(array.shape), "dtype": str(array.dtype)}

    if array.size == 0:
        summary["empty"] = True
        return summary

    if np.issubdtype(array.dtype, np.floating):
        finite = array[np.isfinite(array)]
        summary["stats"] = {
            "sum": round(float(finite.sum()), 6),
            "min": round(float(finite.min()), 6) if finite.size else None,
            "max": round(float(finite.max()), 6) if finite.size else None,
            "nonfinite": int(array.size - finite.size),
        }

    if array.size <= MAX_INLINE_ELEMENTS:
        if np.issubdtype(array.dtype, np.floating):
            summary["values"] = np.round(array, INLINE_DECIMALS).tolist()
        else:
            summary["values"] = array.tolist()
    else:
        summary["digest"] = digest(array)

    return summary


def summarise_sample(sample: Mapping[str, Any]) -> Dict[str, Any]:
    """Reduce one dataset sample to a comparable, committable structure."""
    if sample is None:
        return {"sample": None}

    summary: Dict[str, Any] = {"keys": sorted(sample.keys()), "arrays": {}, "scalars": {}}
    for key in sorted(sample):
        value = sample[key]
        if key == "img_metas":
            summary["img_metas"] = _summarise_metas(value)
        elif hasattr(value, "shape") or isinstance(value, (list, tuple)):
            summary["arrays"][key] = summarise_array(value)
        else:
            summary["scalars"][key] = _plain(value)
    return summary


def _summarise_metas(metas: Any) -> Dict[str, Any]:
    if not isinstance(metas, Mapping):
        return {"repr": repr(metas)}
    return {key: _plain(metas[key]) for key in sorted(metas)}


def _plain(value: Any) -> Any:
    """JSON-safe form of a metadata value, rounding floats so noise does not fail a test."""
    if isinstance(value, (bool, int, str)) or value is None:
        return value
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in sorted(value.items())}
    if hasattr(value, "item") and getattr(value, "size", 2) == 1:
        return _plain(value.item())
    if hasattr(value, "tolist"):
        return _plain(value.tolist())
    return repr(value)


def compare_summaries(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
    path: str = "",
    tolerance: float = 0.0,
) -> List[str]:
    """Every difference between two summaries, as readable one-line messages.

    Returns all of them rather than stopping at the first, because when a refactor does move
    something you want the whole blast radius in one run.

    :param tolerance: Absolute difference two numbers may have and still count as equal. Use it
        for known changes in numerical precision, not to paper over a change in behaviour — it is
        small enough that a real difference still fails.
    """
    problems: List[str] = []

    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        for key in sorted(set(expected) | set(actual)):
            where = f"{path}.{key}" if path else str(key)
            if key not in actual:
                problems.append(f"{where}: missing (expected {_short(expected[key])})")
            elif key not in expected:
                problems.append(f"{where}: unexpected (got {_short(actual[key])})")
            else:
                problems.extend(compare_summaries(expected[key], actual[key], where, tolerance))
        return problems

    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            problems.append(f"{path}: length {len(expected)} -> {len(actual)}")
            return problems
        for index, (want, got) in enumerate(zip(expected, actual)):
            problems.extend(compare_summaries(want, got, f"{path}[{index}]", tolerance))
        return problems

    if _equal(expected, actual, tolerance):
        return problems
    problems.append(f"{path}: {_short(expected)} -> {_short(actual)}")
    return problems


def _equal(expected: Any, actual: Any, tolerance: float) -> bool:
    if expected == actual:
        return True
    if tolerance and isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if isinstance(expected, bool) or isinstance(actual, bool):
            return False
        return abs(float(expected) - float(actual)) <= tolerance
    return False


def _short(value: Any, limit: int = 110) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."
