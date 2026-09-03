"""Unit tests for pickle writing, portability validation, and the array round trip.

These cover the cross-version story: the exported pickle must load in an mmdetection3d
environment running a different Python and a different numpy, without py123d installed.
"""

from __future__ import annotations

import pickle
import pickletools
from enum import Enum
from pathlib import Path

import numpy as np
import pytest

from py123detection.mmcv_export.writer import (
    PortabilityError,
    build_metadata,
    dump_infos,
    inspect_pickle,
    numpy_compatibility_warnings,
    to_portable_arrays,
    validate_portable,
)
from py123detection.mmcv_plugin import rebuild_arrays


class _Colour(Enum):
    RED = 1


def make_info() -> dict:
    """A minimal info with every array kind the exporter emits."""
    return {
        "token": "abc",
        "timestamp": 1533151603547590,
        "lidar_path": "/data/x.bin",
        "gt_boxes": np.zeros((3, 7), dtype=np.float64),
        "gt_names": np.array(["car", "truck", "car"], dtype="<U32"),
        "gt_velocity": np.zeros((3, 2), dtype=np.float64),
        "num_lidar_pts": np.array([1, 0, 5], dtype=np.int64),
        "num_radar_pts": np.zeros((3,), dtype=np.int64),
        "valid_flag": np.array([True, False, True]),
        "instance_tokens": ["t0", "t1", "t2"],
        "bboxes2d": [np.zeros((2, 4), dtype=np.float32), np.zeros((0, 4), dtype=np.float32)],
        "labels2d": [np.zeros((2,), dtype=np.int64), np.zeros((0,), dtype=np.int64)],
        "centers2d": [np.zeros((2, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)],
        "depths": [np.zeros((2,), dtype=np.float32), np.zeros((0,), dtype=np.float32)],
        "bboxes3d_cams": [np.zeros((2, 7), dtype=np.float32), np.zeros((0, 7), dtype=np.float32)],
        "bboxes_ignore": [np.zeros((0, 4), dtype=np.float32), np.zeros((0, 4), dtype=np.float32)],
        "cams": {
            "CAM_FRONT": {
                "data_path": "/data/f.jpg",
                "sensor2lidar_rotation": np.eye(3),
                "sensor2lidar_translation": np.zeros(3),
                "cam_intrinsic": np.eye(3),
            }
        },
        "sweeps": [
            {
                "data_path": "/data/s.bin",
                "sensor2lidar_rotation": np.eye(3),
                "sensor2lidar_translation": np.zeros(3),
            }
        ],
    }


def pickle_globals(path: Path) -> set:
    """Every module-level name the pickle stream references."""
    found, recent = set(), []
    for opcode, arg, _ in pickletools.genops(path.read_bytes()):
        if opcode.name in ("SHORT_BINUNICODE", "BINUNICODE"):
            recent.append(str(arg))
            recent[:] = recent[-2:]
        elif opcode.name == "STACK_GLOBAL":
            found.add(".".join(recent))
        elif opcode.name == "GLOBAL":
            found.add(str(arg))
    return found


# -- portability validation ------------------------------------------------------------------


def test_validate_accepts_a_normal_info() -> None:
    validate_portable(make_info())


@pytest.mark.parametrize(
    "payload",
    [
        {"path": Path("/data")},
        {"colour": _Colour.RED},
        {"nested": [{"bad": Path("/x")}]},
        {"array": np.array([{"a": 1}], dtype=object)},
    ],
)
def test_validate_rejects_non_portable_values(payload: dict) -> None:
    with pytest.raises(PortabilityError):
        validate_portable(payload)


def test_validate_rejects_non_string_dict_keys() -> None:
    with pytest.raises(PortabilityError, match="dict keys must be strings"):
        validate_portable({1: "x"})


def test_validate_error_names_the_offending_path() -> None:
    with pytest.raises(PortabilityError, match=r"payload\['a'\]\[2\]"):
        validate_portable({"a": [1, 2, Path("/x")]})


# -- writing ---------------------------------------------------------------------------------


def test_dump_writes_a_loadable_payload(tmp_path: Path) -> None:
    metadata = build_metadata("t", ["car"], "v1", ["src"])
    result = dump_infos([make_info()], tmp_path / "out.pkl", metadata)

    assert result.num_infos == 1
    assert result.protocol == 4
    assert result.size_bytes > 0

    loaded_metadata, count = inspect_pickle(result.path)
    assert count == 1
    assert loaded_metadata["class_names"] == ["car"]
    assert loaded_metadata["array_format"] == "numpy"
    assert loaded_metadata["format"] == "py123detection.mmdet3d_infos.v1"


def test_numpy_format_references_only_numpy(tmp_path: Path) -> None:
    """No py123d class may leak in — the training environment does not have py123d installed."""
    result = dump_infos([make_info()], tmp_path / "out.pkl", build_metadata("t", ["car"], "v1", []))
    references = pickle_globals(result.path)
    assert references, "a numpy-format pickle must reference numpy"
    assert all(name.startswith(("numpy", "ndarray")) for name in references), references


def test_portable_format_references_nothing(tmp_path: Path) -> None:
    """The portable format is the escape hatch from numpy's cross-major-version pickle break."""
    result = dump_infos(
        [make_info()], tmp_path / "out.pkl", build_metadata("t", ["car"], "v1", []), array_format="portable"
    )
    assert pickle_globals(result.path) == set()


def test_default_protocol_is_readable_by_python38(tmp_path: Path) -> None:
    result = dump_infos([make_info()], tmp_path / "out.pkl", build_metadata("t", ["car"], "v1", []))
    assert pickletools.genops(result.path.read_bytes()).__next__()[1] == 4


def test_protocol_above_default_warns(tmp_path: Path) -> None:
    result = dump_infos([make_info()], tmp_path / "o.pkl", build_metadata("t", ["car"], "v1", []), protocol=5)
    assert any("Python >= 3.8" in warning for warning in result.warnings)


def test_protocol_bounds_are_enforced(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="protocol must be"):
        dump_infos([], tmp_path / "o.pkl", build_metadata("t", [], "v1", []), protocol=6)


def test_unknown_array_format_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="array_format must be"):
        dump_infos([], tmp_path / "o.pkl", build_metadata("t", [], "v1", []), array_format="parquet")


def test_metadata_records_the_producing_environment() -> None:
    metadata = build_metadata("tax", ["a", "b"], "v1", ["s1", "s2"], extra={"k": 1})
    assert metadata["numpy_version"] == np.__version__
    assert metadata["taxonomy"] == "tax"
    assert metadata["sources"] == ["s1", "s2"]
    assert metadata["k"] == 1


def test_numpy2_export_is_flagged(monkeypatch: pytest.MonkeyPatch) -> None:
    """numpy 2 pickles cannot be read by numpy 1, which is what mmdet3d environments run."""
    monkeypatch.setattr(np, "__version__", "2.1.0")
    warnings = numpy_compatibility_warnings("numpy")
    assert len(warnings) == 1
    assert "numpy<2" in warnings[0]
    assert numpy_compatibility_warnings("portable") == []


# -- portable round trip ---------------------------------------------------------------------


def test_to_portable_arrays_strips_every_numpy_type() -> None:
    portable = to_portable_arrays(make_info())

    def has_numpy(obj) -> bool:
        if isinstance(obj, (np.ndarray, np.generic)):
            return True
        if isinstance(obj, dict):
            return any(has_numpy(value) for value in obj.values())
        if isinstance(obj, (list, tuple)):
            return any(has_numpy(value) for value in obj)
        return False

    assert not has_numpy(portable)


def test_portable_round_trip_restores_arrays_exactly(tmp_path: Path) -> None:
    original = make_info()
    original["gt_boxes"] = np.arange(21, dtype=np.float64).reshape(3, 7)
    original["gt_velocity"] = np.arange(6, dtype=np.float64).reshape(3, 2)

    result = dump_infos(
        [original], tmp_path / "o.pkl", build_metadata("t", ["car"], "v1", []), array_format="portable"
    )
    with open(result.path, "rb") as handle:
        restored = rebuild_arrays(pickle.load(handle)["infos"][0])

    np.testing.assert_array_equal(restored["gt_boxes"], original["gt_boxes"])
    np.testing.assert_array_equal(restored["gt_velocity"], original["gt_velocity"])
    np.testing.assert_array_equal(restored["gt_names"], original["gt_names"])
    np.testing.assert_array_equal(restored["valid_flag"], original["valid_flag"])
    assert restored["gt_boxes"].dtype == np.float64
    assert restored["num_lidar_pts"].dtype == np.int64
    assert restored["cams"]["CAM_FRONT"]["sensor2lidar_rotation"].shape == (3, 3)
    assert restored["sweeps"][0]["sensor2lidar_translation"].shape == (3,)


def test_rebuild_preserves_the_shape_of_empty_arrays() -> None:
    """A frame with no boxes must still produce (0, 7), or mmdet3d's indexing breaks."""
    rebuilt = rebuild_arrays({"gt_boxes": [], "gt_velocity": [], "bboxes2d": [[]], "centers2d": [[]]})
    assert rebuilt["gt_boxes"].shape == (0, 7)
    assert rebuilt["gt_velocity"].shape == (0, 2)
    assert rebuilt["bboxes2d"][0].shape == (0, 4)
    assert rebuilt["centers2d"][0].shape == (0, 2)


def test_rebuild_is_idempotent_on_numpy_format_infos() -> None:
    info = make_info()
    once = rebuild_arrays(dict(info))
    twice = rebuild_arrays(once)
    np.testing.assert_array_equal(twice["gt_boxes"], info["gt_boxes"])
