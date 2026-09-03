"""Unit tests for source resolution, the info schema check, and timestamp-ordering detection."""

from __future__ import annotations

import numpy as np
import pytest
from py123d.api import SceneFilter

from py123detection.mmcv_export.export import _assign_disjoint_offsets, check_timestamp_ordering
from py123detection.mmcv_export.schema import validate_info
from py123detection.mmcv_export.tokens import MappingTokenResolver, TokenResolver
from py123detection.sources import Source

# -- Source ------------------------------------------------------------------------------------


def test_filter_requests_one_scene_per_log() -> None:
    """Leaving future_* unset is what makes the scene builder emit whole logs, not sliding windows."""
    scene_filter = Source(data_root="/x", splits=["nuscenes_train"]).build_scene_filter()
    assert scene_filter.future_num_iterations is None
    assert scene_filter.future_duration_s is None
    assert scene_filter.history_num_iterations == 0
    assert scene_filter.split_names == ["nuscenes_train"]


def test_sample_rate_becomes_an_iteration_duration() -> None:
    scene_filter = Source(data_root="/x", sample_rate_hz=2.0).build_scene_filter()
    assert scene_filter.target_iteration_duration_s == pytest.approx(0.5)


def test_frame_stride_is_passed_through() -> None:
    scene_filter = Source(data_root="/x", frame_stride=5).build_scene_filter()
    assert scene_filter.target_iteration_stride == 5
    assert scene_filter.target_iteration_duration_s is None


def test_native_rate_filter_drops_the_sampling_knobs() -> None:
    source = Source(data_root="/x", sample_rate_hz=2.0, frame_stride=5)
    native = source.build_scene_filter(native_rate=True)
    assert native.target_iteration_duration_s is None
    assert native.target_iteration_stride is None


def test_native_rate_also_applies_to_a_custom_filter() -> None:
    source = Source(data_root="/x", scene_filter=SceneFilter(target_iteration_stride=4, split_names=["s"]))
    assert source.build_scene_filter().target_iteration_stride == 4
    assert source.build_scene_filter(native_rate=True).target_iteration_stride is None
    assert source.build_scene_filter(native_rate=True).split_names == ["s"]


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (Source(data_root="/x"), False),
        (Source(data_root="/x", sample_rate_hz=2.0), True),
        (Source(data_root="/x", frame_stride=1), False),
        (Source(data_root="/x", frame_stride=3), True),
    ],
)
def test_is_subsampled(source: Source, expected: bool) -> None:
    assert source.is_subsampled is expected


def test_rejects_a_non_positive_sample_rate() -> None:
    with pytest.raises(ValueError, match="sample_rate_hz must be > 0"):
        Source(data_root="/x", sample_rate_hz=0.0).build_scene_filter()


def test_missing_data_root_is_reported_clearly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PY123D_DATA_ROOT", raising=False)
    with pytest.raises(ValueError, match="PY123D_DATA_ROOT"):
        Source().load_scenes()


def test_label_falls_back_through_name_then_splits() -> None:
    assert Source(name="custom").label == "custom"
    assert Source(splits=["a", "b"]).label == "a+b"
    assert Source(datasets=["nuscenes"]).label == "nuscenes"
    assert Source().label == "all"


# -- token resolvers ----------------------------------------------------------------------------


def test_default_resolver_returns_the_123d_uuid() -> None:
    resolver = TokenResolver()
    assert resolver.resolve("nuscenes", "s", "log", 1, "uuid-1") == "uuid-1"
    assert resolver.name == "uuid"


def test_mapping_resolver_restores_native_tokens_and_falls_back() -> None:
    resolver = MappingTokenResolver({"scene-0103:42": "native-token"})
    assert resolver.resolve("nuscenes", "s", "scene-0103", 42, "uuid") == "native-token"
    assert resolver.resolve("nuscenes", "s", "scene-0103", 43, "uuid") == "uuid"
    assert resolver.num_misses == 1


def test_strict_mapping_resolver_raises_on_a_miss() -> None:
    resolver = MappingTokenResolver({}, strict=True)
    with pytest.raises(KeyError, match="No native token"):
        resolver.resolve("nuscenes", "s", "log", 1, "uuid")


# -- schema -------------------------------------------------------------------------------------


def make_valid_info() -> dict:
    """A schema-complete info with one camera and one box."""
    camera = {
        "data_path": "/x.jpg",
        "type": "CAM_FRONT",
        "sensor2ego_translation": [0, 0, 0],
        "sensor2ego_rotation": [1, 0, 0, 0],
        "ego2global_translation": [0, 0, 0],
        "ego2global_rotation": [1, 0, 0, 0],
        "timestamp": 1,
        "sensor2lidar_rotation": np.eye(3),
        "sensor2lidar_translation": np.zeros(3),
        "cam_intrinsic": np.eye(3),
    }
    return {
        "lidar_path": "/x.bin",
        "token": "t",
        "prev": "",
        "next": "",
        "sweeps": [],
        "frame_idx": 0,
        "cams": {"CAM_FRONT": camera},
        "scene_token": "split/log",
        "lidar2ego_translation": [0, 0, 0],
        "lidar2ego_rotation": [1, 0, 0, 0],
        "ego2global_translation": [0, 0, 0],
        "ego2global_rotation": [1, 0, 0, 0],
        "timestamp": 1,
        "gt_boxes": np.zeros((1, 7)),
        "gt_names": np.array(["car"]),
        "gt_velocity": np.zeros((1, 2)),
        "num_lidar_pts": np.array([3]),
        "num_radar_pts": np.array([0]),
        "valid_flag": np.array([True]),
        "bboxes2d": [np.zeros((0, 4))],
        "labels2d": [np.zeros((0,))],
        "centers2d": [np.zeros((0, 2))],
        "depths": [np.zeros((0,))],
        "bboxes_ignore": [np.zeros((0, 4))],
    }


def test_valid_info_passes() -> None:
    assert validate_info(make_valid_info()) == []


def test_missing_key_is_reported() -> None:
    info = make_valid_info()
    del info["gt_velocity"]
    assert "missing key 'gt_velocity'" in validate_info(info)


def test_streampetr_only_keys_are_optional_without_the_flag() -> None:
    info = make_valid_info()
    for key in ("bboxes2d", "labels2d", "centers2d", "depths", "bboxes_ignore"):
        del info[key]
    assert validate_info(info, streampetr=False) == []
    assert validate_info(info, streampetr=True)


def test_mismatched_annotation_lengths_are_reported() -> None:
    info = make_valid_info()
    info["gt_names"] = np.array(["car", "truck"])
    assert any("gt_names" in problem for problem in validate_info(info))


def test_per_camera_list_length_must_match_the_camera_count() -> None:
    info = make_valid_info()
    info["bboxes2d"] = [np.zeros((0, 4)), np.zeros((0, 4))]
    assert any("bboxes2d" in problem for problem in validate_info(info))


def test_missing_camera_key_is_reported() -> None:
    info = make_valid_info()
    del info["cams"]["CAM_FRONT"]["cam_intrinsic"]
    assert any("cam_intrinsic" in problem for problem in validate_info(info))


# -- timestamp ordering ---------------------------------------------------------------------------


def infos_at(scene_token: str, timestamps) -> list:
    """Minimal infos carrying only what the ordering check reads."""
    return [{"scene_token": scene_token, "timestamp": t} for t in timestamps]


def test_non_overlapping_logs_are_fine() -> None:
    infos = infos_at("a", [1, 2, 3]) + infos_at("b", [10, 11, 12])
    assert check_timestamp_ordering(infos) is None


def test_interleaved_logs_are_detected() -> None:
    infos = infos_at("a", [1, 3, 5]) + infos_at("b", [2, 4, 6])
    warning = check_timestamp_ordering(infos)
    assert warning is not None
    assert "disjoint_timestamps" in warning


def test_disjoint_offsets_leave_the_first_source_untouched() -> None:
    sources = [Source(name="a"), Source(name="b"), Source(name="c")]
    _assign_disjoint_offsets(sources)
    assert sources[0].timestamp_offset_us == 0
    assert sources[1].timestamp_offset_us > 0
    assert sources[2].timestamp_offset_us > sources[1].timestamp_offset_us


def test_disjoint_offsets_respect_an_explicit_value() -> None:
    sources = [Source(name="a"), Source(name="b", timestamp_offset_us=42)]
    _assign_disjoint_offsets(sources)
    assert sources[1].timestamp_offset_us == 42
