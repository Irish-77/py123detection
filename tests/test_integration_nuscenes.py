"""End-to-end test: nuScenes -> 123D -> pkl must land where nuScenes -> pkl lands.

Marked ``integration`` and skipped unless the data is on disk. To run it::

    export NUSCENES_DATA_ROOT=/path/to/nuscenes      # holds v1.0-mini/, samples/, sweeps/
    export PY123D_DATA_ROOT=/path/to/py123d          # the 123D conversion of that nuScenes
    pytest -m integration tests/test_integration_nuscenes.py

The reference pickle is produced on the fly by ``tools/reference_nuscenes_converter.py``, a port
of the converter StreamPETR and mmdetection3d ship, so the comparison is against the real thing
rather than against a snapshot that could drift.

Everything geometric is asserted to be equal within floating-point noise. The two knowing
exceptions are asserted as *bounded* rather than exact, and each is a documented consequence of
what 123D stores:

* ``gt_velocity`` — 123D recomputes box velocity by finite differences at conversion time
  instead of copying ``nusc.box_velocity``. Both are the same central difference over the same
  keyframes, so the two agree to microns per second.
* ``sweeps`` — a keyframe-only 123D conversion holds 2 Hz lidar, while nuScenes' own sweeps are
  20 Hz. Sequence starts (an empty list) must still line up exactly, because StreamPETR's
  sequence grouping depends on them.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest

pytestmark = pytest.mark.integration

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

BENCHMARK_CATEGORIES = {
    "car",
    "truck",
    "trailer",
    "bus",
    "construction_vehicle",
    "bicycle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "barrier",
}

NUSCENES_ROOT = os.environ.get("NUSCENES_DATA_ROOT")
PY123D_ROOT = os.environ.get("PY123D_DATA_ROOT")
SPLIT = os.environ.get("PY123DET_TEST_SPLIT", "nuscenes-mini_val")

requires_data = pytest.mark.skipif(
    not NUSCENES_ROOT or not PY123D_ROOT,
    reason="set NUSCENES_DATA_ROOT and PY123D_DATA_ROOT to run the integration test",
)


@pytest.fixture(scope="module")
def reference_infos(tmp_path_factory) -> Dict[str, dict]:
    """Infos from the native nuScenes converter, keyed by sample token."""
    from reference_nuscenes_converter import create_nuscenes_infos

    out_dir = tmp_path_factory.mktemp("reference")
    _, val_path = create_nuscenes_infos(root_path=NUSCENES_ROOT, out_dir=str(out_dir), version="v1.0-mini")

    import pickle

    which = val_path if SPLIT.endswith("val") else val_path.parent / val_path.name.replace("_val", "_train")
    with open(which, "rb") as handle:
        return {info["token"]: info for info in pickle.load(handle)["infos"]}


@pytest.fixture(scope="module")
def exported_infos(tmp_path_factory) -> Dict[str, dict]:
    """Infos from the 123D export, keyed by the same (restored) sample token."""
    from py123detection import Source
    from py123detection.mmcv_export import ExportConfig, NuScenesTokenResolver, export_to_mmdet3d

    out = tmp_path_factory.mktemp("export") / "infos.pkl"
    report = export_to_mmdet3d(
        Source(data_root=PY123D_ROOT, splits=[SPLIT]),
        output_path=out,
        config=ExportConfig(token_resolver=NuScenesTokenResolver(NUSCENES_ROOT)),
        progress=False,
    )
    assert report.stats.num_frames > 0

    import pickle

    with open(out, "rb") as handle:
        return {info["token"]: info for info in pickle.load(handle)["infos"]}


def benchmark_mask(info: dict) -> np.ndarray:
    """Mask selecting the reference boxes a 10-class taxonomy keeps."""
    return np.array([name in BENCHMARK_CATEGORIES for name in np.asarray(info["gt_names"])], dtype=bool)


def matched_boxes(reference: dict, candidate: dict) -> List[tuple]:
    """Pair reference and candidate boxes by instance token."""
    mask = benchmark_mask(reference)
    reference_tokens = [token for token, keep in zip(reference["instance_tokens"], mask) if keep]
    candidate_index = {token: index for index, token in enumerate(candidate["instance_tokens"])}
    return [
        (reference_index, candidate_index[token])
        for reference_index, token in enumerate(reference_tokens)
        if token in candidate_index
    ]


@requires_data
def test_every_frame_is_present_with_the_same_token(reference_infos, exported_infos) -> None:
    assert set(reference_infos) == set(exported_infos)


@requires_data
def test_frame_scalars_match_exactly(reference_infos, exported_infos) -> None:
    for token, reference in reference_infos.items():
        candidate = exported_infos[token]
        assert candidate["timestamp"] == reference["timestamp"]
        assert list(candidate["cams"]) == list(reference["cams"])
        assert os.path.realpath(candidate["lidar_path"]) == os.path.realpath(reference["lidar_path"])


@requires_data
def test_frame_poses_match(reference_infos, exported_infos) -> None:
    from pyquaternion import Quaternion

    for token, reference in reference_infos.items():
        candidate = exported_infos[token]
        for key in ("lidar2ego", "ego2global"):
            np.testing.assert_allclose(
                candidate[f"{key}_translation"], reference[f"{key}_translation"], atol=1e-9
            )
            # Compare rotations as matrices: q and -q are the same rotation.
            np.testing.assert_allclose(
                Quaternion(np.asarray(candidate[f"{key}_rotation"])).rotation_matrix,
                Quaternion(np.asarray(reference[f"{key}_rotation"])).rotation_matrix,
                atol=1e-12,
            )


@requires_data
def test_camera_calibration_and_transforms_match(reference_infos, exported_infos) -> None:
    for token, reference in reference_infos.items():
        candidate = exported_infos[token]
        for camera_name, reference_cam in reference["cams"].items():
            candidate_cam = candidate["cams"][camera_name]
            assert candidate_cam["timestamp"] == reference_cam["timestamp"]
            assert os.path.realpath(candidate_cam["data_path"]) == os.path.realpath(reference_cam["data_path"])
            np.testing.assert_allclose(
                candidate_cam["cam_intrinsic"], reference_cam["cam_intrinsic"], atol=1e-12
            )
            # sensor2lidar is what PETR / StreamPETR turn into lidar2img.
            np.testing.assert_allclose(
                candidate_cam["sensor2lidar_rotation"], reference_cam["sensor2lidar_rotation"], atol=1e-10
            )
            np.testing.assert_allclose(
                candidate_cam["sensor2lidar_translation"],
                reference_cam["sensor2lidar_translation"],
                atol=1e-9,
            )


@requires_data
def test_box_counts_and_classes_match(reference_infos, exported_infos) -> None:
    for token, reference in reference_infos.items():
        candidate = exported_infos[token]
        mask = benchmark_mask(reference)
        assert len(candidate["gt_names"]) == int(mask.sum())

        pairs = matched_boxes(reference, candidate)
        assert len(pairs) == int(mask.sum()), "every benchmark box must pair by instance token"

        reference_names = np.asarray(reference["gt_names"])[mask]
        for reference_index, candidate_index in pairs:
            assert reference_names[reference_index] == candidate["gt_names"][candidate_index]


@requires_data
def test_box_geometry_matches(reference_infos, exported_infos) -> None:
    for token, reference in reference_infos.items():
        candidate = exported_infos[token]
        mask = benchmark_mask(reference)
        reference_boxes = np.asarray(reference["gt_boxes"])[mask]
        reference_points = np.asarray(reference["num_lidar_pts"])[mask]

        for reference_index, candidate_index in matched_boxes(reference, candidate):
            expected = reference_boxes[reference_index]
            actual = candidate["gt_boxes"][candidate_index]
            np.testing.assert_allclose(actual[:3], expected[:3], atol=1e-9)  # center
            np.testing.assert_allclose(actual[3:6], expected[3:6], atol=1e-12)  # l, w, h
            yaw_error = np.arctan2(np.sin(actual[6] - expected[6]), np.cos(actual[6] - expected[6]))
            assert abs(yaw_error) < 1e-9
            assert candidate["num_lidar_pts"][candidate_index] == reference_points[reference_index]


@requires_data
def test_box_velocity_matches_within_the_recomputation_tolerance(reference_infos, exported_infos) -> None:
    """123D re-derives velocity by finite differences; the same central difference, so ~1e-5 m/s."""
    worst = 0.0
    for token, reference in reference_infos.items():
        candidate = exported_infos[token]
        mask = benchmark_mask(reference)
        reference_velocity = np.asarray(reference["gt_velocity"])[mask]

        for reference_index, candidate_index in matched_boxes(reference, candidate):
            expected = reference_velocity[reference_index]
            if np.isnan(expected).any():  # the devkit returns NaN at track boundaries
                continue
            worst = max(worst, float(np.abs(expected - candidate["gt_velocity"][candidate_index]).max()))
    assert worst < 1e-3, f"velocity deviation {worst} m/s is larger than recomputation noise"


@requires_data
def test_2d_annotations_match(reference_infos, exported_infos) -> None:
    """StreamPETR reads these during training, so they must reproduce the reference exactly."""
    total = 0
    for token, reference in reference_infos.items():
        candidate = exported_infos[token]
        for camera_index in range(len(reference["cams"])):
            reference_boxes = np.asarray(reference["bboxes2d"][camera_index]).reshape(-1, 4)
            candidate_boxes = np.asarray(candidate["bboxes2d"][camera_index]).reshape(-1, 4)
            assert candidate_boxes.shape == reference_boxes.shape
            if reference_boxes.size == 0:
                continue

            reference_centers = np.asarray(reference["centers2d"][camera_index]).reshape(-1, 2)
            candidate_centers = np.asarray(candidate["centers2d"][camera_index]).reshape(-1, 2)
            order = np.argsort(reference_centers[:, 0], kind="stable")
            candidate_order = np.argsort(candidate_centers[:, 0], kind="stable")

            np.testing.assert_allclose(candidate_boxes[candidate_order], reference_boxes[order], atol=1e-4)
            np.testing.assert_allclose(
                candidate_centers[candidate_order], reference_centers[order], atol=1e-4
            )
            np.testing.assert_allclose(
                np.asarray(candidate["depths"][camera_index])[candidate_order],
                np.asarray(reference["depths"][camera_index])[order],
                atol=1e-4,
            )
            total += len(reference_boxes)
    assert total > 0, "the fixture produced no 2D boxes to compare"


@requires_data
def test_sequence_starts_line_up(reference_infos, exported_infos) -> None:
    """An empty sweep list marks a sequence start; StreamPETR's grouping depends on it."""
    reference_starts = {token for token, info in reference_infos.items() if len(info["sweeps"]) == 0}
    candidate_starts = {token for token, info in exported_infos.items() if len(info["sweeps"]) == 0}
    assert reference_starts == candidate_starts


@requires_data
def test_sweeps_are_capped_and_ordered_newest_first(exported_infos) -> None:
    for info in exported_infos.values():
        assert len(info["sweeps"]) <= 10
        timestamps = [sweep["timestamp"] for sweep in info["sweeps"]]
        assert timestamps == sorted(timestamps, reverse=True)
        assert all(timestamp < info["timestamp"] for timestamp in timestamps)


@requires_data
def test_prev_next_links_are_consistent(exported_infos) -> None:
    for token, info in exported_infos.items():
        if info["next"]:
            assert exported_infos[info["next"]]["prev"] == token
        if info["prev"]:
            assert exported_infos[info["prev"]]["next"] == token


@requires_data
def test_exported_infos_satisfy_the_schema(exported_infos) -> None:
    from py123detection.mmcv_export import validate_info

    for info in exported_infos.values():
        assert validate_info(info) == []
