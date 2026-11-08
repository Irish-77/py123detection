"""The refactored dataset has to reproduce what the master-thesis one produced.

``tools/freeze_dataset_golden.py`` recorded the legacy ``DatasetNuScenesFormat`` output on real
123D logs. These tests rebuild the same configurations with
:class:`~py123detection.torch.data.Py123DDetectionDataset` and compare.

Some differences are intended — a bug fixed, a class order changed — and
:data:`tests.golden.configs.EXPECTED_DIFFERENCES` names each one with its reason. The comparison
fails both ways: an unexpected difference is a regression, and an expected one that stopped
appearing means the fixture or the reason has gone stale.

These need the 123D logs on disk, so they are marked ``integration`` and deselected by default.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import pytest

from tests.golden import compare_summaries, summarise_sample
from tests.golden.configs import (
    DATA_ROOT,
    GOLDEN_CONFIGS,
    NUMERIC_TOLERANCE,
    SEED,
    expected_differences,
)

torch = pytest.importorskip("torch", reason="the torch branch needs torch")

from py123detection.core.sources import Source  # noqa: E402
from py123detection.core.taxonomy import get_taxonomy  # noqa: E402
from py123detection.torch.data import (  # noqa: E402
    Augmentation,
    BevTransform,
    BoxFilters,
    CameraRig,
    Draw,
    ImagePipeline,
    ImageTransform,
    Py123DDetectionDataset,
)
from py123detection.torch.data.rigs import PREPROCESSING  # noqa: E402
from py123detection.torch.models import multiview  # noqa: E402

IMAGENET_MEAN = np.asarray(multiview.IMAGENET_MEAN, dtype=np.float32)
IMAGENET_STD = np.asarray(multiview.IMAGENET_STD, dtype=np.float32)

FIXTURE = Path(__file__).parent / "golden" / "dataset_golden.json"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not Path(DATA_ROOT).exists(), reason=f"no 123D logs at {DATA_ROOT}"),
    pytest.mark.torch,
]


@pytest.fixture(scope="module")
def golden() -> Dict[str, Any]:
    if not FIXTURE.exists():
        pytest.skip(f"no fixture at {FIXTURE}; run tools/freeze_dataset_golden.py")
    return json.loads(FIXTURE.read_text())


def build_dataset(config: Dict[str, Any]) -> Py123DDetectionDataset:
    """The current dataset for one configuration."""
    image = config["image"]
    filters = config.get("filters", {})
    visibility = filters.get("visibility") or {}

    return Py123DDetectionDataset(
        source=Source(data_root=DATA_ROOT, **config["source"]),
        taxonomy=get_taxonomy(config["taxonomy"]),
        rig=CameraRig(
            cameras=tuple(config["cameras"]),
            preprocessing=PREPROCESSING.get(config.get("camera_preprocessing")),
        ),
        images=ImagePipeline(size=tuple(image["size"]), mode="crop" if image["mode"] == "ida" else "stretch"),
        augment=Augmentation() if config.get("augment") else None,
        filters=BoxFilters(
            drop_boxes_without_lidar_points=filters.get("skip_empty_boxes", True),
            drop_frames_without_boxes=filters.get("skip_empty_frames", True),
            pc_range=tuple(filters["pc_range"]) if filters.get("pc_range") else None,
            visibility=visibility.get("mode"),
            visibility_margin_deg=visibility.get("fov_margin_deg", 0.0),
        ),
        compute_velocity=config.get("compute_velocity", False),
        seed=SEED,
    )


def replay(config: Dict[str, Any], sample: Dict[str, Any]) -> Draw:
    """The draw the legacy dataset made for one recorded sample.

    The two implementations draw their randomness differently, so replaying the legacy draw order
    is neither possible nor the point: what has to agree is what the pipeline does with a given
    transform, not which transform a generator happened to pick.
    """
    recorded = sample["image_transform"]
    resize = recorded["resize"]
    if resize is None:  # the stretch path scales each axis independently
        source_height, source_width = sample["img_metas"]["ori_shape"][0][:2]
        if config.get("camera_preprocessing"):
            source_width, source_height = PREPROCESSING[config["camera_preprocessing"]].target_size
        resize_pair = (recorded["resize_dims"][0] / source_width, recorded["resize_dims"][1] / source_height)
    else:
        resize_pair = (resize, resize)

    return Draw(
        image=ImageTransform(resize=resize_pair, crop=tuple(recorded["crop"]), flip=bool(recorded["flip"])),
        bev=BevTransform(
            angle=float(sample["img_metas"]["rotation"]),
            scale=float(sample["img_metas"]["scale_factor"]),
        ),
    )


def normalized(sample: Dict[str, Any]) -> np.ndarray:
    """The ``uint8`` image as the model sees it: normalised, zero outside the image."""
    images = sample["img"].numpy().astype(np.float32)
    normalised = (images - IMAGENET_MEAN.reshape(1, 3, 1, 1)) / IMAGENET_STD.reshape(1, 3, 1, 1)
    height, width = (int(value) for value in sample["img_shape"])
    normalised[..., height:, :] = 0.0
    normalised[..., :, width:] = 0.0
    normalised[~sample["cam_valid_mask"].numpy()] = 0.0
    return normalised


def summarise(dataset: Py123DDetectionDataset, index: int, draw: Draw) -> Dict[str, Any]:
    """The summary to compare with the legacy one, images left out: they have their own test."""
    sample = dict(dataset.get(index, draw))
    sample["gt_class_names"] = list(sample["img_metas"].get("gt_class_names", []))
    sample["img_metas"] = {k: v for k, v in sample["img_metas"].items() if k != "gt_class_names"}
    summary = summarise_sample(sample)
    summary["arrays"].pop("img")
    return summary


def without_image(summary: Dict[str, Any]) -> Dict[str, Any]:
    summary = {key: value for key, value in summary.items() if key != "image_transform"}
    summary["arrays"] = {key: value for key, value in summary["arrays"].items() if key != "img"}
    return summary


def legacy_two_pass(image: np.ndarray, rig: CameraRig, camera: str, transform: ImageTransform) -> np.ndarray:
    """The thesis image path: crop and resize to the rig's shape, resize again, crop, flip."""
    preprocessing = rig.preprocessing
    if preprocessing is not None:
        first, last = preprocessing.row_ranges.get(camera, (0, image.shape[0]))
        image = cv2.resize(image[first:last], preprocessing.target_size, interpolation=cv2.INTER_LINEAR)
    scale_x, scale_y = transform.resize
    resized = cv2.resize(
        image, (int(image.shape[1] * scale_x), int(image.shape[0] * scale_y)), interpolation=cv2.INTER_LINEAR
    )
    x0, y0, x1, y1 = transform.crop
    cropped = resized[max(0, y0):y1, max(0, x0):x1]
    return cv2.flip(cropped, 1) if transform.flip else cropped


def classify(problems: List[str], allowed: Dict[str, str]) -> Tuple[List[str], List[str]]:
    """Split reported differences into the ones that were expected and the ones that were not."""
    unexpected, matched = [], []
    for problem in problems:
        path = problem.split(":", 1)[0]
        for pattern in allowed:
            if re.match(pattern + r"($|[.\[])", path):
                matched.append(pattern)
                break
        else:
            unexpected.append(problem)
    return unexpected, matched


@pytest.mark.parametrize("config_name", sorted(GOLDEN_CONFIGS))
def test_matches_the_legacy_dataset(golden: Dict[str, Any], config_name: str) -> None:
    config = GOLDEN_CONFIGS[config_name]
    recorded = golden["configs"][config_name]
    allowed = expected_differences(config_name)

    unexpected: List[str] = []
    seen: set = set()

    dataset = build_dataset(config)
    for index_text, expected in recorded["samples"].items():
        actual = summarise(dataset, int(index_text), replay(config, expected))

        problems = compare_summaries(without_image(expected), actual, tolerance=NUMERIC_TOLERANCE)
        sample_unexpected, matched = classify(problems, allowed)
        unexpected.extend(f"[{config_name}#{index_text}] {problem}" for problem in sample_unexpected)
        seen.update(matched)

    assert not unexpected, (
        f"{len(unexpected)} unexpected difference(s) from the legacy dataset:\n  "
        + "\n  ".join(unexpected[:40])
    )

    stale = sorted(set(allowed) - seen)
    assert not stale, (
        f"These differences were declared expected but did not occur, so either the fixture or "
        f"the reason is stale: {stale}"
    )


@pytest.mark.parametrize("config_name", sorted(GOLDEN_CONFIGS))
def test_images_match_the_legacy_pipeline(golden: Dict[str, Any], config_name: str) -> None:
    """The same picture, resampled once instead of twice.

    The thesis resized to the rig's shape and then again by the augmentation scale; this package
    warps once through the composed matrix, and decodes at reduced size when the network input is
    at most half the sensor's resolution. So the bytes differ and the fixture's digest cannot be
    compared. What must hold instead:

    * the mean normalised value agrees with the legacy fixture, so nothing is darker, brighter or
      differently normalised;
    * each camera agrees pixel for pixel with the legacy two-pass path, re-run here on the same
      source image, to within interpolation noise;
    * and the two are aligned. The legacy pixels sat up to ~0.6 px off the legacy intrinsics —
      ``cv2.resize`` samples at pixel centres and truncates the resized size to an integer,
      neither of which the matrix knew about — so a shift below one pixel is the legacy's
      inconsistency, not this pipeline's. A crop, flip or scale error would move whole pixels.
    """
    config = GOLDEN_CONFIGS[config_name]
    dataset = build_dataset(config)

    for index_text, expected in golden["configs"][config_name]["samples"].items():
        draw = replay(config, expected)
        sample = dataset.get(int(index_text), draw)

        legacy = expected["arrays"]["img"]
        ours = normalized(sample)
        assert list(ours.shape) == legacy["shape"]
        assert abs(ours.sum() - legacy["stats"]["sum"]) / ours.size < 0.005, index_text

        scene_index, iteration = dataset.frames[int(sample["img_metas"]["sample_idx"])]
        scene = dataset.scenes[scene_index]
        for position, camera in enumerate(dataset.rig.cameras):
            reference = legacy_two_pass(
                scene.get_camera_at_iteration(iteration, camera).image, dataset.rig, camera, draw.image
            )
            image = sample["img"][position].numpy().transpose(1, 2, 0)[: reference.shape[0], : reference.shape[1]]
            where = f"{config_name}#{index_text} {camera}"

            difference = np.abs(image.astype(np.int16) - reference.astype(np.int16))
            assert difference.mean() < 5.0, f"{where}: mean |difference| {difference.mean():.2f}"

            (shift_x, shift_y), _ = cv2.phaseCorrelate(
                image.astype(np.float32).mean(axis=2), reference.astype(np.float32).mean(axis=2)
            )
            assert np.hypot(shift_x, shift_y) < 1.0, f"{where}: shifted by ({shift_x:.2f}, {shift_y:.2f}) px"


def test_frame_count_matches(golden: Dict[str, Any]) -> None:
    """A different number of frames would make every per-index comparison meaningless."""
    for config_name, recorded in golden["configs"].items():
        assert len(build_dataset(GOLDEN_CONFIGS[config_name])) == recorded["length"], config_name


def test_box_columns_are_documented() -> None:
    from py123detection.torch.data import BOX_COLUMNS

    assert BOX_COLUMNS == ("x", "y", "z", "w", "l", "h", "yaw", "vx", "vy")
    assert len(BOX_COLUMNS) == 9
