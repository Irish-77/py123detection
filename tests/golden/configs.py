"""The dataset configurations the golden fixture pins.

Each entry describes what to build in implementation-neutral terms. The legacy adapter in
``tools/freeze_dataset_golden.py`` turns one into a ``DatasetNuScenesFormat``; the adapter in
``tests/test_dataset_equivalence.py`` turns the same one into the current dataset. Comparing the
two is only meaningful because both sides read this one description.

Between them the five configurations exercise every part of the sample pipeline that can change:
both image paths, augmentation on and off, native and derived velocity, class merging, and each
of the three ways a box gets dropped.
"""

from __future__ import annotations

from typing import Any, Dict

DATA_ROOT = "/mnt/lustre/work/geiger/gwb883/py123d_workspace/data"

NUSCENES_CAMERAS = ("PCAM_F0", "PCAM_L0", "PCAM_R0", "PCAM_L1", "PCAM_R1", "PCAM_B0")
AV2_CAMERAS = ("PCAM_F0", "PCAM_L0", "PCAM_R0", "PCAM_L1", "PCAM_R1", "PCAM_L2", "PCAM_R2")
WOD_CAMERAS = ("PCAM_F0", "PCAM_L0", "PCAM_R0", "PCAM_L1", "PCAM_R1")

#: Seed applied before each sample so the augmented configurations are reproducible.
SEED = 20260919


GOLDEN_CONFIGS: Dict[str, Dict[str, Any]] = {
    "nuscenes_resize": {
        "doc": "Plain resize path: no IDA, no augmentation. The simplest sample there is.",
        "source": {"splits": ["nuscenes_val"], "sample_rate_hz": 2.0, "max_logs": 2},
        "cameras": NUSCENES_CAMERAS,
        "taxonomy": "nuscenes_detection_10cls",
        "image": {"mode": "resize", "size": [320, 800]},
        "augment": None,
        "filters": {"skip_empty_boxes": True, "skip_empty_frames": True},
        "indices": [0, 1, 17, 39, 55],
    },
    "nuscenes_resize_aug": {
        "doc": "The plain-resize path with augmentation on. The only configuration where a "
               "horizontal flip goes through the non-IDA branch, which is where the two branches "
               "used to disagree by one pixel.",
        "source": {"splits": ["nuscenes_val"], "sample_rate_hz": 2.0, "max_logs": 2},
        "cameras": NUSCENES_CAMERAS,
        "taxonomy": "nuscenes_detection_10cls",
        "image": {"mode": "resize", "size": [320, 800]},
        "augment": {"flip": True, "rot_range": [-0.3925, 0.3925], "scale_range": [0.95, 1.05]},
        "filters": {"skip_empty_boxes": True, "skip_empty_frames": True},
        "indices": [0, 1, 17, 39, 55],
    },
    "nuscenes_ida": {
        "doc": "IDA at the PETR reference size with augmentation off, i.e. the deterministic "
               "centre-crop validation path.",
        "source": {"splits": ["nuscenes_val"], "sample_rate_hz": 2.0, "max_logs": 2},
        "cameras": NUSCENES_CAMERAS,
        "taxonomy": "nuscenes_detection_10cls",
        "image": {"mode": "ida", "size": [704, 1280], "resize_lim": [0.8, 1.0], "bot_pct_lim": [0.0, 0.0]},
        "augment": None,
        "filters": {"skip_empty_boxes": True, "skip_empty_frames": True},
        "indices": [0, 1, 17, 39, 55],
    },
    "nuscenes_ida_aug": {
        "doc": "The training path: random resize/crop/flip on the images and random rotation and "
               "scale in BEV, seeded per sample.",
        "source": {"splits": ["nuscenes_val"], "sample_rate_hz": 2.0, "max_logs": 2},
        "cameras": NUSCENES_CAMERAS,
        "taxonomy": "nuscenes_detection_10cls",
        "image": {"mode": "ida", "size": [704, 1280], "resize_lim": [0.8, 1.0], "bot_pct_lim": [0.0, 0.0]},
        "augment": {"flip": True, "rot_range": [-0.3925, 0.3925], "scale_range": [0.95, 1.05]},
        "filters": {"skip_empty_boxes": True, "skip_empty_frames": True},
        "indices": [0, 1, 17, 39, 55],
    },
    "nuscenes_range_filtered": {
        "doc": "Range and camera-visibility filtering, the two ways a box is dropped for where it "
               "is rather than what it is.",
        "source": {"splits": ["nuscenes_val"], "sample_rate_hz": 2.0, "max_logs": 2},
        "cameras": NUSCENES_CAMERAS,
        "taxonomy": "nuscenes_detection_10cls",
        "image": {"mode": "ida", "size": [704, 1280], "resize_lim": [0.8, 1.0], "bot_pct_lim": [0.0, 0.0]},
        "augment": None,
        "filters": {
            "skip_empty_boxes": True,
            "skip_empty_frames": True,
            "pc_range": [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            "visibility": {"mode": "center", "fov_margin_deg": 2.0},
        },
        "indices": [0, 1, 17, 39, 55],
    },
    "av2_merged_velocity": {
        "doc": "AV2: seven cameras, per-camera crop preprocessing, velocity derived from tracks "
               "because the logs carry none, and riders merged onto their two-wheeler.",
        "source": {"splits": ["av2-sensor_val"], "sample_rate_hz": 2.0, "max_logs": 2},
        "cameras": AV2_CAMERAS,
        "taxonomy": "mixed2_3cls",
        "camera_preprocessing": "av2",
        "image": {"mode": "ida", "size": [704, 1280], "resize_lim": [0.8, 1.0], "bot_pct_lim": [0.0, 0.0]},
        "augment": None,
        "compute_velocity": True,
        "merge": {"vehicle": {"sources": ["motorcycle", "motorcyclist"], "discard_unmerged_base": False},
                  "bicycle": {"sources": ["bicycle", "bicyclist"], "discard_unmerged_base": True}},
        "filters": {"skip_empty_boxes": True, "skip_empty_frames": True},
        "indices": [0, 1, 11, 23, 31],
    },
    "wod_visibility_filtered": {
        "doc": "Waymo's five cameras face forward, so unlike a 360 nuScenes rig the "
               "camera-visibility filter actually removes boxes here. Also the only "
               "configuration where most ground truth maps to void.",
        "source": {"splits": ["wod-perception_val"], "sample_rate_hz": 2.0, "max_logs": 2},
        "cameras": WOD_CAMERAS,
        "taxonomy": "mixed2_3cls",
        # The thesis needed a separate Waymo table for the same three classes; this package
        # reaches them through default_map, so only the freezer still names the old one.
        "legacy_taxonomy": "mixed2_3cls_wod",
        "camera_preprocessing": "wod",
        "image": {"mode": "ida", "size": [704, 1280], "resize_lim": [0.8, 1.0], "bot_pct_lim": [0.0, 0.0]},
        "augment": None,
        "compute_velocity": True,
        "filters": {
            "skip_empty_boxes": True,
            "skip_empty_frames": True,
            "visibility": {"mode": "corners", "fov_margin_deg": 0.0},
        },
        "indices": [0, 1, 17, 30, 39],
    },
}


#: How far two numbers may differ and still count as equal.
#:
#: The legacy dataset cast the world-to-ego transform to float32 before applying it to global
#: coordinates, which in a UTM-like frame is of order 1e7 and so carries about a millimetre of
#: quantisation into every box centre. It also took yaw from py123d's Euler angle where this
#: package takes the heading of the box's x axis, which differ by up to 2 mrad on a dataset whose
#: boxes have roll and pitch. Both are below annotation precision; anything larger is a real
#: change and still fails.
NUMERIC_TOLERANCE = 3e-3


#: Differences that only apply to the ten-class nuScenes taxonomy, whose order changed.
_CLASS_ORDER_CHANGED = {
    r"arrays\.gt_labels_3d": (
        "Class ids are positional and the taxonomy now uses mmdet3d's class order, not the "
        "devkit's. The class names are compared instead, and the permutation between the two "
        "orders is pinned by test_class_permutation_swaps_exactly_bus_trailer_and_pedestrian_bicycle."
    ),
    r"img_metas\.class_names": "Same reason: the class list is reordered, not changed.",
}

#: Differences the refactor is *meant* to introduce, per configuration.
#:
#: A key is a regular expression matched against the dotted path ``compare_summaries`` reports.
#: The equivalence test tolerates matching differences and fails when one of them does *not*
#: appear, so the list cannot quietly rot once a bug is fixed a second time.
EXPECTED_DIFFERENCES: Dict[str, Dict[str, str]] = {
    "*": {
        r"img_metas\.box_mode_3d": (
            "Renamed from 'IMU' to 'ego'. Neither is an mmdet3d Box3DMode value, and 'ego' is "
            "what 123D calls the frame these boxes are in."
        ),
        r"img_metas\.dataset_name": (
            "The legacy dataset hard-coded 'nuscenes' and relied on the data module to overwrite "
            "it afterwards, so a dataset used on its own mislabelled every sample — Argoverse "
            "frames were also called 'nuscenes'. It now comes from the source."
        ),
    },
    "nuscenes_ida_aug": {
        r"arrays\.gt_bboxes_3d\.(values\[\d+\]\[6\]|stats)": (
            "Rotation augmentation used to turn box centres by R(+a) and yaw by -a, leaving every "
            "augmented box facing the wrong way by twice the angle. Yaw now turns with the "
            "centres, so column 6 of an augmented frame differs."
        ),
        r"arrays\.intrinsics": (
            "The IDA flip mirrored the intrinsics as x -> W - x but the pixels with cv2.flip, which "
            "is x -> W - 1 - x, so a flipped camera projected one pixel off its own image. The "
            "image is now warped through the intrinsics' own matrix, which mirrors as W - 1 - x "
            "like the plain-resize path always did, so a flipped camera's cx moves by one pixel."
        ),
        r"arrays\.imu2img": "Follows the flip convention fix above.",
    },
    "nuscenes_resize_aug": {
        r"arrays\.gt_bboxes_3d\.(values\[\d+\]\[6\]|stats)": (
            "The rotation-augmentation yaw fix, as for nuscenes_ida_aug."
        ),
    },
    "nuscenes_range_filtered": {
        r"arrays\.gt_bboxes_3d": (
            "pc_range's z bounds were accepted and never applied. They are applied now, so boxes "
            "outside [z_min, z_max] are dropped and the remaining rows shift up."
        ),
        r"arrays\.gt_class_names": "Follows the z-range fix above.",
    },
    "av2_merged_velocity": {
        r"arrays\.gt_bboxes_3d\.(values\[\d+\]\[[78]\]|stats)": (
            "compute_velocity was effectively a no-op: the legacy dataset only derived a velocity "
            "for boxes that went through a merge rule, so every other Argoverse box kept the zero "
            "the logs store. Velocities are now derived for all of them, from the log's native "
            "frame rate."
        ),
    },
    "wod_visibility_filtered": {
        r"arrays\.gt_bboxes_3d\.(values\[\d+\]\[[78]\]|stats)": (
            "The same compute_velocity no-op as for Argoverse; Waymo perception logs also store "
            "no velocity."
        ),
    },
}


def expected_differences(config_name: str) -> Dict[str, str]:
    """Allowed-difference patterns for one configuration, including the shared ones."""
    merged = dict(EXPECTED_DIFFERENCES.get("*", {}))
    if GOLDEN_CONFIGS[config_name]["taxonomy"] == "nuscenes_detection_10cls":
        merged.update(_CLASS_ORDER_CHANGED)
    merged.update(EXPECTED_DIFFERENCES.get(config_name, {}))
    return merged
