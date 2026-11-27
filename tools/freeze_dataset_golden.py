"""Freeze the legacy thesis dataset's output as the fixture the rewrite must reproduce.

Run this once, against the master-thesis implementation, *before* the dataset is rewritten::

    PYTHONPATH=../master-thesis/src \
    ../envs/modern/bin/python tools/freeze_dataset_golden.py \
        --out tests/golden/dataset_golden.json

It builds each entry of :data:`tests.golden.configs.GOLDEN_CONFIGS` as a
``DatasetNuScenesFormat``, pulls the listed sample indices, and writes the summaries produced by
:func:`tests.golden.summarise_sample`. ``tests/test_dataset_equivalence.py`` then builds the same
descriptions with the new dataset and compares.

This is a migration tool: once the legacy tree is gone the fixture stands on its own and this
script only matters for regenerating it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))

from golden import summarise_sample  # noqa: E402
from golden.configs import DATA_ROOT, GOLDEN_CONFIGS, SEED  # noqa: E402

# Legacy class tables, kept here rather than in the shared config so the fixture records what the
# thesis actually produced and the new taxonomy is free to disagree.
LEGACY_TAXONOMIES = {
    "nuscenes_detection_10cls": ("mappings.nuscenes", "DEFAULT_NUSCENES_PY123D_CLASSES_TO_MODEL_CLASSES",
                                 "DEFAULT_MODEL_CLASSES_TO_IDS"),
    "mixed2_3cls": ("mappings.av2", "MIXED2_PY123D_CLASSES_TO_MODEL_CLASSES",
                    "MIXED2_MODEL_CLASSES_TO_IDS"),
    "mixed2_3cls_wod": ("mappings.wod", "MIXED2_PY123D_CLASSES_TO_MODEL_CLASSES",
                        "MIXED2_MODEL_CLASSES_TO_IDS"),
}

CAMERA_PREPROCESSING = {
    "av2": ("av2_preprocessing", "av2_camera_preprocessing"),
    "wod": ("wod_preprocessing", "wod_camera_preprocessing"),
    "nuscenes": ("nusc_preprocessing", "nusc_camera_preprocessing"),
}


def _import(module_path: str, name: str) -> Any:
    from importlib import import_module

    return getattr(import_module(f"thesis.py123d_plugin.{module_path}"), name)


def build_legacy_dataset(config: Dict[str, Any]):
    """A ``DatasetNuScenesFormat`` matching one neutral configuration."""
    from thesis.py123d_plugin.torch_integration.dataset_nuscenes_format import DatasetNuScenesFormat

    from py123detection.core.sources import Source

    scenes = Source(data_root=DATA_ROOT, **config["source"]).load_scenes()

    module, class_map_name, ids_name = LEGACY_TAXONOMIES[config.get("legacy_taxonomy", config["taxonomy"])]
    class_map = _import(module, class_map_name)
    class_ids = _import("mappings.base", ids_name)

    image = config["image"]
    augment = config.get("augment")
    filters = config.get("filters", {})
    visibility = filters.get("visibility") or {}

    camera_preprocessing = None
    if config.get("camera_preprocessing"):
        module_name, function_name = CAMERA_PREPROCESSING[config["camera_preprocessing"]]
        camera_preprocessing = getattr(
            __import__(f"thesis.py123d_plugin.torch_integration.{module_name}", fromlist=[function_name]),
            function_name,
        )

    merge_classes = None
    if config.get("merge"):
        merge_classes = {
            target: (rule["sources"], rule.get("discard_unmerged_base", False))
            for target, rule in config["merge"].items()
        }

    return DatasetNuScenesFormat(
        scenes=scenes,
        camera_order=tuple(config["cameras"]),
        sample_frequency=1,
        image_size=tuple(image["size"]),
        py123d_classes_to_model_classes=class_map,
        model_classes_to_ids=class_ids,
        merge_classes=merge_classes,
        compute_velocity=config.get("compute_velocity", False),
        use_image_processing=image["mode"] == "ida",
        use_ida_aug=image["mode"] == "ida",
        ida_final_dim=tuple(image["size"]),
        ida_resize_lim=tuple(image.get("resize_lim", (0.8, 1.0))),
        ida_bot_pct_lim=tuple(image.get("bot_pct_lim", (0.0, 0.0))),
        use_augmentations=augment is not None,
        rand_flip=bool(augment and augment.get("flip", True)),
        rot_range=tuple(augment["rot_range"]) if augment else (0.0, 0.0),
        scale_range=tuple(augment["scale_range"]) if augment else (1.0, 1.0),
        camera_preprocessing=camera_preprocessing,
        skip_empty_bboxes=filters.get("skip_empty_boxes", True),
        skip_empty_gt_samples=filters.get("skip_empty_frames", True),
        pc_range=filters.get("pc_range"),
        filter_invisible_boxes=bool(visibility),
        visibility_filter_mode=visibility.get("mode", "center"),
        fov_margin_deg=visibility.get("fov_margin_deg", 0.0),
        cache_path=None,
        frame_cache_verbose=False,
    )


def add_class_names(sample: Dict[str, Any], class_ids: Dict[str, int]) -> None:
    """Record each box's class *name* alongside its id.

    Ids are positional and the taxonomy reorders them, so names are what the equivalence test can
    actually compare.
    """
    labels = sample.get("gt_labels_3d")
    if labels is None:
        return
    id_to_name = {index: name for name, index in class_ids.items()}
    sample["gt_class_names"] = [id_to_name.get(int(label), f"<unmapped:{int(label)}>") for label in labels]


def record_image_transform(dataset) -> Dict[str, Any]:
    """Capture the resize/crop/flip the legacy dataset samples, as it samples it.

    The new dataset draws its randomness differently, so replaying the legacy draw order is not
    possible and not the point. Recording the transform lets the equivalence test apply the *same*
    transform through the new pipeline, which is what actually needs to agree.
    """
    captured: Dict[str, Any] = {}
    original = dataset._sample_ida_augmentation

    def capturing(ori_h: int, ori_w: int):
        params = original(ori_h, ori_w)
        captured.setdefault("first", dict(params))
        return params

    dataset._sample_ida_augmentation = capturing
    return captured


def freeze(config_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    dataset = build_legacy_dataset(config)
    legacy_name = config.get("legacy_taxonomy", config["taxonomy"])
    class_ids = _import("mappings.base", LEGACY_TAXONOMIES[legacy_name][2])

    samples: Dict[str, Any] = {}
    for index in config["indices"]:
        if index >= len(dataset):
            raise IndexError(f"{config_name}: index {index} beyond dataset of length {len(dataset)}")
        captured = record_image_transform(dataset)
        np.random.seed(SEED + index)
        sample = dict(dataset[index])
        add_class_names(sample, class_ids)
        summary = summarise_sample(sample)

        params = captured.get("first", {})
        summary["image_transform"] = {
            "resize": params.get("resize"),
            "resize_dims": list(params["resize_dims"]) if "resize_dims" in params else None,
            "crop": list(params["crop"]) if "crop" in params else None,
            "flip": bool(params.get("flip", False)),
        }
        samples[str(index)] = summary

    return {"doc": config["doc"], "length": len(dataset), "samples": samples}


def git_revision(path: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=REPO / "tests" / "golden" / "dataset_golden.json")
    parser.add_argument("--configs", nargs="*", default=None, help="Subset of configuration names to freeze.")
    args = parser.parse_args()

    import py123d

    names = args.configs or list(GOLDEN_CONFIGS)
    fixture: Dict[str, Any] = {
        "note": "Generated from the master-thesis DatasetNuScenesFormat. Do not edit by hand; "
                "regenerate with tools/freeze_dataset_golden.py.",
        "py123d_version": py123d.__version__,
        "thesis_revision": git_revision(REPO.parent / "master-thesis"),
        "seed": SEED,
        "configs": {},
    }

    for name in names:
        print(f"freezing {name} ...", flush=True)
        fixture["configs"][name] = freeze(name, GOLDEN_CONFIGS[name])
        counts = [len(s.get("arrays", {}).get("gt_bboxes_3d", {}).get("shape", [0])) and
                  s["arrays"]["gt_bboxes_3d"]["shape"][0] for s in fixture["configs"][name]["samples"].values()]
        print(f"   {len(fixture['configs'][name]['samples'])} samples, "
              f"length {fixture['configs'][name]['length']}, boxes per sample {counts}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(fixture, indent=1, sort_keys=True) + "\n")
    print(f"\nwrote {args.out} ({args.out.stat().st_size / 1024:.0f} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
