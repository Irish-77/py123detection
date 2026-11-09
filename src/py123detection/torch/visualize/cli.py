"""``py123det-viz``: turn a predictions file into pictures.

    py123det-viz bev work_dirs/petr_nuscenes_10cls/eval/nuscenes/predictions.jsonl -o frames/
    py123det-viz cameras .../predictions.jsonl -o frames/ --experiment nuscenes --taxonomy nuscenes_10cls

``bev`` needs only the file: after an evaluation you have one number per dataset, and this is how
you find out why it is that number. ``cameras`` additionally needs the 123D logs the run was scored
on, because the file holds boxes and not pixels.

Argparse rather than Hydra, unlike ``py123det-train`` and ``py123det-eval``: the common case needs
no configuration at all, and a parser gives the documentation site a generated reference page,
which the two Hydra entry points cannot have.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, List, Optional, Sequence

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """The ``py123det-viz`` command line."""
    parser = argparse.ArgumentParser(
        prog="py123det-viz",
        description="Render the predictions a py123det-eval run wrote.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    def common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("predictions", type=Path, help="a predictions.jsonl from py123det-eval")
        sub.add_argument("-o", "--out", type=Path, required=True, help="directory for the PNGs")
        sub.add_argument(
            "--score-threshold",
            type=float,
            default=0.3,
            help="leave out predictions below this score (default: %(default)s). A detector emits "
            "one box per query, so an unfiltered picture shows nothing.",
        )
        sub.add_argument("--limit", type=int, default=20, help="render at most this many frames")
        sub.add_argument("--dataset", default=None, help="only frames of this dataset")
        sub.add_argument(
            "--taxonomy",
            default=None,
            help="name a taxonomy to label classes by name instead of index",
        )

    bev = subcommands.add_parser("bev", help="bird's-eye view; needs nothing but the file")
    common(bev)
    bev.add_argument("--radius", type=float, default=60.0, help="half-width of the view in metres")

    cameras = subcommands.add_parser(
        "cameras", help="boxes drawn on the camera images; needs the 123D logs"
    )
    common(cameras)
    cameras.add_argument(
        "--experiment", required=True, help="the experiment config naming the datasets and splits"
    )
    cameras.add_argument("--images", default="petr_704", help="the images config group to use")
    cameras.add_argument("--columns", type=int, default=3, help="cameras per row in the tiled image")
    return parser


def _class_names(name: Optional[str]) -> Optional[List[str]]:
    """Class names for a taxonomy, named either way round.

    ``conf/taxonomy/nuscenes_10cls.yaml`` declares ``name: nuscenes_detection_10cls``, so the
    spelling you pass to ``py123det-train`` as ``taxonomy=`` is not always the one the registry
    knows. Accepting both means nobody has to discover that.
    """
    if name is None:
        return None
    from py123detection.core.taxonomy import get_taxonomy

    try:
        return list(get_taxonomy(name).class_names)
    except (KeyError, ValueError):
        pass

    group = Path(__file__).resolve().parents[1] / "conf" / "taxonomy" / f"{name}.yaml"
    if group.exists():
        from omegaconf import OmegaConf

        declared = OmegaConf.load(group).get("name")
        if declared:
            logger.info("taxonomy %r is the config group for %r", name, declared)
            return list(get_taxonomy(declared).class_names)
    raise SystemExit(f"unknown taxonomy {name!r}; it is neither a registry name nor a config group")


def _frames(path: Path, dataset: Optional[str], limit: int):
    from py123detection.torch.predictions import frames_for

    for index, frame in enumerate(frames_for(path, dataset)):
        if index >= limit:
            return
        yield frame


def _run_bev(args: argparse.Namespace) -> int:
    from py123detection.torch.visualize.bev import save_bev

    args.out.mkdir(parents=True, exist_ok=True)
    names = _class_names(args.taxonomy)
    written = 0
    for frame in _frames(args.predictions, args.dataset, args.limit):
        target = args.out / f"{frame.dataset}_{frame.token.replace(':', '_')}_bev.png"
        save_bev(
            frame,
            target,
            class_names=names,
            score_threshold=args.score_threshold,
            radius=args.radius,
        )
        written += 1
    logger.info("wrote %d bird's-eye view(s) to %s", written, args.out)
    if written == 0:
        logger.warning("no frames matched; is --dataset right for this file?")
    return 0


def _dataset_index(args: argparse.Namespace) -> Any:
    """The evaluated datasets, keyed by frame token, so a prediction can find its images.

    Composed through Hydra the way ``py123det-eval`` composes it, so the frames line up with the
    ones that were scored rather than a differently filtered set.
    """
    from hydra import compose, initialize_config_module

    from py123detection.torch.build import build_datasets

    overrides = [f"experiment={args.experiment}", f"images={args.images}"]
    if args.taxonomy:
        overrides.append(f"taxonomy={args.taxonomy}")
    with initialize_config_module(config_module="py123detection.torch.conf", version_base=None):
        cfg = compose(config_name="config", overrides=overrides)
    return build_datasets(cfg, "val")


def _run_cameras(args: argparse.Namespace) -> int:
    import numpy as np

    from py123detection.torch.evaluate import frame_token
    from py123detection.torch.visualize.boxes import to_ego
    from py123detection.torch.visualize.cameras import render_camera, save_cameras

    args.out.mkdir(parents=True, exist_ok=True)
    datasets = _dataset_index(args)
    logger.info("built %d validation dataset(s)", len(datasets))

    # One pass over each dataset, building token -> index, because a Py123DDetectionDataset is
    # addressed by position and a predictions file by token.
    locations = {}
    for dataset in datasets:
        for index in range(len(dataset)):
            sample = dataset[index]
            locations[frame_token(sample["img_metas"])] = (dataset, index)

    written = 0
    for frame in _frames(args.predictions, args.dataset, args.limit):
        found = locations.get(frame.token)
        if found is None:
            logger.warning("%s is not in the dataset built here; skipping", frame.token)
            continue
        dataset, index = found
        sample = dataset[index]

        if frame.ego_pose is None:
            logger.warning("%s has no ego pose, so its boxes cannot be put back in the ego frame", frame.token)
            continue
        predictions = to_ego(frame.boxes, frame.ego_pose)
        truth = to_ego(frame.gt_boxes, frame.ego_pose)

        images = np.asarray(sample["img"])
        rendered = [
            render_camera(
                np.ascontiguousarray(images[camera].transpose(1, 2, 0)[..., ::-1]),
                np.asarray(sample["imu2img"])[camera],
                predictions,
                ground_truth=truth,
                scores=frame.scores,
                score_threshold=args.score_threshold,
            )
            for camera in range(images.shape[0])
        ]
        target = args.out / f"{frame.dataset}_{frame.token.replace(':', '_')}_cameras.png"
        save_cameras(target, rendered, columns=args.columns)
        written += 1

    logger.info("wrote %d camera view(s) to %s", written, args.out)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    if not args.predictions.exists():
        logger.error("no predictions file at %s", args.predictions)
        return 1
    return _run_bev(args) if args.command == "bev" else _run_cameras(args)


if __name__ == "__main__":
    sys.exit(main())
