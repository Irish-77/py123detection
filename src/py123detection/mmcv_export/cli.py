"""``py123det-export-mmcv``: export 123D splits to an mmdetection3d pickle.

``export`` writes the pickle, ``inspect`` prints its metadata and first frame (run it in the
training environment to check the file survived the Python/numpy hop), and ``token-map`` dumps
nuScenes sample tokens so exports can restore them without the devkit installed.
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from py123detection.annotations import BOX_LAYOUTS
from py123detection.mmcv_export.converter import (
    DEFAULT_CAMERA_ORDER,
    EGO_POSE_SOURCES,
    VELOCITY_SOURCES,
    ExportConfig,
)
from py123detection.mmcv_export.export import export_to_mmdet3d
from py123detection.mmcv_export.schema import validate_info
from py123detection.mmcv_export.sensors import PathStyle, SensorMode, SensorResolverConfig
from py123detection.mmcv_export.tokens import (
    LogTimestampTokenResolver,
    MappingTokenResolver,
    NuScenesTokenResolver,
    TokenResolver,
    dump_nuscenes_token_map,
)
from py123detection.sources import Source
from py123detection.taxonomy import TAXONOMIES, get_taxonomy


def _add_export_arguments(parser: argparse.ArgumentParser) -> None:
    data = parser.add_argument_group("input")
    data.add_argument("--data-root", type=Path, default=None, help="123D data root (default: $PY123D_DATA_ROOT).")
    data.add_argument(
        "--split",
        dest="splits",
        action="append",
        default=None,
        metavar="NAME",
        help="Split to export, e.g. nuscenes-mini_train. Repeatable. Default: every split under the root.",
    )
    data.add_argument("--dataset", dest="datasets", action="append", default=None, help="Restrict to a dataset name.")
    data.add_argument("--log-name", dest="log_names", action="append", default=None, help="Restrict to a log name.")
    data.add_argument("--sample-rate-hz", type=float, default=None, help="Target frame rate, e.g. 2.0.")
    data.add_argument("--frame-stride", type=int, default=None, help="Keep every N-th raw frame.")
    data.add_argument("--max-logs", type=int, default=None, help="Export at most this many logs.")
    data.add_argument(
        "--require-modality",
        dest="required_modalities",
        action="append",
        default=None,
        help="123D modality requirement, e.g. 'camera:all'. Repeatable.",
    )
    data.add_argument(
        "--merge",
        action="append",
        default=None,
        metavar="ROOT:SPLIT[,SPLIT...]",
        help="Additional source to merge into the same pickle, for cross-dataset training. Repeatable.",
    )
    data.add_argument(
        "--disjoint-timestamps",
        action="store_true",
        help="Offset merged sources in time so mmdet3d's timestamp sort cannot interleave them.",
    )

    output = parser.add_argument_group("output")
    output.add_argument("--out", type=Path, required=True, help="Destination .pkl.")
    output.add_argument("--version", default="py123d-v1.0", help="Value stored as metadata['version'].")
    output.add_argument(
        "--array-format",
        choices=("numpy", "portable"),
        default="numpy",
        help="'portable' stores arrays as nested lists, sidestepping numpy's pickle format "
        "(needs py123detection.mmcv_plugin on the training side).",
    )
    output.add_argument("--pickle-protocol", type=int, default=4, help="Pickle protocol (4 reads on Python >= 3.4).")

    content = parser.add_argument_group("content")
    content.add_argument(
        "--taxonomy",
        default="nuscenes_detection_10cls",
        choices=sorted(TAXONOMIES),
        help="Class taxonomy. Cross-dataset exports want a default-label taxonomy like general_3cls.",
    )
    content.add_argument(
        "--camera-order",
        default=",".join(DEFAULT_CAMERA_ORDER),
        help="Comma-separated 123D CameraID names in model input order.",
    )
    content.add_argument(
        "--camera-key",
        choices=("native", "camera_id"),
        default="native",
        help="Key used inside info['cams']. Use camera_id for merged multi-dataset exports.",
    )
    content.add_argument("--reference-lidar", default=None, help="LidarID name defining the reference frame.")
    content.add_argument(
        "--lidar2ego-mode",
        choices=("static", "dynamic"),
        default="static",
        help="'static' uses the rig extrinsic 123D stores (what mmdet3d's converter writes); "
        "'dynamic' re-derives it from the ego pose at the lidar's own timestamp.",
    )
    content.add_argument(
        "--camera2ego-mode",
        choices=("static", "dynamic"),
        default="static",
        help="Where a camera's timing offset is booked: 'static' reports it in ego2global "
        "(mmdet3d's convention), 'dynamic' folds it into sensor2ego. sensor2lidar is unaffected.",
    )
    content.add_argument(
        "--lidar-frame",
        choices=("sensor", "ego"),
        default="sensor",
        help="Frame for boxes and extrinsics: the reference lidar's sensor frame (mmdet3d's nuScenes "
        "convention) or the ego/IMU frame.",
    )
    content.add_argument("--max-sweeps", type=int, default=10, help="Preceding lidar frames per info.")
    content.add_argument(
        "--no-2d-annotations",
        action="store_true",
        help="Skip per-camera 2D/mono-3D annotations. StreamPETR training needs them.",
    )
    content.add_argument(
        "--yaw-convention",
        choices=("mmdet3d", "heading"),
        default="mmdet3d",
        help="Yaw read-out. 'mmdet3d' matches pickles from mmdetection3d's own converter; "
        "'heading' matches the nuScenes evaluation devkit.",
    )
    content.add_argument(
        "--full-3d-velocity",
        action="store_true",
        help="Keep the vertical velocity component when rotating boxes into the lidar frame. "
        "mmdetection3d drops it, so this diverges from stock pickles.",
    )
    content.add_argument(
        "--on-missing-camera",
        choices=("skip_frame", "drop_camera", "error"),
        default="skip_frame",
        help="Behaviour when a configured camera has no data at a frame.",
    )
    content.add_argument(
        "--box-layout",
        choices=BOX_LAYOUTS,
        default="streampetr",
        help="gt_boxes column layout: 'streampetr' = [x,y,z,l,w,h,yaw] (StreamPETR, mmdet3d>=1.0); "
        "'mmdet3d_0.17' = [x,y,z,w,l,h,-yaw-pi/2] (mmdetection3d 0.17's own converter, PETR v1).",
    )
    content.add_argument(
        "--velocity-source",
        choices=VELOCITY_SOURCES,
        default="stored",
        help="'stored' writes the velocity the 123D log carries; 'tracks' derives it as a central "
        "difference over each track at the log's native frame rate (for datasets without annotated "
        "velocity, e.g. Argoverse 2).",
    )
    content.add_argument(
        "--velocity-max-dt",
        type=float,
        default=0.25,
        metavar="SECONDS",
        help="Neighbour tolerance for --velocity-source tracks (default 0.25 s: one dropped 10 Hz frame).",
    )
    content.add_argument(
        "--ego-pose-source",
        choices=EGO_POSE_SOURCES,
        default="sync",
        help="'sync' takes the ego state the log's sync table associated with the frame; 'nearest' "
        "looks it up by the frame timestamp instead, which is robust to sync tables built from "
        "slightly-off ego timestamps (py123d 0.6.0 AV2 logs).",
    )

    sensors = parser.add_argument_group("sensor payloads")
    sensors.add_argument(
        "--camera-mode",
        choices=[mode.value for mode in SensorMode],
        default=SensorMode.AUTO.value,
        help="How camera images become paths: reference the original dataset, or extract embedded payloads.",
    )
    sensors.add_argument(
        "--lidar-mode",
        choices=[mode.value for mode in SensorMode],
        default=SensorMode.AUTO.value,
        help="Same, for point clouds.",
    )
    sensors.add_argument(
        "--path-style",
        choices=[style.value for style in PathStyle],
        default=PathStyle.ABSOLUTE.value,
        help="Absolute paths (needed for merged exports) or relative to --path-relative-to.",
    )
    sensors.add_argument("--path-relative-to", type=Path, default=None, help="Root for relative paths.")
    sensors.add_argument("--extract-root", type=Path, default=None, help="Directory for extracted payloads.")
    sensors.add_argument(
        "--sensor-root",
        dest="sensor_roots",
        action="append",
        default=None,
        metavar="DATASET=PATH",
        help="Override a dataset's sensor root, e.g. nuscenes=/data/nuscenes. Repeatable.",
    )

    tokens = parser.add_argument_group("tokens")
    tokens.add_argument(
        "--token-style",
        choices=("uuid", "log_timestamp"),
        default="uuid",
        help="Token when no native token is restored: the deterministic 123D UUID, or "
        "'{log_name}/{timestamp_us}' (reproducible by any converter; use it to diff against a "
        "reference pickle for datasets without native frame tokens).",
    )
    tokens.add_argument(
        "--nuscenes-root",
        type=Path,
        default=None,
        help="nuScenes data root. Restores native sample_tokens, which official nuScenes evaluation requires.",
    )
    tokens.add_argument("--token-map", type=Path, default=None, help="Token map JSON from the token-map subcommand.")


def _parse_sensor_roots(values: Optional[Sequence[str]]) -> dict:
    roots = {}
    for value in values or []:
        if "=" not in value:
            raise SystemExit(f"--sensor-root expects DATASET=PATH, got {value!r}.")
        dataset, path = value.split("=", 1)
        roots[dataset] = Path(path)
    return roots


def _parse_merge_sources(values: Optional[Sequence[str]], primary: Source) -> List[Source]:
    """``ROOT:SPLIT[,SPLIT...]`` specs. Merged sources inherit sampling from the primary source."""
    sources = []
    for value in values or []:
        if ":" not in value:
            raise SystemExit(f"--merge expects ROOT:SPLIT[,SPLIT...], got {value!r}.")
        root, splits = value.rsplit(":", 1)
        sources.append(
            Source(
                data_root=Path(root),
                splits=[split for split in splits.split(",") if split],
                sample_rate_hz=primary.sample_rate_hz,
                frame_stride=primary.frame_stride,
                required_modalities=primary.required_modalities,
                max_logs=primary.max_logs,
            )
        )
    return sources


def _build_token_resolver(args: argparse.Namespace) -> TokenResolver:
    if args.token_map is not None:
        return MappingTokenResolver.from_json(args.token_map)
    if args.nuscenes_root is not None:
        return NuScenesTokenResolver(args.nuscenes_root)
    if args.token_style == "log_timestamp":
        return LogTimestampTokenResolver()
    return TokenResolver()


def _run_export(args: argparse.Namespace) -> int:
    primary = Source(
        data_root=args.data_root,
        splits=args.splits,
        datasets=args.datasets,
        log_names=args.log_names,
        sample_rate_hz=args.sample_rate_hz,
        frame_stride=args.frame_stride,
        required_modalities=args.required_modalities,
        max_logs=args.max_logs,
    )
    sources = [primary] + _parse_merge_sources(args.merge, primary)

    config = ExportConfig(
        taxonomy=get_taxonomy(args.taxonomy),
        camera_order=tuple(name.strip() for name in args.camera_order.split(",") if name.strip()),
        camera_key=args.camera_key,
        reference_lidar=args.reference_lidar,
        lidar_frame=args.lidar_frame,
        lidar2ego_mode=args.lidar2ego_mode,
        camera2ego_mode=args.camera2ego_mode,
        max_sweeps=args.max_sweeps,
        with_2d_annotations=not args.no_2d_annotations,
        on_missing_camera=args.on_missing_camera,
        yaw_convention=args.yaw_convention,
        planar_velocity=not args.full_3d_velocity,
        box_layout=args.box_layout,
        velocity_source=args.velocity_source,
        velocity_max_dt_s=args.velocity_max_dt,
        ego_pose_source=args.ego_pose_source,
        sensors=SensorResolverConfig(
            camera_mode=SensorMode(args.camera_mode),
            lidar_mode=SensorMode(args.lidar_mode),
            path_style=PathStyle(args.path_style),
            relative_to=args.path_relative_to,
            extract_root=args.extract_root,
            sensor_roots=_parse_sensor_roots(args.sensor_roots),
        ),
        token_resolver=_build_token_resolver(args),
    )

    report = export_to_mmdet3d(
        sources,
        output_path=args.out,
        config=config,
        version=args.version,
        array_format=args.array_format,
        pickle_protocol=args.pickle_protocol,
        disjoint_timestamps=args.disjoint_timestamps,
    )
    print(report.summary())
    print(f"\nSet class_names in your mmdet3d config to:\n  {list(config.taxonomy.class_names)}")
    return 0


def _run_inspect(args: argparse.Namespace) -> int:
    with open(args.path, "rb") as handle:
        payload = pickle.load(handle)

    infos = payload.get("infos", [])
    print(f"{args.path}: {len(infos)} frames")
    print("metadata:")
    for key, value in payload.get("metadata", {}).items():
        print(f"  {key}: {value}")

    if infos:
        info = infos[0]
        print("\nfirst frame:")
        for key in ("token", "scene_token", "frame_idx", "timestamp", "lidar_path"):
            print(f"  {key}: {info.get(key)}")
        print(f"  cams: {list(info.get('cams', {}))}")
        print(f"  boxes: {len(info.get('gt_names', []))}")
        problems = validate_info(info)
        print(f"  schema: {'ok' if not problems else problems}")
    return 0


def _run_token_map(args: argparse.Namespace) -> int:
    count = dump_nuscenes_token_map(args.nuscenes_root, args.out)
    print(f"Wrote {count} tokens to {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="py123det-export-mmcv",
        description="Export 123D datasets to mmdetection3d info pickles.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export", help="Export 123D splits to an mmdet3d .pkl.")
    _add_export_arguments(export_parser)
    export_parser.set_defaults(handler=_run_export)

    inspect_parser = subparsers.add_parser("inspect", help="Print an exported pickle's metadata and first frame.")
    inspect_parser.add_argument("path", type=Path)
    inspect_parser.set_defaults(handler=_run_inspect)

    token_parser = subparsers.add_parser("token-map", help="Dump a nuScenes sample_token map to JSON.")
    token_parser.add_argument("--nuscenes-root", type=Path, required=True)
    token_parser.add_argument("--out", type=Path, required=True)
    token_parser.set_defaults(handler=_run_token_map)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
