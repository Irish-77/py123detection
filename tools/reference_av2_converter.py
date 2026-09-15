#!/usr/bin/env python3
"""Reference Argoverse 2 (sensor) -> mmdetection3d info converter: **arm A** of the AV2
provenance test.

Written straight from the raw AV2 feather tables and JPEG directories, independently of py123d
and py123detection, so that ``raw AV2 -> pkl`` can be diffed against ``raw AV2 -> 123D -> pkl``
with ``tools/compare_infos.py``. No mmdetection3d converter exists for AV2 (mmdet3d 0.17, PETR
and CoIn3D have none); the closest precedent is Far3D's ``create_av2_infos.py``, whose
conventions this follows where they exist (all cameras via the av2-api synchronization rule,
boxes in the egovehicle frame, ``num_interior_pts`` as the point count).

Conventions (each one is also what the 123D export is configured to produce):

* **Frames**: the lidar sweeps of a log in timestamp order, every ``--frame-stride``-th one
  starting from the first (``--frame-stride 5`` turns 10 Hz into 2 Hz).
* **Reference frame**: the egovehicle frame at the sweep timestamp. AV2 sweeps and annotations
  are expressed in it, so ``lidar2ego`` is the identity and ``gt_boxes`` are the annotation
  rows verbatim.
* **Camera sync** (``--camera-matching``): ``nearest`` is av2-api's ``SensorDataloader`` rule —
  the image closest to the sweep timestamp within 25 ms (half the 50 ms shutter interval);
  ``forward`` is py123d's rule — the first image at or after the sweep start within 102 ms
  (the sweep interval plus buffer). The 123D logs used here hold ``forward``.
* **Camera poses**: ``ego2global`` at the *image's* timestamp (AV2 stores a pose per sensor
  timestamp), ``sensor2ego`` from the rig calibration, ``sensor2lidar`` composed through the
  global frame — mmdetection3d's nuScenes convention.
* **Sweeps**: the ``--max-sweeps`` preceding 10 Hz sweeps, newest first, with the same
  sweep -> ego -> global -> ego' -> lidar chain as mmdetection3d.
* **Classes**: CoIn3D's three classes with nuScenes spelling (``car`` / ``pedestrian`` /
  ``motorcycle``); riders, wheelchairs, strollers, animals, signs, cones and bollards are
  dropped — see ``CATEGORY_TO_CLASS`` and ``py123detection.taxonomy.COIN3D_3CLS``.
* **Velocity**: AV2 annotates none. Derived per box as the central difference of the global
  box centre over the neighbouring 10 Hz annotations of the same ``track_uuid`` (one-sided at
  track ends, zero without a neighbour within ``--velocity-max-dt``), rotated into the ego frame
  and reduced to ``(vx, vy)``. The nuScenes devkit's ``box_velocity`` rule.
* **Yaw**: ``pyquaternion.Quaternion.yaw_pitch_roll[0]`` of the annotation quaternion — the
  read-out mmdetection3d's own converter uses.
* **Tokens**: ``"{log_id}/{timestamp_us}"``, matching the export's ``--token-style log_timestamp``.

Usage::

    reference_av2_converter.py --av2-root data/av2/sensor --split val \\
        --out data/av2/av2_infos_val.pkl --frame-stride 5 --camera-matching forward
"""

from __future__ import annotations

import argparse
import bisect
import multiprocessing as mp
import pickle
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from pyquaternion import Quaternion

CAMERAS: Tuple[str, ...] = (
    "ring_front_center",
    "ring_front_right",
    "ring_front_left",
    "ring_side_left",
    "ring_side_right",
    "ring_rear_left",
    "ring_rear_right",
)
"""Model input order: front first, then right/left pairs — 123D ``PCAM_F0, R0, L0, L1, R1, L2, R2``."""

CLASS_NAMES: Tuple[str, ...] = ("car", "pedestrian", "motorcycle")

# Raw AV2 category -> CoIn3D class, or None to drop. Every one of AV2's 30 categories is listed
# on purpose: an unknown category raises, so a dataset revision cannot slip through unmapped.
CATEGORY_TO_CLASS: Dict[str, Optional[str]] = {
    "REGULAR_VEHICLE": "car",
    "LARGE_VEHICLE": "car",
    "BUS": "car",
    "ARTICULATED_BUS": "car",
    "SCHOOL_BUS": "car",
    "BOX_TRUCK": "car",
    "TRUCK": "car",
    "TRUCK_CAB": "car",
    "VEHICULAR_TRAILER": "car",
    "MESSAGE_BOARD_TRAILER": "car",
    "TRAFFIC_LIGHT_TRAILER": "car",
    "RAILED_VEHICLE": "car",
    "PEDESTRIAN": "pedestrian",
    "OFFICIAL_SIGNALER": "pedestrian",
    "BICYCLE": "motorcycle",
    "MOTORCYCLE": "motorcycle",
    "BICYCLIST": None,
    "MOTORCYCLIST": None,
    "WHEELED_RIDER": None,
    "WHEELED_DEVICE": None,
    "WHEELCHAIR": None,
    "STROLLER": None,
    "DOG": None,
    "ANIMAL": None,
    "BOLLARD": None,
    "CONSTRUCTION_BARREL": None,
    "CONSTRUCTION_CONE": None,
    "SIGN": None,
    "STOP_SIGN": None,
    "MOBILE_PEDESTRIAN_CROSSING_SIGN": None,
}

# av2-api sensor_dataloader.py: CAM_FPS = 20 -> CAM_SHUTTER_INTERVAL_MS = 50; LIDAR 10 Hz ->
# LIDAR_SWEEP_INTERVAL_MS = 100 + ALLOWED_TIMESTAMP_BUFFER_MS = 2.
NEAREST_TOLERANCE_NS = 25_000_000  # CAM_SHUTTER_INTERVAL_MS / 2, the lidar->camera tolerance
FORWARD_TOLERANCE_NS = 102_000_000  # LIDAR_SWEEP_INTERVAL_W_BUFFER_NS, py123d's sweep window

BOX_LAYOUTS = ("streampetr", "mmdet3d_0.17")


# --------------------------------------------------------------------------------------------
# Small geometry helpers
# --------------------------------------------------------------------------------------------


def pose_matrix(qw: float, qx: float, qy: float, qz: float, tx: float, ty: float, tz: float) -> np.ndarray:
    """4x4 transform from an AV2 ``(q, t)`` row."""
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = Quaternion(qw, qx, qy, qz).rotation_matrix
    matrix[:3, 3] = (tx, ty, tz)
    return matrix


def invert(matrix: np.ndarray) -> np.ndarray:
    """Inverse of a rigid 4x4 transform."""
    rotation = matrix[:3, :3]
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ matrix[:3, 3]
    return inverse


def pick_image(images_ns: Sequence[int], sweep_ns: int, matching: str) -> Optional[int]:
    """Choose the image timestamp paired with a lidar sweep."""
    index = bisect.bisect_left(images_ns, sweep_ns)
    if matching == "forward":
        if index < len(images_ns) and images_ns[index] - sweep_ns <= FORWARD_TOLERANCE_NS:
            return int(images_ns[index])
        return None
    candidates = [images_ns[j] for j in (index - 1, index) if 0 <= j < len(images_ns)]
    if not candidates:
        return None
    best = min(candidates, key=lambda t: abs(t - sweep_ns))
    return int(best) if abs(best - sweep_ns) <= NEAREST_TOLERANCE_NS else None


def track_velocities(times_ns: np.ndarray, centers: np.ndarray, max_dt_s: float) -> np.ndarray:
    """Central-difference velocities along one track (the nuScenes devkit rule)."""
    count = len(times_ns)
    velocities = np.zeros((count, 3), dtype=np.float64)
    max_dt_ns = max_dt_s * 1e9
    for k in range(count):
        has_prev = k > 0 and times_ns[k] - times_ns[k - 1] <= max_dt_ns
        has_next = k + 1 < count and times_ns[k + 1] - times_ns[k] <= max_dt_ns
        if has_prev and has_next:
            lo, hi = k - 1, k + 1
        elif has_prev:
            lo, hi = k - 1, k
        elif has_next:
            lo, hi = k, k + 1
        else:
            continue
        velocities[k] = (centers[hi] - centers[lo]) / ((times_ns[hi] - times_ns[lo]) * 1e-9)
    return velocities


# --------------------------------------------------------------------------------------------
# Per-log conversion
# --------------------------------------------------------------------------------------------


def convert_log(task: Tuple[str, str, str, Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Convert one AV2 log directory into infos. Runs in a worker process."""
    log_dir_str, split, sensor_root_str, options = task
    log_dir = Path(log_dir_str)
    sensor_root = Path(sensor_root_str)
    log_id = log_dir.name
    stats: Dict[str, Any] = Counter()

    lidar_ts = sorted(int(path.stem) for path in (log_dir / "sensors" / "lidar").glob("*.feather"))
    if not lidar_ts:
        stats["logs_skipped_no_lidar"] += 1
        return [], stats

    ego_df = pd.read_feather(log_dir / "city_SE3_egovehicle.feather")
    ego_pose: Dict[int, np.ndarray] = {}
    ego_quat: Dict[int, List[float]] = {}
    ego_trans: Dict[int, List[float]] = {}
    for row in ego_df.itertuples(index=False):
        ts = int(row.timestamp_ns)
        ego_pose[ts] = pose_matrix(row.qw, row.qx, row.qy, row.qz, row.tx_m, row.ty_m, row.tz_m)
        ego_quat[ts] = [float(row.qw), float(row.qx), float(row.qy), float(row.qz)]
        ego_trans[ts] = [float(row.tx_m), float(row.ty_m), float(row.tz_m)]

    calibration = pd.read_feather(log_dir / "calibration" / "egovehicle_SE3_sensor.feather")
    intrinsics = pd.read_feather(log_dir / "calibration" / "intrinsics.feather")
    cameras: Dict[str, Dict[str, Any]] = {}
    for name in CAMERAS:
        calib_rows = calibration[calibration["sensor_name"] == name]
        intr_rows = intrinsics[intrinsics["sensor_name"] == name]
        if calib_rows.empty or intr_rows.empty:
            # py123d skips a camera without intrinsics; the export then skips the whole log.
            stats["logs_skipped_missing_camera"] += 1
            return [], stats
        calib = calib_rows.iloc[0]
        intr = intr_rows.iloc[0]
        cameras[name] = {
            "cam2ego": pose_matrix(calib.qw, calib.qx, calib.qy, calib.qz, calib.tx_m, calib.ty_m, calib.tz_m),
            "quat": [float(calib.qw), float(calib.qx), float(calib.qy), float(calib.qz)],
            "trans": [float(calib.tx_m), float(calib.ty_m), float(calib.tz_m)],
            "K": np.array(
                [[intr.fx_px, 0.0, intr.cx_px], [0.0, intr.fy_px, intr.cy_px], [0.0, 0.0, 1.0]], dtype=np.float64
            ),
            "width": int(intr.width_px),
            "height": int(intr.height_px),
            "images": sorted(int(path.stem) for path in (log_dir / "sensors" / "cameras" / name).glob("*.jpg")),
        }

    annotations_path = log_dir / "annotations.feather"
    annotations = pd.read_feather(annotations_path) if annotations_path.exists() else None
    boxes_by_ts: Dict[int, pd.DataFrame] = {}
    velocity: Dict[Tuple[str, int], np.ndarray] = {}
    if annotations is not None and len(annotations):
        for ts, group in annotations.groupby("timestamp_ns", sort=True):
            boxes_by_ts[int(ts)] = group
        track_observations: Dict[str, List[Tuple[int, np.ndarray]]] = defaultdict(list)
        for row in annotations.itertuples(index=False):
            ts = int(row.timestamp_ns)
            pose = ego_pose.get(ts)
            if pose is None:
                continue
            center_global = pose[:3, :3] @ np.array([row.tx_m, row.ty_m, row.tz_m], dtype=np.float64) + pose[:3, 3]
            track_observations[str(row.track_uuid)].append((ts, center_global))
        for track, observations in track_observations.items():
            observations.sort(key=lambda item: item[0])
            times = np.array([item[0] for item in observations], dtype=np.int64)
            centers = np.stack([item[1] for item in observations], axis=0)
            for ts, vel in zip(times.tolist(), track_velocities(times, centers, options["velocity_max_dt_s"])):
                velocity[(track, int(ts))] = vel

    scene_token = f"av2-sensor_{split}/{log_id}"
    matching = options["camera_matching"]
    layout = options["box_layout"]
    max_sweeps = options["max_sweeps"]
    infos: List[Dict[str, Any]] = []

    for sweep_index in range(0, len(lidar_ts), options["frame_stride"]):
        ts = lidar_ts[sweep_index]
        ego = ego_pose.get(ts)
        if ego is None:
            stats["frames_skipped_missing_ego"] += 1
            continue
        global_to_ego = invert(ego)

        cams: Dict[str, Dict[str, Any]] = {}
        for name in CAMERAS:
            camera = cameras[name]
            image_ts = pick_image(camera["images"], ts, matching)
            if image_ts is None or image_ts not in ego_pose:
                cams = {}
                break
            cam_to_global = ego_pose[image_ts] @ camera["cam2ego"]
            cam_to_lidar = global_to_ego @ cam_to_global
            cams[name] = {
                "data_path": str(sensor_root / split / log_id / "sensors" / "cameras" / name / f"{image_ts}.jpg"),
                "type": name,
                "sample_data_token": f"{scene_token}/{name}/{image_ts // 1000}",
                "sensor2ego_translation": list(camera["trans"]),
                "sensor2ego_rotation": list(camera["quat"]),
                "ego2global_translation": list(ego_trans[image_ts]),
                "ego2global_rotation": list(ego_quat[image_ts]),
                "timestamp": image_ts // 1000,
                "sensor2lidar_rotation": np.ascontiguousarray(cam_to_lidar[:3, :3]),
                "sensor2lidar_translation": np.ascontiguousarray(cam_to_lidar[:3, 3]),
                "cam_intrinsic": camera["K"].copy(),
                "width": camera["width"],
                "height": camera["height"],
            }
        if not cams:
            stats["frames_skipped_missing_camera"] += 1
            continue

        # --- annotations, verbatim in the ego (= lidar) frame ---------------------------------
        gt_boxes: List[List[float]] = []
        gt_names: List[str] = []
        gt_velocity: List[List[float]] = []
        num_lidar_pts: List[int] = []
        instance_tokens: List[str] = []
        group = boxes_by_ts.get(ts)
        if group is not None:
            for row in group.itertuples(index=False):
                class_name = CATEGORY_TO_CLASS[str(row.category)]  # KeyError on purpose for unknown categories
                if class_name is None:
                    stats[f"dropped:{row.category}"] += 1
                    continue
                yaw = Quaternion(row.qw, row.qx, row.qy, row.qz).yaw_pitch_roll[0]
                if layout == "streampetr":
                    box = [row.tx_m, row.ty_m, row.tz_m, row.length_m, row.width_m, row.height_m, yaw]
                else:
                    box = [row.tx_m, row.ty_m, row.tz_m, row.width_m, row.length_m, row.height_m, -yaw - np.pi / 2]
                gt_boxes.append([float(value) for value in box])
                gt_names.append(class_name)
                vel_global = velocity.get((str(row.track_uuid), ts), np.zeros(3))
                vel_planar = np.array([vel_global[0], vel_global[1], 0.0], dtype=np.float64)
                gt_velocity.append((ego[:3, :3].T @ vel_planar)[:2].tolist())
                num_lidar_pts.append(int(row.num_interior_pts))
                instance_tokens.append(str(row.track_uuid))
                stats[f"class:{class_name}"] += 1

        # --- sweeps: preceding native-rate lidar frames, newest first ------------------------
        sweeps: List[Dict[str, Any]] = []
        for sweep_ts in reversed(lidar_ts[max(0, sweep_index - max_sweeps) : sweep_index]):
            sweep_pose = ego_pose.get(sweep_ts)
            if sweep_pose is None:
                continue
            sweep_to_lidar = global_to_ego @ sweep_pose
            sweeps.append(
                {
                    "data_path": str(sensor_root / split / log_id / "sensors" / "lidar" / f"{sweep_ts}.feather"),
                    "type": "lidar",
                    "sample_data_token": f"{log_id}/{sweep_ts // 1000}",
                    "sensor2ego_translation": [0.0, 0.0, 0.0],
                    "sensor2ego_rotation": [1.0, 0.0, 0.0, 0.0],
                    "ego2global_translation": list(ego_trans[sweep_ts]),
                    "ego2global_rotation": list(ego_quat[sweep_ts]),
                    "timestamp": sweep_ts // 1000,
                    "sensor2lidar_rotation": np.ascontiguousarray(sweep_to_lidar[:3, :3]),
                    "sensor2lidar_translation": np.ascontiguousarray(sweep_to_lidar[:3, 3]),
                }
            )

        count = len(gt_boxes)
        infos.append(
            {
                "lidar_path": str(sensor_root / split / log_id / "sensors" / "lidar" / f"{ts}.feather"),
                "token": f"{log_id}/{ts // 1000}",
                "prev": "",
                "next": "",
                "sweeps": sweeps,
                "frame_idx": 0,
                "cams": cams,
                "scene_token": scene_token,
                "lidar2ego_translation": [0.0, 0.0, 0.0],
                "lidar2ego_rotation": [1.0, 0.0, 0.0, 0.0],
                "ego2global_translation": list(ego_trans[ts]),
                "ego2global_rotation": list(ego_quat[ts]),
                "timestamp": ts // 1000,
                "gt_boxes": np.asarray(gt_boxes, dtype=np.float64).reshape(count, 7),
                "gt_names": np.asarray(gt_names, dtype="<U32"),
                "gt_velocity": np.asarray(gt_velocity, dtype=np.float64).reshape(count, 2),
                "num_lidar_pts": np.asarray(num_lidar_pts, dtype=np.int64),
                "num_radar_pts": np.zeros(count, dtype=np.int64),
                "valid_flag": np.asarray(num_lidar_pts, dtype=np.int64) > 0,
                "instance_tokens": instance_tokens,
            }
        )
        stats["frames"] += 1
        stats["boxes"] += count

    for index, info in enumerate(infos):
        info["frame_idx"] = index
        info["prev"] = infos[index - 1]["token"] if index > 0 else ""
        info["next"] = infos[index + 1]["token"] if index + 1 < len(infos) else ""
    stats["logs"] += 1
    return infos, stats


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--av2-root", type=Path, required=True, help="AV2 sensor root holding train/ val/ test/.")
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frame-stride", type=int, default=5, help="Keep every N-th 10 Hz sweep (5 -> 2 Hz).")
    parser.add_argument("--camera-matching", choices=("forward", "nearest"), default="forward")
    parser.add_argument("--max-sweeps", type=int, default=10)
    parser.add_argument("--velocity-max-dt", type=float, default=0.25)
    parser.add_argument("--box-layout", choices=BOX_LAYOUTS, default="mmdet3d_0.17")
    parser.add_argument("--version", default="av2-sensor", help="Stored as metadata['version'].")
    parser.add_argument("--max-logs", type=int, default=None)
    parser.add_argument("--log-id", action="append", default=None, help="Restrict to these log ids. Repeatable.")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)

    split_dir = args.av2_root / args.split
    log_dirs = sorted(path for path in split_dir.iterdir() if path.is_dir())
    if args.log_id:
        wanted = set(args.log_id)
        log_dirs = [path for path in log_dirs if path.name in wanted]
    if args.max_logs is not None:
        log_dirs = log_dirs[: args.max_logs]
    if not log_dirs:
        print(f"No logs under {split_dir}", file=sys.stderr)
        return 1

    options = {
        "frame_stride": args.frame_stride,
        "camera_matching": args.camera_matching,
        "max_sweeps": args.max_sweeps,
        "velocity_max_dt_s": args.velocity_max_dt,
        "box_layout": args.box_layout,
    }
    tasks = [(str(path), args.split, str(args.av2_root), options) for path in log_dirs]

    start = time.time()
    infos: List[Dict[str, Any]] = []
    stats: Counter = Counter()
    if args.workers > 1:
        with mp.get_context("fork").Pool(args.workers) as pool:
            for index, (log_infos, log_stats) in enumerate(pool.imap(convert_log, tasks), start=1):
                infos.extend(log_infos)
                stats.update(log_stats)
                if index % 50 == 0 or index == len(tasks):
                    print(f"  {index}/{len(tasks)} logs, {len(infos)} frames, {time.time() - start:.0f}s", flush=True)
    else:
        for index, task in enumerate(tasks, start=1):
            log_infos, log_stats = convert_log(task)
            infos.extend(log_infos)
            stats.update(log_stats)

    metadata = {
        "version": args.version,
        "converter": "py123detection/tools/reference_av2_converter.py",
        "taxonomy": "coin3d_3cls",
        "class_names": list(CLASS_NAMES),
        "camera_order": list(CAMERAS),
        "camera_matching": args.camera_matching,
        "frame_stride": args.frame_stride,
        "max_sweeps": args.max_sweeps,
        "lidar_frame": "ego",
        "box_layout": args.box_layout,
        "yaw_convention": "mmdet3d",
        "velocity_source": "tracks",
        "velocity_max_dt_s": args.velocity_max_dt,
        "token_source": "log_timestamp",
        "sources": [f"av2-sensor_{args.split}"],
        "stats": dict(stats),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as handle:
        pickle.dump({"infos": infos, "metadata": metadata}, handle, protocol=4)

    class_counts = {key.split(":", 1)[1]: value for key, value in stats.items() if key.startswith("class:")}
    dropped = {key.split(":", 1)[1]: value for key, value in stats.items() if key.startswith("dropped:")}
    print(f"Wrote {len(infos)} frames from {stats['logs']} logs to {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")
    print(f"  boxes: {stats['boxes']}   classes: {class_counts}")
    print(f"  dropped categories: {dropped}")
    print(
        f"  skipped: frames missing camera={stats['frames_skipped_missing_camera']}, "
        f"missing ego={stats['frames_skipped_missing_ego']}, "
        f"logs={stats['logs_skipped_missing_camera'] + stats['logs_skipped_no_lidar']}"
    )
    print(f"  took {time.time() - start:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
