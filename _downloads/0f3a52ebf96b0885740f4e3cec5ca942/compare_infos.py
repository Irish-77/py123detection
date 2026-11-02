"""Compare two mmdetection3d info pickles field by field.

Built to answer one question: does ``nuScenes -> 123D -> pkl`` land in the same place as
``nuScenes -> pkl``? It aligns frames by token, aligns boxes by instance token, and reports the
maximum absolute deviation of every geometric quantity, so a difference shows up as a number
rather than as "looks fine".

Comparisons that need care:

* **Quaternion sign.** ``q`` and ``-q`` are the same rotation, so orientations are compared as
  rotation matrices, never componentwise.
* **Yaw wrap.** Heading differences are wrapped into ``(-pi, pi]`` before taking a maximum.
* **Unmapped categories.** The reference converter keeps every annotation, leaving categories
  outside the 10-class benchmark under their raw nuScenes name. A taxonomy-driven export drops
  them, so the reference is filtered to benchmark categories before box counts are compared.
"""

from __future__ import annotations

import argparse
import os
import pickle
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from pyquaternion import Quaternion

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


class Deviation:
    """Accumulates the worst absolute deviation seen for one quantity."""

    def __init__(self, name: str, unit: str = "") -> None:
        self.name = name
        self.unit = unit
        self.max_abs = 0.0
        self.count = 0
        self.worst_context: Optional[str] = None

    def update(self, values: np.ndarray, context: str = "") -> None:
        """Fold a batch of absolute deviations into the accumulator."""
        values = np.abs(np.asarray(values, dtype=np.float64)).ravel()
        if values.size == 0:
            return
        self.count += values.size
        peak = float(values.max())
        if peak > self.max_abs:
            self.max_abs = peak
            self.worst_context = context

    def line(self) -> str:
        """Render one report row."""
        unit = f" {self.unit}" if self.unit else ""
        context = f"   @ {self.worst_context}" if self.worst_context else ""
        return f"  {self.name:<34} max |diff| = {self.max_abs:.3e}{unit}   (n={self.count}){context}"


def _same_file(left: str, right: str) -> bool:
    """Compare two sensor paths by the file they resolve to, not by how they are spelled.

    The reference converter emits whatever ``--root-path`` it was given (often relative), while
    exports default to absolute paths; both point at the same file on disk.
    """
    return os.path.realpath(str(left)) == os.path.realpath(str(right))


def load_infos(path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Load an info pickle.

    :param path: Path to the pickle.
    :return: ``(infos, metadata)``.
    """
    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    return payload["infos"], payload.get("metadata", {})


def quaternion_matrix(rotation: Sequence[float]) -> np.ndarray:
    """Rotation matrix of a scalar-first quaternion, immune to the ``q``/``-q`` sign ambiguity."""
    return Quaternion(np.asarray(rotation, dtype=np.float64)).rotation_matrix


def wrap_angle(values: np.ndarray) -> np.ndarray:
    """Wrap angles into ``(-pi, pi]``."""
    return np.arctan2(np.sin(values), np.cos(values))


def filter_reference_boxes(info: Dict[str, Any], categories: Set[str] = BENCHMARK_CATEGORIES) -> Dict[str, Any]:
    """Drop reference boxes whose category is outside the compared class set.

    :param info: A reference info.
    :param categories: Class names to keep — the candidate's taxonomy when it records one,
        otherwise the nuScenes 10-class benchmark.
    :return: A shallow copy with the annotation arrays masked.
    """
    names = np.asarray(info["gt_names"])
    mask = np.array([name in categories for name in names], dtype=bool)
    filtered = dict(info)
    filtered["gt_names"] = names[mask]
    filtered["gt_boxes"] = np.asarray(info["gt_boxes"])[mask]
    filtered["gt_velocity"] = np.asarray(info["gt_velocity"])[mask]
    filtered["num_lidar_pts"] = np.asarray(info["num_lidar_pts"])[mask]
    filtered["num_radar_pts"] = np.asarray(info["num_radar_pts"])[mask]
    filtered["valid_flag"] = np.asarray(info["valid_flag"])[mask]
    if "instance_tokens" in info:
        filtered["instance_tokens"] = [t for t, keep in zip(info["instance_tokens"], mask) if keep]
    return filtered


def compare(
    reference_path: str,
    candidate_path: str,
    max_frames: Optional[int] = None,
    compare_2d: bool = True,
) -> int:
    """Compare two info pickles and print a report.

    :param reference_path: Pickle produced by the native converter.
    :param candidate_path: Pickle produced by the 123D export.
    :param max_frames: Compare only the first N aligned frames.
    :param compare_2d: Also compare the per-camera 2D annotations.
    :return: Process exit code: 0 when the files align structurally, 1 otherwise.
    """
    reference_infos, reference_meta = load_infos(reference_path)
    candidate_infos, candidate_meta = load_infos(candidate_path)

    print("=" * 100)
    print(f"reference : {reference_path}  ({len(reference_infos)} frames, metadata={reference_meta})")
    print(f"candidate : {candidate_path}  ({len(candidate_infos)} frames)")
    print(f"            taxonomy={candidate_meta.get('taxonomy')} classes={candidate_meta.get('class_names')}")
    print("=" * 100)

    # Compare over the candidate's class set: a taxonomy-driven export only emits its own classes,
    # while a reference converter may keep every raw category.
    compared_classes: Set[str] = set(candidate_meta.get("class_names") or BENCHMARK_CATEGORIES)
    print(f"reference boxes filtered to the compared classes: {sorted(compared_classes)}")

    reference_by_token = {info["token"]: info for info in reference_infos}
    candidate_by_token = {info["token"]: info for info in candidate_infos}
    shared = [token for token in reference_by_token if token in candidate_by_token]

    print("\n[1] Frame alignment")
    print(f"  reference frames        : {len(reference_infos)}")
    print(f"  candidate frames        : {len(candidate_infos)}")
    print(f"  matched by token        : {len(shared)}")
    only_reference = len(reference_by_token) - len(shared)
    only_candidate = len(candidate_by_token) - len(shared)
    print(f"  reference-only          : {only_reference}")
    print(f"  candidate-only          : {only_candidate}")
    if not shared:
        print("\n  No frames matched by token. Both pickles must spell tokens the same way: for")
        print("  nuScenes re-export with --nuscenes-root (native sample tokens); for datasets without")
        print("  native frame tokens export with --token-style log_timestamp and have the reference")
        print("  converter write '{log_name}/{timestamp_us}' as well.")
        return 1

    if max_frames is not None:
        shared = shared[:max_frames]

    structural_problems: List[str] = []
    deviations = {
        "timestamp": Deviation("timestamp", "us"),
        "lidar2ego_t": Deviation("lidar2ego translation", "m"),
        "lidar2ego_R": Deviation("lidar2ego rotation matrix"),
        "ego2global_t": Deviation("ego2global translation", "m"),
        "ego2global_R": Deviation("ego2global rotation matrix"),
        "cam_intrinsic": Deviation("cam_intrinsic", "px"),
        "cam_sensor2ego_t": Deviation("cam sensor2ego translation", "m"),
        "cam_sensor2ego_R": Deviation("cam sensor2ego rotation matrix"),
        "cam_ego2global_t": Deviation("cam ego2global translation", "m"),
        "cam_ego2global_R": Deviation("cam ego2global rotation matrix"),
        "cam_sensor2lidar_t": Deviation("cam sensor2lidar translation", "m"),
        "cam_sensor2lidar_R": Deviation("cam sensor2lidar rotation"),
        "cam_timestamp": Deviation("cam timestamp", "us"),
        "box_center": Deviation("box center (lidar frame)", "m"),
        "box_size": Deviation("box size l/w/h", "m"),
        "box_yaw": Deviation("box yaw", "rad"),
        "box_velocity": Deviation("box velocity (lidar frame)", "m/s"),
        "box_num_lidar_pts": Deviation("box num_lidar_pts", "pts"),
        "bbox2d": Deviation("2D bbox corners", "px"),
        "center2d": Deviation("2D center", "px"),
        "depth2d": Deviation("2D depth", "m"),
        "bbox3d_cam": Deviation("mono-3D box (camera frame)", "m/rad"),
    }

    path_mismatches = 0
    path_total = 0
    box_counts = {"reference": 0, "candidate": 0, "matched": 0}
    counts_2d = {"reference": 0, "candidate": 0, "matched_frames_cams": 0}
    label_mismatches = 0
    velocity_nan_reference = 0
    sweep_lengths: Dict[str, List[int]] = defaultdict(list)

    for token in shared:
        reference = filter_reference_boxes(reference_by_token[token], compared_classes)
        candidate = candidate_by_token[token]
        context = f"token={token[:8]}"

        deviations["timestamp"].update(np.array([reference["timestamp"] - candidate["timestamp"]]), context)

        # frame-level poses
        for key in ("lidar2ego", "ego2global"):
            deviations[f"{key}_t"].update(
                np.asarray(reference[f"{key}_translation"], dtype=np.float64)
                - np.asarray(candidate[f"{key}_translation"], dtype=np.float64),
                context,
            )
            deviations[f"{key}_R"].update(
                quaternion_matrix(reference[f"{key}_rotation"]) - quaternion_matrix(candidate[f"{key}_rotation"]),
                context,
            )

        # lidar path
        path_total += 1
        if not _same_file(reference["lidar_path"], candidate["lidar_path"]):
            path_mismatches += 1

        # cameras
        if set(reference["cams"]) != set(candidate["cams"]):
            structural_problems.append(
                f"{context}: camera keys differ: reference={sorted(reference['cams'])} "
                f"candidate={sorted(candidate['cams'])}"
            )
        else:
            if list(reference["cams"]) != list(candidate["cams"]):
                structural_problems.append(
                    f"{context}: camera ORDER differs: reference={list(reference['cams'])} "
                    f"candidate={list(candidate['cams'])}"
                )
            for camera_name, reference_cam in reference["cams"].items():
                candidate_cam = candidate["cams"][camera_name]
                camera_context = f"{context} cam={camera_name}"
                path_total += 1
                if not _same_file(reference_cam["data_path"], candidate_cam["data_path"]):
                    path_mismatches += 1
                deviations["cam_timestamp"].update(
                    np.array([reference_cam["timestamp"] - candidate_cam["timestamp"]]), camera_context
                )
                deviations["cam_intrinsic"].update(
                    np.asarray(reference_cam["cam_intrinsic"], dtype=np.float64)
                    - np.asarray(candidate_cam["cam_intrinsic"], dtype=np.float64),
                    camera_context,
                )
                for key in ("sensor2ego", "ego2global"):
                    deviations[f"cam_{key}_t"].update(
                        np.asarray(reference_cam[f"{key}_translation"], dtype=np.float64)
                        - np.asarray(candidate_cam[f"{key}_translation"], dtype=np.float64),
                        camera_context,
                    )
                    deviations[f"cam_{key}_R"].update(
                        quaternion_matrix(reference_cam[f"{key}_rotation"])
                        - quaternion_matrix(candidate_cam[f"{key}_rotation"]),
                        camera_context,
                    )
                deviations["cam_sensor2lidar_t"].update(
                    np.asarray(reference_cam["sensor2lidar_translation"], dtype=np.float64)
                    - np.asarray(candidate_cam["sensor2lidar_translation"], dtype=np.float64),
                    camera_context,
                )
                deviations["cam_sensor2lidar_R"].update(
                    np.asarray(reference_cam["sensor2lidar_rotation"], dtype=np.float64)
                    - np.asarray(candidate_cam["sensor2lidar_rotation"], dtype=np.float64),
                    camera_context,
                )

        # sweeps
        sweep_lengths["reference"].append(len(reference["sweeps"]))
        sweep_lengths["candidate"].append(len(candidate["sweeps"]))

        # 3D boxes, matched by instance token
        box_counts["reference"] += len(reference["gt_names"])
        box_counts["candidate"] += len(candidate["gt_names"])
        matched = _match_boxes(reference, candidate)
        box_counts["matched"] += len(matched)
        for reference_index, candidate_index in matched:
            reference_box = np.asarray(reference["gt_boxes"][reference_index], dtype=np.float64)
            candidate_box = np.asarray(candidate["gt_boxes"][candidate_index], dtype=np.float64)
            deviations["box_center"].update(reference_box[:3] - candidate_box[:3], context)
            deviations["box_size"].update(reference_box[3:6] - candidate_box[3:6], context)
            deviations["box_yaw"].update(wrap_angle(np.array([reference_box[6] - candidate_box[6]])), context)

            reference_velocity = np.asarray(reference["gt_velocity"][reference_index], dtype=np.float64)
            candidate_velocity = np.asarray(candidate["gt_velocity"][candidate_index], dtype=np.float64)
            if np.isnan(reference_velocity).any():
                velocity_nan_reference += 1
            else:
                deviations["box_velocity"].update(reference_velocity - candidate_velocity, context)

            deviations["box_num_lidar_pts"].update(
                np.array(
                    [
                        float(reference["num_lidar_pts"][reference_index])
                        - float(candidate["num_lidar_pts"][candidate_index])
                    ]
                ),
                context,
            )
            if reference["gt_names"][reference_index] != candidate["gt_names"][candidate_index]:
                label_mismatches += 1

        # per-camera 2D annotations
        if compare_2d and "bboxes2d" in reference and "bboxes2d" in candidate:
            _compare_2d(reference, candidate, deviations, counts_2d, context)
    print("\n[2] Paths")
    print(f"  compared                : {path_total}")
    print(f"  mismatched              : {path_mismatches}")

    print("\n[3] Sweeps (per frame)")
    for name, lengths in sweep_lengths.items():
        array = np.array(lengths)
        empty = int((array == 0).sum())
        print(f"  {name:<22}: mean={array.mean():.2f} max={array.max()} empty={empty}")

    print("\n[4] 3D boxes")
    print(f"  reference (benchmark)   : {box_counts['reference']}")
    print(f"  candidate               : {box_counts['candidate']}")
    print(f"  matched by instance     : {box_counts['matched']}")
    print(f"  class-name mismatches   : {label_mismatches}")
    print(f"  reference NaN velocities: {velocity_nan_reference} (excluded from the velocity comparison)")

    if compare_2d:
        print("\n[5] 2D annotations")
        print(f"  reference boxes         : {counts_2d['reference']}")
        print(f"  candidate boxes         : {counts_2d['candidate']}")
        print(f"  frame-cameras compared  : {counts_2d['matched_frames_cams']}")

    print("\n[6] Numeric deviations")
    for deviation in deviations.values():
        if deviation.count:
            print(deviation.line())

    if structural_problems:
        print(f"\n[7] Structural problems ({len(structural_problems)}):")
        for problem in structural_problems[:20]:
            print(f"  {problem}")
        if len(structural_problems) > 20:
            print(f"  ... and {len(structural_problems) - 20} more")

    ok = (
        not structural_problems
        and only_reference == 0
        and only_candidate == 0
        and path_mismatches == 0
        and label_mismatches == 0
    )
    print("\n" + ("=" * 100))
    print("RESULT:", "structurally identical" if ok else "differences found (see above)")
    print("=" * 100)
    return 0 if ok else 1


def _match_boxes(reference: Dict[str, Any], candidate: Dict[str, Any]) -> List[Tuple[int, int]]:
    """Pair reference and candidate boxes by instance token, falling back to positional order."""
    reference_tokens = reference.get("instance_tokens")
    candidate_tokens = candidate.get("instance_tokens")
    if reference_tokens and candidate_tokens:
        candidate_index_by_token = {token: index for index, token in enumerate(candidate_tokens)}
        return [
            (reference_index, candidate_index_by_token[token])
            for reference_index, token in enumerate(reference_tokens)
            if token in candidate_index_by_token
        ]
    count = min(len(reference["gt_names"]), len(candidate["gt_names"]))
    return [(index, index) for index in range(count)]


def _compare_2d(
    reference: Dict[str, Any],
    candidate: Dict[str, Any],
    deviations: Dict[str, Deviation],
    counts: Dict[str, int],
    context: str,
) -> None:
    """Compare the per-camera 2D annotation lists of one frame.

    The two converters filter marginal boxes slightly differently, so entries are paired by
    matching 2D centers within a pixel; unpaired boxes only affect the counts.
    """
    for camera_index, camera_name in enumerate(reference["cams"]):
        if camera_index >= len(candidate["bboxes2d"]):
            continue
        counts["matched_frames_cams"] += 1

        reference_boxes = np.asarray(reference["bboxes2d"][camera_index], dtype=np.float64).reshape(-1, 4)
        candidate_boxes = np.asarray(candidate["bboxes2d"][camera_index], dtype=np.float64).reshape(-1, 4)
        reference_centers = np.asarray(reference["centers2d"][camera_index], dtype=np.float64).reshape(-1, 2)
        candidate_centers = np.asarray(candidate["centers2d"][camera_index], dtype=np.float64).reshape(-1, 2)
        reference_depths = np.asarray(reference["depths"][camera_index], dtype=np.float64).reshape(-1)
        candidate_depths = np.asarray(candidate["depths"][camera_index], dtype=np.float64).reshape(-1)
        reference_boxes3d = np.asarray(reference["bboxes3d_cams"][camera_index], dtype=np.float64).reshape(-1, 7)
        candidate_boxes3d = np.asarray(candidate["bboxes3d_cams"][camera_index], dtype=np.float64).reshape(-1, 7)

        counts["reference"] += len(reference_boxes)
        counts["candidate"] += len(candidate_boxes)
        if len(reference_centers) == 0 or len(candidate_centers) == 0:
            continue

        distances = np.linalg.norm(reference_centers[:, None, :] - candidate_centers[None, :, :], axis=-1)
        for reference_index in range(len(reference_centers)):
            candidate_index = int(distances[reference_index].argmin())
            if distances[reference_index, candidate_index] > 1.0:
                continue
            camera_context = f"{context} cam#{camera_index}"
            deviations["bbox2d"].update(reference_boxes[reference_index] - candidate_boxes[candidate_index], camera_context)
            deviations["center2d"].update(
                reference_centers[reference_index] - candidate_centers[candidate_index], camera_context
            )
            deviations["depth2d"].update(
                np.array([reference_depths[reference_index] - candidate_depths[candidate_index]]), camera_context
            )
            difference = reference_boxes3d[reference_index] - candidate_boxes3d[candidate_index]
            difference[6] = wrap_angle(np.array([difference[6]]))[0]
            deviations["bbox3d_cam"].update(difference, camera_context)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point for the comparison tool."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("reference", help="Pickle from the native converter.")
    parser.add_argument("candidate", help="Pickle from the 123D export.")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--no-2d", action="store_true")
    args = parser.parse_args(argv)
    return compare(args.reference, args.candidate, max_frames=args.max_frames, compare_2d=not args.no_2d)


if __name__ == "__main__":
    raise SystemExit(main())
