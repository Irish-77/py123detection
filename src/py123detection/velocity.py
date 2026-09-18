"""Box velocities derived from object tracks.

Some datasets (Argoverse 2, KITTI-360, ...) annotate no velocity, so their 123D logs carry zeros.
PETR-style models regress ``gt_velocity``, so the export can derive one the way the nuScenes
devkit's ``box_velocity`` does, per track in the global frame at the log's native frame rate:

* both neighbours within ``max_dt_s``: ``v = (p[k+1] - p[k-1]) / (t[k+1] - t[k-1])``
* one neighbour within ``max_dt_s``: the one-sided difference with that neighbour
* no neighbour within ``max_dt_s``: ``0``

The devkit uses a 1.5 s limit on 2 Hz keyframes and returns ``nan`` in the last case, which
mmdetection3d then replaces by ``0``. The 0.25 s default tolerates one dropped frame at 10 Hz.
The reference converters implement the same rule, so their pickles can be diffed against exports.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Mapping, Tuple

import numpy as np
from py123d.api import SceneAPI


def velocities_from_track(times_us: np.ndarray, centers: np.ndarray, max_dt_s: float = 0.25) -> np.ndarray:
    """Velocities of one track.

    :param times_us: ``(N,)`` timestamps in microseconds, sorted ascending.
    :param centers: ``(N, 3)`` box centres at those timestamps.
    :param max_dt_s: Neighbours further away than this are ignored.
    :return: ``(N, 3)`` velocities in units of ``centers`` per second.
    """
    times_us = np.asarray(times_us, dtype=np.int64).reshape(-1)
    centers = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
    if times_us.shape[0] != centers.shape[0]:
        raise ValueError("times_us and centers must have the same length.")
    if np.any(np.diff(times_us) < 0):
        raise ValueError("times_us must be sorted ascending.")

    count = times_us.shape[0]
    velocities = np.zeros((count, 3), dtype=np.float64)
    max_dt_us = max_dt_s * 1e6
    for k in range(count):
        has_prev = k > 0 and (times_us[k] - times_us[k - 1]) <= max_dt_us
        has_next = k + 1 < count and (times_us[k + 1] - times_us[k]) <= max_dt_us
        if has_prev and has_next:
            lo, hi = k - 1, k + 1
        elif has_prev:
            lo, hi = k - 1, k
        elif has_next:
            lo, hi = k, k + 1
        else:
            continue
        dt_s = (times_us[hi] - times_us[lo]) * 1e-6
        if dt_s > 0:
            velocities[k] = (centers[hi] - centers[lo]) / dt_s
    return velocities


class TrackVelocityTable:
    """Per-frame, per-track velocities of one log."""

    def __init__(self, max_dt_s: float = 0.25) -> None:
        self.max_dt_s = max_dt_s
        self._frames: Dict[int, Dict[str, np.ndarray]] = {}

    @classmethod
    def from_scene(cls, scene: SceneAPI, max_dt_s: float = 0.25) -> TrackVelocityTable:
        """Derive velocities for every track in ``scene``.

        Pass the native-rate view of the log: a subsampled one would stretch the differences over
        several frames.
        """
        table = cls(max_dt_s=max_dt_s)
        tracks: Dict[str, List[Tuple[int, np.ndarray]]] = defaultdict(list)

        for iteration in range(scene.number_of_iterations):
            detections = scene.get_box_detections_se3_at_iteration(iteration)
            if detections is None:
                continue
            timestamp_us = int(scene.get_timestamp_at_iteration(iteration).time_us)
            for detection in detections:
                center = detection.bounding_box_se3.center_se3
                tracks[str(detection.attributes.track_token)].append(
                    (timestamp_us, np.array([center.x, center.y, center.z], dtype=np.float64))
                )

        for track_token, track in tracks.items():
            track.sort(key=lambda observation: observation[0])
            times_us = np.array([timestamp_us for timestamp_us, _ in track], dtype=np.int64)
            centers = np.stack([center for _, center in track])
            velocities = velocities_from_track(times_us, centers, max_dt_s=max_dt_s)
            for timestamp_us, velocity in zip(times_us.tolist(), velocities):
                table._frames.setdefault(timestamp_us, {})[track_token] = velocity
        return table

    def at(self, timestamp_us: int) -> Mapping[str, np.ndarray]:
        """Velocities of the tracks observed at one frame, keyed by track token."""
        return self._frames.get(int(timestamp_us), {})
