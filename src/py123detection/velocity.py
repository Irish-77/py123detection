"""Deriving per-box velocities from object tracks.

Several datasets 123D covers annotate boxes without a velocity (Argoverse 2, KITTI-360, ...),
and their 123D logs carry zeros. mmdetection3d's nuScenes schema expects ``gt_velocity`` and the
PETR family regresses it, so an export can optionally *derive* one the way the nuScenes devkit
does (``NuScenes.box_velocity``): a central difference of the box centre over the neighbouring
annotations of the same track.

The rule, applied per track in the **global frame** at the log's **native** frame rate:

* both neighbours within ``max_dt_s``:  ``v = (p[k+1] - p[k-1]) / (t[k+1] - t[k-1])``
* one neighbour within ``max_dt_s``:    the one-sided difference with that neighbour
* no neighbour within ``max_dt_s``:     ``0``

``max_dt_s`` defaults to 0.25 s, so on a 10 Hz log one dropped frame is tolerated. The nuScenes
devkit applies the same three cases with a 1.5 s limit on its 2 Hz keyframes and returns ``nan``
in the last one; mmdetection3d then replaces that ``nan`` by ``0``, so the two conventions land
in the same place.

Any other converter can reproduce the rule from this description alone — that is deliberate,
so a reference pipeline built from the raw dataset can be diffed against an export.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Mapping, Tuple

import numpy as np
import numpy.typing as npt
from py123d.api import SceneAPI

Array = npt.NDArray[np.float64]


def velocities_from_track(times_us: np.ndarray, centers: np.ndarray, max_dt_s: float = 0.25) -> Array:
    """Central-difference velocities of one track.

    :param times_us: ``(N,)`` observation timestamps in microseconds, sorted ascending.
    :param centers: ``(N, 3)`` box centres at those timestamps, in one common frame.
    :param max_dt_s: A neighbour further away than this is ignored.
    :return: ``(N, 3)`` velocities in the units of ``centers`` per second.
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
    """Per-frame, per-track velocities of one log, derived from its native-rate detections."""

    def __init__(self, max_dt_s: float = 0.25) -> None:
        """Initialize an empty table.

        :param max_dt_s: See :func:`velocities_from_track`.
        """
        self.max_dt_s = max_dt_s
        self._frames: Dict[int, Dict[str, Array]] = {}
        self.num_tracks = 0
        self.num_observations = 0

    @classmethod
    def from_scene(cls, scene: SceneAPI, max_dt_s: float = 0.25) -> "TrackVelocityTable":
        """Walk every iteration of a scene and derive velocities for every track it contains.

        Pass the **native-rate** view of a log (all frames), not a subsampled one: the rule
        wants the closest annotated neighbours, and subsampling would stretch the differences
        over several frames.

        :param scene: The scene to walk. Only box detections are read; no sensor payloads.
        :param max_dt_s: See :func:`velocities_from_track`.
        :return: The populated table.
        """
        table = cls(max_dt_s=max_dt_s)
        observations: Dict[str, List[Tuple[int, Array]]] = defaultdict(list)

        for iteration in range(scene.number_of_iterations):
            detections = scene.get_box_detections_se3_at_iteration(iteration)
            if detections is None:
                continue
            timestamp_us = int(scene.get_timestamp_at_iteration(iteration).time_us)
            for detection in detections:
                center = detection.bounding_box_se3.center_se3
                observations[str(detection.attributes.track_token)].append(
                    (timestamp_us, np.array([center.x, center.y, center.z], dtype=np.float64))
                )

        for track_token, track in observations.items():
            track.sort(key=lambda observation: observation[0])
            times_us = np.array([observation[0] for observation in track], dtype=np.int64)
            centers = np.stack([observation[1] for observation in track], axis=0)
            velocities = velocities_from_track(times_us, centers, max_dt_s=max_dt_s)
            for timestamp_us, velocity in zip(times_us.tolist(), velocities):
                table._frames.setdefault(int(timestamp_us), {})[track_token] = velocity

        table.num_tracks = len(observations)
        table.num_observations = sum(len(track) for track in observations.values())
        return table

    def at(self, timestamp_us: int) -> Mapping[str, Array]:
        """Velocities of every track observed at one frame, keyed by track token.

        :param timestamp_us: The frame's timestamp in microseconds.
        :return: A mapping; empty when the frame holds no detections.
        """
        return self._frames.get(int(timestamp_us), {})
