"""Turning "a 123D data root plus a split name" into the scenes an export iterates over.

The 123D :class:`~py123d.api.SceneFilter` is expressive but has a lot of knobs, and only a
specific corner of it produces what a frame-indexed pkl needs: **one scene per log, spanning
the whole log**. :class:`Source` encodes that corner, exposes the handful of options that
actually matter for an export (which splits, which sample rate), and leaves an escape hatch
(:attr:`Source.scene_filter`) for anyone who wants the full filter.

Passing a folder is therefore the normal path::

    Source(data_root="/data/py123d", splits=["nuscenes-mini_train"], sample_rate_hz=2.0)

and passing scenes you already built yourself is the advanced path::

    Source.from_scenes(get_filtered_scenes(my_own_filter))
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import List, Optional, Sequence, Union

from py123d.api import SceneAPI, SceneFilter, get_filtered_scenes
from py123d.common.execution import Executor, ThreadPoolExecutor

logger = logging.getLogger(__name__)


@dataclass
class Source:
    """One 123D input to export: where the logs are, which of them, and at what rate."""

    data_root: Optional[Union[str, Path]] = None
    """123D data root (the folder holding ``logs/`` and ``maps/``). ``None`` uses ``PY123D_DATA_ROOT``."""

    splits: Optional[List[str]] = None
    """Split names to read, e.g. ``["nuscenes-mini_train"]``. ``None`` reads every split under the root."""

    datasets: Optional[List[str]] = None
    """Restrict to these dataset names (the part before ``_`` in a split name)."""

    split_types: Optional[List[str]] = None
    """Restrict to these split types (``train`` / ``val`` / ``test``). Ignored when :attr:`splits` is set."""

    log_names: Optional[List[str]] = None
    """Restrict to these log names (nuScenes scene names, AV2 log ids, ...)."""

    sample_rate_hz: Optional[float] = None
    """Target frame rate. ``2.0`` on a 10 Hz log keeps every 5th frame. Takes priority over :attr:`frame_stride`."""

    frame_stride: Optional[int] = None
    """Keep every N-th raw frame. Ignored when :attr:`sample_rate_hz` is set."""

    required_modalities: Optional[List[str]] = None
    """123D modality requirements a log must satisfy, e.g. ``["camera:any", "box_detections_se3"]``.
    Uses the ``SceneFilter.required_scene_modalities`` grammar."""

    max_logs: Optional[int] = None
    """Keep at most this many logs. Useful for smoke tests."""

    timestamp_offset_us: int = 0
    """Constant added to every exported timestamp for this source.

    mmdet3d's ``NuScenesDataset.load_annotations`` sorts all infos by timestamp, so merging
    datasets whose absolute timestamps interleave would shuffle logs together. Offsetting one
    source pushes it clear of the others while leaving intra-log time deltas untouched.
    """

    name: Optional[str] = None
    """Human-readable label for logs and exported metadata. Defaults to a join of :attr:`splits`."""

    scene_filter: Optional[SceneFilter] = None
    """Escape hatch: use this filter verbatim instead of building one from the fields above.

    It must still yield one scene per log spanning the full log — i.e. leave ``future_duration_s``
    and ``future_num_iterations`` unset — otherwise frames are exported more than once.
    """

    scenes: Optional[List[SceneAPI]] = field(default=None, repr=False)
    """Escape hatch: pre-built scenes. When set, no filtering happens at all."""

    @classmethod
    def from_scenes(cls, scenes: Sequence[SceneAPI], name: Optional[str] = None, timestamp_offset_us: int = 0) -> Source:
        """Build a source from scenes that were already loaded.

        :param scenes: Scenes to export. Each should span a full log (one scene per log).
        :param name: Optional label for logs and metadata.
        :param timestamp_offset_us: See :attr:`timestamp_offset_us`.
        :return: The source.
        """
        return cls(scenes=list(scenes), name=name or "scenes", timestamp_offset_us=timestamp_offset_us)

    @property
    def label(self) -> str:
        """Human-readable identifier for this source."""
        if self.name is not None:
            return self.name
        if self.splits:
            return "+".join(self.splits)
        if self.datasets:
            return "+".join(self.datasets)
        return "all"

    @property
    def is_subsampled(self) -> bool:
        """Whether this source keeps only a subset of the log's frames."""
        if self.scene_filter is not None:
            return (
                self.scene_filter.target_iteration_duration_s is not None
                or (self.scene_filter.target_iteration_stride or 1) > 1
            )
        return self.sample_rate_hz is not None or (self.frame_stride or 1) > 1

    def build_scene_filter(self, native_rate: bool = False) -> SceneFilter:
        """Build the :class:`SceneFilter` describing this source.

        The filter deliberately leaves ``future_*`` unset, which puts the 123D scene builder in
        its "one scene per log, running to the end of the log" mode — exactly one scene per log,
        with ``number_of_iterations`` frames in it.

        :param native_rate: Drop the sampling knobs, yielding every frame the log holds. The
            export uses this second view to read lidar sweeps at the log's own rate rather than at
            the (possibly much coarser) export rate.
        :return: The scene filter.
        """
        if self.scene_filter is not None:
            if not native_rate:
                return self.scene_filter
            return replace(self.scene_filter, target_iteration_duration_s=None, target_iteration_stride=None)

        target_iteration_duration_s: Optional[float] = None
        if self.sample_rate_hz is not None:
            if self.sample_rate_hz <= 0:
                raise ValueError(f"sample_rate_hz must be > 0, got {self.sample_rate_hz}.")
            target_iteration_duration_s = 1.0 / self.sample_rate_hz

        return SceneFilter(
            datasets=self.datasets,
            split_types=self.split_types,
            split_names=self.splits,
            log_names=self.log_names,
            future_num_iterations=None,
            history_num_iterations=0,
            target_iteration_duration_s=None if native_rate else target_iteration_duration_s,
            target_iteration_stride=None if native_rate else self.frame_stride,
            required_scene_modalities=self.required_modalities,
        )

    def load_scenes(self, executor: Optional[Executor] = None, native_rate: bool = False) -> List[SceneAPI]:
        """Resolve this source to a list of scenes, one per log.

        :param executor: Executor used to parallelize log discovery. Defaults to a thread pool.
        :param native_rate: Ignore the sampling knobs and return every frame of each log.
        :return: Scenes sorted by ``(split, log_name)`` so exports are reproducible.
        """
        if self.scenes is not None:
            return sorted(self.scenes, key=lambda scene: (scene.split, scene.log_name))

        data_root = self.data_root if self.data_root is not None else os.environ.get("PY123D_DATA_ROOT")
        if data_root is None:
            raise ValueError(
                "No 123D data root: pass Source(data_root=...) or set the PY123D_DATA_ROOT environment variable."
            )

        scenes = get_filtered_scenes(
            self.build_scene_filter(native_rate=native_rate),
            data_root=Path(data_root),
            executor=executor if executor is not None else ThreadPoolExecutor(),
        )
        scenes = sorted(scenes, key=lambda scene: (scene.split, scene.log_name))

        if self.max_logs is not None:
            scenes = scenes[: self.max_logs]

        if not native_rate:
            logger.info("Source '%s': resolved %d log(s) from %s", self.label, len(scenes), data_root)
        return scenes
