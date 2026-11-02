"""Resolving "a 123D data root plus split names" into the scenes an export iterates over.

A frame-indexed pkl needs exactly one scene per log, spanning the whole log. :class:`Source`
builds the :class:`~py123d.api.SceneFilter` for that and exposes the few options that matter::

    Source(data_root="/data/py123d", splits=["nuscenes-mini_train"], sample_rate_hz=2.0)

Scenes built elsewhere can be passed in directly with ``Source.from_scenes(scenes)``.
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
    """123D data root holding ``logs/`` and ``maps/``. ``None`` uses ``PY123D_DATA_ROOT``."""

    splits: Optional[List[str]] = None
    """Split names, e.g. ``["nuscenes-mini_train"]``. ``None`` reads every split under the root."""

    datasets: Optional[List[str]] = None

    split_types: Optional[List[str]] = None
    """``train`` / ``val`` / ``test``. Ignored when :attr:`splits` is set."""

    log_names: Optional[List[str]] = None

    sample_rate_hz: Optional[float] = None
    """Target frame rate; ``2.0`` on a 10 Hz log keeps every 5th frame. Wins over :attr:`frame_stride`."""

    frame_stride: Optional[int] = None

    required_modalities: Optional[List[str]] = None
    """``SceneFilter.required_scene_modalities`` entries, e.g. ``["camera:any", "box_detections_se3"]``."""

    max_logs: Optional[int] = None

    timestamp_offset_us: int = 0
    """Added to every exported timestamp of this source.

    mmdet3d's ``NuScenesDataset.load_annotations`` sorts all infos by timestamp, so merged datasets
    whose timestamps overlap would get their logs interleaved. An offset keeps them apart without
    touching intra-log time deltas.
    """

    name: Optional[str] = None
    """Label for logs and metadata. Defaults to the joined :attr:`splits`."""

    scene_filter: Optional[SceneFilter] = None
    """Use this filter instead of building one. Leave ``future_duration_s`` and
    ``future_num_iterations`` unset, or frames get exported more than once."""

    scenes: Optional[List[SceneAPI]] = field(default=None, repr=False)
    """Pre-built scenes, one per full log. No filtering happens when set."""

    @classmethod
    def from_scenes(cls, scenes: Sequence[SceneAPI], name: Optional[str] = None, timestamp_offset_us: int = 0) -> Source:
        return cls(scenes=list(scenes), name=name or "scenes", timestamp_offset_us=timestamp_offset_us)

    @property
    def label(self) -> str:
        if self.name is not None:
            return self.name
        if self.splits:
            return "+".join(self.splits)
        if self.datasets:
            return "+".join(self.datasets)
        return "all"

    @property
    def is_subsampled(self) -> bool:
        if self.scene_filter is not None:
            return (
                self.scene_filter.target_iteration_duration_s is not None
                or (self.scene_filter.target_iteration_stride or 1) > 1
            )
        return self.sample_rate_hz is not None or (self.frame_stride or 1) > 1

    def build_scene_filter(self, native_rate: bool = False) -> SceneFilter:
        """Build the filter for this source.

        Leaving ``future_*`` unset puts the 123D scene builder in its one-scene-per-full-log mode.

        :param native_rate: Drop the sampling options and keep every frame. The export uses this
            view for lidar sweeps and track velocities.
        """
        if self.scene_filter is not None:
            if not native_rate:
                return self.scene_filter
            return replace(self.scene_filter, target_iteration_duration_s=None, target_iteration_stride=None)

        target_iteration_duration_s = None
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
        """Scenes of this source, one per log, sorted by ``(split, log_name)`` for reproducible exports."""
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
