"""One-call export: 123D data root(s) in, an mmdetection3d ``.pkl`` out.

This is the layer most users touch::

    from py123detection import Source
    from py123detection.mmcv_export import export_to_mmdet3d

    export_to_mmdet3d(
        Source(data_root="/data/py123d", splits=["nuscenes-mini_train"]),
        output_path="/data/nuscenes/py123d_infos_train.pkl",
    )

Passing several sources merges them into a single pickle, which is how cross-dataset training
(as in CoIn3D) consumes multiple datasets: one ``ann_file``, one ``class_names`` list, frames
from every source side by side.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from py123detection.mmcv_export.converter import ExportConfig, ExportStats, MMDet3DConverter
from py123detection.mmcv_export.writer import WriteResult, build_metadata, dump_infos
from py123detection.sources import Source

logger = logging.getLogger(__name__)

DAY_US = 24 * 60 * 60 * 1_000_000
"""One day in microseconds — the granularity used when spacing sources apart in time."""


@dataclass
class ExportReport:
    """Everything worth knowing about a finished export."""

    write: WriteResult
    """Where the pickle went and how it was written."""

    stats: ExportStats
    """Converter counters (frames, boxes, skips, class histogram)."""

    metadata: Dict[str, Any]
    """The metadata block stored in the pickle."""

    per_source_frames: Dict[str, int] = field(default_factory=dict)
    """Frames contributed by each source, keyed by :attr:`Source.label`."""

    ordering_warning: Optional[str] = None
    """Set when mmdet3d's timestamp sort would interleave logs. See :func:`check_timestamp_ordering`."""

    def summary(self) -> str:
        """Render a short human-readable summary of the export."""
        lines = [
            f"Wrote {self.write.num_infos} frames to {self.write.path} "
            f"({self.write.size_bytes / 1e6:.1f} MB, protocol {self.write.protocol}, "
            f"arrays={self.write.array_format})",
            f"  logs: {self.stats.num_logs}   boxes: {self.stats.num_boxes}",
            f"  classes: {dict(sorted(self.stats.class_counts.items(), key=lambda kv: -kv[1]))}",
            f"  sensor payloads: {self.stats.num_referenced_payloads} referenced, "
            f"{self.stats.num_extracted_payloads} extracted",
        ]
        skipped = (
            self.stats.frames_skipped_missing_camera
            + self.stats.frames_skipped_missing_ego
            + self.stats.frames_skipped_missing_lidar
        )
        if skipped or self.stats.logs_skipped:
            lines.append(
                f"  skipped: {self.stats.logs_skipped} log(s), {skipped} frame(s) "
                f"(camera={self.stats.frames_skipped_missing_camera}, "
                f"ego={self.stats.frames_skipped_missing_ego}, "
                f"lidar={self.stats.frames_skipped_missing_lidar})"
            )
        if len(self.per_source_frames) > 1:
            lines.append(f"  per source: {self.per_source_frames}")
        if self.ordering_warning:
            lines.append(f"  WARNING: {self.ordering_warning}")
        for warning in self.write.warnings:
            lines.append(f"  WARNING: {warning}")
        return "\n".join(lines)


def check_timestamp_ordering(infos: Sequence[Dict[str, Any]]) -> Optional[str]:
    """Check that mmdet3d's global timestamp sort keeps each log's frames contiguous.

    ``NuScenesDataset.load_annotations`` sorts every info by ``timestamp``. Within one dataset
    that is harmless — logs are recorded at distinct wall-clock times — but merging datasets whose
    absolute timestamps overlap would interleave their frames, breaking sequence-based samplers
    and temporal models.

    :param infos: The merged infos.
    :return: A warning message, or ``None`` when the ordering is safe.
    """
    ordered = sorted(infos, key=lambda info: info["timestamp"])
    seen: Dict[str, int] = {}
    interleaved: List[str] = []
    previous: Optional[str] = None

    for info in ordered:
        scene_token = info["scene_token"]
        if scene_token != previous:
            if scene_token in seen:
                interleaved.append(scene_token)
            seen[scene_token] = seen.get(scene_token, 0) + 1
            previous = scene_token

    if not interleaved:
        return None
    sample = sorted(set(interleaved))[:3]
    return (
        f"{len(set(interleaved))} log(s) are split apart when sorting by timestamp (e.g. {sample}). "
        "mmdetection3d sorts infos by timestamp at load time, so sequence-based samplers would see "
        "interleaved logs. Re-run with disjoint_timestamps=True, or set Source.timestamp_offset_us "
        "per source to space the datasets apart."
    )


def _assign_disjoint_offsets(sources: Sequence[Source]) -> None:
    """Space sources a day apart in exported time, preserving intra-log deltas exactly.

    Only sources that left :attr:`Source.timestamp_offset_us` at its default are touched.
    """
    spacing = 365 * DAY_US
    for index, source in enumerate(sources):
        if source.timestamp_offset_us == 0 and index > 0:
            source.timestamp_offset_us = index * spacing


def export_to_mmdet3d(
    sources: Union[Source, Sequence[Source]],
    output_path: Union[str, Path],
    config: Optional[ExportConfig] = None,
    version: str = "py123d-v1.0",
    array_format: str = "numpy",
    pickle_protocol: int = 4,
    disjoint_timestamps: bool = False,
    progress: bool = True,
) -> ExportReport:
    """Export one or more 123D sources to a single mmdetection3d info pickle.

    :param sources: A :class:`Source`, or several to merge into one pickle.
    :param output_path: Destination ``.pkl``.
    :param config: Export configuration. Defaults to :class:`ExportConfig`.
    :param version: Value stored as ``metadata['version']`` and read back by mmdet3d.
    :param array_format: ``"numpy"`` or ``"portable"`` — see
        :func:`py123detection.mmcv_export.writer.dump_infos`.
    :param pickle_protocol: Pickle protocol; ``4`` reads on every Python from 3.4 up.
    :param disjoint_timestamps: Automatically offset each source's timestamps so mmdet3d's
        global timestamp sort cannot interleave datasets. Intra-log time deltas are unchanged.
    :param progress: Show per-log progress bars.
    :return: A report describing the export.
    """
    source_list = [sources] if isinstance(sources, Source) else list(sources)
    if not source_list:
        raise ValueError("At least one Source is required.")

    if disjoint_timestamps:
        _assign_disjoint_offsets(source_list)

    converter = MMDet3DConverter(config)
    infos: List[Dict[str, Any]] = []
    per_source_frames: Dict[str, int] = {}

    for source in source_list:
        scenes = source.load_scenes()
        if not scenes:
            logger.warning("Source '%s' matched no logs.", source.label)
            per_source_frames[source.label] = 0
            continue

        # A subsampled export still wants sweeps — and track-derived velocities — at the log's
        # own frame rate, so load a second, unsampled view of the same logs to draw them from.
        # Scenes are lazy, so this is cheap.
        native_scenes = None
        needs_native_rate = converter.config.max_sweeps > 0 or converter.config.velocity_source == "tracks"
        if needs_native_rate and source.is_subsampled:
            native_scenes = {
                (scene.split, scene.log_name): scene for scene in source.load_scenes(native_rate=True)
            }

        before = len(infos)
        infos.extend(
            converter.convert_scenes(
                scenes,
                timestamp_offset_us=source.timestamp_offset_us,
                progress=progress,
                native_scenes=native_scenes,
            )
        )
        per_source_frames[source.label] = len(infos) - before

    if not infos:
        raise RuntimeError(
            "The export produced no frames. Check that the splits exist under the data root and that "
            "the configured cameras and lidar are present in the logs (run with logging at WARNING to see "
            "per-log skip reasons)."
        )

    ordering_warning = check_timestamp_ordering(infos)
    if ordering_warning:
        logger.warning("%s", ordering_warning)

    stats = converter.stats
    metadata = build_metadata(
        taxonomy_name=converter.config.taxonomy.name,
        class_names=converter.config.taxonomy.class_names,
        version=version,
        sources=[source.label for source in source_list],
        extra={
            "camera_order": list(converter.config.camera_order),
            "camera_key": converter.config.camera_key,
            "max_sweeps": converter.config.max_sweeps,
            "with_2d_annotations": converter.config.with_2d_annotations,
            "token_source": converter.config.token_resolver.name,
            "lidar_frame": converter.config.lidar_frame,
            "lidar2ego_mode": converter.config.lidar2ego_mode,
            "camera2ego_mode": converter.config.camera2ego_mode,
            "yaw_convention": converter.config.yaw_convention,
            "box_layout": converter.config.box_layout,
            "velocity_source": converter.config.velocity_source,
            "velocity_max_dt_s": converter.config.velocity_max_dt_s,
            "ego_pose_source": converter.config.ego_pose_source,
            "per_source_frames": per_source_frames,
            "stats": stats.as_dict(),
        },
    )

    write = dump_infos(
        infos,
        output_path=output_path,
        metadata=metadata,
        protocol=pickle_protocol,
        array_format=array_format,
    )

    return ExportReport(
        write=write,
        stats=stats,
        metadata=metadata,
        per_source_frames=per_source_frames,
        ordering_warning=ordering_warning,
    )
