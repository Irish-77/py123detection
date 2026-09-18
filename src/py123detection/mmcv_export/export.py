"""One-call export: 123D sources in, an mmdetection3d ``.pkl`` out::

    from py123detection import Source
    from py123detection.mmcv_export import export_to_mmdet3d

    export_to_mmdet3d(
        Source(data_root="/data/py123d", splits=["nuscenes-mini_train"]),
        output_path="/data/nuscenes/py123d_infos_train.pkl",
    )

Several sources are merged into one pickle, which is how cross-dataset training (as in CoIn3D)
consumes multiple datasets.
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


@dataclass
class ExportReport:
    write: WriteResult
    stats: ExportStats
    metadata: Dict[str, Any]
    per_source_frames: Dict[str, int] = field(default_factory=dict)  # keyed by Source.label
    ordering_warning: Optional[str] = None

    def summary(self) -> str:
        stats = self.stats
        lines = [
            f"Wrote {self.write.num_infos} frames to {self.write.path} "
            f"({self.write.size_bytes / 1e6:.1f} MB, protocol {self.write.protocol}, "
            f"arrays={self.write.array_format})",
            f"  logs: {stats.num_logs}   boxes: {stats.num_boxes}",
            f"  classes: {dict(sorted(stats.class_counts.items(), key=lambda kv: -kv[1]))}",
            f"  sensor payloads: {stats.num_referenced_payloads} referenced, "
            f"{stats.num_extracted_payloads} extracted",
        ]
        skipped = (
            stats.frames_skipped_missing_camera + stats.frames_skipped_missing_ego + stats.frames_skipped_missing_lidar
        )
        if skipped or stats.logs_skipped:
            lines.append(
                f"  skipped: {stats.logs_skipped} log(s), {skipped} frame(s) "
                f"(camera={stats.frames_skipped_missing_camera}, "
                f"ego={stats.frames_skipped_missing_ego}, "
                f"lidar={stats.frames_skipped_missing_lidar})"
            )
        if len(self.per_source_frames) > 1:
            lines.append(f"  per source: {self.per_source_frames}")
        if self.ordering_warning:
            lines.append(f"  WARNING: {self.ordering_warning}")
        for warning in self.write.warnings:
            lines.append(f"  WARNING: {warning}")
        return "\n".join(lines)


def check_timestamp_ordering(infos: Sequence[Dict[str, Any]]) -> Optional[str]:
    """Warning message if sorting by timestamp would split a log apart, else ``None``.

    ``NuScenesDataset.load_annotations`` sorts all infos by timestamp. That is harmless within one
    dataset, but merged datasets with overlapping timestamps get their logs interleaved, which
    breaks sequence samplers and temporal models.
    """
    seen = set()
    interleaved = set()
    previous = None
    for info in sorted(infos, key=lambda info: info["timestamp"]):
        scene_token = info["scene_token"]
        if scene_token != previous:
            if scene_token in seen:
                interleaved.add(scene_token)
            seen.add(scene_token)
            previous = scene_token

    if not interleaved:
        return None
    return (
        f"{len(interleaved)} log(s) are split apart when sorting by timestamp (e.g. {sorted(interleaved)[:3]}). "
        "mmdetection3d sorts infos by timestamp at load time, so sequence-based samplers would see "
        "interleaved logs. Re-run with disjoint_timestamps=True, or set Source.timestamp_offset_us "
        "per source to space the datasets apart."
    )


def _assign_disjoint_offsets(sources: Sequence[Source]) -> None:
    """Space sources a year apart. Sources with an explicit offset keep it."""
    for index, source in enumerate(sources):
        if source.timestamp_offset_us == 0 and index > 0:
            source.timestamp_offset_us = index * 365 * DAY_US


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
    """Export one or more 123D sources into a single mmdetection3d info pickle.

    :param version: Stored as ``metadata['version']``, which mmdet3d reads back.
    :param array_format: ``"numpy"`` or ``"portable"``, see :func:`~py123detection.mmcv_export.writer.dump_infos`.
    :param disjoint_timestamps: Offset each source's timestamps so mmdet3d's timestamp sort cannot
        interleave datasets.
    """
    source_list = [sources] if isinstance(sources, Source) else list(sources)
    if not source_list:
        raise ValueError("At least one Source is required.")
    if disjoint_timestamps:
        _assign_disjoint_offsets(source_list)

    converter = MMDet3DConverter(config)
    config = converter.config
    infos: List[Dict[str, Any]] = []
    per_source_frames: Dict[str, int] = {}

    for source in source_list:
        scenes = source.load_scenes()
        if not scenes:
            logger.warning("Source '%s' matched no logs.", source.label)
            per_source_frames[source.label] = 0
            continue

        # Sweeps and track velocities want the log's native frame rate even when the export is
        # subsampled, so load an unsampled view of the same logs too. Scenes are lazy.
        native_scenes = None
        if (config.max_sweeps > 0 or config.velocity_source == "tracks") and source.is_subsampled:
            native_scenes = {(scene.split, scene.log_name): scene for scene in source.load_scenes(native_rate=True)}

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
        taxonomy_name=config.taxonomy.name,
        class_names=config.taxonomy.class_names,
        version=version,
        sources=[source.label for source in source_list],
        extra={
            "camera_order": list(config.camera_order),
            "camera_key": config.camera_key,
            "max_sweeps": config.max_sweeps,
            "with_2d_annotations": config.with_2d_annotations,
            "token_source": config.token_resolver.name,
            "lidar_frame": config.lidar_frame,
            "lidar2ego_mode": config.lidar2ego_mode,
            "camera2ego_mode": config.camera2ego_mode,
            "yaw_convention": config.yaw_convention,
            "box_layout": config.box_layout,
            "velocity_source": config.velocity_source,
            "velocity_max_dt_s": config.velocity_max_dt_s,
            "ego_pose_source": config.ego_pose_source,
            "per_source_frames": per_source_frames,
            "stats": stats.as_dict(),
        },
    )

    write = dump_infos(
        infos, output_path=output_path, metadata=metadata, protocol=pickle_protocol, array_format=array_format
    )
    return ExportReport(
        write=write,
        stats=stats,
        metadata=metadata,
        per_source_frames=per_source_frames,
        ordering_warning=ordering_warning,
    )
