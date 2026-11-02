"""Per-box lidar point counts from a standalone cache, for logs that record none.

A box with no lidar return is an object no sensor saw, usually fully occluded, and
:class:`~py123detection.torch.data.boxes.BoxFilters` drops it. That needs a count, and some 123D
logs do not carry one: every Waymo perception box reads ``-1``, which keeps them all. The thesis
counted the points itself and cached them next to the logs, one table per log::

    <cache_root>/<split>/<log_name>/num_lidar_points.arrow
        sync_uuid         binary(16)      the log's sync frame
        track_tokens      list<string>    the boxes of that frame
        num_lidar_points  list<int>       their point counts

A frame's sync UUID is found through the log's own ``sync.arrow``: row ``initial_idx +
iteration * stride`` of the scene. Where the cache has no entry, the log's own count stands.
"""

from __future__ import annotations

import logging
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Dict, Union

import pyarrow as pa
import pyarrow.compute as pc

logger = logging.getLogger(__name__)

CACHE_FILENAME = "num_lidar_points.arrow"
SYNC_FILENAME = "sync.arrow"


def _uuid_bytes(value) -> bytes:
    if isinstance(value, uuid.UUID):
        return value.bytes
    if isinstance(value, bytes):
        return value
    return uuid.UUID(str(value)).bytes


@lru_cache(maxsize=256)
def _sync_uuids(path: str) -> list:
    from py123d.api.utils.arrow_helper import get_lru_cached_arrow_table

    return [_uuid_bytes(value) for value in get_lru_cached_arrow_table(path)["sync.uuid"].to_pylist()]


class LidarPointCounts:
    """Reads a num-lidar-points cache laid out like the 123D ``logs/`` directory.

    :param cache_root: The cache's root, holding one directory per split.
    :param logs_root: The 123D ``logs/`` directory the cache was computed from.
    """

    def __init__(self, cache_root: Union[str, Path], logs_root: Union[str, Path]) -> None:
        self.cache_root = Path(cache_root)
        self.logs_root = Path(logs_root)
        self._missing: set = set()

    def for_frame(self, scene, iteration: int) -> Dict[str, int]:
        """``{track_token: count}`` for one frame; empty if the cache has nothing for it."""
        from py123d.api.utils.arrow_helper import get_lru_cached_arrow_table

        split, log_name = scene.log_metadata.split, scene.log_metadata.log_name
        cache_path = self.cache_root / split / log_name / CACHE_FILENAME
        sync_path = self.logs_root / split / log_name / SYNC_FILENAME
        if not cache_path.exists() or not sync_path.exists():
            if (split, log_name) not in self._missing:
                self._missing.add((split, log_name))
                logger.warning("No lidar point cache for %s/%s; keeping the log's own counts.", split, log_name)
            return {}

        metadata = scene.scene_metadata
        row = metadata.initial_idx + iteration * metadata.target_iteration_stride
        sync_uuid = _sync_uuids(str(sync_path))[row]

        table = get_lru_cached_arrow_table(str(cache_path))
        match = table.filter(pc.equal(table["sync_uuid"], pa.scalar(sync_uuid, type=pa.binary(16))))
        if len(match) == 0:
            return {}
        tokens = match["track_tokens"][0].as_py()
        counts = match["num_lidar_points"][0].as_py()
        return {token: int(count) for token, count in zip(tokens, counts)}
