"""Choosing what goes into ``info['token']``.

123D does not carry a dataset's native frame identifier. It stamps every synchronized frame
with a deterministic UUIDv5 derived from ``(split, log_name, timestamp_us)``, which is stable
across conversions but is *not* the nuScenes ``sample_token``.

That distinction matters downstream. Training only needs ``token`` to be unique, so the UUID is
fine. The **official nuScenes evaluation**, however, keys submissions by ``sample_token`` and
will reject or mis-score results whose tokens it does not recognise. When the original dataset
is still on disk, :class:`NuScenesTokenResolver` recovers the native tokens by joining on
``(log_name, timestamp_us)`` — the same pair the UUID was derived from — restoring full
evaluation compatibility.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Union

logger = logging.getLogger(__name__)


class TokenResolver:
    """Base class: maps a 123D frame identity to the token written into an exported info."""

    name = "uuid"
    """Identifier recorded in the exported metadata."""

    def resolve(self, dataset: str, split: str, log_name: str, timestamp_us: int, uuid: str) -> str:
        """Return the token for one frame.

        :param dataset: 123D dataset name (e.g. ``"nuscenes"``).
        :param split: 123D split name (e.g. ``"nuscenes-mini_val"``).
        :param log_name: Log name (nuScenes scene name, AV2 log id, ...).
        :param timestamp_us: Frame timestamp in microseconds.
        :param uuid: The deterministic 123D frame UUID.
        :return: The token to store.
        """
        return uuid


class MappingTokenResolver(TokenResolver):
    """Resolver backed by an explicit ``"{log_name}:{timestamp_us}" -> token`` mapping."""

    name = "mapping"

    def __init__(self, mapping: Dict[str, str], strict: bool = False) -> None:
        """Initialize the resolver.

        :param mapping: Keys of the form ``"{log_name}:{timestamp_us}"``.
        :param strict: Raise on a missing key instead of falling back to the 123D UUID.
        """
        self._mapping = mapping
        self._strict = strict
        self._misses = 0

    @classmethod
    def from_json(cls, path: Union[str, Path], strict: bool = False) -> MappingTokenResolver:
        """Load a mapping written by :func:`dump_nuscenes_token_map`.

        :param path: Path to the JSON file.
        :param strict: See :meth:`__init__`.
        :return: The resolver.
        """
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls(payload["tokens"], strict=strict)

    @property
    def num_misses(self) -> int:
        """How many frames fell back to the 123D UUID."""
        return self._misses

    def resolve(self, dataset: str, split: str, log_name: str, timestamp_us: int, uuid: str) -> str:
        """Inherited, see superclass."""
        token = self._mapping.get(f"{log_name}:{timestamp_us}")
        if token is None:
            if self._strict:
                raise KeyError(f"No native token for log '{log_name}' at timestamp {timestamp_us}.")
            self._misses += 1
            return uuid
        return token


class NuScenesTokenResolver(MappingTokenResolver):
    """Recovers nuScenes ``sample_token`` values straight from the devkit database."""

    name = "nuscenes"

    def __init__(
        self,
        dataroot: Union[str, Path],
        versions: Optional[list] = None,
        strict: bool = False,
    ) -> None:
        """Initialize the resolver by indexing the nuScenes sample tables.

        :param dataroot: nuScenes data root (the folder holding ``v1.0-*`` and ``samples/``).
        :param versions: Database versions to index. Defaults to every ``v1.0-*`` folder present.
        :param strict: Raise when a frame has no matching keyframe sample. Leave ``False`` for
            interpolated (10 Hz) 123D logs, where non-keyframe frames legitimately have no token.
        """
        super().__init__(build_nuscenes_token_map(dataroot, versions), strict=strict)


def build_nuscenes_token_map(dataroot: Union[str, Path], versions: Optional[list] = None) -> Dict[str, str]:
    """Index a nuScenes install as ``"{scene_name}:{sample_timestamp_us}" -> sample_token``.

    :param dataroot: nuScenes data root.
    :param versions: Database versions to index. Defaults to every ``v1.0-*`` folder present.
    :return: The token mapping.
    """
    from nuscenes import NuScenes  # imported lazily: only needed when rebuilding native tokens

    dataroot = Path(dataroot)
    if versions is None:
        versions = sorted(path.name for path in dataroot.glob("v1.0-*") if path.is_dir())
    if not versions:
        raise FileNotFoundError(f"No nuScenes 'v1.0-*' database folder found under {dataroot}.")

    mapping: Dict[str, str] = {}
    for version in versions:
        nusc = NuScenes(version=version, dataroot=str(dataroot), verbose=False)
        scene_name_by_token = {scene["token"]: scene["name"] for scene in nusc.scene}
        for sample in nusc.sample:
            scene_name = scene_name_by_token[sample["scene_token"]]
            mapping[f"{scene_name}:{sample['timestamp']}"] = sample["token"]
        logger.info("Indexed %s: %d samples", version, len(nusc.sample))
    return mapping


def dump_nuscenes_token_map(dataroot: Union[str, Path], output_path: Union[str, Path]) -> int:
    """Write a nuScenes token map to JSON so exports can run without the devkit installed.

    :param dataroot: nuScenes data root.
    :param output_path: Destination JSON path.
    :return: The number of tokens written.
    """
    mapping = build_nuscenes_token_map(dataroot)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump({"format": "py123detection.token_map.v1", "tokens": mapping}, handle)
    return len(mapping)
