"""What goes into ``info['token']``.

123D stamps every frame with a deterministic UUID derived from ``(split, log_name, timestamp_us)``,
which is enough for training but is not the nuScenes ``sample_token`` that the official nuScenes
evaluation keys on. :class:`NuScenesTokenResolver` restores those by joining on
``(log_name, timestamp_us)``. For datasets without native frame tokens,
:class:`LogTimestampTokenResolver` writes ``"{log_name}/{timestamp_us}"``, which an independent
converter can reproduce.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class TokenResolver:
    """Default resolver: the 123D frame UUID."""

    name = "uuid"  # recorded in the exported metadata

    def resolve(self, dataset: str, split: str, log_name: str, timestamp_us: int, uuid: str) -> str:
        return uuid


class LogTimestampTokenResolver(TokenResolver):
    name = "log_timestamp"

    def resolve(self, dataset: str, split: str, log_name: str, timestamp_us: int, uuid: str) -> str:
        return f"{log_name}/{int(timestamp_us)}"


class MappingTokenResolver(TokenResolver):
    """Looks tokens up in a ``"{log_name}:{timestamp_us}" -> token`` mapping.

    Frames missing from the mapping fall back to the 123D UUID, or raise when ``strict`` is set.
    """

    name = "mapping"

    def __init__(self, mapping: Dict[str, str], strict: bool = False) -> None:
        self.mapping = mapping
        self.strict = strict
        self.num_misses = 0

    @classmethod
    def from_json(cls, path: Union[str, Path], strict: bool = False) -> MappingTokenResolver:
        """Load a mapping written by :func:`dump_nuscenes_token_map`."""
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls(payload["tokens"], strict=strict)

    def resolve(self, dataset: str, split: str, log_name: str, timestamp_us: int, uuid: str) -> str:
        token = self.mapping.get(f"{log_name}:{timestamp_us}")
        if token is None:
            if self.strict:
                raise KeyError(f"No native token for log '{log_name}' at timestamp {timestamp_us}.")
            self.num_misses += 1
            return uuid
        return token


class NuScenesTokenResolver(MappingTokenResolver):
    """Restores nuScenes ``sample_token`` values from the devkit database.

    Leave ``strict`` off for interpolated (10 Hz) 123D logs, whose non-keyframes have no token.
    """

    name = "nuscenes"

    def __init__(self, dataroot: Union[str, Path], versions: Optional[List[str]] = None, strict: bool = False) -> None:
        super().__init__(build_nuscenes_token_map(dataroot, versions), strict=strict)


def build_nuscenes_token_map(dataroot: Union[str, Path], versions: Optional[List[str]] = None) -> Dict[str, str]:
    """Index a nuScenes install as ``"{scene_name}:{sample_timestamp_us}" -> sample_token``.

    :param versions: Database versions to index. Defaults to every ``v1.0-*`` folder present.
    """
    from nuscenes import NuScenes  # optional dependency

    dataroot = Path(dataroot)
    if versions is None:
        versions = sorted(path.name for path in dataroot.glob("v1.0-*") if path.is_dir())
    if not versions:
        raise FileNotFoundError(f"No nuScenes 'v1.0-*' database folder found under {dataroot}.")

    mapping: Dict[str, str] = {}
    for version in versions:
        nusc = NuScenes(version=version, dataroot=str(dataroot), verbose=False)
        scene_names = {scene["token"]: scene["name"] for scene in nusc.scene}
        for sample in nusc.sample:
            mapping[f"{scene_names[sample['scene_token']]}:{sample['timestamp']}"] = sample["token"]
        logger.info("Indexed %s: %d samples", version, len(nusc.sample))
    return mapping


def dump_nuscenes_token_map(dataroot: Union[str, Path], output_path: Union[str, Path]) -> int:
    """Write the token map to JSON so exports run without the devkit. Returns the token count."""
    mapping = build_nuscenes_token_map(dataroot)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump({"format": "py123detection.token_map.v1", "tokens": mapping}, handle)
    return len(mapping)
