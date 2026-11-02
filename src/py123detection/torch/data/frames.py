"""Which frames of the resolved logs a dataset serves.

Two layers decide it. The :class:`~py123detection.core.sources.Source` thins each log while it is
read — ``sample_rate_hz`` or ``frame_stride`` — which is the right tool when the rate itself
matters: 2 Hz is nuScenes' keyframe rate, and a 10 Hz log sampled at 2 Hz is comparable to it.

:class:`FrameIndex` then optionally takes a fixed number of those frames, spread evenly across all
logs. That is the right tool when the *amount* matters: ``num_frames: 10000`` gives ten thousand
samples without first counting the split and dividing, and gives every dataset in a mix the same
weight regardless of how long or how densely recorded its logs are.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class FrameIndex:
    """The ``(scene, iteration)`` pairs behind a dataset's indices, in log order.

    :param scenes: The resolved logs, already thinned by their source.
    :param num_frames: Keep this many frames, evenly spaced over the concatenation of all logs, so
        each log contributes in proportion to its length. ``None`` keeps every frame. Asking for
        more frames than there are keeps them all and says so.
    :param name: Used in log messages only.
    """

    def __init__(self, scenes: Sequence, num_frames: Optional[int] = None, name: str = "") -> None:
        lengths = np.array([scene.number_of_iterations for scene in scenes], dtype=np.int64)
        scene_ids = np.repeat(np.arange(len(lengths), dtype=np.int64), lengths)
        starts = np.concatenate([[0], np.cumsum(lengths)[:-1]]) if len(lengths) else np.zeros(0, np.int64)
        iterations = np.arange(int(lengths.sum()), dtype=np.int64) - np.repeat(starts, lengths)

        #: Frames available before ``num_frames`` was applied.
        self.available = int(lengths.sum())

        if num_frames is not None:
            if num_frames <= 0:
                raise ValueError(f"num_frames must be positive, got {num_frames}.")
            if num_frames < self.available:
                # floor(k * M / N): exact integer spacing, the first frame always included, and the
                # same frames as frame_stride=M/N whenever that divides evenly.
                positions = np.arange(num_frames, dtype=np.int64) * self.available // num_frames
                scene_ids, iterations = scene_ids[positions], iterations[positions]
            elif num_frames > self.available:
                logger.warning(
                    "%s: num_frames=%d asked for, but only %d frame(s) exist; using all of them.",
                    name or "dataset", num_frames, self.available,
                )

        self._scene_ids = scene_ids
        self._iterations = iterations

    def __len__(self) -> int:
        return int(self._scene_ids.shape[0])

    def __getitem__(self, index: int) -> Tuple[int, int]:
        """``(scene_index, iteration)`` of one dataset index."""
        return int(self._scene_ids[index]), int(self._iterations[index])

    def continues_scene(self, index: int) -> bool:
        """Whether ``index - 1`` is an earlier frame of the same log, for a temporal model."""
        return index > 0 and self._scene_ids[index - 1] == self._scene_ids[index]
