"""Predictions on disk, written one frame at a time and read back the same way.

Evaluation used to hold every frame's boxes in memory until the whole run was scored, which on a
three-dataset validation pass meant tens of millions of box objects and, in one case, an
OOM kill at 67 GB. Writing each frame out as it is predicted keeps the peak at one frame, and
makes the far more useful thing possible: scoring a checkpoint again — with a different class
range, a different true-positive set, a different box cap — without running the model again.

The format is JSON Lines, one object per frame, with the boxes as arrays rather than one object per
box::

    {"token": "<scene uuid>:<frame>", "dataset": "nuscenes", "ego_translation": [x, y, z],
     "boxes": [[x, y, z, w, l, h, yaw, vx, vy], ...], "scores": [...], "labels": [...],
     "gt_boxes": [...], "gt_labels": [...]}

Arrays rather than per-box objects because a frame carries up to 900 predictions and the nuScenes
result format spends about ten keys on each of them. Ground truth travels with the predictions so a
file is enough to re-score on its own; it is a few boxes a frame against hundreds of predictions, so
it costs almost nothing.

Boxes are in the **global** frame, already converted, because that is what the metric measures in
and converting once is what keeps predictions and ground truth honest with each other.

This is not the nuScenes submission format. :func:`py123detection.core.evaluation.load_predictions`
still reads that, for a file produced elsewhere.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

import numpy as np

#: Columns of a box: ``[x, y, z, w, l, h, yaw, vx, vy]``.
BOX_COLUMNS = 9


@dataclass
class FramePrediction:
    """One frame's predictions and ground truth, in the global frame."""

    token: str
    dataset: str
    ego_translation: np.ndarray
    """``(3,)`` ego position, which the metric turns into each box's distance from the vehicle."""
    boxes: np.ndarray
    """``(N, 9)`` predicted boxes."""
    scores: np.ndarray
    """``(N,)`` confidences."""
    labels: np.ndarray
    """``(N,)`` class indices into the taxonomy's ``class_names``."""
    gt_boxes: np.ndarray
    """``(M, 9)`` ground-truth boxes."""
    gt_labels: np.ndarray
    """``(M,)`` class indices."""
    ego_pose: Optional[np.ndarray] = None
    """``(4, 4)`` ego-to-global pose, when the writer had it.

    The metric needs only the translation, to measure each box's distance from the vehicle. A
    bird's-eye view wants the rotation too, so it can draw the scene the way the vehicle was facing
    rather than aligned to global north. Optional so a file written before this field still reads.
    """

    def as_json(self) -> Dict[str, Any]:
        """A plain dict, with full float precision so a re-score reproduces the first one exactly."""
        return {
            "token": self.token,
            "dataset": self.dataset,
            "ego_translation": np.asarray(self.ego_translation, dtype=np.float64).tolist(),
            "boxes": np.asarray(self.boxes, dtype=np.float64).tolist(),
            "scores": np.asarray(self.scores, dtype=np.float64).tolist(),
            "labels": np.asarray(self.labels, dtype=np.int64).tolist(),
            "gt_boxes": np.asarray(self.gt_boxes, dtype=np.float64).tolist(),
            "gt_labels": np.asarray(self.gt_labels, dtype=np.int64).tolist(),
            **(
                {"ego_pose": np.asarray(self.ego_pose, dtype=np.float64).tolist()}
                if self.ego_pose is not None
                else {}
            ),
        }

    @classmethod
    def from_json(cls, content: Dict[str, Any]) -> "FramePrediction":
        return cls(
            token=content["token"],
            dataset=content["dataset"],
            ego_translation=np.asarray(content["ego_translation"], dtype=np.float64),
            boxes=_rows(content["boxes"]),
            scores=np.asarray(content["scores"], dtype=np.float64),
            labels=np.asarray(content["labels"], dtype=np.int64),
            gt_boxes=_rows(content["gt_boxes"]),
            gt_labels=np.asarray(content["gt_labels"], dtype=np.int64),
            ego_pose=(
                np.asarray(content["ego_pose"], dtype=np.float64) if "ego_pose" in content else None
            ),
        )


def _rows(value: Sequence[Any]) -> np.ndarray:
    """An ``(n, 9)`` array, keeping the second axis when there are no rows at all."""
    array = np.asarray(value, dtype=np.float64)
    return array.reshape(0, BOX_COLUMNS) if array.size == 0 else array


class PredictionWriter:
    """Append frames to a JSON Lines file, one line at a time.

    Used as a context manager so a crashed run still leaves a readable file of everything predicted
    up to that point, which is what makes a failed evaluation diagnosable instead of just lost.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.frames = 0
        self._handle = None

    def __enter__(self) -> "PredictionWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", encoding="utf-8")
        return self

    def write(self, frame: FramePrediction) -> None:
        json.dump(frame.as_json(), self._handle, separators=(",", ":"))
        self._handle.write("\n")
        self.frames += 1

    def __exit__(self, *exc: Any) -> None:
        if self._handle is not None:
            self._handle.flush()
            self._handle.close()
            self._handle = None


def read_predictions(path: Path) -> Iterator[FramePrediction]:
    """Stream a predictions file back, one frame at a time."""
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield FramePrediction.from_json(json.loads(line))


def dataset_names(path: Path) -> List[str]:
    """Which datasets a predictions file covers, in the order they first appear."""
    seen: Dict[str, None] = {}
    for frame in read_predictions(path):
        seen.setdefault(frame.dataset, None)
    return list(seen)


def frames_for(path: Path, dataset: Optional[str] = None) -> Iterable[FramePrediction]:
    """Frames of one dataset, or all of them when ``dataset`` is None."""
    for frame in read_predictions(path):
        if dataset is None or frame.dataset == dataset:
            yield frame


def rank_files(directory: Path) -> List[Path]:
    """Per-rank prediction files in a stable order, so a merge is reproducible."""
    return sorted(Path(directory).glob("predictions_rank*.jsonl"))


def merge_rank_files(paths: Sequence[Path], out: Path) -> Dict[str, int]:
    """Concatenate per-rank prediction files into one, keeping each frame once.

    Deduplication is not tidiness, it is correctness. ``DistributedSampler`` pads the last batch so
    every rank sees the same number of samples, which means the first few frames of a split are
    handed to two ranks and would otherwise be scored twice. The dataset can repeat a token on its
    own account too: when a frame is rejected by the box filters it walks forward to the next usable
    one, so two indices can return the same frame.

    Scoring a frame twice inflates both its hits and its misses, and scoring a subset inflates
    recall — neither shows up as an error, so the counts are returned for the caller to log.

    :returns: ``{"frames": kept, "duplicates": dropped}``.
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    seen: Dict[str, None] = {}
    duplicates = 0

    with out.open("w", encoding="utf-8") as handle:
        for path in paths:
            with Path(path).open("r", encoding="utf-8") as source:
                for line in source:
                    line = line.strip()
                    if not line:
                        continue
                    token = json.loads(line)["token"]
                    if token in seen:
                        duplicates += 1
                        continue
                    seen[token] = None
                    handle.write(line + "\n")

    return {"frames": len(seen), "duplicates": duplicates}
