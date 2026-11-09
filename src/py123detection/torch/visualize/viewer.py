"""``py123det-view``: a run's predictions inside py123d's own viewer.

    py123det-view work_dirs/petr_nuscenes/eval/nuscenes/predictions.jsonl --split nuscenes_val

py123d ships a Viser viewer with lidar, map, camera frustums, ground truth and playback. This adds
one element to it — the model's boxes for the frame on screen — rather than writing a second
viewer, so everything else in the picture stays py123d's and keeps working when py123d improves it.

Two things have to line up for a prediction to land on the right frame, and both are checked
rather than assumed:

* **The scenes.** A predictions file names scenes by uuid, so the logs are looked up by uuid and a
  uuid the data root does not hold is reported, not silently dropped.
* **The frame numbering.** A token is ``<scene uuid>:<iteration>``, and the iteration is an index
  into the scene *as the evaluation built it*. A run scored at ``frame_stride=12`` numbered every
  twelfth frame 0, 1, 2, so viewing it against the native-rate log would put frame 1's boxes on the
  second frame of the log instead of the twelfth. Pass the same ``--frame-stride`` the run used;
  when the numbering cannot fit the scene, this says so instead of drawing something plausible.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import viser
from py123d.geometry.geometry_index import PoseSE3Index
from py123d.visualization.viser.elements.base_element import ElementContext, ViewerElement
from py123d.visualization.viser.viser_config import ViserConfig
from py123d.visualization.viser.viser_viewer import ViserViewer

from py123detection.torch.predictions import FramePrediction, frames_for
from py123detection.torch.visualize.py123d_interop import wireframe

logger = logging.getLogger(__name__)

#: Predictions, in the same red the still renderers use.
PREDICTION_COLOUR = (214, 39, 40)

#: Ground truth as this package filtered it, which is not always what the scene itself shows.
TRUTH_COLOUR = (90, 90, 90)


class PredictionsElement(ViewerElement):
    """One scene's predictions, drawn as wireframes over whatever else the viewer shows.

    Wireframes rather than solid boxes: py123d already draws the scene's own detections as shaded
    meshes, and a second set of meshes would hide them. An outline reads as an overlay.

    The frame's own ground truth can be drawn too. It is not the same thing as the scene's
    detections, which is the point of offering it — this is ground truth *after* the box filters
    the run was scored with, so a box the scene shows and this does not is a box the model was
    never charged for missing.
    """

    def __init__(
        self,
        frames: Dict[int, FramePrediction],
        origin: np.ndarray,
        score_threshold: float = 0.3,
        line_width: float = 3.0,
        show_ground_truth: bool = False,
    ) -> None:
        self._frames = frames
        self._origin = np.asarray(origin, dtype=np.float64)[:3]
        self._score_threshold = score_threshold
        self._line_width = line_width
        self._show_ground_truth = show_ground_truth
        self._server: Optional[viser.ViserServer] = None
        self._handles: Dict[str, Any] = {}
        self._iteration = 0
        self._visible: Any = None
        self._threshold: Any = None
        self._truth: Any = None
        self._status: Any = None

    @property
    def name(self) -> str:
        return "Predictions"

    def create_gui(self, server: viser.ViserServer) -> None:
        self._server = server
        self._visible = server.gui.add_checkbox("Visible", True)
        self._threshold = server.gui.add_slider(
            "Score threshold", min=0.0, max=1.0, step=0.01, initial_value=self._score_threshold
        )
        self._truth = server.gui.add_checkbox("Filtered ground truth", self._show_ground_truth)
        self._status = server.gui.add_markdown("")
        for control in (self._visible, self._threshold, self._truth):
            control.on_update(lambda _: self.update(self._iteration))

    def update(self, iteration: int) -> None:
        self._iteration = iteration
        frame = self._frames.get(iteration)

        if frame is None or not self._visible.value:
            self._hide()
            self._report(frame)
            return

        scores = np.asarray(frame.scores, dtype=np.float64)
        keep = scores >= float(self._threshold.value)
        self._draw("predictions", frame.boxes[keep], PREDICTION_COLOUR)
        self._draw(
            "filtered_ground_truth",
            frame.gt_boxes if self._truth.value else frame.gt_boxes[:0],
            TRUTH_COLOUR,
        )
        self._report(frame, drawn=int(keep.sum()))

    def remove(self) -> None:
        for handle in self._handles.values():
            handle.remove()
        self._handles = {}

    def on_dark_mode_changed(self, dark_mode: bool) -> None:
        self.update(self._iteration)

    def _draw(self, name: str, boxes: np.ndarray, colour: Tuple[int, int, int]) -> None:
        """Replace one set of wireframes.

        Positions are relative to the scene centre, exactly as py123d's own elements draw them:
        a scene sits at its UTM-like global coordinates, which are large enough that drawing them
        directly costs visible precision.
        """
        if len(boxes) == 0:
            handle = self._handles.pop(name, None)
            if handle is not None:
                handle.remove()
            return

        points = wireframe(boxes).reshape(-1, 2, 3)
        points = points - self._origin
        colours = np.broadcast_to(np.asarray(colour, dtype=np.uint8), points.shape).copy()
        self._handles[name] = self._server.scene.add_line_segments(
            f"py123detection/{name}",
            points=points.astype(np.float32),
            colors=colours,
            line_width=self._line_width,
        )

    def _hide(self) -> None:
        for handle in self._handles.values():
            handle.visible = False

    def _report(self, frame: Optional[FramePrediction], drawn: int = 0) -> None:
        if self._status is None:
            return
        if frame is None:
            self._status.content = (
                f"No prediction for frame {self._iteration}. "
                "If every frame says this, the run was scored at a different frame stride."
            )
            return
        self._status.content = (
            f"{drawn} of {len(frame.scores)} predictions · {len(frame.gt_labels)} ground truth"
        )


class PredictionViewer(ViserViewer):
    """py123d's viewer with one element added.

    Subclassing rather than copying: ``_build_elements`` is the hook py123d provides for exactly
    this, so lidar, map, cameras, ground truth and playback remain py123d's own and keep working
    as it changes them.
    """

    def __init__(
        self,
        scenes: Sequence,
        predictions: Dict[str, Dict[int, FramePrediction]],
        viser_config: Optional[ViserConfig] = None,
        score_threshold: float = 0.3,
        show_ground_truth: bool = False,
    ) -> None:
        # Set before super().__init__, which runs the first scene and therefore calls
        # _build_elements before returning.
        self._predictions = predictions
        self._score_threshold = score_threshold
        self._show_ground_truth = show_ground_truth
        super().__init__(scenes=list(scenes), viser_config=viser_config or ViserConfig())

    def _build_elements(self, context: ElementContext):
        manager = super()._build_elements(context)
        frames = self._predictions.get(context.scene.scene_uuid, {})
        if not frames:
            logger.warning("No predictions for scene %s", context.scene.scene_uuid)
        manager.register(
            PredictionsElement(
                frames,
                context.initial_ego_state.center_se3.array[PoseSE3Index.XYZ],
                score_threshold=self._score_threshold,
                show_ground_truth=self._show_ground_truth,
            )
        )
        return manager


def frames_by_scene(path, dataset: Optional[str] = None) -> Dict[str, Dict[int, FramePrediction]]:
    """A predictions file as ``scene uuid -> iteration -> frame``, which is how a viewer reads it."""
    grouped: Dict[str, Dict[int, FramePrediction]] = {}
    for frame in frames_for(path, dataset):
        scene, _, index = frame.token.rpartition(":")
        grouped.setdefault(scene, {})[int(index)] = frame
    return grouped


def load_scenes(
    uuids: Sequence[str],
    data_root: Optional[str] = None,
    splits: Optional[Sequence[str]] = None,
    frame_stride: Optional[int] = None,
    sample_rate_hz: Optional[float] = None,
    max_scenes: Optional[int] = None,
) -> List:
    """The 123D logs a predictions file refers to, built the way the evaluation built them.

    Goes through :class:`~py123detection.core.sources.Source` rather than assembling a
    ``SceneFilter`` here, so the scenes are resolved by the same code that resolved them for
    training and evaluation; only the uuid list is added on top.
    """
    from py123detection.core.sources import Source

    source = Source(
        data_root=data_root,
        splits=list(splits) if splits else None,
        frame_stride=frame_stride,
        sample_rate_hz=sample_rate_hz,
        max_logs=max_scenes,
    )
    source = replace(source, scene_filter=replace(source.build_scene_filter(), scene_uuids=list(uuids)))
    return source.load_scenes()


def check_alignment(scenes: Sequence, predictions: Dict[str, Dict[int, FramePrediction]]) -> None:
    """Fail loudly when the frame numbering cannot be the one the predictions were written with.

    A prediction whose iteration is past the end of the scene is proof that the two disagree. The
    reverse — a stride that happens to leave enough frames — cannot be detected from the file, so
    this is a floor, not a guarantee, and the error says which knob fixes it.
    """
    for scene in scenes:
        frames = predictions.get(scene.scene_uuid)
        if not frames:
            continue
        highest = max(frames)
        if highest >= scene.number_of_iterations:
            raise SystemExit(
                f"{scene.log_name}: the predictions go up to frame {highest}, but the scene built "
                f"here has {scene.number_of_iterations}. The run was scored at a different frame "
                f"stride; pass the one it used with --frame-stride or --sample-rate-hz."
            )


def build_parser() -> argparse.ArgumentParser:
    """The ``py123det-view`` command line."""
    parser = argparse.ArgumentParser(
        prog="py123det-view",
        description="Open a run's predictions in py123d's Viser viewer.",
    )
    parser.add_argument("predictions", type=Path, help="a predictions.jsonl from py123det-eval")
    parser.add_argument("--dataset", default=None, help="only frames of this dataset")
    parser.add_argument(
        "--split",
        action="append",
        default=None,
        help="123D split to look the logs up in, e.g. nuscenes_val. Repeatable. Without it every "
        "split under the data root is searched, which is slower.",
    )
    parser.add_argument(
        "--data-root", default=None, help="123D root holding logs/; defaults to $PY123D_DATA_ROOT"
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=None,
        help="the frame stride the run was EVALUATED with, which is what its frame numbering "
        "means. Default: every frame.",
    )
    parser.add_argument(
        "--sample-rate-hz", type=float, default=None, help="the evaluation's sample rate, instead of a stride"
    )
    parser.add_argument("--score-threshold", type=float, default=0.3, help="initial score slider position")
    parser.add_argument("--ground-truth", action="store_true", help="also draw the run's filtered ground truth")
    parser.add_argument("--max-scenes", type=int, default=None, help="open at most this many logs")
    parser.add_argument("--host", default="localhost", help="address to serve on; 0.0.0.0 to allow a tunnel")
    parser.add_argument("--port", type=int, default=8080, help="port to serve on")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    if not args.predictions.exists():
        logger.error("no predictions file at %s", args.predictions)
        return 1

    predictions = frames_by_scene(args.predictions, args.dataset)
    if not predictions:
        logger.error("no frames in %s%s", args.predictions, f" for dataset {args.dataset}" if args.dataset else "")
        return 1
    logger.info(
        "%d frame(s) over %d scene(s) in %s",
        sum(len(frames) for frames in predictions.values()), len(predictions), args.predictions,
    )

    scenes = load_scenes(
        list(predictions),
        data_root=args.data_root,
        splits=args.split,
        frame_stride=args.frame_stride,
        sample_rate_hz=args.sample_rate_hz,
        max_scenes=args.max_scenes,
    )
    if not scenes:
        logger.error(
            "none of the %d scene(s) in the file were found under the data root%s",
            len(predictions), f" in {args.split}" if args.split else "",
        )
        return 1

    missing = set(predictions) - {scene.scene_uuid for scene in scenes}
    if missing and args.max_scenes is None:
        logger.warning("%d scene(s) of the file are not under this data root", len(missing))
    check_alignment(scenes, predictions)

    config = ViserConfig()
    config.server.host = args.host
    config.server.port = args.port
    config.server.label = f"py123detection · {args.predictions.parent.name}"
    logger.info("serving on http://%s:%d", args.host, args.port)

    PredictionViewer(
        scenes,
        predictions,
        viser_config=config,
        score_threshold=args.score_threshold,
        show_ground_truth=args.ground_truth,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
