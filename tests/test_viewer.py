"""The interactive viewer: the element py123d's viewer is given, not py123d's viewer itself.

Nothing here starts a server or loads a log. What is worth pinning is the part this package
actually wrote — which boxes are drawn for a given frame and score threshold, where they are
drawn relative to the scene centre, and the two ways a predictions file can fail to line up with
the scenes it names. py123d's viewer, playback and elements are py123d's to test.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("py123d", reason="the viewer element extends py123d's")
pytest.importorskip("viser", reason="py123d's viewer needs viser")

pytestmark = pytest.mark.torch

from py123detection.torch.predictions import FramePrediction  # noqa: E402
from py123detection.torch.visualize.viewer import (  # noqa: E402
    PredictionsElement,
    build_parser,
    check_alignment,
    frames_by_scene,
)

SCENE = "11111111-2222-3333-4444-555555555555"


def boxes(count: int, offset: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(7)
    out = np.zeros((count, 9), dtype=np.float64)
    out[:, 0:3] = rng.uniform(-30, 30, (count, 3)) + offset
    out[:, 3:6] = rng.uniform(1.0, 5.0, (count, 3))
    out[:, 6] = rng.uniform(-np.pi, np.pi, count)
    return out


def frame(index: int = 0, count: int = 4, scores=None, truth: int = 3) -> FramePrediction:
    return FramePrediction(
        token=f"{SCENE}:{index}",
        dataset="nuscenes",
        ego_translation=np.zeros(3),
        ego_pose=np.eye(4),
        boxes=boxes(count),
        scores=np.linspace(0.1, 0.9, count) if scores is None else np.asarray(scores, dtype=np.float64),
        labels=np.zeros(count, dtype=np.int64),
        gt_boxes=boxes(truth, offset=5.0),
        gt_labels=np.zeros(truth, dtype=np.int64),
    )


class StubHandle:
    def __init__(self, name) -> None:
        self.name = name
        self.visible = True
        self.removed = False

    def remove(self) -> None:
        self.removed = True


class StubControl:
    def __init__(self, value) -> None:
        self.value = value
        self.content = ""
        self.callbacks = []

    def on_update(self, callback) -> None:
        self.callbacks.append(callback)


class StubScene:
    def __init__(self) -> None:
        self.calls = []

    def add_line_segments(self, name, points, colors, line_width):
        self.calls.append({"name": name, "points": points, "colors": colors})
        return StubHandle(name)


class StubGui:
    def __init__(self) -> None:
        self.controls = {}

    def add_checkbox(self, label, value):
        return self.controls.setdefault(label, StubControl(value))

    def add_slider(self, label, min, max, step, initial_value):
        return self.controls.setdefault(label, StubControl(initial_value))

    def add_markdown(self, content):
        return self.controls.setdefault("markdown", StubControl(content))


class StubServer:
    def __init__(self) -> None:
        self.scene = StubScene()
        self.gui = StubGui()


def element(frames=None, origin=np.zeros(3), **options) -> tuple:
    frames = {0: frame()} if frames is None else frames
    item = PredictionsElement(frames, origin, **options)
    server = StubServer()
    item.create_gui(server)
    return item, server


class TestWhichBoxesAreDrawn:
    def test_the_score_slider_decides_how_many_predictions_are_drawn(self):
        item, server = element({0: frame(count=4, scores=[0.1, 0.4, 0.6, 0.9])})

        server.gui.controls["Score threshold"].value = 0.5
        item.update(0)

        assert server.scene.calls[-1]["points"].shape == (2 * 12, 2, 3)

    def test_a_threshold_of_zero_draws_every_query(self):
        item, server = element({0: frame(count=6, scores=np.full(6, 0.01))})

        server.gui.controls["Score threshold"].value = 0.0
        item.update(0)

        assert server.scene.calls[-1]["points"].shape == (6 * 12, 2, 3)

    def test_ground_truth_is_off_until_asked_for(self):
        item, server = element({0: frame(count=2, scores=[0.9, 0.9], truth=3)})
        item.update(0)
        assert [call["name"] for call in server.scene.calls] == ["py123detection/predictions"]

        server.gui.controls["Filtered ground truth"].value = True
        item.update(0)
        assert "py123detection/filtered_ground_truth" in [call["name"] for call in server.scene.calls]

    def test_hiding_the_element_leaves_nothing_visible(self):
        item, server = element()
        item.update(0)
        handles = list(item._handles.values())

        server.gui.controls["Visible"].value = False
        item.update(0)

        assert handles and all(not handle.visible for handle in handles)

    def test_colours_are_one_per_endpoint_not_one_per_box(self):
        """viser wants colors shaped exactly like points; a per-box array is twelve times short."""
        item, server = element({0: frame(count=3, scores=np.ones(3))})
        item.update(0)

        call = server.scene.calls[-1]
        assert call["colors"].shape == call["points"].shape


class TestWhereTheyAreDrawn:
    def test_the_scene_centre_is_subtracted_so_the_overlay_sits_with_the_scene(self):
        """py123d draws a scene relative to its centre; an overlay that skips this is far away."""
        origin = np.array([400000.0, 5800000.0, 20.0])
        at_origin, server_a = element({0: frame(count=2, scores=np.ones(2))})
        shifted, server_b = element({0: frame(count=2, scores=np.ones(2))}, origin=origin)

        at_origin.update(0)
        shifted.update(0)

        moved = server_a.scene.calls[-1]["points"] - server_b.scene.calls[-1]["points"]
        assert np.allclose(moved, origin)

    def test_a_longer_origin_is_truncated_to_three_numbers(self):
        """A caller holding a pose row rather than a point should not silently break broadcasting."""
        item, server = element({0: frame(count=1, scores=[1.0])}, origin=np.array([1.0, 2.0, 3.0, 1.0]))
        item.update(0)

        assert server.scene.calls[-1]["points"].shape == (12, 2, 3)


class TestWhenAFrameIsMissing:
    def test_nothing_is_drawn_and_the_reason_is_shown(self):
        item, server = element({0: frame()})
        item.update(0)

        item.update(7)

        assert "different frame stride" in server.gui.controls["markdown"].content

    def test_the_status_counts_what_passed_the_threshold(self):
        item, server = element({0: frame(count=4, scores=[0.1, 0.4, 0.6, 0.9], truth=3)})
        server.gui.controls["Score threshold"].value = 0.5
        item.update(0)

        assert server.gui.controls["markdown"].content == "2 of 4 predictions · 3 ground truth"


class Scene:
    def __init__(self, uuid, iterations, log_name="log") -> None:
        self.scene_uuid = uuid
        self.number_of_iterations = iterations
        self.log_name = log_name


class TestReadingTheFile:
    def test_frames_are_grouped_by_scene_and_keyed_by_iteration(self, tmp_path):
        from py123detection.torch.predictions import PredictionWriter

        path = tmp_path / "predictions.jsonl"
        with PredictionWriter(path) as writer:
            for index in (0, 1, 2):
                writer.write(frame(index))

        grouped = frames_by_scene(path)

        assert list(grouped) == [SCENE]
        assert sorted(grouped[SCENE]) == [0, 1, 2]

    def test_a_frame_past_the_end_of_the_scene_is_a_stride_mismatch(self):
        """Proof the two disagree: the run numbered strided frames 0, 1, 2, the log has more."""
        with pytest.raises(SystemExit, match="frame stride"):
            check_alignment([Scene(SCENE, 20)], {SCENE: {0: frame(0), 40: frame(40)}})

    def test_a_numbering_that_fits_is_accepted(self):
        check_alignment([Scene(SCENE, 40)], {SCENE: {0: frame(0), 39: frame(39)}})

    def test_a_scene_with_no_predictions_is_not_an_error(self):
        check_alignment([Scene("other", 10)], {SCENE: {0: frame(0)}})


class TestTheCommandLine:
    def test_the_stride_is_described_as_the_evaluations_not_the_logs(self):
        """The one setting that silently draws the wrong frame, so its help must say which it is."""
        text = build_parser().format_help()
        assert "EVALUATED" in text

    def test_it_takes_a_predictions_file_and_a_split(self):
        args = build_parser().parse_args(["a.jsonl", "--split", "nuscenes_val", "--port", "9000"])
        assert args.split == ["nuscenes_val"]
        assert args.port == 9000
