"""Rendering: the box geometry, the bird's-eye view, the camera overlay, and the CLI.

The one test here that earns its place above the others is that ``to_ego`` is the exact inverse of
the ``to_global`` evaluation applies. Everything a picture shows is placed by that pair, so if they
disagree the renders are quietly wrong in a way no amount of "it produced a PNG" would reveal — and
being wrong by a rotation is exactly the sort of thing that still looks like a plausible scene.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="the torch branch needs torch")
matplotlib = pytest.importorskip("matplotlib", reason="rendering needs the viz extra")
cv2 = pytest.importorskip("cv2", reason="the camera overlay needs cv2")

matplotlib.use("Agg")  # no display on a runner, and none wanted

#: Needs the torch extra; see the `torch` marker in pyproject.toml.
pytestmark = pytest.mark.torch

from py123detection.torch.evaluate import to_global  # noqa: E402
from py123detection.torch.predictions import FramePrediction, PredictionWriter  # noqa: E402
from py123detection.torch.visualize.bev import (  # noqa: E402
    PREDICTION_COLOUR,
    TRUTH_COLOUR,
    render_bev,
    save_bev,
)
from py123detection.torch.visualize.boxes import EDGES, corners, footprints, to_ego  # noqa: E402
from py123detection.torch.visualize.cameras import (  # noqa: E402
    draw_boxes,
    project_corners,
    render_camera,
    tile,
)
from py123detection.torch.visualize.cli import main  # noqa: E402

SEED = 20261109
CLASS_NAMES = ["vehicle", "pedestrian", "bicycle"]


def boxes(count: int = 4, rng=None) -> np.ndarray:
    rng = rng or np.random.default_rng(SEED)
    out = np.zeros((count, 9))
    out[:, 0] = rng.uniform(-30, 30, count)
    out[:, 1] = rng.uniform(-30, 30, count)
    out[:, 2] = rng.uniform(-2, 0, count)
    out[:, 3:6] = rng.uniform(1.5, 4.5, (count, 3))
    out[:, 6] = rng.uniform(-np.pi, np.pi, count)
    out[:, 7:9] = rng.uniform(-3, 3, (count, 2))
    return out


def pose(angle: float = 0.7, translation=(120.0, -40.0, 1.6)) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:2, :2] = [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    matrix[:3, 3] = translation
    return matrix


def frame(count: int = 4, with_pose: bool = True, scores=None, token: str = "scene-abc:7") -> FramePrediction:
    ego = pose()
    local = boxes(count)
    return FramePrediction(
        token=token,
        dataset="nuscenes",
        ego_translation=ego[:3, 3],
        ego_pose=ego if with_pose else None,
        boxes=to_global(local, ego),
        scores=np.linspace(0.95, 0.05, count) if scores is None else np.asarray(scores),
        labels=np.arange(count) % len(CLASS_NAMES),
        gt_boxes=to_global(boxes(3, np.random.default_rng(SEED + 1)), ego),
        gt_labels=np.arange(3) % len(CLASS_NAMES),
    )


# --------------------------------------------------------------------------------------------
# Box geometry
# --------------------------------------------------------------------------------------------


class TestCorners:
    def test_an_axis_aligned_box_has_the_corners_you_can_work_out_by_hand(self):
        #            x    y    z    w    l    h  yaw
        box = np.array([[1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 0.0]])
        got = corners(box)
        assert got.shape == (1, 8, 3)
        # l spans x (+/-2), w spans y (+/-1), h spans z (+/-3), about the centre.
        assert got[0, :, 0].min() == pytest.approx(-1.0) and got[0, :, 0].max() == pytest.approx(3.0)
        assert got[0, :, 1].min() == pytest.approx(1.0) and got[0, :, 1].max() == pytest.approx(3.0)
        assert got[0, :, 2].min() == pytest.approx(0.0) and got[0, :, 2].max() == pytest.approx(6.0)

    def test_a_quarter_turn_swaps_the_horizontal_extents(self):
        box = np.array([[0.0, 0.0, 0.0, 2.0, 6.0, 1.0, math.pi / 2]])
        got = corners(box)[0]
        assert got[:, 0].max() - got[:, 0].min() == pytest.approx(2.0)
        assert got[:, 1].max() - got[:, 1].min() == pytest.approx(6.0)

    def test_the_first_four_corners_are_the_bottom_face(self):
        got = corners(np.array([[0.0, 0.0, 0.0, 2.0, 4.0, 2.0, 0.3]]))[0]
        assert np.allclose(got[:4, 2], -1.0)
        assert np.allclose(got[4:, 2], 1.0)

    def test_the_edges_describe_a_box(self):
        """Every corner meets three edges, and no edge is repeated — that is a cuboid."""
        assert len(EDGES) == 12
        assert len({tuple(sorted(edge)) for edge in EDGES}) == 12
        for corner in range(8):
            assert sum(corner in edge for edge in EDGES) == 3

    def test_footprints_are_the_bottom_face_seen_from_above(self):
        box = boxes(3)
        assert footprints(box).shape == (3, 4, 2)
        assert np.allclose(footprints(box), corners(box)[:, :4, :2])

    def test_no_boxes_gives_empty_arrays_of_the_right_rank(self):
        assert corners(np.zeros((0, 9))).shape == (0, 8, 3)
        assert footprints(np.zeros((0, 9))).shape == (0, 4, 2)


class TestToEgo:
    def test_it_is_the_exact_inverse_of_the_conversion_evaluation_applies(self):
        """The test this module exists for.

        Every rendered box is placed by ``to_global`` on the way out and ``to_ego`` on the way back.
        A disagreement between them draws a scene that looks entirely plausible and is rotated.
        """
        original = boxes(12)
        ego = pose()
        restored = to_ego(to_global(original, ego), ego)
        assert np.allclose(restored, original, atol=1e-9), np.abs(restored - original).max()

    @pytest.mark.parametrize("angle", [-3.0, -0.5, 0.0, 1.2, 3.0])
    def test_it_inverts_at_any_heading(self, angle):
        original = boxes(6)
        ego = pose(angle=angle)
        assert np.allclose(to_ego(to_global(original, ego), ego), original, atol=1e-9)

    def test_a_box_at_the_vehicle_lands_at_the_origin(self):
        ego = pose()
        at_vehicle = np.zeros((1, 9))
        at_vehicle[0, 3:6] = 1.0
        placed = to_ego(to_global(at_vehicle, ego), ego)
        assert np.allclose(placed[0, :3], 0.0, atol=1e-9)

    def test_no_boxes_passes_through(self):
        assert to_ego(np.zeros((0, 9)), pose()).shape == (0, 9)


# --------------------------------------------------------------------------------------------
# Bird's-eye view
# --------------------------------------------------------------------------------------------


class TestBev:
    def outlines(self, ax, colour):
        """Patches drawn in one colour, which is stabler than a total: the ego vehicle adds a
        rectangle and an arrow, and decorations should not be able to break a box count."""
        from matplotlib.colors import to_rgba

        # RGB only: predictions carry their confidence as alpha, so a full comparison never
        # matches one.
        wanted = to_rgba(colour)[:3]
        return [
            patch for patch in ax.patches
            if getattr(patch, "get_edgecolor", None) and to_rgba(patch.get_edgecolor())[:3] == wanted
        ]

    def test_it_draws_ground_truth_predictions_and_the_vehicle(self):
        figure = render_bev(frame(), class_names=CLASS_NAMES, score_threshold=0.0)
        ax = figure.axes[0]
        assert len(self.outlines(ax, TRUTH_COLOUR)) == 3
        assert len(self.outlines(ax, PREDICTION_COLOUR)) == 4
        assert "nuscenes" in ax.get_title() and "scene-abc:7" in ax.get_title()
        matplotlib.pyplot.close(figure)

    def test_the_score_threshold_removes_boxes_and_the_title_says_how_many(self):
        low = render_bev(frame(), score_threshold=0.0)
        high = render_bev(frame(), score_threshold=0.5)
        assert len(self.outlines(high.axes[0], PREDICTION_COLOUR)) < len(
            self.outlines(low.axes[0], PREDICTION_COLOUR)
        )
        assert "below" in high.axes[0].get_title()
        matplotlib.pyplot.close(low)
        matplotlib.pyplot.close(high)

    def test_with_a_pose_the_view_is_vehicle_forward(self):
        figure = render_bev(frame(with_pose=True))
        assert "vehicle forward" in figure.axes[0].get_xlabel()
        matplotlib.pyplot.close(figure)

    def test_without_a_pose_it_still_renders_and_says_the_axes_are_global(self):
        """A file written before the pose was recorded must still be viewable."""
        figure = render_bev(frame(with_pose=False))
        assert "global" in figure.axes[0].get_xlabel()
        matplotlib.pyplot.close(figure)

    def test_a_frame_with_nothing_in_it_renders(self):
        empty = FramePrediction(
            token="t:0", dataset="wod", ego_translation=np.zeros(3), ego_pose=np.eye(4),
            boxes=np.zeros((0, 9)), scores=np.zeros(0), labels=np.zeros(0, dtype=int),
            gt_boxes=np.zeros((0, 9)), gt_labels=np.zeros(0, dtype=int),
        )
        figure = render_bev(empty)
        assert self.outlines(figure.axes[0], TRUTH_COLOUR) == []
        assert self.outlines(figure.axes[0], PREDICTION_COLOUR) == []
        matplotlib.pyplot.close(figure)

    def test_saving_writes_a_png_and_leaves_no_open_figure(self, tmp_path):
        import matplotlib.pyplot as plt

        before = len(plt.get_fignums())
        target = tmp_path / "bev.png"
        save_bev(frame(), target, class_names=CLASS_NAMES)
        assert target.exists() and target.stat().st_size > 1000
        assert len(plt.get_fignums()) == before, "a long loop would leak figures"


# --------------------------------------------------------------------------------------------
# Camera overlay
# --------------------------------------------------------------------------------------------


class TestCameraOverlay:
    def pinhole(self, focal=400.0, cx=320.0, cy=240.0) -> np.ndarray:
        matrix = np.eye(4)
        matrix[0, 0], matrix[0, 2] = focal, cx
        matrix[1, 1], matrix[1, 2] = focal, cy
        return matrix

    def in_front(self) -> np.ndarray:
        #                x    y    z    w    l    h  yaw
        return np.array([[0.0, 0.0, 20.0, 2.0, 4.0, 1.6, 0.0]])

    def test_a_box_ahead_projects_around_the_principal_point(self):
        pixels, valid = project_corners(corners(self.in_front()), self.pinhole(), (480, 640))
        assert pixels.shape == (1, 8, 2) and valid.shape == (1, 8)
        assert valid.all()
        assert pixels[0, :, 0].mean() == pytest.approx(320.0, abs=1.0)
        assert pixels[0, :, 1].mean() == pytest.approx(240.0, abs=1.0)

    def test_a_box_behind_the_camera_is_entirely_invalid(self):
        behind = self.in_front().copy()
        behind[0, 2] = -20.0
        _, valid = project_corners(corners(behind), self.pinhole(), (480, 640))
        assert not valid.any()

    def test_drawing_changes_pixels_where_the_box_is(self):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        drawn = draw_boxes(image, self.in_front(), self.pinhole())
        assert drawn.shape == image.shape and drawn.dtype == image.dtype
        assert (drawn != 0).any(), "nothing was drawn"
        assert not np.shares_memory(drawn, image), "the input was modified in place"

    def test_a_box_behind_the_camera_draws_nothing(self):
        behind = self.in_front().copy()
        behind[0, 2] = -20.0
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        assert np.array_equal(draw_boxes(image, behind, self.pinhole()), image)

    def test_the_score_threshold_reaches_the_camera_render(self):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        two = np.repeat(self.in_front(), 2, axis=0)
        two[1, 0] = 3.0
        many = render_camera(image, self.pinhole(), two, scores=[0.9, 0.9], score_threshold=0.5)
        few = render_camera(image, self.pinhole(), two, scores=[0.9, 0.1], score_threshold=0.5)
        assert (many != 0).sum() > (few != 0).sum() > 0

    def test_ground_truth_is_drawn_as_well_as_predictions(self):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        only_predictions = render_camera(image, self.pinhole(), self.in_front(), scores=[0.9])
        with_truth = render_camera(
            image, self.pinhole(), self.in_front(), ground_truth=self.in_front(), scores=[0.9]
        )
        assert (with_truth != 0).sum() >= (only_predictions != 0).sum()

    def test_tiling_handles_cameras_of_different_sizes(self):
        """Argoverse 2's rig does not share one resolution, so this is the normal case, not an edge."""
        images = [
            np.full((100, 160, 3), 10, np.uint8),
            np.full((80, 160, 3), 20, np.uint8),
            np.full((100, 120, 3), 30, np.uint8),
        ]
        grid = tile(images, columns=2)
        assert grid.ndim == 3 and grid.shape[2] == 3
        assert grid.shape[0] >= 100 and grid.shape[1] >= 160

    def test_tiling_nothing_is_an_error_rather_than_an_empty_image(self):
        with pytest.raises(ValueError, match="nothing to tile"):
            tile([])


# --------------------------------------------------------------------------------------------
# The command line
# --------------------------------------------------------------------------------------------


class TestCli:
    def written(self, tmp_path, frames) -> "object":
        path = tmp_path / "predictions.jsonl"
        with PredictionWriter(path) as writer:
            for item in frames:
                writer.write(item)
        return path

    def test_bev_renders_one_png_per_frame(self, tmp_path):
        source = self.written(tmp_path, [frame(token="a:1"), frame(token="b:2")])
        out = tmp_path / "pics"
        assert main(["bev", str(source), "-o", str(out), "--taxonomy", "nuscenes_10cls"]) == 0
        pngs = sorted(out.glob("*_bev.png"))
        assert len(pngs) == 2 and all(png.stat().st_size > 1000 for png in pngs)

    def test_the_limit_is_honoured(self, tmp_path):
        source = self.written(tmp_path, [frame(token=f"s:{index}") for index in range(5)])
        out = tmp_path / "pics"
        assert main(["bev", str(source), "-o", str(out), "--limit", "2"]) == 0
        assert len(list(out.glob("*.png"))) == 2

    def test_a_dataset_filter_selects_only_that_dataset(self, tmp_path):
        other = frame(token="other:1")
        other.dataset = "av2"
        source = self.written(tmp_path, [frame(token="mine:1"), other])
        out = tmp_path / "pics"
        assert main(["bev", str(source), "-o", str(out), "--dataset", "av2"]) == 0
        pngs = list(out.glob("*.png"))
        assert len(pngs) == 1 and pngs[0].name.startswith("av2_")

    def test_either_taxonomy_spelling_is_accepted(self, tmp_path):
        """``nuscenes_10cls`` is the config group; ``nuscenes_detection_10cls`` is the registry
        name. Someone who passes what they pass to ``py123det-train`` should not have to know."""
        source = self.written(tmp_path, [frame(token="t:1")])
        for spelling in ("nuscenes_10cls", "nuscenes_detection_10cls"):
            out = tmp_path / spelling
            assert main(["bev", str(source), "-o", str(out), "--taxonomy", spelling]) == 0
            assert len(list(out.glob("*.png"))) == 1

    def test_an_unknown_taxonomy_is_refused_with_a_usable_message(self, tmp_path):
        source = self.written(tmp_path, [frame(token="t:1")])
        with pytest.raises(SystemExit, match="neither a registry name nor a config group"):
            main(["bev", str(source), "-o", str(tmp_path / "x"), "--taxonomy", "nope"])

    def test_a_missing_file_is_reported_rather_than_traced(self, tmp_path):
        assert main(["bev", str(tmp_path / "nope.jsonl"), "-o", str(tmp_path / "out")]) == 1

    def test_the_rendered_frames_came_from_the_file(self, tmp_path):
        """A sanity check that the CLI reads the file rather than inventing frames."""
        source = self.written(tmp_path, [frame()])
        token = json.loads(source.read_text().splitlines()[0])["token"]
        out = tmp_path / "pics"
        main(["bev", str(source), "-o", str(out)])
        assert any(token.replace(":", "_") in png.name for png in out.glob("*.png"))
