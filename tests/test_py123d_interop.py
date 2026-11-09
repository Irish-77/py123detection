"""Converting predictions into py123d's detection type.

The conversion is where this package's box layout meets py123d's, and they differ in more than
column order: ours carries yaw as an angle and sizes as ``w, l, h``, py123d a quaternion and
``length, width, height``. Swapping width and length still renders a perfectly plausible scene with
every car the wrong way round, so the geometry is not checked against hand-written expectations
alone — it is checked against py123d's own corner maths, which is an independent implementation of
the same thing.

The viser calls are exercised against a stub rather than a running server. What that covers is the
shape of what would be drawn; it does not cover the viewer actually opening, which needs 123D logs.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("py123d", reason="the conversion targets py123d's types")

#: Needs the torch extra: FramePrediction lives under py123detection.torch.
pytest.importorskip("torch", reason="the torch branch needs torch")
pytestmark = pytest.mark.torch

from py123detection.core.taxonomy import get_taxonomy  # noqa: E402
from py123detection.torch.predictions import FramePrediction  # noqa: E402
from py123detection.torch.visualize.boxes import corners as our_corners  # noqa: E402
from py123detection.torch.visualize.py123d_interop import (  # noqa: E402
    frames_by_token,
    label_lookup,
    to_box_array,
    to_detections,
    tokens_in_order,
    wireframe,
)

TAXONOMY = get_taxonomy("general_3cls")


def boxes(count: int = 5) -> np.ndarray:
    rng = np.random.default_rng(20261109)
    out = np.zeros((count, 9))
    out[:, 0:2] = rng.uniform(-40, 40, (count, 2))
    out[:, 2] = rng.uniform(-2, 0, count)
    out[:, 3:6] = rng.uniform(1.4, 4.8, (count, 3))
    out[:, 6] = rng.uniform(-np.pi, np.pi, count)
    return out


def frame(count: int = 5, scores=None) -> FramePrediction:
    box = boxes(count)
    return FramePrediction(
        token="scene-xyz:4",
        dataset="nuscenes",
        ego_translation=np.zeros(3),
        ego_pose=np.eye(4),
        boxes=box,
        scores=np.linspace(0.9, 0.1, count) if scores is None else np.asarray(scores),
        labels=np.arange(count) % len(TAXONOMY.class_names),
        gt_boxes=boxes(3),
        gt_labels=np.arange(3) % len(TAXONOMY.class_names),
    )


class TestBoxArray:
    def test_the_layout_is_py123ds(self):
        from py123d.geometry.geometry_index import BoundingBoxSE3Index as Index

        #                x    y    z    w    l    h   yaw
        box = np.array([[1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 0.0]])
        row = to_box_array(box)[0]
        assert row[Index.X] == pytest.approx(1.0)
        assert row[Index.Y] == pytest.approx(2.0)
        assert row[Index.Z] == pytest.approx(3.0)
        assert row[Index.LENGTH] == pytest.approx(4.0), "length is our l, not our w"
        assert row[Index.WIDTH] == pytest.approx(2.0)
        assert row[Index.HEIGHT] == pytest.approx(6.0)
        assert row[Index.QW] == pytest.approx(1.0)
        assert (row[Index.QX], row[Index.QY], row[Index.QZ]) == pytest.approx((0.0, 0.0, 0.0))

    @pytest.mark.parametrize("yaw", [-2.5, -0.4, 0.0, 1.1, 3.0])
    def test_yaw_becomes_a_rotation_about_z(self, yaw):
        row = to_box_array(np.array([[0.0, 0.0, 0.0, 2.0, 4.0, 1.5, yaw]]))[0]
        assert row[3] == pytest.approx(math.cos(yaw / 2))
        assert row[6] == pytest.approx(math.sin(yaw / 2))
        assert (row[4], row[5]) == pytest.approx((0.0, 0.0)), "no roll or pitch"

    def test_the_geometry_agrees_with_py123ds_own_corner_maths(self):
        """The test this module exists for.

        Two independent implementations of the same cuboid. If the layout conversion gets width and
        length the wrong way round, or the quaternion backwards, this is what notices — a rendered
        scene would not.
        """
        from py123d.geometry.utils.bounding_box_utils import bbse3_array_to_corners_array

        box = boxes(20)
        theirs = bbse3_array_to_corners_array(to_box_array(box))
        ours = our_corners(box)
        assert theirs.shape == ours.shape == (20, 8, 3)
        assert np.allclose(theirs, ours, atol=1e-9), np.abs(theirs - ours).max()

    def test_no_boxes_gives_an_empty_array_of_the_right_width(self):
        assert to_box_array(np.zeros((0, 9))).shape == (0, 10)


class TestLabels:
    def test_the_map_is_the_taxonomy_inverted(self):
        from py123d.datatypes.detections import DefaultBoxDetectionLabel

        lookup = label_lookup(TAXONOMY)
        assert lookup["vehicle"] is DefaultBoxDetectionLabel.VEHICLE
        assert lookup["pedestrian"] is DefaultBoxDetectionLabel.PERSON
        assert lookup["cyclist"] is DefaultBoxDetectionLabel.TWO_WHEELER

    def test_void_is_not_a_class_anything_maps_back_to(self):
        assert "void" not in label_lookup(TAXONOMY)

    def test_the_first_label_of_a_merged_class_wins(self):
        """``general_3cls`` sends both VEHICLE and TRAIN to ``vehicle``; going back is lossy."""
        from py123d.datatypes.detections import DefaultBoxDetectionLabel

        assert TAXONOMY.default_map["TRAIN"] == "vehicle"
        assert label_lookup(TAXONOMY)["vehicle"] is DefaultBoxDetectionLabel.VEHICLE

    @pytest.mark.parametrize(
        "name", ["general_3cls", "vehicle_1cls", "mixed2_3cls", "mixed3_1cls", "coin3d_3cls",
                 "default_123d"]
    )
    def test_a_coarse_taxonomy_maps_completely(self, name):
        from py123detection.core.taxonomy import get_taxonomy as lookup_taxonomy
        from py123detection.torch.visualize.py123d_interop import unmapped_classes

        assert unmapped_classes(lookup_taxonomy(name)) == []

    @pytest.mark.parametrize(
        "name,lost",
        [("mixed_8cls", {"truck", "bus", "trailer", "motorcycle"}),
         ("nuscenes_detection_10cls", None)],
    )
    def test_a_fine_taxonomy_cannot_round_trip_through_py123ds_labels(self, name, lost):
        """Not a defect here: py123d's vocabulary is VEHICLE and TWO_WHEELER, not car and bus.

        ``nuscenes_detection_10cls`` loses everything because it is built from ``native_map``,
        which does not name py123d labels at all. The boxes still draw; they lose their colour.
        """
        from py123detection.core.taxonomy import get_taxonomy as lookup_taxonomy
        from py123detection.torch.visualize.py123d_interop import unmapped_classes

        taxonomy = lookup_taxonomy(name)
        missing = set(unmapped_classes(taxonomy))
        assert missing == (lost if lost is not None else set(taxonomy.class_names) - {"void"})

    def test_an_unmapped_class_draws_as_a_generic_object_and_says_so(self, caplog):
        import logging

        from py123d.datatypes.detections import DefaultBoxDetectionLabel

        import py123detection.torch.visualize.py123d_interop as interop
        from py123detection.core.taxonomy import get_taxonomy as lookup_taxonomy

        interop._WARNED.clear()
        taxonomy = lookup_taxonomy("nuscenes_detection_10cls")
        source = frame(2)
        source.labels = np.zeros(2, dtype=int)
        with caplog.at_level(logging.WARNING):
            detections = to_detections(source, taxonomy)

        assert all(
            d.attributes.label is DefaultBoxDetectionLabel.GENERIC_OBJECT
            for d in detections.box_detections
        )
        assert "GENERIC_OBJECT" in caplog.text and "nuscenes_detection_10cls" in caplog.text

    def test_the_warning_is_not_repeated_for_every_frame(self, caplog):
        import logging

        import py123detection.torch.visualize.py123d_interop as interop
        from py123detection.core.taxonomy import get_taxonomy as lookup_taxonomy

        interop._WARNED.clear()
        taxonomy = lookup_taxonomy("mixed_8cls")
        with caplog.at_level(logging.WARNING):
            for _ in range(5):
                to_detections(frame(1), taxonomy)
        assert caplog.text.count("GENERIC_OBJECT") == 1, "stepping a scene would flood the log"


class TestDetections:
    def test_it_builds_py123ds_type_with_one_detection_per_box(self):
        from py123d.datatypes.detections import BoxDetectionsSE3

        detections = to_detections(frame(5), TAXONOMY)
        assert isinstance(detections, BoxDetectionsSE3)
        assert len(detections.box_detections) == 5

    def test_the_centres_survive_the_conversion(self):
        source = frame(4)
        detections = to_detections(source, TAXONOMY)
        for original, built in zip(source.boxes, detections.box_detections):
            centre = built.bounding_box_se3.center_se3
            assert (centre.x, centre.y, centre.z) == pytest.approx(tuple(original[:3]))

    def test_sizes_land_in_the_right_fields(self):
        source = frame(3)
        for original, built in zip(source.boxes, to_detections(source, TAXONOMY).box_detections):
            box = built.bounding_box_se3
            assert box.length == pytest.approx(original[4])
            assert box.width == pytest.approx(original[3])
            assert box.height == pytest.approx(original[5])

    def test_the_score_threshold_drops_boxes(self):
        source = frame(5, scores=[0.9, 0.8, 0.2, 0.1, 0.05])
        assert len(to_detections(source, TAXONOMY, score_threshold=0.5).box_detections) == 2
        assert len(to_detections(source, TAXONOMY, score_threshold=0.0).box_detections) == 5

    def test_ground_truth_is_converted_on_request_and_ignores_the_threshold(self):
        source = frame(5)
        truth = to_detections(source, TAXONOMY, ground_truth=True, score_threshold=0.99)
        assert len(truth.box_detections) == len(source.gt_boxes)

    def test_track_tokens_are_distinct_so_the_viewer_can_key_on_them(self):
        source = frame(6)
        tokens = [d.attributes.track_token for d in to_detections(source, TAXONOMY).box_detections]
        assert len(set(tokens)) == len(tokens)
        assert all(source.token in token for token in tokens)

    def test_predictions_and_ground_truth_do_not_collide(self):
        source = frame(3)
        predicted = {d.attributes.track_token for d in to_detections(source, TAXONOMY).box_detections}
        truth = {
            d.attributes.track_token
            for d in to_detections(source, TAXONOMY, ground_truth=True).box_detections
        }
        assert predicted.isdisjoint(truth)

    def test_an_empty_frame_converts_to_no_detections(self):
        empty = FramePrediction(
            token="t:0", dataset="wod", ego_translation=np.zeros(3), ego_pose=np.eye(4),
            boxes=np.zeros((0, 9)), scores=np.zeros(0), labels=np.zeros(0, dtype=int),
            gt_boxes=np.zeros((0, 9)), gt_labels=np.zeros(0, dtype=int),
        )
        assert to_detections(empty, TAXONOMY).box_detections == []


class TestWireframe:
    def test_each_box_becomes_twelve_segments(self):
        assert wireframe(boxes(7)).shape == (7, 12, 2, 3)

    def test_no_boxes_gives_an_empty_array_of_the_right_rank(self):
        assert wireframe(np.zeros((0, 9))).shape == (0, 12, 2, 3)

    def test_the_segments_span_the_box(self):
        box = np.array([[0.0, 0.0, 0.0, 2.0, 4.0, 1.5, 0.0]])
        points = wireframe(box).reshape(-1, 3)
        assert points[:, 0].min() == pytest.approx(-2.0) and points[:, 0].max() == pytest.approx(2.0)
        assert points[:, 1].min() == pytest.approx(-1.0) and points[:, 1].max() == pytest.approx(1.0)


class StubScene:
    def __init__(self) -> None:
        self.calls = []

    def add_line_segments(self, name, points, colors, line_width):
        self.calls.append({"name": name, "points": points, "colors": colors, "width": line_width})
        return f"handle:{name}"


class StubServer:
    def __init__(self) -> None:
        self.scene = StubScene()


class TestViserOverlay:
    def test_it_asks_the_server_for_the_right_shape_of_line_segments(self):
        from py123detection.torch.visualize.py123d_interop import add_to_viser

        server = StubServer()
        handle = add_to_viser(server, frame(4), score_threshold=0.0)

        assert handle == "handle:/predictions"
        call = server.scene.calls[0]
        assert call["points"].shape == (4 * 12, 2, 3)
        assert call["colors"].shape == call["points"].shape
        assert call["width"] == 3.0

    def test_the_threshold_reaches_the_overlay(self):
        from py123detection.torch.visualize.py123d_interop import add_to_viser

        server = StubServer()
        add_to_viser(server, frame(5, scores=[0.9, 0.8, 0.2, 0.1, 0.05]), score_threshold=0.5)
        assert server.scene.calls[0]["points"].shape == (2 * 12, 2, 3)


class TestFrameIndex:
    def test_frames_are_indexed_by_token(self, tmp_path):
        from py123detection.torch.predictions import PredictionWriter

        path = tmp_path / "p.jsonl"
        with PredictionWriter(path) as writer:
            for index in (0, 1, 2):
                item = frame(2)
                item.token = f"scene:{index}"
                writer.write(item)
        found = frames_by_token(path)
        assert sorted(found) == ["scene:0", "scene:1", "scene:2"]

    def test_tokens_sort_by_frame_number_not_as_text(self):
        """``scene:10`` after ``scene:2``. Sorting as strings makes a scene jump about."""
        tokens = {f"scene:{index}": None for index in (0, 2, 10, 1, 21, 3)}
        assert tokens_in_order(tokens) == ["scene:0", "scene:1", "scene:2", "scene:3",
                                           "scene:10", "scene:21"]

    def test_several_scenes_stay_grouped(self):
        tokens = {"b:2": None, "a:10": None, "a:2": None, "b:1": None}
        assert tokens_in_order(tokens) == ["a:2", "a:10", "b:1", "b:2"]
