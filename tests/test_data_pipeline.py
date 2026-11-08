"""The torch data pipeline, piece by piece, on synthetic logs.

The integration tests compare against the thesis dataset on real logs. These check the contracts
that comparison cannot: that the frame selection means what it says, that pixels and projection
matrices agree whatever the augmentation did, and that augmentation actually varies.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="the torch branch needs torch")

#: Needs the torch extra; see the `torch` marker in pyproject.toml.
pytestmark = pytest.mark.torch

from py123detection.core.sources import Source  # noqa: E402
from py123detection.core.taxonomy import get_taxonomy  # noqa: E402
from py123detection.torch.data import (  # noqa: E402
    Augmentation,
    BevTransform,
    BoxFilters,
    CameraPreprocessing,
    CameraRig,
    Draw,
    FrameIndex,
    ImageDraw,
    ImagePipeline,
    Py123DDetectionDataset,
    collate_rigs,
)
from py123detection.torch.data.cameras import decode_factor  # noqa: E402
from py123detection.torch.data.datamodule import _init_worker  # noqa: E402
from py123detection.torch.models.multiview import IMAGENET_MEAN, IMAGENET_STD, MultiViewDetector  # noqa: E402
from tests.fakes import MARKER, FakeCamera, FakeScene, look_along, marker_centroid  # noqa: E402


def logs(*lengths: int):
    return [SimpleNamespace(number_of_iterations=length) for length in lengths]


class TestFrameIndex:
    def test_without_a_count_every_frame_is_served_in_log_order(self) -> None:
        index = FrameIndex(logs(3, 2))
        assert [index[i] for i in range(len(index))] == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]

    def test_a_count_is_met_exactly(self) -> None:
        index = FrameIndex(logs(700, 300, 1234), num_frames=1000)
        assert len(index) == 1000
        assert index.available == 2234

    def test_the_count_is_spread_over_logs_in_proportion_to_their_length(self) -> None:
        index = FrameIndex(logs(600, 300, 100), num_frames=100)
        per_log = np.bincount([index[i][0] for i in range(len(index))], minlength=3)
        assert per_log.tolist() == [60, 30, 10]

    def test_an_even_divisor_is_the_same_as_a_stride(self) -> None:
        index = FrameIndex(logs(30, 30), num_frames=20)
        assert [index[i] for i in range(len(index))] == [(scene, frame) for scene in (0, 1) for frame in range(0, 30, 3)]

    def test_asking_for_more_than_exists_keeps_everything_and_says_so(self, caplog) -> None:
        with caplog.at_level(logging.WARNING):
            index = FrameIndex(logs(5, 5), num_frames=50, name="tiny")
        assert len(index) == 10
        assert "only 10 frame(s) exist" in caplog.text

    def test_a_non_positive_count_is_refused(self) -> None:
        with pytest.raises(ValueError):
            FrameIndex(logs(5), num_frames=0)

    def test_continuity_is_per_log(self) -> None:
        index = FrameIndex(logs(2, 2))
        assert [index.continues_scene(i) for i in range(4)] == [False, True, False, True]


class TestImageTransform:
    pipeline = ImagePipeline(size=(704, 1280))

    def test_validation_just_covers_the_input_and_keeps_the_bottom(self) -> None:
        transform = self.pipeline.resolve(ImageDraw(), 920, 1640)
        scale = transform.resize[0]
        assert scale == pytest.approx(max(704 / 920, 1280 / 1640))
        assert transform.crop[3] == int(920 * scale), "bottom-aligned"
        assert transform.size == (704, 1280)

    def test_zoom_is_relative_so_one_range_fits_every_input_size(self) -> None:
        for size in ((704, 1280), (320, 800)):
            pipeline = ImagePipeline(size=size)
            base = pipeline.resolve(ImageDraw(), 900, 1600).resize[0]
            zoomed = pipeline.resolve(ImageDraw(zoom=1.25), 900, 1600).resize[0]
            assert zoomed / base == pytest.approx(1.25)

    def test_a_flip_mirrors_about_the_last_pixel_like_cv2_flip(self) -> None:
        transform = self.pipeline.resolve(ImageDraw(flip=True), 704, 1280)
        matrix = transform.matrix()
        assert matrix @ np.array([0.0, 0.0, 1.0]) == pytest.approx([1279.0, 0.0, 1.0])
        assert matrix @ np.array([1279.0, 0.0, 1.0]) == pytest.approx([0.0, 0.0, 1.0])

    def test_stretch_fills_the_input_whatever_the_aspect(self) -> None:
        transform = ImagePipeline(size=(320, 800), mode="stretch").resolve(ImageDraw(zoom=2.0), 900, 1600)
        assert transform.resize == pytest.approx((0.5, 320 / 900))
        assert transform.crop == (0, 0, 800, 320)

    def test_decoding_is_reduced_only_while_nothing_is_lost(self) -> None:
        assert decode_factor(np.diag([0.8, 0.8, 1.0])) == 1
        assert decode_factor(np.diag([0.5, 0.4, 1.0])) == 2
        assert decode_factor(np.diag([0.3, 0.2, 1.0])) == 2
        assert decode_factor(np.diag([0.1, 0.1, 1.0])) == 8


# -- the dataset on a synthetic log -------------------------------------------------------------

CAMERAS = ("PCAM_F0", "PCAM_L0", "PCAM_R0")


def fake_scene(**kwargs) -> FakeScene:
    """A three-camera rig; the side cameras are portrait, so a rig of mixed sizes."""
    return FakeScene(
        [
            FakeCamera("PCAM_F0", 1600, 900, 1260.0, look_along(0.0)),
            FakeCamera("PCAM_L0", 900, 1600, 1100.0, look_along(0.5)),
            FakeCamera("PCAM_R0", 900, 1600, 1100.0, look_along(-0.5)),
        ],
        **kwargs,
    )


def fake_dataset(
    preprocessing=None, size=(320, 800), augment=None, scenes=None, **kwargs
) -> Py123DDetectionDataset:
    return Py123DDetectionDataset(
        source=Source.from_scenes(scenes or [fake_scene()], name="fake"),
        taxonomy=get_taxonomy("mixed3_1cls"),
        rig=CameraRig(cameras=CAMERAS, preprocessing=preprocessing),
        images=ImagePipeline(size=size),
        augment=augment,
        filters=BoxFilters(drop_frames_without_boxes=False),
        seed=0,
        **kwargs,
    )


def assert_marker_where_projected(sample, tolerance: float = 0.1) -> None:
    """Project the marker with the sample's own matrices; the blob must be there."""
    bev = BevTransform(angle=sample["img_metas"]["rotation"], scale=sample["img_metas"]["scale_factor"])
    marker = bev.forward_matrix().astype(np.float64) @ np.append(MARKER, 1.0)
    checked = 0
    for position in range(sample["img"].shape[0]):
        pixel = sample["imu2img"][position].double().numpy() @ marker
        u, v = pixel[0] / pixel[2], pixel[1] / pixel[2]
        height, width = (int(value) for value in sample["img_shape"])
        if not (15 < u < width - 15 and 15 < v < height - 15):
            continue  # a blob cut by the image edge has a biased centroid
        centroid = marker_centroid(sample["img"][position].numpy()[:, :height, :width])
        assert centroid is not None
        assert np.hypot(centroid[0] - u, centroid[1] - v) < tolerance, (position, (u, v), centroid)
        checked += 1
    assert checked > 0, "the marker should be visible in at least one camera"


PRESET = CameraPreprocessing(
    row_ranges={"PCAM_F0": (100, 900), "PCAM_L0": (300, 1500), "PCAM_R0": (300, 1500)},
    target_size=(1600, 960),
)


class TestPixelsAndMatricesAgree:
    """Wherever the pipeline moves the pixels, the matrices must have moved them the same way."""

    @pytest.mark.parametrize("preprocessing", [None, PRESET], ids=["own_sizes", "rig_preset"])
    @pytest.mark.parametrize("size", [(704, 1280), (320, 800)], ids=["704", "320"])
    def test_validation(self, preprocessing, size) -> None:
        assert_marker_where_projected(fake_dataset(preprocessing=preprocessing, size=size)[0])

    @pytest.mark.parametrize("preprocessing", [None, PRESET], ids=["own_sizes", "rig_preset"])
    @pytest.mark.parametrize(
        "draw",
        [
            Draw(image=ImageDraw(zoom=1.2, crop_x=0.9, flip=True)),
            Draw(image=ImageDraw(zoom=1.05, crop_x=0.1), bev=BevTransform(angle=0.3, scale=1.04)),
            Draw(image=ImageDraw(zoom=1.25, crop_x=0.5, flip=True), bev=BevTransform(angle=-0.35, scale=0.95)),
        ],
        ids=["flip", "rotate", "flip_and_rotate"],
    )
    def test_training(self, preprocessing, draw) -> None:
        dataset = fake_dataset(preprocessing=preprocessing, size=(704, 1280), augment=Augmentation())
        assert_marker_where_projected(dataset.get(0, draw))

    def test_a_reduced_decode_lands_in_the_same_place(self) -> None:
        scene = fake_scene()
        sample = fake_dataset(size=(320, 800), scenes=[scene])[0]
        assert 2 in scene.decode_scales, "320x800 from 1600 px is a 2x reduction"
        assert_marker_where_projected(sample, tolerance=0.15)


class TestDataset:
    def test_images_are_uint8_and_padded_to_the_divisor(self) -> None:
        sample = fake_dataset(size=(300, 790))[0]
        assert sample["img"].dtype == torch.uint8
        assert tuple(sample["img"].shape) == (3, 3, 320, 800)
        assert sample["img_shape"].tolist() == [300, 790]
        assert int(sample["img"][:, :, 300:, :].max()) == 0

    def test_num_frames_reaches_the_index(self) -> None:
        scenes = [fake_scene(iterations=40, uuid="a"), fake_scene(iterations=60, uuid="b")]
        dataset = fake_dataset(scenes=scenes, num_frames=10)
        assert len(dataset) == 10
        assert [dataset.frames[i] for i in (0, 3, 4, 5)] == [(0, 0), (0, 30), (1, 0), (1, 10)]

    def test_a_missing_camera_skips_the_frame_or_blanks_the_camera(self) -> None:
        scene = fake_scene(missing=[(0, "PCAM_L0")])
        skipped = fake_dataset(scenes=[scene])[0]
        assert skipped["img_metas"]["frame_idx"] == 1, "frame 0 lacks a camera, so frame 1 is served"

        blanked = fake_dataset(scenes=[fake_scene(missing=[(0, "PCAM_L0")])], missing_camera="zero")[0]
        assert blanked["img_metas"]["frame_idx"] == 0
        assert blanked["cam_valid_mask"].tolist() == [True, False, True]
        assert int(blanked["img"][1].max()) == 0
        assert torch.equal(blanked["intrinsics"][1], torch.eye(4))

    def test_each_epoch_draws_a_new_augmentation(self) -> None:
        dataset = fake_dataset(augment=Augmentation())
        first, second = dataset[0]["img_metas"], dataset[0]["img_metas"]
        assert first["rotation"] != second["rotation"]

    def test_the_same_seed_draws_the_same_augmentations(self) -> None:
        one, other = fake_dataset(augment=Augmentation()), fake_dataset(augment=Augmentation())
        one.reseed(7, 0)
        other.reseed(7, 0)
        assert [one.draw() for _ in range(5)] == [other.draw() for _ in range(5)]
        other.reseed(7, 1)
        assert one.draw() != other.draw(), "a different rank must draw a different stream"

    def test_validation_is_never_augmented(self) -> None:
        dataset = fake_dataset()
        assert dataset.draw() == Draw()
        assert dataset[0]["img_metas"]["rotation"] == 0.0

    def test_each_worker_reseeds_its_copy(self, monkeypatch) -> None:
        dataset = fake_dataset(augment=Augmentation())
        before = dataset.draw()
        info = SimpleNamespace(dataset=torch.utils.data.ConcatDataset([dataset]), seed=12345)
        monkeypatch.setattr("py123detection.torch.data.datamodule.get_worker_info", lambda: info)

        _init_worker(0, rank=3)
        reseeded = dataset.draw()
        dataset.reseed(12345, 3)
        assert dataset.draw() == reseeded
        assert reseeded != before


class TestNormalisationOnTheDevice:
    stub = SimpleNamespace(
        pixel_mean=torch.tensor(IMAGENET_MEAN).view(1, 1, 3, 1, 1),
        pixel_std=torch.tensor(IMAGENET_STD).view(1, 1, 3, 1, 1),
    )

    def test_it_matches_normalising_on_the_cpu_and_blanks_what_is_not_image(self) -> None:
        wide = fake_dataset(size=(300, 790))[0]
        narrow = dict(wide, img=wide["img"][:2], intrinsics=wide["intrinsics"][:2],
                      extrinsics=wide["extrinsics"][:2], imu2img=wide["imu2img"][:2],
                      cam_valid_mask=torch.tensor([True, False]))
        batch = collate_rigs([wide, narrow])
        result = MultiViewDetector.normalize_images(self.stub, batch)

        pixels = wide["img"].float()
        expected = (pixels - self.stub.pixel_mean[0]) / self.stub.pixel_std[0]
        assert torch.allclose(result[0, :, :, :300, :790], expected[:, :, :300, :790], atol=1e-5)
        assert float(result[0, :, :, 300:, :].abs().max()) == 0.0, "stride padding"
        assert float(result[0, :, :, :, 790:].abs().max()) == 0.0, "stride padding"
        assert float(result[1, 1:].abs().max()) == 0.0, "a missing camera and a padded one"

    def test_floating_point_input_is_taken_as_normalised(self) -> None:
        images = torch.randn(1, 2, 3, 8, 8)
        assert MultiViewDetector.normalize_images(self.stub, {"img": images}) is images


class TestBuildFromConfig:
    @staticmethod
    def built(monkeypatch, overrides):
        pytest.importorskip("hydra", reason="the torch branch needs hydra")
        from py123detection.torch import build
        from tests.test_configs import composed

        captured = []
        monkeypatch.setattr(build, "Py123DDetectionDataset", lambda **kwargs: captured.append(kwargs) or [])
        build.build_datasets(composed(["experiment=nuscenes", *overrides]), "train")
        return captured[0]

    def test_frame_stride_and_num_frames_reach_the_dataset(self, monkeypatch) -> None:
        arguments = self.built(
            monkeypatch, ["~data.train.0.sample_rate_hz", "+data.train.0.frame_stride=3", "+data.train.0.num_frames=10"]
        )
        assert arguments["source"].frame_stride == 3
        assert arguments["source"].sample_rate_hz is None
        assert arguments["num_frames"] == 10
        assert arguments["augment"] == Augmentation()

    def test_a_rate_and_a_stride_together_are_refused(self, monkeypatch) -> None:
        with pytest.raises(ValueError, match="choose one"):
            self.built(monkeypatch, ["+data.train.0.frame_stride=3"])


class TestLidarPointCache:
    """Point counts from a standalone cache, for logs that record none (Waymo in 123D)."""

    @staticmethod
    def write_log(tmp_path, counts_by_row):
        import uuid

        import pyarrow as pa
        import pyarrow.ipc as ipc

        uuids = [uuid.uuid4().bytes for _ in range(len(counts_by_row))]
        log = ("wod_split", "log_a")
        for root, table in (
            ("logs", pa.table({"sync.uuid": pa.array(uuids, type=pa.binary(16))})),
            (
                "cache",
                pa.table({
                    "sync_uuid": pa.array(uuids, type=pa.binary(16)),
                    "track_tokens": [list(counts) for counts in counts_by_row],
                    "num_lidar_points": [list(counts.values()) for counts in counts_by_row],
                }),
            ),
        ):
            directory = tmp_path / root / log[0] / log[1]
            directory.mkdir(parents=True)
            name = "sync.arrow" if root == "logs" else "num_lidar_points.arrow"
            with ipc.new_file(str(directory / name), table.schema) as writer:
                writer.write_table(table)
        return log

    def test_a_strided_scene_reads_the_right_sync_row(self, tmp_path) -> None:
        from py123detection.core.lidar_points import LidarPointCounts

        rows = [{"car": row, "bus": 10 * row} for row in range(12)]
        split, log_name = self.write_log(tmp_path, rows)
        scene = SimpleNamespace(
            log_metadata=SimpleNamespace(split=split, log_name=log_name),
            scene_metadata=SimpleNamespace(initial_idx=2, target_iteration_stride=3),
        )
        counts = LidarPointCounts(tmp_path / "cache", tmp_path / "logs")
        assert counts.for_frame(scene, 0) == {"car": 2, "bus": 20}
        assert counts.for_frame(scene, 3) == {"car": 11, "bus": 110}

    def test_a_log_without_a_cache_keeps_its_own_counts(self, tmp_path, caplog) -> None:
        from py123detection.core.lidar_points import LidarPointCounts

        scene = SimpleNamespace(
            log_metadata=SimpleNamespace(split="s", log_name="missing"),
            scene_metadata=SimpleNamespace(initial_idx=0, target_iteration_stride=1),
        )
        with caplog.at_level(logging.WARNING):
            assert LidarPointCounts(tmp_path / "cache", tmp_path / "logs").for_frame(scene, 0) == {}
        assert "No lidar point cache" in caplog.text
