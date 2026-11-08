"""The parts of the training loop that are easy to get quietly wrong.

Per-dataset loss weighting, the distributed box count, and the parameter groups all produce a
number that looks plausible whatever they do, so none of them fails loudly when wrong.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="the torch branch needs torch")

#: Needs the torch extra; see the `torch` marker in pyproject.toml.
pytestmark = pytest.mark.torch

from py123detection.torch.models.base import Detector3D  # noqa: E402


class StubHead(torch.nn.Module):
    """A head whose loss does not depend on how many samples it is given.

    Real losses normalise by the synchronised box count, so splitting a batch does not change
    what each group reports. This stub has the same property, which is what makes the weighted
    sum comparable to the unsplit loss.
    """

    def __init__(self, per_sample: bool = False) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(4, 4))
        self.bias = torch.nn.Parameter(torch.zeros(4))
        self.seen = []
        self.per_sample = per_sample

    def loss(self, boxes, labels, predictions, **kwargs):
        self.seen.append(len(labels))
        scale = float(len(labels)) if self.per_sample else 1.0
        return {"loss_ce": torch.tensor(2.0 * scale), "loss_bbox": torch.tensor(1.0 * scale)}


class StubDetector(Detector3D):
    def __init__(self, per_sample: bool = False) -> None:
        super().__init__()
        self.backbone = torch.nn.Linear(4, 4)
        self.head = StubHead(per_sample=per_sample)

    def forward(self, batch):
        return {"all_cls_scores": torch.zeros(2, len(batch["gt_labels_3d"]), 3, 2)}


def batch_of(dataset_names):
    count = len(dataset_names)
    return {
        "gt_bboxes_3d": [torch.zeros(1, 9) for _ in range(count)],
        "gt_labels_3d": [torch.zeros(1, dtype=torch.long) for _ in range(count)],
        "img_metas": [{"dataset_name": name} for name in dataset_names],
    }


class TestDatasetGrouping:
    def test_one_dataset_is_a_single_group(self) -> None:
        groups = Detector3D.group_by_dataset(batch_of(["nuscenes"] * 4))
        assert list(groups.values()) == [[0, 1, 2, 3]]

    def test_samples_are_grouped_by_their_dataset(self) -> None:
        groups = Detector3D.group_by_dataset(batch_of(["nuscenes", "av2", "nuscenes", "wod"]))
        assert groups == {"nuscenes": [0, 2], "av2": [1], "wod": [3]}

    def test_a_batch_without_metadata_still_forms_one_group(self) -> None:
        """A bare tensor batch has no dataset names, and must not crash the loss."""
        groups = Detector3D.group_by_dataset(
            {"gt_labels_3d": [torch.zeros(1)] * 3, "img_metas": None}
        )
        assert list(groups.values()) == [[0, 1, 2]]

    def test_every_sample_lands_in_exactly_one_group(self) -> None:
        names = ["a", "b", "a", "c", "b", "a"]
        groups = Detector3D.group_by_dataset(batch_of(names))
        assigned = sorted(index for indices in groups.values() for index in indices)
        assert assigned == list(range(len(names)))


class TestGroupedLossWeighting:
    """Splitting a batch by dataset must not change the loss, only make it readable."""

    def test_splitting_a_batch_does_not_change_the_loss(self) -> None:
        model = StubDetector()
        model.trainer = None

        mixed = batch_of(["nuscenes", "nuscenes", "av2", "av2"])
        single = batch_of(["nuscenes"] * 4)

        grouped = model._loss(mixed, model(mixed), stage="train", sync=False)
        ungrouped = model._loss(single, model(single), stage="train", sync=False)

        assert model.head.seen == [2, 2, 4], "the mixed batch should reach the head once per dataset"
        assert grouped == pytest.approx(ungrouped)

    def test_an_uneven_split_weights_by_share(self) -> None:
        """Three of one dataset and one of another is still one batch, not two equal halves."""
        model = StubDetector(per_sample=True)
        model.trainer = None
        batch = batch_of(["nuscenes", "nuscenes", "nuscenes", "av2"])

        total = model._loss(batch, model(batch), stage="train", sync=False)

        # Groups report 3 * 3 and 1 * 3, weighted by 3/4 and 1/4.
        assert total == pytest.approx(9.0 * 0.75 + 3.0 * 0.25)

    def test_an_empty_group_never_appears(self) -> None:
        model = StubDetector()
        model.trainer = None
        batch = batch_of(["nuscenes", "av2"])
        model._loss(batch, model(batch), stage="train", sync=False)
        assert all(count > 0 for count in model.head.seen)


class TestBoxCount:
    def test_counts_every_box_in_the_batch(self) -> None:
        labels = [torch.zeros(3), torch.zeros(5), torch.zeros(0)]
        assert Detector3D.sync_box_count(labels, torch.device("cpu")).item() == pytest.approx(8.0)

    def test_never_returns_zero(self) -> None:
        """The loss divides by this, so a batch with no boxes must not produce a division by zero."""
        count = Detector3D.sync_box_count([torch.zeros(0)], torch.device("cpu"))
        assert count.item() == pytest.approx(1.0)


class TestParameterGroups:
    def test_backbone_trains_slower_than_the_head(self) -> None:
        """A pretrained trunk under a randomly initialised head wants a lower rate."""
        model = StubDetector()
        model.learning_rate = 1e-3
        by_name = {group["name"]: group for group in model.parameter_groups()}

        assert by_name["backbone_decay"]["lr"] == pytest.approx(1e-4)
        assert by_name["head_decay"]["lr"] == pytest.approx(1e-3)

    def test_norms_and_biases_escape_weight_decay(self) -> None:
        """Decaying a bias changes what the layer computes rather than regularising it."""
        model = StubDetector()
        model.weight_decay = 0.05
        by_name = {group["name"]: group for group in model.parameter_groups()}

        assert by_name["head_no_decay"]["weight_decay"] == 0.0
        assert by_name["head_decay"]["weight_decay"] == pytest.approx(0.05)
        assert all(p.ndim == 1 for p in by_name["head_no_decay"]["params"])

    def test_every_trainable_parameter_is_in_exactly_one_group(self) -> None:
        model = StubDetector()
        grouped = [id(p) for group in model.parameter_groups() for p in group["params"]]
        trainable = [id(p) for p in model.parameters() if p.requires_grad]

        assert sorted(grouped) == sorted(trainable)
        assert len(grouped) == len(set(grouped))


class TestValidationAggregation:
    """Validation runs one loader per dataset, so the single number has to be built by hand."""

    def test_the_epoch_loss_is_weighted_by_frames(self) -> None:
        model = StubDetector()
        model._validation_losses = [
            (torch.tensor(10.0), 3),   # three frames of one dataset
            (torch.tensor(2.0), 1),    # one frame of another
        ]
        logged = {}
        model.log = lambda name, value, **kwargs: logged.__setitem__(name, float(value))

        model.on_validation_epoch_end()

        assert logged["val_loss"] == pytest.approx((10.0 * 3 + 2.0 * 1) / 4)

    def test_a_dataset_with_more_frames_counts_for_more(self) -> None:
        """An equal average would let a tiny validation set swing the checkpoint choice."""
        model = StubDetector()
        model._validation_losses = [(torch.tensor(1.0), 99), (torch.tensor(100.0), 1)]
        logged = {}
        model.log = lambda name, value, **kwargs: logged.__setitem__(name, float(value))

        model.on_validation_epoch_end()

        assert logged["val_loss"] < 5.0

    def test_the_accumulator_is_cleared_between_epochs(self) -> None:
        """Left uncleared, every epoch would average in every earlier epoch."""
        model = StubDetector()
        model._validation_losses = [(torch.tensor(5.0), 1)]
        model.log = lambda *args, **kwargs: None

        model.on_validation_epoch_end()
        assert model._validation_losses == []

        model.on_validation_epoch_start()
        assert model._validation_losses == []

    def test_an_empty_epoch_logs_nothing(self) -> None:
        model = StubDetector()
        model._validation_losses = []
        logged = {}
        model.log = lambda name, value, **kwargs: logged.__setitem__(name, value)

        model.on_validation_epoch_end()
        assert logged == {}


class TestResume:
    """A three-day job gets requeued, and reruns the same command when it does."""

    @staticmethod
    def cfg(tmp_path, resume):
        from omegaconf import OmegaConf

        return OmegaConf.create(
            {"run": {"name": "petr_nuscenes_10cls", "resume": resume}, "paths": {"work_dir": str(tmp_path)}}
        )

    def test_nothing_to_resume_from_by_default(self, tmp_path) -> None:
        from py123detection.torch.train import resume_from

        assert resume_from(self.cfg(tmp_path, None)) is None

    def test_last_starts_from_scratch_on_the_first_attempt(self, tmp_path) -> None:
        """The same command has to work before there is a checkpoint, or the first attempt of
        every requeued job fails on a missing file."""
        from py123detection.torch.train import resume_from

        assert resume_from(self.cfg(tmp_path, "last")) is None

    def test_last_finds_the_run_s_own_checkpoint(self, tmp_path) -> None:
        from py123detection.torch.train import resume_from

        checkpoint = tmp_path / "petr_nuscenes_10cls" / "checkpoints" / "last.ckpt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.touch()

        assert resume_from(self.cfg(tmp_path, "last")) == str(checkpoint)

    def test_an_explicit_path_is_passed_through(self, tmp_path) -> None:
        from py123detection.torch.train import resume_from

        assert resume_from(self.cfg(tmp_path, "/data/other_run/epoch=05.ckpt")) == "/data/other_run/epoch=05.ckpt"


class TestLearningRateSchedule:
    """PETR's published schedule, which the thesis also trained with."""

    def thesis(self, step: int, steps_per_epoch: int, max_epochs: int) -> float:
        """The master thesis's lr_lambda, transcribed."""
        import math

        if step < 500:
            return 1.0 / 3 + (1.0 - 1.0 / 3) * step / 500
        progress = min(max((step // steps_per_epoch) / max_epochs, 0.0), 1.0)
        return 1e-3 + 0.5 * (1.0 - 1e-3) * (1.0 + math.cos(math.pi * progress))

    def test_it_is_the_thesis_schedule(self) -> None:
        from py123detection.torch.models.base import petr_schedule

        for step in (0, 1, 250, 499, 500, 4111, 4112, 9000, 24671):
            assert petr_schedule(step, steps_per_epoch=4112, max_epochs=6) == pytest.approx(
                self.thesis(step, 4112, 6)
            ), step

    def test_the_rate_is_constant_within_an_epoch_and_ends_at_the_floor(self) -> None:
        from py123detection.torch.models.base import petr_schedule

        assert petr_schedule(1000, 1000, 6) == petr_schedule(1999, 1000, 6)
        assert petr_schedule(1000, 1000, 6) > petr_schedule(2000, 1000, 6)
        assert petr_schedule(6000, 1000, 6) == pytest.approx(1e-3)
