"""The set-prediction loss: the Hungarian matcher and the criterion built on it.

This module had no test of its own and its docstring still said so. Both detectors use it, and it is
the kind of code that returns a plausible number whatever it does — a matcher that pairs the wrong
query with the wrong target still produces a loss that falls, just more slowly and to a worse place.

Three things here are load-bearing and silent when broken: the matching is a bijection onto the
targets, the per-layer loss keys name the layer they came from, and the precomputed-normalisation
fast path has to agree with recomputing it. The last one matters most in a mixed-dataset batch,
where some samples have no ground truth at all and the two paths take visibly different branches.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="the torch branch needs torch")
pytest.importorskip("scipy")
pytest.importorskip("torchvision")

#: Needs the torch extra; see the `torch` marker in pyproject.toml.
pytestmark = pytest.mark.torch

from py123detection.torch.models.common.loss import HungarianMatcher, SetCriterion  # noqa: E402
from py123detection.torch.models.common.util import normalize_bbox  # noqa: E402

NUM_CLASSES = 3
CODE_SIZE = 10
PC_RANGE = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
SEED = 20261108


def gt_boxes(count: int, generator: torch.Generator) -> torch.Tensor:
    """``(count, 9)`` ground-truth boxes: ``[x, y, z, w, l, h, yaw, vx, vy]``."""
    boxes = torch.zeros(count, 9, dtype=torch.float32)
    boxes[:, 0] = torch.rand(count, generator=generator) * 80 - 40
    boxes[:, 1] = torch.rand(count, generator=generator) * 80 - 40
    boxes[:, 2] = torch.rand(count, generator=generator) * 3 - 2
    boxes[:, 3:6] = torch.rand(count, 3, generator=generator) * 2 + 1
    boxes[:, 6] = torch.rand(count, generator=generator) * 6 - 3
    boxes[:, 7:9] = torch.rand(count, 2, generator=generator) * 4 - 2
    return boxes


def perfect_outputs(targets, num_queries: int, num_layers: int = 2, at=None):
    """Predictions that already are the targets, parked at chosen query slots.

    The matcher should then pair query ``at[k]`` with target ``k``, which is a fact about the
    assignment rather than about the numbers, so it holds for any boxes.
    """
    batch = len(targets)
    cls_scores = torch.full((num_layers, batch, num_queries, NUM_CLASSES), -8.0)
    bbox_preds = torch.zeros(num_layers, batch, num_queries, CODE_SIZE)

    for sample, target in enumerate(targets):
        count = len(target["labels"])
        slots = list(range(count)) if at is None else at[sample]
        if count == 0:
            continue
        normalised = normalize_bbox(target["boxes_3d"])
        for index, slot in enumerate(slots):
            bbox_preds[:, sample, slot] = normalised[index]
            cls_scores[:, sample, slot, target["labels"][index]] = 8.0
    return {"all_cls_scores": cls_scores, "all_bbox_preds": bbox_preds}


def targets_for(counts, generator):
    out = []
    for count in counts:
        out.append(
            {
                "boxes_3d": gt_boxes(count, generator),
                "labels": torch.randint(0, NUM_CLASSES, (count,), generator=generator),
            }
        )
    return out


@pytest.fixture
def generator():
    return torch.Generator().manual_seed(SEED)


def criterion(matcher=None, weight_dict=None, code_weights=None):
    return SetCriterion(
        num_classes=NUM_CLASSES,
        matcher=matcher or HungarianMatcher(),
        weight_dict=weight_dict or {"loss_ce": 1.0, "loss_bbox": 1.0},
        losses=["labels", "boxes"],
        pc_range=PC_RANGE,
        code_weights=code_weights,
    )


# --------------------------------------------------------------------------------------------
# The matcher
# --------------------------------------------------------------------------------------------


class TestHungarianMatcher:
    def test_a_sample_with_no_ground_truth_matches_nothing(self, generator):
        targets = targets_for([0], generator)
        indices = HungarianMatcher()(perfect_outputs(targets, num_queries=8), targets)
        assert len(indices) == 1
        rows, columns = indices[0]
        assert len(rows) == 0 and len(columns) == 0
        assert rows.dtype == torch.int64 and columns.dtype == torch.int64

    def test_an_empty_sample_does_not_disturb_its_neighbours(self, generator):
        """The mixed-dataset case: one dataset's frame has no box, the next frame does."""
        targets = targets_for([0, 3, 0, 2], generator)
        indices = HungarianMatcher()(perfect_outputs(targets, num_queries=10), targets)
        assert [len(rows) for rows, _ in indices] == [0, 3, 0, 2]

    def test_predictions_that_already_are_the_targets_match_them(self, generator):
        targets = targets_for([4], generator)
        indices = HungarianMatcher()(perfect_outputs(targets, num_queries=12), targets)
        rows, columns = indices[0]
        assert dict(zip(columns.tolist(), rows.tolist())) == {0: 0, 1: 1, 2: 2, 3: 3}

    def test_moving_a_target_to_another_query_moves_the_match_with_it(self, generator):
        """Permutation equivariance: the assignment follows the data, not the query index."""
        targets = targets_for([4], generator)
        slots = [7, 2, 11, 0]
        indices = HungarianMatcher()(perfect_outputs(targets, 12, at=[slots]), targets)
        rows, columns = indices[0]
        assert dict(zip(columns.tolist(), rows.tolist())) == dict(enumerate(slots))

    def test_every_target_is_matched_exactly_once(self, generator):
        """A bijection onto the targets. A query matched twice would double-count its box."""
        targets = targets_for([6, 1, 9], generator)
        outputs = {
            "all_cls_scores": torch.randn(2, 3, 20, NUM_CLASSES, generator=generator),
            "all_bbox_preds": torch.randn(2, 3, 20, CODE_SIZE, generator=generator),
        }
        for (rows, columns), target in zip(HungarianMatcher()(outputs, targets), targets):
            count = len(target["labels"])
            assert sorted(columns.tolist()) == list(range(count)), "every target matched once"
            assert len(set(rows.tolist())) == count, "no query matched twice"

    def test_precomputing_the_normalised_targets_changes_no_assignment(self, generator):
        """The fast path the criterion takes, against the path it skips."""
        targets = targets_for([5, 0, 3], generator)
        outputs = {
            "all_cls_scores": torch.randn(2, 3, 16, NUM_CLASSES, generator=generator),
            "all_bbox_preds": torch.randn(2, 3, 16, CODE_SIZE, generator=generator),
        }
        matcher = HungarianMatcher()
        precomputed = [
            normalize_bbox(t["boxes_3d"].float()) if len(t["labels"]) else None for t in targets
        ]
        inline = matcher(outputs, targets)
        fast = matcher(outputs, targets, precomputed_norm_bboxes=precomputed)
        for (rows_a, cols_a), (rows_b, cols_b) in zip(inline, fast):
            assert torch.equal(rows_a, rows_b) and torch.equal(cols_a, cols_b)

    def test_pc_range_is_accepted_and_ignored(self, generator):
        """Documents a dead parameter rather than pretending it does something.

        ``normalize_bbox`` takes ``pc_range`` and never reads it — centres are deliberately left in
        world coordinates to match PETR, see the NOTE in ``common/util.py``. The parameter is still
        threaded through the matcher and the criterion, so anyone reading the signature would assume
        it changes the cost. It does not.
        """
        targets = targets_for([4], generator)
        outputs = perfect_outputs(targets, num_queries=10)
        without = HungarianMatcher(pc_range=None)(outputs, targets)
        with_range = HungarianMatcher(pc_range=PC_RANGE)(outputs, targets)
        assert torch.equal(without[0][0], with_range[0][0])
        assert torch.equal(without[0][1], with_range[0][1])

    def test_a_cost_matrix_full_of_nans_still_returns_an_assignment(self, generator):
        """The sanitising is deliberate; without it one bad step would kill the run."""
        targets = targets_for([3], generator)
        outputs = perfect_outputs(targets, num_queries=8)
        outputs["all_bbox_preds"] = outputs["all_bbox_preds"] * float("nan")
        rows, columns = HungarianMatcher()(outputs, targets)[0]
        assert len(rows) == 3 and sorted(columns.tolist()) == [0, 1, 2]


# --------------------------------------------------------------------------------------------
# The criterion
# --------------------------------------------------------------------------------------------


class TestSetCriterion:
    def test_every_decoder_layer_but_the_last_is_prefixed_with_its_index(self, generator):
        targets = targets_for([2], generator)
        losses = criterion()(perfect_outputs(targets, 8, num_layers=4), targets)
        assert set(losses) == {
            "d0.loss_ce", "d0.loss_bbox",
            "d1.loss_ce", "d1.loss_bbox",
            "d2.loss_ce", "d2.loss_bbox",
            "loss_ce", "loss_bbox",
        }

    def test_a_single_layer_reports_only_the_unprefixed_loss(self, generator):
        targets = targets_for([2], generator)
        losses = criterion()(perfect_outputs(targets, 8, num_layers=1), targets)
        assert set(losses) == {"loss_ce", "loss_bbox"}

    def test_predictions_that_already_are_the_targets_cost_nothing_to_regress(self, generator):
        targets = targets_for([3], generator)
        losses = criterion()(perfect_outputs(targets, 10, num_layers=1), targets)
        assert losses["loss_bbox"].item() == pytest.approx(0.0, abs=1e-5)
        assert losses["loss_ce"].item() > 0.0, "classification still pays for the empty queries"

    def test_the_weight_dict_scales_each_term(self, generator):
        targets = targets_for([3], generator)
        outputs = perfect_outputs(targets, 10, num_layers=2)
        plain = criterion()(outputs, targets)
        scaled = criterion(weight_dict={"loss_ce": 3.0, "loss_bbox": 0.5})(outputs, targets)
        for key in plain:
            factor = 3.0 if key.endswith("loss_ce") else 0.5
            assert scaled[key].item() == pytest.approx(plain[key].item() * factor, rel=1e-6)

    def test_a_batch_with_no_ground_truth_still_produces_a_differentiable_zero(self, generator):
        """The branch that keeps a step from crashing on an empty frame.

        ``loss_bbox`` must be exactly zero and still carry a gradient, or an empty frame would
        either break the backward pass or quietly contribute a regression target it does not have.
        """
        targets = targets_for([0, 0], generator)
        outputs = perfect_outputs(targets, 8, num_layers=2)
        outputs["all_bbox_preds"].requires_grad_(True)
        losses = criterion()(outputs, targets)

        assert losses["loss_bbox"].item() == 0.0
        assert losses["loss_bbox"].requires_grad
        assert losses["loss_ce"].item() > 0.0, "every query is a false positive here"
        sum(losses.values()).backward()
        assert outputs["all_bbox_preds"].grad is not None

    def test_the_synchronised_box_count_is_used_when_given(self, generator):
        """Why it matters: ranks must divide by the same number or their losses are not comparable."""
        targets = targets_for([4], generator)
        outputs = perfect_outputs(targets, 10, num_layers=1)
        local = criterion()(outputs, targets)
        supplied = criterion()(outputs, targets, num_boxes_synced=torch.tensor(8.0))
        assert supplied["loss_ce"].item() == pytest.approx(local["loss_ce"].item() / 2.0, rel=1e-6)

    def test_code_weights_scale_the_box_terms(self, generator):
        targets = targets_for([3], generator)
        outputs = {
            "all_cls_scores": torch.randn(1, 1, 10, NUM_CLASSES, generator=generator),
            "all_bbox_preds": torch.randn(1, 1, 10, CODE_SIZE, generator=generator),
        }
        plain = criterion()(outputs, targets)
        halved = criterion(code_weights=torch.full((CODE_SIZE,), 0.5))(outputs, targets)
        assert halved["loss_bbox"].item() == pytest.approx(plain["loss_bbox"].item() * 0.5, rel=1e-6)
        assert halved["loss_ce"].item() == pytest.approx(plain["loss_ce"].item(), rel=1e-6)

    def test_gradients_reach_both_heads(self, generator):
        targets = targets_for([2, 3], generator)
        outputs = {
            "all_cls_scores": torch.randn(2, 2, 12, NUM_CLASSES, generator=generator).requires_grad_(True),
            "all_bbox_preds": torch.randn(2, 2, 12, CODE_SIZE, generator=generator).requires_grad_(True),
        }
        sum(criterion()(outputs, targets).values()).backward()
        for name in ("all_cls_scores", "all_bbox_preds"):
            grad = outputs[name].grad
            assert grad is not None and torch.isfinite(grad).all(), name
            assert grad.abs().sum() > 0, f"{name} received no gradient"

    def test_the_precomputed_fast_path_matches_recomputing_it(self, generator):
        """A batch where the two branches differ visibly: empty samples are skipped by one.

        ``get_layer_loss`` builds its regression targets either from the precomputed list, skipping
        samples with no match, or by concatenating per-target slices, which includes empty slices.
        Those are different code paths and have to produce the same tensor.
        """
        targets = targets_for([0, 4, 0, 2], generator)
        outputs = {
            "all_cls_scores": torch.randn(1, 4, 14, NUM_CLASSES, generator=generator),
            "all_bbox_preds": torch.randn(1, 4, 14, CODE_SIZE, generator=generator),
        }
        model = criterion()
        indices = model.matcher(outputs, targets)
        num_boxes = torch.tensor(float(sum(len(t["labels"]) for t in targets)))
        precomputed = [
            normalize_bbox(t["boxes_3d"].float()) if len(t["labels"]) else None for t in targets
        ]

        inline = model.get_layer_loss(
            outputs["all_cls_scores"][0], outputs["all_bbox_preds"][0], targets, indices, num_boxes
        )
        fast = model.get_layer_loss(
            outputs["all_cls_scores"][0],
            outputs["all_bbox_preds"][0],
            targets,
            indices,
            num_boxes,
            precomputed_norm_bboxes=precomputed,
        )
        for key in ("loss_ce", "loss_bbox"):
            assert fast[key].item() == pytest.approx(inline[key].item(), rel=1e-6), key

    def test_the_loss_falls_as_predictions_approach_the_targets(self, generator):
        """A sanity check on direction, which no amount of shape checking gives you."""
        targets = targets_for([4], generator)
        exact = perfect_outputs(targets, 10, num_layers=1)
        noisy = {
            "all_cls_scores": exact["all_cls_scores"].clone(),
            "all_bbox_preds": exact["all_bbox_preds"]
            + torch.randn(exact["all_bbox_preds"].shape, generator=generator) * 2.0,
        }
        model = criterion()
        assert model(exact, targets)["loss_bbox"].item() < model(noisy, targets)["loss_bbox"].item()
