"""The model's pure helpers: box encoding, the BEV grid, projection, and grid masking.

None of these had a test. They are all small, all deterministic, and all reachable only through a
forward pass today — so the two golden fixtures cover whichever branch of them that one fixed batch
happens to take, and nothing covers the rest. The box encoding in particular has an unusual column
order that is easy to "tidy" into a different one while every shape check still passes.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch", reason="the torch branch needs torch")

#: Needs the torch extra; see the `torch` marker in pyproject.toml.
pytestmark = pytest.mark.torch

from py123detection.torch.models.bevformer.utils import (  # noqa: E402
    create_3d_reference_points,
    create_bev_grid,
    project_3d_to_cameras,
)
from py123detection.torch.models.common.grid_mask import GridMask  # noqa: E402
from py123detection.torch.models.common.util import (  # noqa: E402
    denormalize_bbox,
    inverse_sigmoid,
    normalize_bbox,
    pos2posemb3d,
)

PC_RANGE = [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]
SEED = 20261108


# --------------------------------------------------------------------------------------------
# Box encoding
# --------------------------------------------------------------------------------------------


class TestBoxEncoding:
    def box(self) -> torch.Tensor:
        #        x     y     z     w    l    h   yaw   vx    vy
        return torch.tensor([[12.5, -3.25, -1.0, 1.9, 4.5, 1.6, 0.75, 2.5, -1.25]])

    def test_the_column_order_is_the_documented_one(self):
        """Not the input order. ``cz`` sits at index 4, between ``log l`` and ``log h``.

        This is PETR's layout and it looks like a mistake, which is exactly why it needs pinning:
        reordering it to the obvious ``[x, y, z, w, l, h, ...]`` would keep every shape valid and
        silently mis-train every box.
        """
        encoded = normalize_bbox(self.box())[0]
        box = self.box()[0]
        assert encoded[0].item() == pytest.approx(box[0].item())  # cx
        assert encoded[1].item() == pytest.approx(box[1].item())  # cy
        assert encoded[2].item() == pytest.approx(math.log(box[3].item()))  # log w
        assert encoded[3].item() == pytest.approx(math.log(box[4].item()))  # log l
        assert encoded[4].item() == pytest.approx(box[2].item())  # cz
        assert encoded[5].item() == pytest.approx(math.log(box[5].item()))  # log h
        assert encoded[6].item() == pytest.approx(math.sin(box[6].item()))
        assert encoded[7].item() == pytest.approx(math.cos(box[6].item()))
        assert encoded[8].item() == pytest.approx(box[7].item())  # vx
        assert encoded[9].item() == pytest.approx(box[8].item())  # vy

    def test_encoding_then_decoding_returns_the_box(self):
        original = self.box()
        restored = denormalize_bbox(normalize_bbox(original))
        assert torch.allclose(restored, original, atol=1e-5)

    @pytest.mark.parametrize("yaw", [-3.0, -1.5, 0.0, 0.5, 1.5, 3.0])
    def test_yaw_survives_the_round_trip_inside_one_turn(self, yaw):
        """It goes through sin/cos and comes back through atan2, so only (-pi, pi] is recoverable."""
        box = self.box()
        box[0, 6] = yaw
        assert denormalize_bbox(normalize_bbox(box))[0, 6].item() == pytest.approx(yaw, abs=1e-5)

    def test_a_yaw_outside_one_turn_comes_back_wrapped(self):
        box = self.box()
        box[0, 6] = 0.5 + 2 * math.pi
        assert denormalize_bbox(normalize_bbox(box))[0, 6].item() == pytest.approx(0.5, abs=1e-5)

    def test_a_box_without_velocity_round_trips_too(self):
        box = self.box()[:, :7]
        encoded = normalize_bbox(box)
        assert encoded.shape == (1, 8)
        assert torch.allclose(denormalize_bbox(encoded), box, atol=1e-5)

    def test_a_degenerate_size_is_clamped_rather_than_producing_minus_infinity(self):
        box = self.box()
        box[0, 3:6] = 0.0
        encoded = normalize_bbox(box)
        assert torch.isfinite(encoded).all()

    def test_pc_range_is_accepted_and_ignored(self):
        """A dead parameter, documented. The NOTE in ``util.py`` says centres stay in world
        coordinates to match PETR, so ``pc_range`` is threaded through both this and the loss
        without ever being read."""
        box = self.box()
        assert torch.equal(normalize_bbox(box, None), normalize_bbox(box, PC_RANGE))
        encoded = normalize_bbox(box)
        assert torch.equal(denormalize_bbox(encoded, None), denormalize_bbox(encoded, PC_RANGE))


class TestInverseSigmoid:
    def test_it_inverts_sigmoid(self):
        values = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9])
        assert torch.allclose(torch.sigmoid(inverse_sigmoid(values)), values, atol=1e-5)

    def test_the_ends_are_bounded_rather_than_infinite(self):
        out = inverse_sigmoid(torch.tensor([0.0, 1.0]))
        assert torch.isfinite(out).all()
        assert out[0] < 0 < out[1]

    def test_inputs_outside_the_unit_interval_are_clamped_in(self):
        assert inverse_sigmoid(torch.tensor([-5.0])).item() == pytest.approx(
            inverse_sigmoid(torch.tensor([0.0])).item()
        )
        assert inverse_sigmoid(torch.tensor([5.0])).item() == pytest.approx(
            inverse_sigmoid(torch.tensor([1.0])).item()
        )

    def test_a_larger_eps_brings_the_ends_closer_in(self):
        tight = inverse_sigmoid(torch.tensor([0.0]), eps=1e-9).item()
        loose = inverse_sigmoid(torch.tensor([0.0]), eps=1e-2).item()
        assert tight < loose < 0


class TestPositionalEmbedding:
    def test_the_shape_is_three_axes_of_features(self):
        pos = torch.rand(2, 7, 3, generator=torch.Generator().manual_seed(SEED))
        assert pos2posemb3d(pos, num_pos_feats=16).shape == (2, 7, 48)

    def test_it_is_deterministic(self):
        pos = torch.rand(1, 4, 3, generator=torch.Generator().manual_seed(SEED))
        assert torch.equal(pos2posemb3d(pos), pos2posemb3d(pos))

    def test_different_positions_embed_differently(self):
        pos = torch.tensor([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]])
        embedded = pos2posemb3d(pos, num_pos_feats=8)
        assert not torch.allclose(embedded[0, 0], embedded[0, 1])

    def test_the_embedding_is_finite_across_the_unit_cube(self):
        pos = torch.cartesian_prod(*[torch.linspace(0, 1, 5)] * 3).unsqueeze(0)
        assert torch.isfinite(pos2posemb3d(pos, num_pos_feats=16)).all()


# --------------------------------------------------------------------------------------------
# The BEV grid and projection
# --------------------------------------------------------------------------------------------


class TestBevGrid:
    def test_a_two_by_two_grid_is_the_four_cell_centres(self):
        grid = create_bev_grid(2, 2)
        assert grid.shape == (4, 2)
        expected = torch.tensor([[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]])
        assert torch.allclose(grid, expected)

    def test_x_varies_fastest(self):
        """Row-major over (y, x). A head with per-cell parameters learns this order, so swapping it
        transposes the BEV plane while every shape still matches."""
        grid = create_bev_grid(3, 4)
        assert grid.shape == (12, 2)
        assert grid[0, 1].item() == pytest.approx(grid[3, 1].item()), "first four share one y"
        assert grid[0, 0].item() < grid[1, 0].item() < grid[2, 0].item() < grid[3, 0].item()

    def test_the_centres_are_inside_the_unit_square(self):
        grid = create_bev_grid(5, 7)
        assert grid.min().item() > 0.0 and grid.max().item() < 1.0

    def test_reference_points_are_pillars_over_the_grid(self):
        grid = create_bev_grid(2, 2)
        points = create_3d_reference_points(grid, num_heights=4, pc_range=PC_RANGE)
        assert points.shape == (4, 4, 3)
        for cell in range(4):
            assert len(set(points[cell, :, 0].tolist())) == 1, "a pillar shares one x"
            assert len(set(points[cell, :, 1].tolist())) == 1, "a pillar shares one y"

    def test_the_pillars_span_the_height_range(self):
        points = create_3d_reference_points(create_bev_grid(2, 2), 4, PC_RANGE)
        assert points[0, 0, 2].item() == pytest.approx(PC_RANGE[2])
        assert points[0, -1, 2].item() == pytest.approx(PC_RANGE[5])

    def test_the_grid_is_denormalised_into_the_detection_range(self):
        points = create_3d_reference_points(create_bev_grid(2, 2), 2, PC_RANGE)
        # x_norm 0.25 over [-50, 50] is -25.
        assert points[0, 0, 0].item() == pytest.approx(-25.0)
        assert points[0, 0, 1].item() == pytest.approx(-25.0)
        assert points[3, 0, 0].item() == pytest.approx(25.0)


class TestProjection:
    def pinhole(self, focal=400.0, cx=320.0, cy=240.0) -> torch.Tensor:
        """``imu2img`` for a camera looking down +z with no extrinsic motion."""
        matrix = torch.eye(4)
        matrix[0, 0], matrix[0, 2] = focal, cx
        matrix[1, 1], matrix[1, 2] = focal, cy
        return matrix.view(1, 1, 4, 4)

    def test_a_point_on_the_optical_axis_lands_at_the_principal_point(self):
        points = torch.tensor([[[[0.0, 0.0, 10.0]]]])
        coords, valid = project_3d_to_cameras(points, self.pinhole(), pad_h=480, pad_w=640)
        assert coords.shape == (1, 1, 1, 1, 2) and valid.shape == (1, 1, 1, 1)
        assert bool(valid[0, 0, 0, 0])
        assert coords[0, 0, 0, 0, 0].item() == pytest.approx(0.5)
        assert coords[0, 0, 0, 0, 1].item() == pytest.approx(0.5)

    def test_an_offset_point_projects_where_the_pinhole_says(self):
        points = torch.tensor([[[[2.0, 1.0, 20.0]]]])
        coords, valid = project_3d_to_cameras(points, self.pinhole(), pad_h=480, pad_w=640)
        assert bool(valid[0, 0, 0, 0])
        assert coords[0, 0, 0, 0, 0].item() == pytest.approx((400 * 2 / 20 + 320) / 640)
        assert coords[0, 0, 0, 0, 1].item() == pytest.approx((400 * 1 / 20 + 240) / 480)

    def test_a_point_behind_the_camera_is_invalid(self):
        points = torch.tensor([[[[0.0, 0.0, -10.0]]]])
        _, valid = project_3d_to_cameras(points, self.pinhole(), pad_h=480, pad_w=640)
        assert not bool(valid[0, 0, 0, 0])

    def test_a_point_outside_the_frame_is_invalid(self):
        points = torch.tensor([[[[100.0, 0.0, 10.0]]]])
        _, valid = project_3d_to_cameras(points, self.pinhole(), pad_h=480, pad_w=640)
        assert not bool(valid[0, 0, 0, 0])

    def test_invalid_points_still_come_back_inside_the_unit_square(self):
        """The clamp matters: ``F.grid_sample`` downstream would otherwise see wild coordinates."""
        points = torch.tensor([[[[0.0, 0.0, -10.0], [1e4, 1e4, 1.0]]]])
        coords, valid = project_3d_to_cameras(points, self.pinhole(), pad_h=480, pad_w=640)
        assert not valid.any()
        assert coords.min().item() >= 0.0 and coords.max().item() <= 1.0
        assert torch.isfinite(coords).all()

    def test_every_camera_of_a_rig_is_projected(self):
        rig = torch.cat([self.pinhole(), self.pinhole(focal=200.0)], dim=1)
        points = torch.tensor([[[[0.0, 0.0, 10.0], [1.0, 0.0, 10.0]]]])
        coords, valid = project_3d_to_cameras(points, rig, pad_h=480, pad_w=640)
        assert coords.shape == (1, 2, 1, 2, 2) and valid.shape == (1, 2, 1, 2)
        # The shorter focal length puts the same off-axis point nearer the centre.
        assert coords[0, 1, 0, 1, 0].item() < coords[0, 0, 0, 1, 0].item()


# --------------------------------------------------------------------------------------------
# Grid masking
# --------------------------------------------------------------------------------------------


class TestGridMask:
    def image(self) -> torch.Tensor:
        return torch.ones(2, 3, 32, 32)

    def test_it_does_nothing_outside_training(self):
        mask = GridMask(use_h=True, use_w=True, prob=1.0).eval()
        image = self.image()
        assert torch.equal(mask(image), image)

    def test_it_does_nothing_at_probability_zero(self):
        mask = GridMask(use_h=True, use_w=True, prob=0.0).train()
        image = self.image()
        assert torch.equal(mask(image), image)

    def test_it_masks_some_of_the_image_when_enabled(self):
        torch.manual_seed(SEED)
        mask = GridMask(use_h=True, use_w=True, prob=1.0, ratio=0.5).train()
        image = self.image()
        out = mask(image)
        assert out.shape == image.shape
        assert (out == 0).any(), "nothing was masked"
        assert (out != 0).any(), "everything was masked"

    def test_the_masked_fraction_follows_the_ratio(self):
        torch.manual_seed(SEED)
        light = GridMask(use_h=True, use_w=True, prob=1.0, ratio=0.2).train()
        torch.manual_seed(SEED)
        heavy = GridMask(use_h=True, use_w=True, prob=1.0, ratio=0.8).train()
        image = self.image()
        assert (light(image) == 0).float().mean() < (heavy(image) == 0).float().mean()

    def test_set_prob_ramps_from_nothing_to_the_configured_probability(self):
        mask = GridMask(use_h=True, use_w=True, prob=0.7)
        mask.set_prob(0, 24)
        assert mask.prob == pytest.approx(0.0)
        mask.set_prob(12, 24)
        assert mask.prob == pytest.approx(0.35)
        mask.set_prob(24, 24)
        assert mask.prob == pytest.approx(0.7)
        assert mask.st_prob == pytest.approx(0.7), "the configured value is kept to ramp from"

    def test_nothing_in_the_package_calls_set_prob(self):
        """So the ramp above never happens in a real run.

        The thesis annealed the probability each epoch; the port kept the method and dropped the
        call, which leaves a feature that looks present and is not. Recorded here so the next person
        either wires it to a config key or deletes it, rather than assuming it works.
        """
        from pathlib import Path

        source = Path(__file__).resolve().parents[1] / "src"
        mentions = [
            f"{path.relative_to(source)}:{number}"
            for path in sorted(source.rglob("*.py"))
            for number, line in enumerate(path.read_text().splitlines(), 1)
            if "set_prob" in line and "def set_prob" not in line
        ]
        assert mentions == [], (
            f"set_prob now has a caller ({mentions}); wire it to a config key and update this test"
        )
