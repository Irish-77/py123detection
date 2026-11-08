"""Moving BEVFormer out from under PETR must not move a single number.

In the thesis, ``BEVFormer`` subclassed ``PETR``: it inherited PETR's forward, so every change to
PETR was a silent change to BEVFormer. The port gives both detectors one shared
``MultiViewDetector`` and a head each — a restructuring, not a rewrite, so the parameter count, the
forward outputs and every loss term must come out exactly as they did. That is checked here
against ``tests/golden/bevformer_golden.json``, frozen from the thesis model by
``tools/freeze_bevformer_golden.py``, in both camera-embedding modes the port kept.

``tests/test_model_equivalence.py`` makes the same claim for PETR, on the same batch and seed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

torch = pytest.importorskip("torch", reason="the torch branch needs torch")

#: Needs the torch extra; see the `torch` marker in pyproject.toml.
pytestmark = pytest.mark.torch

import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from freeze_bevformer_golden import CAMERA_EMBEDDINGS  # noqa: E402
from freeze_model_golden import SEED, SHAPE, fixed_batch, summarise_tensor  # noqa: E402

from py123detection.torch.models import BEVFormer  # noqa: E402

FIXTURE = Path(__file__).parent / "golden" / "bevformer_golden.json"

#: Floating-point work reorders slightly between runs; anything real is far larger than this.
TOLERANCE = 1e-4


def port_kwargs(golden: Dict[str, Any], camera_embedding: str) -> Dict[str, Any]:
    """The frozen arguments, spelled as the ported constructor takes them."""
    kwargs = dict(golden["petr_kwargs"], **golden["bevformer_kwargs"])
    # The legacy constructor described the input through four separate arguments; this one takes
    # the size and derives the rest.
    kwargs["image_size"] = (kwargs.pop("image_height"), kwargs.pop("image_width"))
    kwargs.pop("patches_x"), kwargs.pop("patches_y")
    kwargs["backbone_checkpointing"] = kwargs.pop("backbone_with_cp")
    # The thesis BEVFormer took ``head_with_cp`` and ignored PETR's ``transformer_with_cp``; both
    # were frozen at False, and the port has a single flag for the head.
    kwargs["head_checkpointing"] = kwargs.pop("transformer_with_cp")
    kwargs.pop("optimizer_fused")
    # ``position_range`` spans PETR's 3D positional embedding. A bird's-eye grid has none, so the
    # thesis only inherited the argument and left it inert; the port does not accept it.
    kwargs.pop("position_range")
    kwargs["camera_embedding"] = camera_embedding
    return kwargs


@pytest.fixture(scope="module")
def golden() -> Dict[str, Any]:
    if not FIXTURE.exists():
        pytest.skip(f"no fixture at {FIXTURE}; run tools/freeze_bevformer_golden.py")
    return json.loads(FIXTURE.read_text())


@pytest.fixture(scope="module")
def ported(golden) -> Dict[str, Dict[str, Any]]:
    """Run the ported BEVFormer on the fixture's exact input and seed, in both modes."""
    import numpy as np

    results = {}
    for camera_embedding in CAMERA_EMBEDDINGS:
        device = torch.device("cpu")
        torch.manual_seed(SEED)
        np.random.seed(SEED)

        model = BEVFormer(**port_kwargs(golden, camera_embedding)).to(device).eval()
        batch = fixed_batch(device)

        with torch.no_grad():
            outputs = model(
                {
                    "img": batch["img"],
                    "intrinsics": batch["intrinsics"],
                    "extrinsics": batch["extrinsics"],
                    "imu2img": batch["imu2img"],
                    "cam_mask": batch["cam_mask"],
                    "img_metas": None,
                }
            )
            losses = model.head.loss(
                batch["gt_bboxes_3d"], batch["gt_labels_3d"], outputs, sync=False
            )

        results[camera_embedding] = {
            "parameters": int(sum(p.numel() for p in model.parameters())),
            "outputs": {
                key: summarise_tensor(value)
                for key, value in sorted(outputs.items())
                if torch.is_tensor(value)
            },
            "losses": {key: float(value) for key, value in sorted(losses.items())},
        }

    return results


@pytest.mark.parametrize("camera_embedding", CAMERA_EMBEDDINGS)
class TestTheRestructuringIsInvisible:
    def test_the_parameter_count_is_unchanged(self, golden, ported, camera_embedding) -> None:
        """No module was lost in the move, and none was added.

        Only ``"bevformer"`` has parameters of its own — the learned per-camera table — so the two
        modes must also differ by exactly ``num_cameras * hidden_dim``.
        """
        assert ported[camera_embedding]["parameters"] == golden["runs"][camera_embedding]["parameters"]

    def test_the_forward_outputs_are_unchanged(self, golden, ported, camera_embedding) -> None:
        expected = golden["runs"][camera_embedding]["outputs"]
        actual = ported[camera_embedding]["outputs"]
        assert sorted(actual) == sorted(expected)

        for key in expected:
            assert actual[key]["shape"] == expected[key]["shape"], key
            for statistic in ("sum", "mean", "min", "max"):
                assert actual[key][statistic] == pytest.approx(
                    expected[key][statistic], abs=TOLERANCE, rel=TOLERANCE
                ), f"{key}.{statistic}"

    def test_every_loss_term_is_unchanged(self, golden, ported, camera_embedding) -> None:
        expected = golden["runs"][camera_embedding]["losses"]
        actual = ported[camera_embedding]["losses"]
        assert sorted(actual) == sorted(expected), "the set of loss terms changed"
        for term, value in expected.items():
            assert actual[term] == pytest.approx(value, abs=TOLERANCE, rel=TOLERANCE), term


def test_the_learned_table_is_the_only_difference_between_the_modes(golden) -> None:
    """A per-camera token per camera, and nothing else, as the paper has it."""
    plain = golden["runs"]["none"]["parameters"]
    learned = golden["runs"]["bevformer"]["parameters"]
    assert learned - plain == SHAPE["cameras"] * golden["petr_kwargs"]["hidden_dim"]


def test_the_fixture_was_frozen_on_the_batch_this_test_builds(golden) -> None:
    """``fixed_batch`` comes from the tool, not from the fixture, so the two can drift apart."""
    assert golden["seed"] == SEED
    assert golden["shape"] == SHAPE


def test_the_dropped_camera_embedding_is_rejected() -> None:
    """The thesis had a third mode, ``"petr"``, which went with the extensions.

    An unknown mode has to fail loudly: silently returning no embedding would train a model that
    quietly differs from the one that was asked for.
    """
    from py123detection.torch.models.bevformer.camera_embedding import build_camera_embedding

    with pytest.raises(ValueError, match="petr"):
        build_camera_embedding("petr", embed_dims=256, num_cameras=6)


def test_no_thesis_era_spellings_survive_in_the_constructor() -> None:
    """The port renamed these; keeping both spellings alive is how configs start disagreeing."""
    import inspect

    parameters = set(inspect.signature(BEVFormer.__init__).parameters)
    assert not (parameters & {"head_with_cp", "cam_embed_type", "pe_position_range"})
    assert {"camera_embedding", "with_box_refine"} <= parameters
