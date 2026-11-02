"""Stripping the thesis extensions must not move a single number.

Every extension was gated behind a flag defaulting to off, so the default configuration already
was plain PETR and deleting the gated code should be invisible. "Should be" is not a proof, and
the gated code touched shared things — the positional-embedding input width, the decoder's
signature, the forward body — so this checks it directly against
``tests/golden/model_golden.json``, frozen from the legacy model before the strip.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

torch = pytest.importorskip("torch", reason="the torch branch needs torch")

import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from freeze_model_golden import MODEL_KWARGS, SEED, fixed_batch, summarise_tensor  # noqa: E402

from py123detection.torch.models import PETR  # noqa: E402

FIXTURE = Path(__file__).parent / "golden" / "model_golden.json"

#: Floating-point work reorders slightly between runs; anything real is far larger than this.
TOLERANCE = 1e-4


@pytest.fixture(scope="module")
def golden() -> Dict[str, Any]:
    if not FIXTURE.exists():
        pytest.skip(f"no fixture at {FIXTURE}; run tools/freeze_model_golden.py")
    return json.loads(FIXTURE.read_text())


@pytest.fixture(scope="module")
def stripped_run(golden) -> Dict[str, Any]:
    """Run the current PETR on the fixture's exact input and seed."""
    import numpy as np

    kwargs = dict(MODEL_KWARGS)
    # The legacy constructor described the input through four separate arguments; this one takes
    # the size and derives the rest.
    kwargs["image_size"] = (kwargs.pop("image_height"), kwargs.pop("image_width"))
    kwargs.pop("patches_x"), kwargs.pop("patches_y")
    kwargs["backbone_checkpointing"] = kwargs.pop("backbone_with_cp")
    kwargs["head_checkpointing"] = kwargs.pop("transformer_with_cp")
    kwargs.pop("optimizer_fused")

    device = torch.device("cpu")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = PETR(**kwargs).to(device).eval()
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
        losses = model.head.loss(batch["gt_bboxes_3d"], batch["gt_labels_3d"], outputs, sync=False)

    return {
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "outputs": {k: summarise_tensor(v) for k, v in sorted(outputs.items()) if torch.is_tensor(v)},
        "losses": {k: float(v) for k, v in sorted(losses.items())},
    }


def test_parameter_count_is_unchanged(golden, stripped_run) -> None:
    """Every removed module was inactive, so none of them held a parameter."""
    assert stripped_run["parameters"] == golden["petr"]["parameters"]


def test_forward_outputs_are_unchanged(golden, stripped_run) -> None:
    expected, actual = golden["petr"]["outputs"], stripped_run["outputs"]
    assert sorted(actual) == sorted(expected)

    for key in expected:
        assert actual[key]["shape"] == expected[key]["shape"], key
        for statistic in ("sum", "mean", "min", "max"):
            assert actual[key][statistic] == pytest.approx(
                expected[key][statistic], abs=TOLERANCE, rel=TOLERANCE
            ), f"{key}.{statistic}"


def test_every_loss_term_is_unchanged(golden, stripped_run) -> None:
    expected, actual = golden["petr"]["losses"], stripped_run["losses"]
    assert sorted(actual) == sorted(expected), "the set of loss terms changed"
    for term, value in expected.items():
        assert actual[term] == pytest.approx(value, abs=TOLERANCE, rel=TOLERANCE), term


def test_no_extension_flags_survive_in_the_constructor() -> None:
    """The point of the strip was the flag surface, not only the code behind it."""
    import inspect

    parameters = set(inspect.signature(PETR.__init__).parameters)
    forbidden = {
        "enable_coin3d", "enable_mv_encoder", "enable_rig_encoder", "enable_rig_aux_loss",
        "enable_rig_xattn", "enable_proj_ground", "use_feature_focal_scaling", "input_canon",
        "pe_strategy", "backbone_lora", "backbone_train_mode", "skip_3d_pe",
    }
    assert not (parameters & forbidden)
    assert len(parameters) < 30, f"PETR still takes {len(parameters)} arguments"
