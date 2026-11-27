"""Freeze the thesis BEVFormer's numerics as the fixture the ported model must reproduce.

Run this against the master-thesis implementation::

    PYTHONPATH=../master-thesis/src \
    ../envs/modern/bin/python tools/freeze_bevformer_golden.py --out tests/golden/bevformer_golden.json

PETR's port is pinned by ``tools/freeze_model_golden.py``, which froze the legacy model before the
extensions were stripped. BEVFormer was not only stripped but moved: the thesis class subclassed
PETR and inherited its forward, so the two detectors shared whatever PETR happened to do, while
the port gives them one ``MultiViewDetector`` and a head each. That restructuring is supposed to be
invisible — same parameters, same outputs, same loss terms — and this is where the claim is
checked, on the batch and seed PETR's fixture already uses.

Both camera-embedding modes the port kept are frozen. The thesis also offered ``"petr"``, which
went with the extensions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from freeze_model_golden import MODEL_KWARGS, SEED, SHAPE, fixed_batch, summarise_tensor

REPO = Path(__file__).resolve().parents[1]

#: BEVFormer's own arguments. The grid is 25x25 instead of the config's 50x50 and the encoder has
#: two layers instead of three — small enough to run on a machine without a GPU, still more than
#: one layer, so layer stacking and ``with_box_refine``'s per-layer heads are exercised.
BEVFORMER_KWARGS = dict(
    bev_h=25,
    bev_w=25,
    num_encoder_layers=2,
    num_heights=4,
    num_points=4,
    feedforward_channels_encoder=512,
    feedforward_channels_decoder=2048,
    with_box_refine=True,
    num_cameras=SHAPE["cameras"],
)

#: ``"none"`` is what lets one model train across rigs with different camera counts; ``"bevformer"``
#: is the paper's learned per-camera token, and the only mode with parameters of its own.
CAMERA_EMBEDDINGS = ("none", "bevformer")


def thesis_kwargs(camera_embedding: str) -> Dict[str, Any]:
    """PETR's frozen arguments plus BEVFormer's, spelled as the thesis constructor takes them."""
    kwargs = dict(MODEL_KWARGS, **BEVFORMER_KWARGS)
    # The thesis BEVFormer ignores PETR's transformer_with_cp in favour of its own flag.
    kwargs["head_with_cp"] = False
    kwargs["cam_embed_type"] = camera_embedding
    return kwargs


def run(device_name: str, camera_embedding: str) -> Dict[str, Any]:
    from thesis.models.bevformer.bevformer import BEVFormer

    device = torch.device(device_name)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = BEVFormer(**thesis_kwargs(camera_embedding)).to(device).eval()
    batch = fixed_batch(device)

    with torch.no_grad():
        outputs = model(
            batch["img"],
            batch["intrinsics"],
            batch["extrinsics"],
            cam_mask=batch["cam_mask"],
            imu2img=batch["imu2img"],
        )
        losses = model.head.loss(
            batch["gt_bboxes_3d"], batch["gt_labels_3d"], outputs, sync=False
        )

    return {
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "outputs": {
            key: summarise_tensor(value)
            for key, value in sorted(outputs.items())
            if torch.is_tensor(value)
        },
        "losses": {key: round(float(value), 6) for key, value in sorted(losses.items())},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=REPO / "tests" / "golden" / "bevformer_golden.json")
    parser.add_argument("--device", default="cpu", help="cpu keeps this reproducible across machines")
    args = parser.parse_args()

    fixture = {
        "note": "Generated from the master-thesis BEVFormer with every extension at its default "
                "(off). Do not edit by hand; regenerate with tools/freeze_bevformer_golden.py.",
        "seed": SEED,
        "shape": SHAPE,
        "petr_kwargs": MODEL_KWARGS,
        "bevformer_kwargs": BEVFORMER_KWARGS,
        "runs": {name: run(args.device, name) for name in CAMERA_EMBEDDINGS},
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(fixture, indent=1, sort_keys=True) + "\n")
    for name, result in fixture["runs"].items():
        print(f"cam_embed={name:9s} parameters: {result['parameters']:,}  losses: {result['losses']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
