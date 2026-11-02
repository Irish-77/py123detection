"""Freeze the legacy PETR's numerics as the fixture the stripped model must reproduce.

Run this against the master-thesis implementation *before* the extensions are removed::

    PYTHONPATH=../master-thesis/src \
    ../envs/modern/bin/python tools/freeze_model_golden.py --out tests/golden/model_golden.json

Every thesis extension is gated behind a flag that defaults to off, so the default configuration
already is plain PETR. Deleting the gated code therefore must not move a single number, and that
is a stronger claim than "the tests still pass" — so it is checked directly: forward outputs and
each loss term, at a fixed seed, on fixed input.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]

#: Small enough to run on any GPU-less machine, large enough that every module is exercised.
SHAPE = {"batch": 2, "cameras": 6, "height": 320, "width": 800}
SEED = 20260919

MODEL_KWARGS = dict(
    num_classes=10,
    num_query=50,
    num_decoder_layers=2,
    hidden_dim=256,
    backbone_name="resnet50",
    pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
    position_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
    use_grid_mask=False,  # stochastic, and its own module is tested separately
    backbone_with_cp=False,
    transformer_with_cp=False,
    optimizer_fused=False,
    image_width=800,
    image_height=320,
    patches_x=50,
    patches_y=20,
)


def fixed_batch(device: torch.device) -> Dict[str, Any]:
    """Deterministic input: images, a plausible rig, and two frames of boxes."""
    generator = torch.Generator().manual_seed(SEED)
    batch, cameras = SHAPE["batch"], SHAPE["cameras"]

    images = torch.randn(
        batch, cameras, 3, SHAPE["height"], SHAPE["width"], generator=generator
    )

    intrinsics = torch.eye(4).repeat(batch, cameras, 1, 1)
    intrinsics[..., 0, 0] = 800.0
    intrinsics[..., 1, 1] = 800.0
    intrinsics[..., 0, 2] = SHAPE["width"] / 2
    intrinsics[..., 1, 2] = SHAPE["height"] / 2

    extrinsics = torch.eye(4).repeat(batch, cameras, 1, 1)
    for camera in range(cameras):
        angle = 2.0 * np.pi * camera / cameras
        rotation = torch.tensor(
            [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        extrinsics[:, camera, :3, :3] = rotation
        extrinsics[:, camera, :3, 3] = torch.tensor([0.0, 0.0, -1.6])

    boxes, labels = [], []
    for index in range(batch):
        count = 4 + index
        box = torch.zeros(count, 9)
        box[:, 0] = torch.linspace(-20.0, 20.0, count)
        box[:, 1] = torch.linspace(10.0, -10.0, count)
        box[:, 2] = -1.0
        box[:, 3:6] = torch.tensor([1.9, 4.5, 1.6])
        box[:, 6] = torch.linspace(-1.0, 1.0, count)
        boxes.append(box)
        labels.append(torch.arange(count) % MODEL_KWARGS["num_classes"])

    return {
        "img": images.to(device),
        "intrinsics": intrinsics.to(device),
        "extrinsics": extrinsics.to(device),
        "imu2img": torch.matmul(intrinsics, extrinsics).to(device),
        "cam_mask": torch.ones(batch, cameras, dtype=torch.bool, device=device),
        "gt_bboxes_3d": [box.to(device) for box in boxes],
        "gt_labels_3d": [label.to(device) for label in labels],
    }


def summarise_tensor(tensor: torch.Tensor) -> Dict[str, Any]:
    values = tensor.detach().float().cpu()
    return {
        "shape": list(values.shape),
        "sum": round(float(values.sum()), 4),
        "mean": round(float(values.mean()), 6),
        "min": round(float(values.min()), 6),
        "max": round(float(values.max()), 6),
    }


def run(device_name: str) -> Dict[str, Any]:
    from thesis.models.petr.petr import PETR

    device = torch.device(device_name)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = PETR(**MODEL_KWARGS).to(device).eval()
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
    parser.add_argument("--out", type=Path, default=REPO / "tests" / "golden" / "model_golden.json")
    parser.add_argument("--device", default="cpu", help="cpu keeps this reproducible across machines")
    args = parser.parse_args()

    fixture = {
        "note": "Generated from the master-thesis PETR with every extension at its default (off). "
                "Do not edit by hand; regenerate with tools/freeze_model_golden.py.",
        "seed": SEED,
        "shape": SHAPE,
        "model_kwargs": MODEL_KWARGS,
        "petr": run(args.device),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(fixture, indent=1, sort_keys=True) + "\n")
    print(f"parameters: {fixture['petr']['parameters']:,}")
    print("losses:", fixture["petr"]["losses"])
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
