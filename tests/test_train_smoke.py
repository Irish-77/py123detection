"""One real training step, end to end.

Nothing here ran a backward pass before. ``test_training_loop.py`` drives a stub head whose loss is
a constant, and the two golden fixtures do a forward pass and compare numbers — so the optimiser,
the schedule, ``configure_optimizers`` and the gradient path through the real backbone and head were
exercised only by hand. ``conf/trainer/smoke.yaml`` exists to prove the loop runs end to end and
nothing automated invoked it.

The model here is as small as the architecture allows: one decoder layer, two cameras, a 64x160
input. It is not meant to learn anything, only to show that a step completes, the loss is finite,
gradients reach the backbone, and the parameters actually move. The overfitting test then checks the
one thing shapes cannot: that the loss goes down.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="the torch branch needs torch")
pytest.importorskip("torchvision")
pytest.importorskip("scipy")

#: Needs the torch extra; see the `torch` marker in pyproject.toml.
pytestmark = pytest.mark.torch

import lightning as L  # noqa: E402

from py123detection.torch.models import PETR  # noqa: E402

SEED = 20261108
#: Small but structurally real: the backbone stride is 16, so both sides stay divisible by it.
SHAPE = {"batch": 1, "cameras": 2, "height": 64, "width": 160}
NUM_CLASSES = 3
PC_RANGE = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
POSITION_RANGE = [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0]


def tiny_model(**overrides) -> PETR:
    kwargs = dict(
        num_classes=NUM_CLASSES,
        num_query=10,
        num_decoder_layers=1,
        # 256 and nothing else: see test_hidden_dim_only_works_at_its_default.
        hidden_dim=256,
        backbone_name="resnet50",
        pc_range=PC_RANGE,
        position_range=POSITION_RANGE,
        image_size=(SHAPE["height"], SHAPE["width"]),
        use_grid_mask=False,
        backbone_checkpointing=False,
        head_checkpointing=False,
    )
    kwargs.update(overrides)
    return PETR(**kwargs)


def tiny_batch(boxes_per_frame: int = 3, dataset: str = "nuscenes"):
    """A batch shaped the way ``collate_rigs`` produces one, including ``img_metas``."""
    generator = torch.Generator().manual_seed(SEED)
    batch, cameras = SHAPE["batch"], SHAPE["cameras"]

    images = torch.randn(
        batch, cameras, 3, SHAPE["height"], SHAPE["width"], generator=generator
    )

    intrinsics = torch.eye(4).repeat(batch, cameras, 1, 1)
    intrinsics[..., 0, 0] = intrinsics[..., 1, 1] = 200.0
    intrinsics[..., 0, 2] = SHAPE["width"] / 2
    intrinsics[..., 1, 2] = SHAPE["height"] / 2

    extrinsics = torch.eye(4).repeat(batch, cameras, 1, 1)
    for camera in range(cameras):
        angle = 2.0 * torch.pi * camera / cameras
        extrinsics[:, camera, :3, :3] = torch.tensor(
            [
                [torch.cos(torch.tensor(angle)), -torch.sin(torch.tensor(angle)), 0.0],
                [torch.sin(torch.tensor(angle)), torch.cos(torch.tensor(angle)), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        extrinsics[:, camera, :3, 3] = torch.tensor([0.0, 0.0, -1.6])

    boxes, labels = [], []
    for _ in range(batch):
        box = torch.zeros(boxes_per_frame, 9)
        box[:, 0] = torch.linspace(-20.0, 20.0, boxes_per_frame)
        box[:, 1] = torch.linspace(10.0, -10.0, boxes_per_frame)
        box[:, 2] = -1.0
        box[:, 3:6] = torch.tensor([1.9, 4.5, 1.6])
        box[:, 6] = torch.linspace(-1.0, 1.0, boxes_per_frame)
        boxes.append(box)
        labels.append(torch.arange(boxes_per_frame) % NUM_CLASSES)

    return {
        "img": images,
        "intrinsics": intrinsics,
        "extrinsics": extrinsics,
        "imu2img": torch.matmul(intrinsics, extrinsics),
        "cam_mask": torch.ones(batch, cameras, dtype=torch.bool),
        "gt_bboxes_3d": boxes,
        "gt_labels_3d": labels,
        # image_shapes() returns img_metas as-is when present, so these have to be complete:
        # the head reads pad_shape to map feature positions back to pixels.
        "img_metas": [
            {
                "dataset_name": dataset,
                "img_shape": [(SHAPE["height"], SHAPE["width"])] * cameras,
                "pad_shape": [(SHAPE["height"], SHAPE["width"])] * cameras,
            }
            for _ in range(batch)
        ],
    }


class OneBatch(torch.utils.data.Dataset):
    """The same batch every time, handed over already collated."""

    def __init__(self, batch, length: int = 2) -> None:
        self.batch = batch
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index):
        return self.batch


def loader_for(batch, length: int = 2) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        OneBatch(batch, length), batch_size=None, collate_fn=lambda value: value
    )


def test_a_training_step_produces_a_finite_loss_and_moves_the_weights():
    """Forward, loss, backward, step — by hand, so a failure says which part broke."""
    torch.manual_seed(SEED)
    model = tiny_model().train()
    batch = tiny_batch()

    # A plain optimiser, because configure_optimizers derives its schedule from
    # trainer.estimated_stepping_batches and needs a live Trainer; that path is covered by
    # test_the_loop_runs_end_to_end_under_a_trainer below.
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    before = {name: tensor.detach().clone() for name, tensor in model.named_parameters()}

    loss = model.training_step(batch, 0)
    assert torch.is_tensor(loss) and loss.ndim == 0, "training_step should return a scalar"
    assert torch.isfinite(loss), f"loss was {loss}"
    assert loss.requires_grad

    loss.backward()
    head_grads = [
        parameter.grad for name, parameter in model.named_parameters()
        if name.startswith("head.") and parameter.grad is not None
    ]
    assert head_grads, "no gradient reached the head"
    assert all(torch.isfinite(grad).all() for grad in head_grads)

    optimizer.step()
    moved = [
        name for name, tensor in model.named_parameters()
        if not torch.equal(tensor.detach(), before[name])
    ]
    assert moved, "the optimiser step changed nothing"


def test_gradients_reach_the_backbone_and_not_its_frozen_norms():
    """The backbone is trainable but its BatchNorms are deliberately frozen."""
    torch.manual_seed(SEED)
    model = tiny_model().train()
    model.training_step(tiny_batch(), 0).backward()

    backbone = [
        (name, parameter) for name, parameter in model.named_parameters()
        if name.startswith("backbone.")
    ]
    assert backbone, "no backbone parameters"
    with_grad = [name for name, parameter in backbone if parameter.grad is not None]
    assert with_grad, "no gradient reached the backbone at all"

    frozen = [name for name, parameter in backbone if not parameter.requires_grad]
    assert frozen, "the backbone's norm layers are supposed to be frozen"


def test_a_frame_with_no_boxes_still_steps():
    """The empty-frame path, through the real loss rather than a stub."""
    torch.manual_seed(SEED)
    model = tiny_model().train()
    loss = model.training_step(tiny_batch(boxes_per_frame=0), 0)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_the_loop_runs_end_to_end_under_a_trainer():
    """What ``trainer=smoke`` is for: configure_optimizers, the schedule, train and validate."""
    torch.manual_seed(SEED)
    model = tiny_model()
    batch = tiny_batch()
    before = {name: tensor.detach().clone() for name, tensor in model.named_parameters()}

    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        fast_dev_run=True,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model, train_dataloaders=loader_for(batch), val_dataloaders=loader_for(batch))

    assert trainer.state.finished
    moved = [
        name for name, tensor in model.named_parameters()
        if not torch.equal(tensor.detach(), before[name])
    ]
    assert moved, "fast_dev_run did not take an optimiser step"


def test_the_loss_falls_when_overfitting_one_batch():
    """The one property shapes cannot show: the thing actually trains.

    Five steps on a single batch at a high learning rate. Not a convergence claim — just that the
    gradient points somewhere useful rather than nowhere, which a wrongly wired matcher or a
    detached loss term would break while every other test here still passed.
    """
    torch.manual_seed(SEED)
    model = tiny_model().train()
    batch = tiny_batch()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

    losses = []
    for _ in range(5):
        optimizer.zero_grad()
        loss = model.training_step(batch, 0)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert all(torch.isfinite(torch.tensor(value)) for value in losses), losses
    assert losses[-1] < losses[0], f"loss did not fall over five steps: {losses}"


def test_validation_reports_a_loss_per_dataset():
    """A mixed batch must stay separable, which is the whole reason validation is grouped."""
    torch.manual_seed(SEED)
    model = tiny_model().eval()
    batch = tiny_batch(dataset="av2")
    model.on_validation_epoch_start()
    with torch.no_grad():
        loss = model.validation_step(batch, 0)

    assert torch.isfinite(loss)
    # The frame-weighted aggregate is built at epoch end from what each step records, so the record
    # has to carry the frame count as well as the loss.
    assert len(model._validation_losses) == 1
    recorded, frames = model._validation_losses[0]
    assert torch.isfinite(recorded) and frames == len(batch["gt_labels_3d"])


def test_hidden_dim_only_works_at_its_default():
    """``hidden_dim`` is a constructor parameter that accepts exactly one value.

    ``PETRHead`` hardcodes ``self.embed_dims = 256`` (``petr/head.py:54``) and projects the
    backbone's ``in_channels`` up to it, but the position-embedding reshape later uses the
    detector's ``hidden_dim``. The two agree only at 256, and anything else dies in a forward pass
    with a shape error that names neither parameter.

    Every shipped config sets 256, so nothing in the tree is broken — this is a trap rather than a
    bug, and it is recorded here so that whoever makes ``hidden_dim`` mean something sees a failing
    test rather than having to rediscover this.
    """
    model = tiny_model(hidden_dim=64)
    with pytest.raises(RuntimeError, match="invalid for input of size"):
        model(tiny_batch())
