"""What every camera-based detector in this package does before its head.

Flatten the rig's images into a batch, run one shared backbone over them, fold them back into
per-camera features, and make sure the projection matrices are there. PETR and BEVFormer differ
entirely in what happens next — queries attending to image features, or an explicit bird's-eye
grid — so that is the only thing a subclass supplies.

Keeping this separate from :class:`~py123detection.torch.models.base.Detector3D` keeps the
Lightning loop free of anything camera-specific, and keeps the two detectors siblings. In the
thesis ``BEVFormer`` subclassed ``PETR``, so every change to PETR's forward was a silent change
to BEVFormer's.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import torch

from py123detection.torch.models.base import Detector3D
from py123detection.torch.models.common.backbone import FeatureExtractor
from py123detection.torch.models.common.grid_mask import GridMask

logger = logging.getLogger(__name__)

#: Stride of the shared ResNet backbone: an image becomes a grid this many pixels per cell.
BACKBONE_STRIDE = 16

#: Detection range in the ego (IMU) frame. The z bounds are mmdet3d's lidar-frame ``[-5, 3]``
#: shifted up by two metres, because the IMU sits over the rear axle, well below a roof lidar.
DEFAULT_PC_RANGE = (-51.2, -51.2, -3.0, 51.2, 51.2, 5.0)

#: ImageNet statistics in RGB order, which the pretrained backbones expect.
IMAGENET_MEAN = (123.675, 116.28, 103.53)
IMAGENET_STD = (58.395, 57.12, 57.375)


class MultiViewDetector(Detector3D):
    """A shared backbone over a rig of cameras, plus a head a subclass chooses.

    :param num_classes: Size of the classification head; must match the taxonomy's class count.
    :param num_query: Object queries. Each may become one detection, so this caps how many
        objects a frame can hold.
    :param pc_range: ``(x_min, y_min, z_min, x_max, y_max, z_max)`` the box regression is
        normalised against. A prediction outside it cannot be expressed.
    :param image_size: ``(height, width)`` of the padded network input.
    :param use_grid_mask: Blank a random grid of squares in each image during training. Cheap and
        effective here, where a model otherwise learns to lean on whichever view is easiest.
    :param pixel_mean: Per-channel RGB mean subtracted from ``uint8`` input, on the device. The
        statistics belong to the backbone's pretraining, so they live here and not in the dataset.
    :param pixel_std: Per-channel RGB standard deviation, see ``pixel_mean``.
    """

    def __init__(
        self,
        num_classes: int = 10,
        num_query: int = 900,
        num_decoder_layers: int = 6,
        hidden_dim: int = 256,
        backbone_name: str = "resnet50",
        pc_range: Optional[List[float]] = None,
        image_size: tuple = (320, 800),
        use_grid_mask: bool = True,
        grid_mask_ratio: float = 0.5,
        grid_mask_prob: float = 0.7,
        backbone_checkpointing: bool = False,
        head_checkpointing: bool = False,
        pretrained_weights: Optional[str] = None,
        pixel_mean: Sequence[float] = IMAGENET_MEAN,
        pixel_std: Sequence[float] = IMAGENET_STD,
        **trainer_kwargs: Any,
    ) -> None:
        super().__init__(**trainer_kwargs)

        # Not persistent: they are configuration, and a checkpoint from before they existed must
        # still load.
        for name, values in (("pixel_mean", pixel_mean), ("pixel_std", pixel_std)):
            self.register_buffer(name, torch.tensor(values, dtype=torch.float32).view(1, 1, 3, 1, 1), persistent=False)

        self.num_classes = num_classes
        self.pc_range = pc_range or list(DEFAULT_PC_RANGE)
        self.image_size = tuple(image_size)

        self.grid_mask = (
            GridMask(True, True, rotate=1, offset=False, ratio=grid_mask_ratio, mode=1, prob=grid_mask_prob)
            if use_grid_mask
            else None
        )
        self.backbone = FeatureExtractor(
            out_channels=hidden_dim,
            return_layers={"layer3": "0", "layer4": "1"},
            backbone=backbone_name,
            frozen_stages=-1,
            norm_eval=True,  # the official config freezes batch norm
            with_cp=backbone_checkpointing,
        )

        height, width = self.image_size
        self.head = self.build_head(
            num_classes=num_classes,
            in_channels=hidden_dim,
            num_query=num_query,
            num_decoder_layers=num_decoder_layers,
            pc_range=self.pc_range,
            with_cp=head_checkpointing,
            patches_x=width // BACKBONE_STRIDE,
            patches_y=height // BACKBONE_STRIDE,
            image_width=width,
            image_height=height,
        )
        self.head.init_weights()

        if pretrained_weights:
            self.load_pretrained(pretrained_weights)

    def build_head(self, **kwargs: Any):
        """The detector's head. This is the one thing a subclass has to provide."""
        raise NotImplementedError

    def extract_features(self, images: torch.Tensor) -> List[torch.Tensor]:
        """Backbone features per camera, as ``(B, N, C, H, W)`` per level.

        Features come back in fp32 even under mixed precision: the heads run attention over them
        and a trained backbone's magnitudes overflow fp16 there.
        """
        batch_size, cameras = images.shape[:2]
        flattened = images.flatten(0, 1)
        if self.grid_mask is not None and self.training:
            flattened = self.grid_mask(flattened)
        return [
            level.view(batch_size, cameras, *level.shape[1:]).float()
            for level in self.backbone(flattened)
        ]

    def normalize_images(self, batch: Dict[str, Any]) -> torch.Tensor:
        """``batch["img"]`` as the backbone's input.

        The dataset delivers raw ``uint8`` pixels, a quarter of the bytes of normalised floats
        through the worker queues and the host-to-device copy, and cheaper to produce: on the CPU,
        normalising cost more than decoding the JPEG. Here it is one fused operation on the device.

        Everything that is not an image pixel — the padding up to the stride divisor, a camera
        padded in to match a wider rig, a camera with no image this frame — is set to zero after
        normalisation, i.e. to the mean colour, as the float pipeline always did.

        A floating-point ``img`` is taken as already normalised and returned unchanged.
        """
        images = batch["img"]
        if images.is_floating_point():
            return images

        normalized = (images.float() - self.pixel_mean) / self.pixel_std
        batch_size, cameras, _, height, width = images.shape

        keep = torch.ones(batch_size, cameras, 1, height, width, dtype=torch.bool, device=images.device)
        shapes = batch.get("img_shape")
        if torch.is_tensor(shapes):
            shapes = shapes.to(images.device).reshape(batch_size, 2)
            rows = torch.arange(height, device=images.device).view(1, height, 1) < shapes[:, 0].view(-1, 1, 1)
            columns = torch.arange(width, device=images.device).view(1, 1, width) < shapes[:, 1].view(-1, 1, 1)
            keep = keep & (rows & columns).view(batch_size, 1, 1, height, width)
        cam_mask = batch.get("cam_mask")
        if torch.is_tensor(cam_mask):
            keep = keep & cam_mask.to(images.device).view(batch_size, cameras, 1, 1, 1)
        return normalized.masked_fill_(~keep, 0.0)

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        images = self.normalize_images(batch)
        features = self.extract_features(images)

        intrinsics, extrinsics = batch["intrinsics"], batch["extrinsics"]
        ego_to_image = batch.get("imu2img")
        if ego_to_image is None:
            ego_to_image = torch.matmul(intrinsics, extrinsics)

        with torch.autocast(device_type=images.device.type, enabled=False):
            return self.head(
                features,
                img_metas=self.image_shapes(batch, images.shape[1]),
                cam_mask=batch.get("cam_mask"),
                imu2img=ego_to_image,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
            )

    @staticmethod
    def image_shapes(batch: Dict[str, Any], cameras: int) -> List[Dict[str, Any]]:
        """Per-sample image and padded shapes, which the head masks padding with.

        A sample from the dataset carries them; a bare tensor batch does not, so they are derived
        from the tensor and the padding mask becomes a no-op.
        """
        metas = batch.get("img_metas")
        if isinstance(metas, list) and metas and isinstance(metas[0], dict):
            return metas

        images = batch["img"]
        padded = (int(images.shape[-2]), int(images.shape[-1]))
        shapes = batch.get("img_shape")
        if torch.is_tensor(shapes):
            shapes = shapes.reshape(-1, 2).detach().cpu()
            unpadded = [(int(row[0]), int(row[1])) for row in shapes]
        else:
            unpadded = [padded] * images.shape[0]

        return [
            {"img_shape": [shape] * cameras, "pad_shape": [padded] * cameras}
            for shape in unpadded
        ]

    def load_pretrained(self, path: str) -> None:
        """Load weights, keeping only what fits.

        A checkpoint trained on a different class count has a head that cannot be reused; the
        backbone still can, and that is most of the value. What did not fit is reported rather
        than dropped in silence.
        """
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        weights = checkpoint.get("state_dict", checkpoint)

        current = self.state_dict()
        usable = {
            key: value for key, value in weights.items()
            if key in current and current[key].shape == value.shape
        }
        skipped = sorted(set(weights) - set(usable))

        self.load_state_dict(usable, strict=False)
        logger.info("Loaded %d of %d tensors from %s", len(usable), len(weights), path)
        if skipped:
            logger.info("Skipped %d tensor(s) that do not fit: %s", len(skipped), skipped[:8])
