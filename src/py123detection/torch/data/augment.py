"""The random part of a training sample, in one place.

A sample is augmented twice over, and both halves are drawn together, once per sample:

* in the image: a zoom, a crop position and a horizontal flip (:class:`~py123detection.torch.data.images.ImageDraw`),
  shared by every camera of the frame and folded into each camera's intrinsics;
* in bird's-eye view: a rotation and a scale of the world about the ego vehicle
  (:class:`BevTransform`), applied to the boxes and folded, inverted, into every matrix that reads
  from the ego frame, so the images need no change.

Validation is the default :class:`Draw`: no zoom, a centred crop, no flip, no rotation.

**Randomness.** A dataset draws from its own generator, which the data module reseeds in every
worker from the worker's seed and the trainer's rank. So each epoch sees a fresh augmentation of
every frame, the same run repeated gives the same augmentations, and two ranks never draw the
same stream. (The first version seeded each sample from its index, which is reproducible but gave
every frame the *same* augmentation in every epoch.) To reproduce one sample exactly, pass its
:class:`Draw` to the dataset.

**Zoom is relative.** PETR's ``resize_lim`` is an absolute scale, meaningful only for one source
resolution and one output size: ``(0.8, 1.0)`` suits 1600x900 images cropped to 704x1280 and is
wrong by a factor of two at 320x800. Here the zoom multiplies the scale at which the image just
covers the network input, so one range works for every rig and every size. ``(1.0, 1.25)`` is
exactly what the thesis trained with on Argoverse 2 and Waymo (on nuScenes it was
``(1.025, 1.28)``), and PETR's own configs, rewritten this way, come to ``(0.94, 1.25)`` at both
320x800 and 640x1600.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np

from py123detection.torch.data.images import ImageDraw


@dataclass(frozen=True)
class BevTransform:
    """The rotation and scale chosen for one sample. The identity when not augmenting."""

    angle: float = 0.0
    scale: float = 1.0

    @property
    def is_identity(self) -> bool:
        return self.angle == 0.0 and self.scale == 1.0

    def rotation_matrix(self) -> np.ndarray:
        cos_angle, sin_angle = np.cos(self.angle), np.sin(self.angle)
        matrix = np.eye(4, dtype=np.float32)
        matrix[0, 0], matrix[0, 1] = cos_angle, -sin_angle
        matrix[1, 0], matrix[1, 1] = sin_angle, cos_angle
        return matrix

    def inverse_matrix(self) -> np.ndarray:
        """Augmented ego frame back to the original one, i.e. ``R(-angle) / scale``.

        A matrix mapping *from* the ego frame — ``imu2img``, ``imu2cam``, ``imu2global`` — is
        right-multiplied by this, because its input now arrives already rotated and scaled.
        """
        inverse_rotation = self.rotation_matrix().T
        inverse_scale = np.diag(np.array([1.0 / self.scale, 1.0 / self.scale, 1.0 / self.scale, 1.0], np.float32))
        return inverse_rotation @ inverse_scale

    def forward_matrix(self) -> np.ndarray:
        """Original ego frame to the augmented one, for matrices that map *into* it."""
        scale = np.diag(np.array([self.scale, self.scale, self.scale, 1.0], dtype=np.float32))
        return scale @ self.rotation_matrix()

    def apply_to_boxes(self, boxes: np.ndarray) -> np.ndarray:
        """Rotate and scale ``[x, y, z, w, l, h, yaw, vx, vy]`` rows.

        Centres, sizes, velocities and yaw all turn the same way. The legacy implementation turned
        the centres by ``+angle`` and the yaw by ``-angle``, which left every augmented box facing
        the wrong way by twice the angle; a symmetric range hid it on average.
        """
        if boxes.shape[0] == 0 or self.is_identity:
            return boxes

        boxes = boxes.copy()
        cos_angle, sin_angle = np.cos(self.angle), np.sin(self.angle)

        boxes[:, :3] *= self.scale
        boxes[:, 3:6] *= self.scale
        boxes[:, 7:9] *= self.scale

        for first, second in ((0, 1), (7, 8)):
            x, y = boxes[:, first].copy(), boxes[:, second].copy()
            boxes[:, first] = x * cos_angle - y * sin_angle
            boxes[:, second] = x * sin_angle + y * cos_angle

        yaw = boxes[:, 6] + self.angle
        boxes[:, 6] = np.arctan2(np.sin(yaw), np.cos(yaw))
        return boxes


@dataclass(frozen=True)
class Draw:
    """Everything random about one sample. The default is the unaugmented, validation sample."""

    image: ImageDraw = field(default_factory=ImageDraw)
    bev: BevTransform = field(default_factory=BevTransform)

    def metadata(self) -> Dict[str, object]:
        """What was drawn, for ``img_metas``."""
        return {
            "flip": bool(self.image.flip),
            "scale_factor": float(self.bev.scale),
            "rotation": float(self.bev.angle),
        }


@dataclass(frozen=True)
class Augmentation:
    """The distribution a training sample's :class:`Draw` comes from.

    A dataset without one is not augmented, which is the whole difference between training and
    validation data.
    """

    zoom_range: Tuple[float, float] = (1.0, 1.25)
    """Image scale relative to the covering one; see the module notes. Ignored in ``stretch`` mode."""

    flip: bool = True
    """Mirror the images of half the samples."""

    rotation_range: Tuple[float, float] = (-0.3925, 0.3925)
    """Bird's-eye yaw in radians, about +/- 22.5 degrees by default."""

    scale_range: Tuple[float, float] = (0.95, 1.05)
    """Bird's-eye scale of the world about the ego vehicle."""

    def draw(self, rng: np.random.Generator) -> Draw:
        return Draw(
            image=ImageDraw(
                zoom=float(rng.uniform(*self.zoom_range)),
                crop_x=float(rng.uniform(0.0, 1.0)),
                flip=bool(self.flip and rng.random() < 0.5),
            ),
            bev=BevTransform(
                angle=float(rng.uniform(*self.rotation_range)),
                scale=float(rng.uniform(*self.scale_range)),
            ),
        )
