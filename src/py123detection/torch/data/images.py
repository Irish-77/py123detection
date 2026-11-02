"""How a camera image becomes network input: one affine map, applied once.

Every step between the sensor and the network is an axis-aligned affine map of pixel
coordinates: the rig's fixed row crop and resize (:mod:`~py123detection.torch.data.rigs`), the
scale that makes the image cover the network input, the crop window, the horizontal flip. So the
whole chain is one 3x3 matrix ``A``, and it is used twice::

    image_out  = warpAffine(image, A)     # one resampling of the source pixels
    intrinsics = A @ intrinsics           # the camera that would have taken image_out

Because both come from the same matrix, pixels and projection cannot disagree. The thesis
pipeline resized twice (to the rig's size, then by the augmentation scale), which cost a second
interpolation and left the intrinsics and the pixels a fraction of a pixel apart, because
``cv2.resize`` samples at pixel centres and the matrix scaled about the corner.

What is chosen per sample lives in :class:`ImageDraw` — a zoom, a horizontal crop position, a
flip — and is shared by every camera of the frame. :meth:`ImagePipeline.resolve` turns a draw into
the concrete transform for one camera's size. The validation path is the default draw: no zoom,
centred, not flipped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Union

import cv2
import numpy as np


@dataclass(frozen=True)
class ImageDraw:
    """The per-sample choices of the image transform, independent of any camera's size."""

    zoom: float = 1.0
    """Scale relative to the smallest one at which the image covers the network input. Above one
    the network sees a magnified part of the view; the validation path uses exactly one."""

    crop_x: float = 0.5
    """Where the crop window sits in the horizontal slack, from 0 (left) to 1 (right)."""

    flip: bool = False


@dataclass(frozen=True)
class ImageTransform:
    """The resize, crop and flip for one camera, resolved against its size."""

    resize: Tuple[float, float]
    """Scale applied to width and height. Equal in ``crop`` mode, independent in ``stretch`` mode."""

    crop: Tuple[int, int, int, int]
    """``(x0, y0, x1, y1)`` of the window taken out of the resized image. It may reach past the
    image's edge; what lies outside is black."""

    flip: bool

    @property
    def size(self) -> Tuple[int, int]:
        """``(height, width)`` of the result."""
        return self.crop[3] - self.crop[1], self.crop[2] - self.crop[0]

    def matrix(self) -> np.ndarray:
        """The 3x3 map from the input's pixel coordinates to the output's.

        Pixel centres sit on integer coordinates, so a window ``W`` pixels wide mirrors as
        ``x -> W - 1 - x``, which is also what ``cv2.flip`` does. PETR mirrors as ``x -> W - x``
        in the intrinsics while flipping the pixels with ``cv2.flip``, which left every flipped
        camera's projection one pixel off its image.
        """
        matrix = np.eye(3, dtype=np.float64)
        matrix[0, 0], matrix[1, 1] = self.resize
        matrix[:2, 2] = -np.asarray(self.crop[:2], dtype=np.float64)
        if self.flip:
            width = self.crop[2] - self.crop[0]
            mirror = np.array([[-1.0, 0.0, width - 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
            matrix = mirror @ matrix
        return matrix


@dataclass(frozen=True)
class ImagePipeline:
    """The network input every camera is mapped to.

    :param mode: ``"crop"`` scales by one factor until the image covers :attr:`size`, then cuts
        :attr:`size` out, bottom-aligned so the road is kept and the sky is lost. It keeps the
        pixel scale the same across cameras and is what PETR does. ``"stretch"`` resizes straight
        to :attr:`size`, distorting the aspect ratio but keeping the whole field of view; it ignores
        the zoom and the crop position of a draw.
    """

    size: Tuple[int, int] = (320, 800)
    """``(height, width)`` of the network input, before padding."""

    mode: str = "crop"

    pad_divisor: int = 32
    """Pad up to a multiple of this, so a convolutional backbone's strides divide it."""

    def __post_init__(self) -> None:
        if self.mode not in ("crop", "stretch"):
            raise ValueError(f"ImagePipeline mode must be 'crop' or 'stretch', got {self.mode!r}.")

    def resolve(self, draw: Union[ImageDraw, ImageTransform], height: int, width: int) -> ImageTransform:
        """The transform ``draw`` means for an image of ``height`` by ``width`` pixels.

        An :class:`ImageTransform` is already resolved and comes back unchanged, which is how a
        test or a debugging session replays one exact transform.
        """
        if isinstance(draw, ImageTransform):
            return draw

        target_height, target_width = self.size
        if self.mode == "stretch":
            return ImageTransform(
                resize=(target_width / width, target_height / height),
                crop=(0, 0, target_width, target_height),
                flip=draw.flip,
            )

        scale = max(target_height / height, target_width / width) * draw.zoom
        scaled_width, scaled_height = int(width * scale), int(height * scale)
        top = scaled_height - target_height
        left = int(draw.crop_x * max(0, scaled_width - target_width))
        return ImageTransform(
            resize=(scale, scale),
            crop=(left, top, left + target_width, top + target_height),
            flip=draw.flip,
        )

    def padded_size(self, height: int, width: int) -> Tuple[int, int]:
        divisor = max(1, self.pad_divisor)
        return (
            int(np.ceil(height / divisor)) * divisor,
            int(np.ceil(width / divisor)) * divisor,
        )


def warp(image: np.ndarray, matrix: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resample ``image`` through the 3x3 ``matrix`` into ``size = (height, width)``, once.

    Output pixels whose source lies outside the image are black, as in PETR's crop.
    """
    height, width = size
    return cv2.warpAffine(
        image,
        np.ascontiguousarray(matrix[:2], dtype=np.float64),
        (int(width), int(height)),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
