"""Telling a BEVFormer head which camera a feature came from.

The paper adds a learned vector per camera to the image features before the BEV encoder, so the
encoder can treat the front camera differently from the rear one. That table is indexed by camera
position, which locks a model to one rig: train with it on nuScenes' six cameras and the weights
mean nothing on Argoverse's seven.

``"none"`` is therefore the right choice for cross-dataset training, and ``"bevformer"`` for
reproducing the paper on a single dataset. The embedding is added once, before the encoder, which
is where the official implementation puts it.
"""

from typing import Optional

import torch
import torch.nn as nn

#: Accepted values of ``cam_embed_type``.
VALID_CAM_EMBED_TYPES = ("none", "bevformer")


class CameraEmbedding(nn.Module):
    """Abstract base. Forward returns a tensor to be ADDED to
    ``image_features`` (same shape ``(B, N, embed_dims, H, W)``)."""

    def forward(
        self,
        image_features: torch.Tensor,
        *,
        imu2img: Optional[torch.Tensor] = None,
        cam_mask: Optional[torch.Tensor] = None,
        pad_h: Optional[int] = None,
        pad_w: Optional[int] = None,
    ) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError


class BEVFormerCamEmbed(CameraEmbedding):
    """BEVFormer's own learnable per-camera embedding.

    This is the ``cams_embeds`` parameter of the official implementation's transformer: one vector
    per camera, normal-initialised, broadcast-added to that camera's features.

    The table is indexed by camera position, so the number of cameras is fixed at construction and
    a batch with a different ``N`` raises rather than silently reusing the wrong rows. That is the
    official design, and the error is the point: these weights mean nothing on another rig.

    There is one FPN level here, so there are no ``level_embeds`` to go with it. Adding multi-scale
    features later means adding both parameters together, since the official code adds them in the
    same place.

    :param num_cameras: How many cameras the model is trained on, 6 for nuScenes. A forward pass
        with any other count raises.
    :param embed_dims: Feature dimension, matching ``image_features``.
    """

    def __init__(self, num_cameras: int, embed_dims: int):
        super().__init__()
        self.num_cameras = num_cameras
        self.embed_dims = embed_dims
        self.cams_embeds = nn.Parameter(
            torch.empty(num_cameras, embed_dims)
        )
        # Matches official BEVFormer init (nn.init.normal_ with defaults).
        nn.init.normal_(self.cams_embeds)

    def forward(
        self,
        image_features: torch.Tensor,
        *,
        cam_mask: Optional[torch.Tensor] = None,
        **_ignored,
    ) -> torch.Tensor:
        B, N, C, H, W = image_features.shape
        if N != self.num_cameras:
            raise ValueError(
                f"BEVFormerCamEmbed is locked to num_cameras="
                f"{self.num_cameras} but got N={N}. "
                "Use cam_embed_type='none' for variable camera counts "
                "(cross-dataset training)."
            )
        # (1, N, C, 1, 1) → broadcast-expand to image_features shape.
        pe = self.cams_embeds.view(1, N, C, 1, 1).to(
            dtype=image_features.dtype
        ).expand(B, N, C, H, W)

        # Zero out padded cameras so their random learned embedding does
        # not bias SCA's valid-query aggregation. Gradients still flow
        # into cams_embeds only from valid cams.
        if cam_mask is not None:
            pe = pe * cam_mask.view(B, N, 1, 1, 1).to(pe.dtype)
        return pe


def build_camera_embedding(
    cam_embed_type: str,
    *,
    embed_dims: int,
    num_cameras: int,
) -> Optional[CameraEmbedding]:
    """Build the per-camera embedding a BEVFormer head adds to its features.

    :param cam_embed_type: ``"bevformer"`` for the paper's learned per-camera token, or
        ``"none"``. The learned token is indexed by camera position, so it fixes the model to one
        rig; ``"none"`` is what lets a model train across datasets with different camera counts.
    :param num_cameras: Size of the learned table. Only meaningful for ``"bevformer"``.
    """
    if cam_embed_type not in VALID_CAM_EMBED_TYPES:
        raise ValueError(
            f"Unknown cam_embed_type={cam_embed_type!r}; must be one of {VALID_CAM_EMBED_TYPES}."
        )
    if cam_embed_type == "none":
        return None
    return BEVFormerCamEmbed(num_cameras=num_cameras, embed_dims=embed_dims)
