"""
Stacks encoder layers that apply:
1. Deformable self-attention among BEV queries
2. Spatial cross-attention from BEV queries to multi-view image features
3. Feed-forward network
"""


import torch
import torch.nn as nn
import torch.utils.checkpoint as cp

from py123detection.torch.models.bevformer.deformable_attention import (
    MultiScaleDeformableAttention,
)
from py123detection.torch.models.bevformer.spatial_cross_attention import (
    SpatialCrossAttention,
)


class BEVFormerEncoderLayer(nn.Module):
    """One layer of the BEV encoder.

    Pre-norm, three blocks, each with a residual: deformable self-attention over the BEV grid, so
    a cell can read its neighbours; spatial cross-attention into the camera features, so it can
    read the images; then a feed-forward network.

    :param embed_dims: Feature dimension.
    :param num_heads: Attention heads.
    :param feedforward_channels: Hidden width of the feed-forward network.
    :param dropout: Dropout rate.
    :param num_points_self: Sampling points for the self-attention.
    :param num_points_cross: Sampling points for the cross-attention.
    :param num_heights: Pillar height count, which has to match the ``num_Z_anchors`` dimension
        of the 3D reference points.
    :param with_cp: Gradient-checkpoint the layer.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        feedforward_channels: int = 512,
        dropout: float = 0.1,
        num_points_self: int = 4,
        num_points_cross: int = 8,
        num_heights: int = 4,
        with_cp: bool = True,
    ):
        super().__init__()
        self.use_checkpoint = with_cp

        # Deformable self-attention on the BEV grid
        self.self_attn = MultiScaleDeformableAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_levels=1,
            num_points=num_points_self,
        )
        self.norm1 = nn.LayerNorm(embed_dims)

        # Spatial cross-attention to image features (height-aware via
        # MSDeformableAttention3D inside SCA).
        self.cross_attn = SpatialCrossAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_points=num_points_cross,
            num_heights=num_heights,
            dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(embed_dims)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, feedforward_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feedforward_channels, embed_dims),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(embed_dims)

    def _forward(
        self,
        bev_queries: torch.Tensor,
        bev_pos: torch.Tensor,
        image_features: torch.Tensor,
        reference_points_3d: torch.Tensor,
        bev_ref_2d: torch.Tensor,
        spatial_shapes_bev: torch.Tensor,
        level_start_index_bev: torch.Tensor,
        imu2img: torch.Tensor,
        cam_mask: torch.Tensor,
        pad_h: int,
        pad_w: int,
    ) -> torch.Tensor:
        """
        :param bev_queries: ``(B, Q_bev, C)``, the BEV features as they stand.
        :param bev_pos: ``(B, Q_bev, C)`` positional encoding.
        :param image_features: ``(B, N, C, H, W)``.
        :param reference_points_3d: ``(Q_bev, num_heights, 3)`` pillar points.
        :param bev_ref_2d: ``(1, Q_bev, 1, 2)`` reference points for the self-attention.
        :param spatial_shapes_bev: ``(1, 2)``, the grid's ``(bev_h, bev_w)``.
        :param level_start_index_bev: ``(1,)``, always ``[0]``: there is one BEV level.
        :param imu2img: ``(B, N, 4, 4)``.
        :param cam_mask: ``(B, N)`` camera validity.
        :param pad_h: Padded image height.
        :param pad_w: Padded image width.
        :returns: ``(B, Q_bev, C)``, the updated BEV features.
        """
        B = bev_queries.shape[0]

        # 1. Deformable self-attention on BEV
        residual = bev_queries
        bev_queries = self.norm1(bev_queries)
        bev_queries = residual + self.self_attn(
            query=bev_queries + bev_pos,
            reference_points=bev_ref_2d.expand(B, -1, -1, -1),
            value=bev_queries,
            spatial_shapes=spatial_shapes_bev,
            level_start_index=level_start_index_bev,
        )

        # 2. Spatial cross-attention to image features
        residual = bev_queries
        bev_queries = self.norm2(bev_queries)
        bev_queries = residual + self.cross_attn(
            bev_queries=bev_queries + bev_pos,
            image_features=image_features,
            reference_points_3d=reference_points_3d,
            imu2img=imu2img,
            cam_mask=cam_mask,
            pad_h=pad_h,
            pad_w=pad_w,
        )

        # 3. FFN
        residual = bev_queries
        bev_queries = self.norm3(residual + self.ffn(bev_queries))

        return bev_queries

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward with optional gradient checkpointing."""
        if self.use_checkpoint and self.training:
            return cp.checkpoint(
                self._forward, *args, use_reentrant=False, **kwargs
            )
        return self._forward(*args, **kwargs)


class BEVFormerEncoder(nn.Module):
    """A stack of :class:`BEVFormerEncoderLayer`.

    :param num_layers: How many layers; 3 is BEVFormer-S.
    :param embed_dims: Feature dimension.
    :param num_heads: Attention heads.
    :param feedforward_channels: Hidden width of each layer's feed-forward network.
    :param dropout: Dropout rate.
    :param num_points_self: Sampling points for the self-attention.
    :param num_points_cross: Sampling points for the cross-attention.
    :param num_heights: Pillar height count for the spatial cross-attention.
    :param with_cp: Gradient-checkpoint each layer.
    """

    def __init__(
        self,
        num_layers: int = 3,
        embed_dims: int = 256,
        num_heads: int = 8,
        feedforward_channels: int = 512,
        dropout: float = 0.1,
        num_points_self: int = 4,
        num_points_cross: int = 8,
        num_heights: int = 4,
        with_cp: bool = True,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.embed_dims = embed_dims

        self.layers = nn.ModuleList([
            BEVFormerEncoderLayer(
                embed_dims=embed_dims,
                num_heads=num_heads,
                feedforward_channels=feedforward_channels,
                dropout=dropout,
                num_points_self=num_points_self,
                num_points_cross=num_points_cross,
                num_heights=num_heights,
                with_cp=with_cp,
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        bev_queries: torch.Tensor,
        bev_pos: torch.Tensor,
        image_features: torch.Tensor,
        reference_points_3d: torch.Tensor,
        bev_ref_2d: torch.Tensor,
        spatial_shapes_bev: torch.Tensor,
        level_start_index_bev: torch.Tensor,
        imu2img: torch.Tensor,
        cam_mask: torch.Tensor,
        pad_h: int,
        pad_w: int,
    ) -> torch.Tensor:
        """Run BEV queries through all encoder layers.

        Returns:
            bev_queries: (B, Q_bev, C) final BEV features.
        """
        for layer in self.layers:
            bev_queries = layer(
                bev_queries=bev_queries,
                bev_pos=bev_pos,
                image_features=image_features,
                reference_points_3d=reference_points_3d,
                bev_ref_2d=bev_ref_2d,
                spatial_shapes_bev=spatial_shapes_bev,
                level_start_index_bev=level_start_index_bev,
                imu2img=imu2img,
                cam_mask=cam_mask,
                pad_h=pad_h,
                pad_w=pad_w,
            )
        return bev_queries
