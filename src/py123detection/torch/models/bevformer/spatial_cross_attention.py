"""
Core "backward projection" mechanism: BEV queries reach into multi-view image
features by projecting 3D pillar reference points to each camera view and
applying deformable attention at the projected locations.
"""

import torch
import torch.nn as nn

from py123detection.torch.models.bevformer.deformable_attention import (
    MultiScaleDeformableAttention3D,
)
from py123detection.torch.models.bevformer.utils import project_3d_to_cameras


class SpatialCrossAttention(nn.Module):
    """Spatial Cross-Attention module with valid-query gating.

    For each BEV query:
    1. Projects its 3D pillar reference points to all camera views.
    2. Gates: only queries with at least one valid projection in a camera
       participate in that camera's attention.
    3. Batches all cameras into a single 3D deformable attention call.
    4. Scatters per-camera contributions back to the full BEV grid.
    5. Normalizes by the number of cameras each query appeared in.

    Args:
        embed_dims: Feature dimension (default: 256).
        num_heads: Number of attention heads (default: 8).
        num_points: Deformable attention sampling points per head (default: 8).
        num_heights: Height samples per pillar (num_Z_anchors) (default: 4).
        dropout: Dropout rate (default: 0.1).
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        num_points: int = 8,
        num_heights: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_heights = num_heights

        # Height-aware deformable attention: one set of num_points offsets
        # per query per head, shared across num_Z_anchors heights.
        self.deformable_attn = MultiScaleDeformableAttention3D(
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_levels=1,
            num_points=num_points,
            num_Z_anchors=num_heights,
        )
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        bev_queries: torch.Tensor,
        image_features: torch.Tensor,
        reference_points_3d: torch.Tensor,
        imu2img: torch.Tensor,
        cam_mask: torch.Tensor,
        pad_h: int,
        pad_w: int,
    ) -> torch.Tensor:
        """Forward pass with valid-query gating.

        Args:
            bev_queries: (B, Q, C) BEV query features (with positional
                encoding already added).
            image_features: (B, N, C, H, W) multi-view image features.
            reference_points_3d: (Q, num_heights, 3) 3D pillar reference
                points in world coordinates (shared across batch).
            imu2img: (B, N, 4, 4) projection matrices.
            cam_mask: (B, N) bool. True = valid camera, False = padded.
            pad_h: Padded image height.
            pad_w: Padded image width.

        Returns:
            output: (B, Q, C) aggregated BEV features.
        """
        B, Q, C = bev_queries.shape
        N = image_features.shape[1]
        H, W = image_features.shape[3], image_features.shape[4]
        P = reference_points_3d.shape[1]  # num_heights
        assert P == self.num_heights, (
            f"Reference points height count ({P}) does not match "
            f"SCA.num_heights ({self.num_heights})"
        )

        # Expand 3D reference points for batch: (B, Q, P, 3)
        ref_3d = reference_points_3d[None].expand(B, -1, -1, -1)

        # Project to all cameras
        #   coords_2d: (B, N, Q, P, 2) in [0, 1]
        #   valid:     (B, N, Q, P)
        coords_2d, valid = project_3d_to_cameras(
            ref_3d, imu2img, pad_h, pad_w
        )

        # Apply camera validity mask: padded cameras (cam_mask=False) are
        # fully invalid for all queries/heights.
        if cam_mask is not None:
            valid = valid & cam_mask[:, :, None, None]

        # Per-(batch, camera) BEV mask: True where ANY height projects validly.
        # Shape: (B, N, Q)
        bev_mask = valid.any(dim=-1)

        # ---- Gating: find valid queries per (b, cam) ----
        #
        # This uses .nonzero(), which is data-dependent and causes a graph
        # break under torch.compile. However it now happens ONCE per encoder
        # layer rather than once per camera, reducing total breaks 6x.
        #
        # indexes[cam_idx][b] is a 1-D LongTensor of BEV query indices that
        # camera `cam_idx` observes for batch element `b`.
        # TODO: re-consider this approach and find torch compile-friendly code
        indexes = []
        for cam_idx in range(N):
            batch_indexes = [
                bev_mask[b, cam_idx].nonzero().squeeze(-1)
                for b in range(B)
            ]
            indexes.append(batch_indexes)

        max_len = 1  # guard against all-zero masks
        for batch_list in indexes:
            for idx in batch_list:
                if idx.numel() > max_len:
                    max_len = int(idx.numel())

        # ---- Pack valid queries and refs into (B, N, max_len, ...) ----
        queries_rebatch = bev_queries.new_zeros(B, N, max_len, C)
        ref_rebatch = coords_2d.new_zeros(B, N, max_len, P, 2)
        for b in range(B):
            for cam_idx in range(N):
                idx = indexes[cam_idx][b]
                L = int(idx.numel())
                if L == 0:
                    continue
                queries_rebatch[b, cam_idx, :L] = bev_queries[b, idx]
                ref_rebatch[b, cam_idx, :L] = coords_2d[b, cam_idx, idx]

        # ---- Flatten image features: (B, N, C, H, W) -> (B*N, H*W, C) ----
        feat_flat = image_features.flatten(0, 1).flatten(2).permute(0, 2, 1)

        spatial_shapes = torch.tensor(
            [[H, W]], device=bev_queries.device, dtype=torch.long
        )
        level_start_index = torch.tensor(
            [0], device=bev_queries.device, dtype=torch.long
        )

        # ---- ONE batched 3D deformable attention across all cameras ----
        attn_out = self.deformable_attn(
            query=queries_rebatch.view(B * N, max_len, C),
            reference_points=ref_rebatch.view(B * N, max_len, P, 2),
            value=feat_flat,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
        ).view(B, N, max_len, C)

        # ---- Scatter back to the full BEV grid ----
        output = bev_queries.new_zeros(B, Q, C)
        for b in range(B):
            for cam_idx in range(N):
                idx = indexes[cam_idx][b]
                L = int(idx.numel())
                if L == 0:
                    continue
                # index_add-style accumulation (a query is reached by
                # multiple cameras, so we sum contributions).
                output[b].index_add_(0, idx, attn_out[b, cam_idx, :L])

        # ---- Normalize by camera count per query ----
        # Height averaging already happened inside MSDA3D, so we divide by
        # the number of cameras each BEV query was visible in. This matches
        # BEVFormer/.../spatial_cross_attention.py
        count = bev_mask.sum(dim=1).float().clamp(min=1.0)  # (B, Q)
        output = output / count.unsqueeze(-1)

        return self.output_proj(self.dropout(output))
