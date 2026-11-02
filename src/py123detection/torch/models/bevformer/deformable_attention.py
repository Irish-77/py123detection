import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleDeformableAttention(nn.Module):
    """Multi-Scale Deformable Attention Module.

    For each query, this module:
    1. Predicts sampling offsets around reference points
    2. Predicts attention weights for each sampling point
    3. Samples value features at the offset locations via bilinear interpolation
    4. Computes weighted sum of sampled features

    Args:
        embed_dims: Feature dimension (default: 256).
        num_heads: Number of attention heads (default: 8).
        num_levels: Number of feature map levels (default: 1).
        num_points: Number of sampling points per head per level (default: 4).
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        num_levels: int = 1,
        num_points: int = 4,
    ):
        super().__init__()
        assert embed_dims % num_heads == 0, \
            f"embed_dims ({embed_dims}) must be divisible by num_heads ({num_heads})"

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.head_dim = embed_dims // num_heads

        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2
        )
        self.attention_weights = nn.Linear(
            embed_dims, num_heads * num_levels * num_points
        )
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)

        self._reset_parameters()

    def _reset_parameters(self):
        # Initialize sampling offsets to small values centered around 0
        nn.init.constant_(self.sampling_offsets.weight, 0.0)

        # Initialize offset biases to form a grid pattern per head
        # Each head starts with offsets forming a regular grid
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.num_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], dim=-1)
        grid_init = (
            grid_init / grid_init.abs().max(-1, keepdim=True)[0]
        )  # normalize to [-1, 1]
        grid_init = grid_init.view(
            self.num_heads, 1, 1, 2
        ).repeat(1, self.num_levels, self.num_points, 1)
        # Scale by point index so points spread out
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))

        # Initialize attention weights uniformly
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)

        # Xavier init for projections
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)

    def forward(
        self,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        value: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            query: (B, Q, C) query features.
            reference_points: (B, Q, num_levels, 2) normalized [0,1] reference
                points for each query at each level.
            value: (B, S, C) flattened multi-scale value features, where
                S = sum(H_l * W_l) across levels.
            spatial_shapes: (num_levels, 2) spatial dimensions (H, W) per level.
            level_start_index: (num_levels,) start index of each level in the
                flattened sequence.
            key_padding_mask: (B, S) True for padded/invalid positions.

        Returns:
            output: (B, Q, C) attended features.
        """
        B, Q, _ = query.shape
        B, S, _ = value.shape

        # Project values: (B, S, C) -> (B, S, num_heads, head_dim)
        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(
                key_padding_mask.unsqueeze(-1), 0.0
            )
        value = value.view(B, S, self.num_heads, self.head_dim)

        # Predict offsets: (B, Q, num_heads * num_levels * num_points * 2)
        sampling_offsets = self.sampling_offsets(query).view(
            B, Q, self.num_heads, self.num_levels, self.num_points, 2
        )

        # Predict attention weights: softmax across all points and levels
        attention_weights = self.attention_weights(query).view(
            B, Q, self.num_heads, self.num_levels * self.num_points
        )
        attention_weights = F.softmax(attention_weights, dim=-1).view(
            B, Q, self.num_heads, self.num_levels, self.num_points
        )

        # Compute sampling locations
        # reference_points: (B, Q, num_levels, 2) in [0, 1]
        # offsets are normalized by spatial_shapes so they're resolution-aware
        offset_normalizer = spatial_shapes.flip(-1)[None, None, None, :, None, :]
        sampling_locations = (
            reference_points[:, :, None, :, None, :]
            + sampling_offsets / offset_normalizer.to(sampling_offsets.dtype)
        )
        # sampling_locations: (B, Q, num_heads, num_levels, num_points, 2)
        # in [0, 1] (approximately, offsets can push outside)

        # Sample features per level using F.grid_sample
        output = torch.zeros(
            B, Q, self.num_heads, self.head_dim,
            device=query.device, dtype=value.dtype,
        )

        for level_idx in range(self.num_levels):
            H_l, W_l = spatial_shapes[level_idx]
            H_l, W_l = int(H_l), int(W_l)
            start = int(level_start_index[level_idx])
            end = start + H_l * W_l

            # Extract this level's value features
            # (B, H_l * W_l, num_heads, head_dim) -> (B*num_heads, head_dim, H_l, W_l)
            value_l = value[:, start:end].permute(0, 2, 3, 1).reshape(
                B * self.num_heads, self.head_dim, H_l, W_l
            )

            # Get sampling locations for this level
            # (B, Q, num_heads, num_points, 2)
            locs_l = sampling_locations[:, :, :, level_idx, :, :]

            # Convert from [0, 1] to [-1, 1] for grid_sample
            grid = 2.0 * locs_l - 1.0
            # (B, Q, num_heads, num_points, 2) -> (B*num_heads, Q, num_points, 2)
            grid = grid.permute(0, 2, 1, 3, 4).reshape(
                B * self.num_heads, Q, self.num_points, 2
            )

            # F.grid_sample expects (N, C, H_in, W_in) and grid (N, H_out, W_out, 2)
            # Here: N=B*num_heads, C=head_dim, H_out=Q, W_out=num_points
            sampled = F.grid_sample(
                value_l, grid,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=False,
            )
            # sampled: (B*num_heads, head_dim, Q, num_points)

            # Reshape back: (B, num_heads, head_dim, Q, num_points)
            sampled = sampled.view(
                B, self.num_heads, self.head_dim, Q, self.num_points
            )

            # Weight and accumulate
            # attention_weights for this level: (B, Q, num_heads, num_points)
            weights_l = attention_weights[:, :, :, level_idx, :]
            # -> (B, num_heads, 1, Q, num_points) for broadcasting
            weights_l = weights_l.permute(0, 2, 1, 3).unsqueeze(2)

            # Weighted sum over points: (B, num_heads, head_dim, Q)
            output += (sampled * weights_l).sum(dim=-1).permute(0, 3, 1, 2)

        # output: (B, Q, num_heads, head_dim) -> (B, Q, C)
        output = output.reshape(B, Q, self.embed_dims)
        output = self.output_proj(output)
        return output


class MultiScaleDeformableAttention3D(nn.Module):
    """Height-aware Multi-Scale Deformable Attention for BEVFormer SCA.

    Unlike plain ``MultiScaleDeformableAttention``, each BEV query carries
    ``num_Z_anchors`` 3D reference points (one per pillar height) that
    project to distinct 2D locations in each camera view. The ``num_points``
    sampling offsets predicted per query are **shared** across all height
    anchors — only the 2D reference location changes with height — and the
    sampled features are averaged across heights before weighted-summing
    across points. This matches ``MSDeformableAttention3D`` in
    ``external/BEVFormer/projects/mmdet3d_plugin/bevformer/modules/spatial_cross_attention.py``.

    Args:
        embed_dims: Feature dimension (default: 256).
        num_heads: Number of attention heads (default: 8).
        num_levels: Number of feature map levels (default: 1).
        num_points: Sampling points per query per head per level (default: 8).
        num_Z_anchors: Number of height samples per pillar (default: 4).
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        num_levels: int = 1,
        num_points: int = 8,
        num_Z_anchors: int = 4,
    ):
        super().__init__()
        assert embed_dims % num_heads == 0, \
            f"embed_dims ({embed_dims}) must be divisible by num_heads ({num_heads})"

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.num_Z_anchors = num_Z_anchors
        self.head_dim = embed_dims // num_heads

        # Offsets: one (dx, dy) per head/level/point (shared across heights)
        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2
        )
        # Weights: softmax over (levels × points); NOT over heights — heights
        # are averaged before the weighted sum.
        self.attention_weights = nn.Linear(
            embed_dims, num_heads * num_levels * num_points
        )
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)

        self._reset_parameters()

    def _reset_parameters(self):
        # Offsets initialized to a grid pattern per head (as in plain MSDA)
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.num_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], dim=-1)
        grid_init = (
            grid_init / grid_init.abs().max(-1, keepdim=True)[0]
        )
        grid_init = grid_init.view(
            self.num_heads, 1, 1, 2
        ).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))

        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)

    def forward(
        self,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        value: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            query: (B, Q, C) query features.
            reference_points: (B, Q, num_Z_anchors, 2) normalized [0, 1] 2D
                reference points, one per pillar height.
            value: (B, S, C) flattened multi-scale value features.
            spatial_shapes: (num_levels, 2) -> (H_l, W_l) per level.
            level_start_index: (num_levels,) start indices in the flattened
                value sequence.
            key_padding_mask: (B, S) True at padded positions.

        Returns:
            output: (B, Q, C) attended features.
        """
        B, Q, _ = query.shape
        B_v, S, _ = value.shape
        assert B == B_v
        Z = self.num_Z_anchors
        assert reference_points.shape == (B, Q, Z, 2), (
            f"Expected reference_points shape (B, Q, {Z}, 2), "
            f"got {tuple(reference_points.shape)}"
        )

        # Project values: (B, S, C) -> (B, S, num_heads, head_dim)
        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(
                key_padding_mask.unsqueeze(-1), 0.0
            )
        value = value.view(B, S, self.num_heads, self.head_dim)

        # Offsets: shared across heights.
        # (B, Q, num_heads, num_levels, num_points, 2)
        sampling_offsets = self.sampling_offsets(query).view(
            B, Q, self.num_heads, self.num_levels, self.num_points, 2
        )

        # Weights: softmax over (levels × points); NOT over heights.
        attention_weights = self.attention_weights(query).view(
            B, Q, self.num_heads, self.num_levels * self.num_points
        )
        attention_weights = F.softmax(attention_weights, dim=-1).view(
            B, Q, self.num_heads, self.num_levels, self.num_points
        )

        # Build sampling locations:
        # (B, Q, num_heads, num_levels, num_points, Z, 2)
        # reference_points: (B, Q, Z, 2) -> (B, Q, 1, 1, 1, Z, 2)
        ref_exp = reference_points[:, :, None, None, None, :, :]
        # offset_normalizer: (1, 1, 1, num_levels, 1, 2)
        offset_normalizer = spatial_shapes.flip(-1)[None, None, None, :, None, :]
        offsets_scaled = (
            sampling_offsets / offset_normalizer.to(sampling_offsets.dtype)
        ).unsqueeze(-2)  # (B, Q, num_heads, num_levels, num_points, 1, 2)
        sampling_locations = ref_exp + offsets_scaled

        output = torch.zeros(
            B, Q, self.num_heads, self.head_dim,
            device=query.device, dtype=value.dtype,
        )

        for level_idx in range(self.num_levels):
            H_l, W_l = spatial_shapes[level_idx]
            H_l, W_l = int(H_l), int(W_l)
            start = int(level_start_index[level_idx])
            end = start + H_l * W_l

            value_l = value[:, start:end].permute(0, 2, 3, 1).reshape(
                B * self.num_heads, self.head_dim, H_l, W_l
            )

            # Locs for this level: (B, Q, num_heads, num_points, Z, 2)
            locs_l = sampling_locations[:, :, :, level_idx, :, :, :]

            # [0, 1] -> [-1, 1] for grid_sample
            grid = 2.0 * locs_l - 1.0
            # (B, Q, num_heads, num_points, Z, 2)
            #   -> (B*num_heads, Q, num_points*Z, 2)
            grid = grid.permute(0, 2, 1, 3, 4, 5).reshape(
                B * self.num_heads, Q, self.num_points * Z, 2
            )

            sampled = F.grid_sample(
                value_l, grid,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=False,
            )
            # (B*num_heads, head_dim, Q, num_points*Z)
            # -> (B, num_heads, head_dim, Q, num_points, Z)
            sampled = sampled.view(
                B, self.num_heads, self.head_dim, Q, self.num_points, Z
            )

            # Average across Z-anchors: heights share offsets, so heights are
            # merged by mean before the weighted sum over points.
            sampled = sampled.mean(dim=-1)
            # (B, num_heads, head_dim, Q, num_points)

            # Weight and accumulate:
            # attention_weights: (B, Q, num_heads, num_levels, num_points)
            weights_l = attention_weights[:, :, :, level_idx, :]
            # -> (B, num_heads, 1, Q, num_points)
            weights_l = weights_l.permute(0, 2, 1, 3).unsqueeze(2)
            # Weighted sum over points -> (B, num_heads, head_dim, Q)
            output += (sampled * weights_l).sum(dim=-1).permute(0, 3, 1, 2)

        output = output.reshape(B, Q, self.embed_dims)
        output = self.output_proj(output)
        return output
