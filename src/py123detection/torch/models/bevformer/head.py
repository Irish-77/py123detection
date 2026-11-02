"""
Stage 1 (BEV Encoder): Builds a dense BEV feature map from multi-view images
    via Spatial Cross-Attention
Stage 2 (Detection Decoder): Object queries cross-attend to the BEV feature
    map using deformable attention to produce 3D detection predictions
"""

from typing import List

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp

from py123detection.torch.models.bevformer.camera_embedding import (
    VALID_CAM_EMBED_TYPES,
    build_camera_embedding,
)
from py123detection.torch.models.bevformer.deformable_attention import (
    MultiScaleDeformableAttention,
)
from py123detection.torch.models.bevformer.encoder import BEVFormerEncoder
from py123detection.torch.models.bevformer.utils import (
    create_3d_reference_points,
    create_bev_grid,
)
from py123detection.torch.models.common.loss import HungarianMatcher, SetCriterion
from py123detection.torch.models.common.util import denormalize_bbox, inverse_sigmoid

# ---------------------------------------------------------------------------
# Detection Decoder (Deformable DETR-style on BEV feature map)
# ---------------------------------------------------------------------------

class DeformableDETRDecoderLayer(nn.Module):
    """Single Deformable DETR decoder layer.

    Order: self-attn -> norm -> deformable cross-attn (on BEV) -> norm ->
           FFN -> norm.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        feedforward_channels: int = 2048,
        dropout: float = 0.1,
        num_points: int = 4,
        with_cp: bool = True,
    ):
        super().__init__()
        self.use_checkpoint = with_cp

        # Standard self-attention among object queries (900 queries is small)
        self.self_attn = nn.MultiheadAttention(
            embed_dims, num_heads, dropout=dropout, batch_first=False,
        )
        self.norm1 = nn.LayerNorm(embed_dims)
        self.dropout1 = nn.Dropout(dropout)

        # Deformable cross-attention on the BEV feature map
        self.cross_attn = MultiScaleDeformableAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_levels=1,
            num_points=num_points,
        )
        self.norm2 = nn.LayerNorm(embed_dims)

        # FFN
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
        query: torch.Tensor,
        query_pos: torch.Tensor,
        reference_points: torch.Tensor,
        bev_features: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            query: (B, num_query, C) object query content features.
            query_pos: (B, num_query, C) object query positional encoding.
            reference_points: (B, num_query, 1, 2) normalized [0,1] BEV
                reference points.
            bev_features: (B, bev_h*bev_w, C) flattened BEV feature map.
            spatial_shapes: (1, 2) -> (bev_h, bev_w).
            level_start_index: (1,) -> [0].

        Returns:
            query: (B, num_query, C) updated query features.
        """
        # 1. Self-attention among object queries
        # nn.MultiheadAttention expects (seq, batch, dim) when batch_first=False
        q = (query + query_pos).permute(1, 0, 2)  # (num_query, B, C)
        k = q
        v = query.permute(1, 0, 2)
        residual = query
        attn_out, _ = self.self_attn(q, k, v)
        attn_out = attn_out.permute(1, 0, 2)  # (B, num_query, C)
        query = self.norm1(residual + self.dropout1(attn_out))

        # 2. Deformable cross-attention on BEV feature map
        residual = query
        query = self.norm2(
            residual + self.cross_attn(
                query=query + query_pos,
                reference_points=reference_points,
                value=bev_features,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
            )
        )

        # 3. FFN
        residual = query
        query = self.norm3(residual + self.ffn(query))

        return query

    def forward(self, *args, **kwargs) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            return cp.checkpoint(
                self._forward, *args, use_reentrant=False, **kwargs
            )
        return self._forward(*args, **kwargs)


class DeformableDETRDecoder(nn.Module):
    """Deformable DETR decoder that cross-attends to BEV features.

    Returns intermediate outputs from each layer for auxiliary loss.
    """

    def __init__(
        self,
        num_layers: int = 6,
        embed_dims: int = 256,
        num_heads: int = 8,
        feedforward_channels: int = 2048,
        dropout: float = 0.1,
        num_points: int = 4,
        with_cp: bool = True,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.embed_dims = embed_dims

        self.layers = nn.ModuleList([
            DeformableDETRDecoderLayer(
                embed_dims=embed_dims,
                num_heads=num_heads,
                feedforward_channels=feedforward_channels,
                dropout=dropout,
                num_points=num_points,
                with_cp=with_cp,
            )
            for _ in range(num_layers)
        ])
        self.post_norm = nn.LayerNorm(embed_dims)

    def forward(
        self,
        query: torch.Tensor,
        query_pos: torch.Tensor,
        reference_points: torch.Tensor,
        bev_features: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns:
            intermediate: (num_layers, B, num_query, C) intermediate outputs.
        """
        intermediate = []
        for layer in self.layers:
            query = layer(
                query=query,
                query_pos=query_pos,
                reference_points=reference_points,
                bev_features=bev_features,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
            )
            intermediate.append(self.post_norm(query))

        return torch.stack(intermediate)  # (num_layers, B, num_query, C)


# ---------------------------------------------------------------------------
# BEVFormerHead
# ---------------------------------------------------------------------------

DEFAULT_CODE_WEIGHTS = [
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2
]


class BEVFormerHead(nn.Module):
    """BEVFormer detection head: BEV encoder + detection decoder.

    Stage 1: BEV encoder builds dense BEV features from multi-view images.
    Stage 2: Detection decoder predicts 3D bounding boxes from BEV features.

    Output format is identical to PETRHead for loss/eval compatibility.

    Args:
        num_classes: Number of object classes.
        in_channels: Input feature channels from FPN (default: 256).
        embed_dims: Internal feature dimension (default: 256).
        bev_h: BEV grid height (default: 50).
        bev_w: BEV grid width (default: 50).
        num_query: Number of detection queries (default: 900).
        num_encoder_layers: BEV encoder layers (default: 3).
        num_decoder_layers: Detection decoder layers (default: 6).
        num_heads: Attention heads (default: 8).
        num_heights: Height samples per pillar (default: 4).
        num_points: Deformable attention sampling points (default: 4).
        pc_range: Detection range [x_min, y_min, z_min, x_max, y_max, z_max].
        code_weights: Per-component loss weights for bbox regression.
        feedforward_channels_encoder: FFN dim in encoder (default: 512).
        feedforward_channels_decoder: FFN dim in decoder (default: 2048).
        dropout: Dropout rate (default: 0.1).
        with_box_refine: If True, iteratively refine 2D BEV reference
            points between decoder layers and give each layer its own
            prediction branches (Deformable DETR-style). Default: True.
        with_cp: Gradient checkpointing (default: True).
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 256,
        embed_dims: int = 256,
        bev_h: int = 50,
        bev_w: int = 50,
        num_query: int = 900,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 6,
        num_heads: int = 8,
        num_heights: int = 4,
        num_points: int = 4,
        pc_range: List[float] = None,
        code_weights: List[float] = None,
        feedforward_channels_encoder: int = 512,
        feedforward_channels_decoder: int = 2048,
        dropout: float = 0.1,
        with_box_refine: bool = True,
        with_cp: bool = True,
        debug_nan_checks: bool = False,
        # Camera-embedding options (cross-dataset config).
        cam_embed_type: str = "none",
        num_cameras: int = 6,
        **kwargs,
    ):
        super().__init__()

        if pc_range is None:
            pc_range = [-51.2, -51.2, -3.0, 51.2, 51.2, 5.0]
        if code_weights is None:
            code_weights = DEFAULT_CODE_WEIGHTS
        if cam_embed_type not in VALID_CAM_EMBED_TYPES:
            raise ValueError(
                f"Unknown cam_embed_type={cam_embed_type!r}; must be one of "
                f"{VALID_CAM_EMBED_TYPES}."
            )

        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.num_query = num_query
        self.num_decoder_layers = num_decoder_layers
        self.num_heights = num_heights
        self.pc_range = pc_range
        self.with_box_refine = with_box_refine
        self.debug_nan_checks = debug_nan_checks

        self.code_weights = nn.Parameter(
            torch.tensor(code_weights, requires_grad=False),
            requires_grad=False,
        )

        # --- BEV queries and positional encoding ---
        self.bev_embedding = nn.Embedding(bev_h * bev_w, embed_dims)
        self.bev_pos_embed = nn.Embedding(bev_h * bev_w, embed_dims)

        # Image feature projection
        self.input_proj = nn.Conv2d(in_channels, embed_dims, kernel_size=1)

        # --- Optional per-camera / 3D positional embedding ---
        # Applied once to image_features after input_proj; the embedded
        # features are then reused across all encoder layers (matches the
        # official BEVFormer's cams_embeds placement).
        #   - "none"      → self.cam_embed is None (no-op at runtime).
        #   - "bevformer" → learnable per-camera param (hard-locked to N).
        #   - "petr"      → pose-derived frustum PE (handles variable N).
        self.cam_embed_type = cam_embed_type
        self.cam_embed = build_camera_embedding(
            cam_embed_type,
            embed_dims=embed_dims,
            num_cameras=num_cameras,
        )

        # --- Stage 1: BEV Encoder ---
        self.encoder = BEVFormerEncoder(
            num_layers=num_encoder_layers,
            embed_dims=embed_dims,
            num_heads=num_heads,
            feedforward_channels=feedforward_channels_encoder,
            dropout=dropout,
            num_points_self=num_points,
            num_points_cross=num_points,
            num_heights=num_heights,
            with_cp=with_cp,
        )

        # --- Stage 2: Detection Decoder ---
        self.detection_query_embed = nn.Embedding(num_query, embed_dims * 2)
        self.detection_reference_points = nn.Linear(embed_dims, 2)

        self.decoder = DeformableDETRDecoder(
            num_layers=num_decoder_layers,
            embed_dims=embed_dims,
            num_heads=num_heads,
            feedforward_channels=feedforward_channels_decoder,
            dropout=dropout,
            num_points=num_points,
            with_cp=with_cp,
        )

        # --- Prediction heads ---
        # With box refinement, each decoder layer needs its OWN prediction
        # branches so successive layers can produce progressively better
        # reference-point corrections. Without refinement, branches are
        # shared (same module instance repeated).
        self.num_pred = num_decoder_layers
        if self.with_box_refine:
            self.cls_branches = nn.ModuleList([
                self._build_cls_branch(num_classes)
                for _ in range(self.num_pred)
            ])
            self.reg_branches = nn.ModuleList([
                self._build_reg_branch() for _ in range(self.num_pred)
            ])
        else:
            fc_cls = self._build_cls_branch(num_classes)
            reg_branch = self._build_reg_branch()
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(self.num_pred)]
            )
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(self.num_pred)]
            )

        # --- Loss ---
        self.matcher = HungarianMatcher(
            cost_class=2.0,
            cost_bbox=0.25,
            pc_range=self.pc_range,
            debug_nan_checks=debug_nan_checks,
        )
        self.criterion = SetCriterion(
            num_classes=num_classes,
            matcher=self.matcher,
            weight_dict={'loss_ce': 2.0, 'loss_bbox': 0.25},
            losses=['labels', 'boxes'],
            pc_range=self.pc_range,
            code_weights=self.code_weights,
        )

        # --- Pre-compute and register BEV grid buffers ---
        self._register_bev_buffers()

    def _build_cls_branch(self, num_classes: int) -> nn.Sequential:
        cls_branch = []
        for _ in range(2):
            cls_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(nn.Linear(self.embed_dims, num_classes))
        return nn.Sequential(*cls_branch)

    def _build_reg_branch(self) -> nn.Sequential:
        reg_branch = []
        for _ in range(2):
            reg_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(nn.Linear(self.embed_dims, 10))  # code_size=10
        return nn.Sequential(*reg_branch)

    def _register_bev_buffers(self):
        """Pre-compute BEV grid and reference points as persistent buffers."""
        # 2D BEV grid: (Q_bev, 2) normalized [0, 1]
        bev_grid = create_bev_grid(self.bev_h, self.bev_w)
        self.register_buffer('_bev_grid', bev_grid, persistent=False)

        # 3D reference points for SCA: (Q_bev, num_heights, 3)
        ref_3d = create_3d_reference_points(bev_grid, self.num_heights, self.pc_range)
        self.register_buffer('_reference_points_3d', ref_3d, persistent=False)

        # 2D BEV reference points for deformable self-attention: (1, Q_bev, 1, 2)
        bev_ref_2d = bev_grid.unsqueeze(0).unsqueeze(2)
        self.register_buffer('_bev_ref_2d', bev_ref_2d, persistent=False)

        # Spatial shapes and level indices for BEV self-attention
        spatial_shapes = torch.tensor(
            [[self.bev_h, self.bev_w]], dtype=torch.long
        )
        level_start_index = torch.tensor([0], dtype=torch.long)
        self.register_buffer(
            '_spatial_shapes_bev', spatial_shapes, persistent=False
        )
        self.register_buffer(
            '_level_start_index_bev', level_start_index, persistent=False
        )

    def init_weights(self):
        """Initialize weights following Deformable DETR conventions."""
        # Xavier init for all transformer parameters
        for p in self.encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for p in self.decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # Detection reference points: uniform init
        nn.init.xavier_uniform_(self.detection_reference_points.weight)
        nn.init.constant_(self.detection_reference_points.bias, 0.0)

        # Classification bias for focal loss stability
        bias_init = -4.59512  # log(0.01 / 0.99)
        for m in self.cls_branches:
            nn.init.constant_(m[-1].bias, bias_init)

    def forward(
        self,
        x,
        img_metas,
        cam_mask=None,
        imu2img=None,
        **kwargs,
    ):
        """Forward pass.

        Args:
            x: List of FPN features, first level used: (B, N, C, H, W).
            img_metas: List of metadata dicts with 'pad_shape'.
            cam_mask: (B, N) bool. True = valid camera.
            imu2img: (B, N, 4, 4) projection matrices.

        Returns:
            dict with 'all_cls_scores' (L, B, num_query, num_classes) and
                'all_bbox_preds' (L, B, num_query, 10).
        """
        feat = x[0]  # (B, N, C, H, W)
        B, N, C, H, W = feat.shape

        # Project image features
        feat_flat = self.input_proj(feat.flatten(0, 1))  # (B*N, embed, H, W)
        image_features = feat_flat.view(B, N, self.embed_dims, H, W)

        # Get padded image dimensions for reference point projection
        # (extracted early so the optional camera embedding can use them).
        pad_h, pad_w = img_metas[0]['pad_shape'][0][:2]

        # Optional per-camera / 3D position embedding. Applied ONCE before
        # the encoder; all encoder layers then reuse the embedded features.
        if self.cam_embed is not None:
            cam_pe = self.cam_embed(
                image_features,
                imu2img=imu2img,
                cam_mask=cam_mask,
                pad_h=pad_h,
                pad_w=pad_w,
            )
            image_features = image_features + cam_pe

        # BEV queries and positional encoding
        bev_queries = self.bev_embedding.weight.unsqueeze(0).expand(
            B, -1, -1
        )  # (B, Q_bev, C)
        bev_pos = self.bev_pos_embed.weight.unsqueeze(0).expand(
            B, -1, -1
        )  # (B, Q_bev, C)

        # ==================== Stage 1: BEV Encoder ====================
        bev_feat = self.encoder(
            bev_queries=bev_queries,
            bev_pos=bev_pos,
            image_features=image_features,
            reference_points_3d=self._reference_points_3d,
            bev_ref_2d=self._bev_ref_2d,
            spatial_shapes_bev=self._spatial_shapes_bev,
            level_start_index_bev=self._level_start_index_bev,
            imu2img=imu2img,
            cam_mask=cam_mask,
            pad_h=pad_h,
            pad_w=pad_w,
        )
        # bev_feat: (B, Q_bev, C)

        # ==================== Stage 2: Detection Decoder ====================

        # Prepare detection queries
        det_embed = self.detection_query_embed.weight  # (num_query, 2C)
        query_pos, query_content = det_embed.split(self.embed_dims, dim=-1)
        query_content = query_content.unsqueeze(0).expand(B, -1, -1)
        query_pos = query_pos.unsqueeze(0).expand(B, -1, -1)

        # Initial detection reference points on BEV: (B, num_query, 2) in [0, 1]
        ref = self.detection_reference_points(query_pos).sigmoid()

        # BEV spatial shapes for the decoder
        bev_spatial = self._spatial_shapes_bev
        bev_start = self._level_start_index_bev

        # Manual decoder iteration with optional box refinement.
        # We bypass self.decoder.forward() so we can update reference
        # points between layers and still reuse the decoder's layers +
        # post-norm module.
        outputs_classes = []
        outputs_coords = []
        query = query_content

        for lvl, layer in enumerate(self.decoder.layers):
            ref_in = ref.unsqueeze(2)  # (B, num_query, 1, 2)
            query = layer(
                query=query,
                query_pos=query_pos,
                reference_points=ref_in,
                bev_features=bev_feat,
                spatial_shapes=bev_spatial,
                level_start_index=bev_start,
            )
            out = self.decoder.post_norm(query)
            out = torch.nan_to_num(out)

            # Predictions for this layer
            outputs_class = self.cls_branches[lvl](out)
            tmp = self.reg_branches[lvl](out).clone()

            # Apply x, y offsets relative to the *current* reference point
            # in logit space, matching Deformable DETR's box refinement.
            ref_inv = inverse_sigmoid(ref.clone())
            new_ref_logit = ref_inv.clone()
            new_ref_logit[..., 0:1] = new_ref_logit[..., 0:1] + tmp[..., 0:1]
            new_ref_logit[..., 1:2] = new_ref_logit[..., 1:2] + tmp[..., 1:2]
            new_ref = new_ref_logit.sigmoid()

            tmp[..., 0:1] = new_ref[..., 0:1]
            tmp[..., 1:2] = new_ref[..., 1:2]
            # z prediction: absolute via sigmoid (no 3D reference)
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()

            outputs_classes.append(outputs_class)
            outputs_coords.append(tmp)

            # Feed refined reference points to the next layer when
            # box refinement is enabled. Detach to prevent gradient
            # instability across layers (matches Deformable DETR).
            if self.with_box_refine:
                ref = new_ref.detach()
            # Otherwise: keep the initial reference points unchanged.

        all_cls_scores = torch.stack(outputs_classes)
        all_bbox_preds = torch.stack(outputs_coords)

        # Denormalize x, y, z from sigmoid [0,1] to world coordinates
        all_bbox_preds[..., 0:1] = (
            all_bbox_preds[..., 0:1]
            * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
        )
        all_bbox_preds[..., 1:2] = (
            all_bbox_preds[..., 1:2]
            * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
        )
        all_bbox_preds[..., 4:5] = (
            all_bbox_preds[..., 4:5]
            * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
        )

        return {
            'all_cls_scores': all_cls_scores,
            'all_bbox_preds': all_bbox_preds,
        }

    def loss(
        self,
        gt_bboxes_list,
        gt_labels_list,
        preds_dicts,
        sync=True,
        num_boxes_synced=None,
    ):
        """Compute loss. Same interface as PETRHead."""
        targets = [
            {'boxes_3d': boxes, 'labels': labels}
            for boxes, labels in zip(gt_bboxes_list, gt_labels_list)
        ]
        return self.criterion(
            preds_dicts,
            targets,
            sync=sync,
            num_boxes_synced=num_boxes_synced,
        )

    def get_bboxes(self, preds_dicts, img_metas=None, rescale=False):
        """Convert predictions to bboxes in world coordinates.
        Same interface as PETRHead."""
        preds = preds_dicts['all_bbox_preds'][-1]
        scores = preds_dicts['all_cls_scores'][-1].sigmoid()

        B = preds.shape[0]
        results = []
        for i in range(B):
            boxes = denormalize_bbox(preds[i], self.pc_range)
            results.append((boxes, scores[i]))
        return results

    def reset_classification_head(self, num_classes: int):
        """Reinitialize classification head for a new class taxonomy."""
        self.num_classes = int(num_classes)

        ref_param = next(self.parameters())
        bias_init = -4.59512

        if self.with_box_refine:
            # One independent branch per decoder layer.
            new_branches = []
            for _ in range(self.num_pred):
                br = self._build_cls_branch(self.num_classes).to(
                    device=ref_param.device, dtype=ref_param.dtype,
                )
                nn.init.constant_(br[-1].bias, bias_init)
                new_branches.append(br)
            self.cls_branches = nn.ModuleList(new_branches)
        else:
            shared_cls_branch = self._build_cls_branch(self.num_classes).to(
                device=ref_param.device, dtype=ref_param.dtype,
            )
            nn.init.constant_(shared_cls_branch[-1].bias, bias_init)
            self.cls_branches = nn.ModuleList(
                [shared_cls_branch for _ in range(self.num_pred)]
            )

        self._init_loss()

    def _init_loss(self):
        self.matcher = HungarianMatcher(
            cost_class=2.0,
            cost_bbox=0.25,
            pc_range=self.pc_range,
            debug_nan_checks=self.debug_nan_checks,
        )
        self.criterion = SetCriterion(
            num_classes=self.num_classes,
            matcher=self.matcher,
            weight_dict={'loss_ce': 2.0, 'loss_bbox': 0.25},
            losses=['labels', 'boxes'],
            pc_range=self.pc_range,
            code_weights=self.code_weights,
        )
