"""
TODO:
- Add docstrings
- Do we still need eps adjustable? Should be fixed by back casting to float32
"""

from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from py123detection.torch.models.common.loss import HungarianMatcher, SetCriterion
from py123detection.torch.models.common.position_embedding import (
    compute_frustum_coords3d,
)
from py123detection.torch.models.common.util import denormalize_bbox, inverse_sigmoid, pos2posemb3d
from py123detection.torch.models.petr.positional_encoding import SinePositionalEncoding3D
from py123detection.torch.models.petr.transformer import PETRTransformer

DEFAULT_CODE_WEIGHTS = [
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2
]
DEFAULT_POSITION_RANGE = [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0]


class PETRHead(nn.Module):
    def __init__(
        self,
        num_classes,
        in_channels,
        num_query=900,
        num_decoder_layers: int = 6,
        pc_range: List[float] = None,
        code_weights: List[float] = DEFAULT_CODE_WEIGHTS,
        position_range: List[float] = DEFAULT_POSITION_RANGE,
        with_cp: bool = True,
        mask_strict: bool = True,
        debug_nan_checks: bool = False,
        # Image and patch geometry.
        patches_x: int = None,
        patches_y: int = None,
        image_width: int = None,
        image_height: int = None,
        **kwargs
    ):
        super(PETRHead, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.num_query = num_query
        self.num_decoder_layers = num_decoder_layers
        self.pc_range = pc_range
        self.embed_dims = 256
        # One prediction head per decoder layer
        self.num_pred = num_decoder_layers
        self.position_range = position_range  # kept for compatibility
        self.register_buffer(
            '_position_range_t',
            torch.tensor(position_range, dtype=torch.float32),
            persistent=False,
        )
        if pc_range is not None:
            self.register_buffer(
                '_pc_range_t',
                torch.tensor(pc_range, dtype=torch.float32),
                persistent=False,
            )
        else:
            self._pc_range_t = None
        self.with_cp = with_cp
        self.mask_strict = mask_strict
        self.debug_nan_checks = debug_nan_checks
        self.patches_x = patches_x
        self.patches_y = patches_y
        self.image_width = image_width
        self.image_height = image_height

        # Per-component weights on the box regression loss. Held as a non-trainable Parameter so
        # they move to the device with the model but never receive a gradient.
        self.code_weights = nn.Parameter(
            torch.tensor(code_weights, requires_grad=False), requires_grad=False
        )

        # Hyperparameters for 3D PE
        self.depth_step = 0.8
        self.depth_num = 64
        self.depth_start = 1
        self.LID = True
        self.position_dim = 3 * self.depth_num

        self.positional_encoding = SinePositionalEncoding3D(
            num_feats=self.embed_dims // 2, normalize=True
        )
        self.transformer = PETRTransformer(
            embed_dims=self.embed_dims,
            num_layers=self.num_decoder_layers,
            with_cp=self.with_cp,
        )

        self._init_layers()
        self._init_loss()

    def _init_layers(self):
        self.input_proj = nn.Conv2d(
            self.in_channels, self.embed_dims, kernel_size=1
        )

        # Input channel count depends on the PE strategy:
        #   frustum:        D*3  (e.g. 64*3 = 192), heavy MLP 192->1024->256
        #   da3_depth_lift: 3    (single xyz per pixel from DA3 depth)
        #   da3_ray:        6    (ray direction + origin from DA3)
        # PETR lifts each pixel to a column of 3D points and embeds the column.
        pe_in_channels = self.depth_num * 3
        pe_hidden = self.embed_dims * 4
        self.position_encoder = nn.Sequential(
            nn.Conv2d(
                pe_in_channels, pe_hidden, kernel_size=1,
                stride=1, padding=0,
            ),
            nn.ReLU(),
            nn.Conv2d(
                pe_hidden, self.embed_dims, kernel_size=1,
                stride=1, padding=0,
            ),
        )

        self.adapt_pos3d = nn.Sequential(
            nn.Conv2d(
                self.embed_dims*3//2, self.embed_dims*4, kernel_size=1,
                stride=1, padding=0
            ),
            nn.ReLU(),
            nn.Conv2d(
                self.embed_dims*4, self.embed_dims, kernel_size=1,
                stride=1, padding=0
            ),
        )

        fc_cls = self._build_cls_branch(self.num_classes)

        reg_branch = []
        for _ in range(2):
            reg_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(nn.Linear(self.embed_dims, 10))   # code_size=10
        reg_branch = nn.Sequential(*reg_branch)

        # Share weights across all decoder layers (matches original PETR)
        self.cls_branches = nn.ModuleList(
            [fc_cls for _ in range(self.num_pred)]
        )
        self.reg_branches = nn.ModuleList(
            [reg_branch for _ in range(self.num_pred)]
        )

        self.reference_points = nn.Embedding(self.num_query, 3)
        self.query_embedding = nn.Sequential(
            nn.Linear(self.embed_dims*3//2, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )

    def _build_cls_branch(self, num_classes: int) -> nn.Sequential:
        cls_branch = []
        for _ in range(2):
            cls_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(nn.Linear(self.embed_dims, num_classes))
        return nn.Sequential(*cls_branch)

    def _init_loss(self):
        # Standard Hungarian Matcher and SetCriterion like DETR3D
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
            code_weights=self.code_weights
        )

    def init_weights(self):
        self.transformer.init_weights()
        nn.init.uniform_(self.reference_points.weight.data, 0, 1)
        # Initialize classification bias for focal loss stability
        # bias = log(prior / (1 - prior)) where prior = 0.01
        # for rare positive samples
        bias_init = -4.59512  # log(0.01 / 0.99)
        for m in self.cls_branches:
            nn.init.constant_(m[-1].bias, bias_init)

    def reset_classification_head(self, num_classes: int):
        """Reinitialize class-dependent modules for a new class taxonomy."""
        self.num_classes = int(num_classes)

        shared_cls_branch = self._build_cls_branch(self.num_classes)

        # Keep dtype/device consistent with the existing module parameters.
        ref_param = next(self.parameters())
        shared_cls_branch = shared_cls_branch.to(
            device=ref_param.device,
            dtype=ref_param.dtype,
        )

        # Share weights across decoder layers (matches original PETR).
        self.cls_branches = nn.ModuleList(
            [shared_cls_branch for _ in range(self.num_pred)]
        )

        # Keep focal-loss prior consistent with regular initialization.
        bias_init = -4.59512  # log(0.01 / 0.99)
        nn.init.constant_(shared_cls_branch[-1].bias, bias_init)

        # Rebuild matcher/criterion because num_classes changed.
        self._init_loss()

    def position_embeding(
        self,
        img_feats,
        img_metas,
        masks=None,
        imu2img=None,
        depth=None,
        ray=None,
        intrinsics=None,
        extrinsics=None,
    ):
        """PETR's 3D positional encoding.

        Every pixel is lifted to a column of points at increasing depth, transformed into the ego
        frame, and the column embedded. That is what gives a 2D feature a 3D position without any
        depth prediction.
        """
        eps = 1e-5
        # img_feats is (B, N, C, H, W)
        B, N, C, H, W = img_feats.shape

        # Use pad_shape (post-padding dimensions) for frustum coordinate
        # mapping. The feature map is downsampled from the PADDED image, so
        # we must use pad_shape to correctly map feature positions back to
        # pixel space. This matches the original PETR.
        pad_h, pad_w = img_metas[0]['pad_shape'][0][:2]

        # Resolve imu2img: prefer the tensor path (avoids GPU→CPU sync from
        # numpy conversion in the training hot path). Fall back to the
        # metadata dict for legacy callers.
        if imu2img is None:
            imu2img_list = []
            for img_meta in img_metas:
                per_sample = []
                for i in range(len(img_meta['imu2img'])):
                    per_sample.append(
                        np.asarray(img_meta['imu2img'][i], dtype=np.float32)
                    )
                imu2img_list.append(np.asarray(per_sample))
            imu2img = img_feats.new_tensor(np.asarray(imu2img_list))

        # Delegate the pure frustum math to the shared helper. Module
        # ownership of position_encoder / adapt_pos3d / positional_encoding
        # stays on PETRHead, so state_dict keys are unchanged.
        coords3d, coords_mask = compute_frustum_coords3d(
            B=B, N=N, H=H, W=W,
            pad_h=pad_h, pad_w=pad_w,
            imu2img=imu2img,
            depth_num=self.depth_num,
            depth_start=self.depth_start,
            LID=self.LID,
            position_range_t=self._position_range_t,
            mask_strict=self.mask_strict,
            device=img_feats.device,
            masks=masks,
            eps=eps,
        )

        # Force FP32 for entire position encoding pipeline
        # (matches official PETR's out_fp32=True). FP16 eps=1e-5 loses
        # precision (FP16 min ~6e-5), causing Inf in inverse_sigmoid.
        with torch.amp.autocast(device_type="cuda", enabled=False):
            coords3d = inverse_sigmoid(coords3d.float(), eps=eps)
            if self.position_encoder is not None:
                coords_position_embeding = self.position_encoder(coords3d)
            else:
                coords_position_embeding = None

        if coords_position_embeding is not None:
            return coords_position_embeding.view(
                B, N, self.embed_dims, H, W
            ), coords_mask
        else:
            return None, coords_mask
        
    # ------------------------------------------------------------------
    # DA3-specific PE strategies
    # ------------------------------------------------------------------

    def _normalize_world_coords(self, coords3d: torch.Tensor) -> torch.Tensor:
        """Normalize world-frame coords to [0, 1] using ``position_range``."""
        pr = self._position_range_t
        coords3d[..., 0:1] = (coords3d[..., 0:1] - pr[0]) / (pr[3] - pr[0])
        coords3d[..., 1:2] = (coords3d[..., 1:2] - pr[1]) / (pr[4] - pr[1])
        coords3d[..., 2:3] = (coords3d[..., 2:3] - pr[2]) / (pr[5] - pr[2])
        return coords3d

    def forward(
        self,
        x,
        img_metas,
        cam_mask=None,
        imu2img=None,
        extrinsics=None,
        intrinsics=None,
        depth=None,
        ray=None,
        coverage_mask=None,
    ):
        """Forward pass.

        Args:
            x: List of features from FPN, first level used: (B, N, C, H, W)
            img_metas: List of metadata dicts per sample
            cam_mask: Optional camera validity mask (B, N).
                True = valid camera, False = padded camera.
                If None, all cameras are assumed valid.
            imu2img: IMU-to-image projection matrices (B, N, 4, 4).
                Used to build the 3-D position embedding.
            extrinsics: Camera extrinsics (B, N, 4, 4), cam <- world.
                Used by GTA/PRoPE attention.
            intrinsics: Camera intrinsics (B, N, 4, 4) or (B, N, 3, 3).
                Used by PRoPE attention (normalised internally).
            coverage_mask: Cylindrical-input only. (B, 1, Hc, Wc) bool, True
                where a real camera covers the canvas pixel. Folded into the
                key-padding mask so blind sectors are excluded from attention.
        """
        # x is a list of features from FPN
        # PETRHead source uses mlvl_feats[0].
        x = x[0]
        B, N, C, H, W = x.shape

        # Use fixed eps
        eps = 1e-5

        # Initialize masks: (B, N, H, W)
        # Matches original PETR: create full-resolution mask from pad_shape,
        # then mark valid region using img_shape, then resize to feature map
        # resolution. Start with all-masked (True = masked for attention)
        if cam_mask is not None:
            # Camera-level mask: invalid cameras are fully masked
            masks = (~cam_mask).unsqueeze(-1).unsqueeze(-1).expand(
                B, N, H, W
            ).clone()
        else:
            masks = torch.zeros(
                (B, N, H, W), dtype=torch.bool, device=x.device
            )

        # Add padding-based masking if pad_shape differs from img_shape
        if 'img_shape' in img_metas[0] and 'pad_shape' in img_metas[0]:
            cam_mask_cpu = None
            if cam_mask is not None:
                cam_mask_cpu = cam_mask.detach().cpu()

            needs_padding_mask = False
            for b_idx in range(B):
                sample_img_shapes = img_metas[b_idx].get('img_shape', [])
                sample_pad_shapes = img_metas[b_idx].get('pad_shape', [])
                for cam_idx in range(N):
                    # Skip padded/invalid cameras from collate-time padding.
                    if cam_mask_cpu is not None and not bool(
                        cam_mask_cpu[b_idx, cam_idx]
                    ):
                        continue

                    if (
                        cam_idx >= len(sample_pad_shapes) or
                        cam_idx >= len(sample_img_shapes)
                    ):
                        raise ValueError(
                            "img_metas camera metadata is shorter than the "
                            "number of valid cameras in this batch. "
                            f"sample={b_idx}, camera={cam_idx}, "
                            f"num_img_shape={len(sample_img_shapes)}, "
                            f"num_pad_shape={len(sample_pad_shapes)}"
                        )

                    # Check if pre-padding shape differs from padded shape
                    # We only care about spatial dimensions (first 2 elements)
                    pad_h, pad_w = sample_pad_shapes[cam_idx][:2]
                    pre_h, pre_w = sample_img_shapes[cam_idx][:2]
                    if pad_h != pre_h or pad_w != pre_w:
                        needs_padding_mask = True
                        break
                if needs_padding_mask:
                    break

            if needs_padding_mask:
                pad_h, pad_w = H, W
                for img_meta in img_metas:
                    pad_shapes = img_meta.get('pad_shape', [])
                    if len(pad_shapes) > 0:
                        pad_h, pad_w = pad_shapes[0][:2]
                        break

                # Create mask at full image resolution,
                # then resize to feature map
                full_mask = x.new_ones((B, N, pad_h, pad_w), dtype=torch.bool)
                for b_idx in range(B):
                    sample_img_shapes = img_metas[b_idx].get('img_shape', [])
                    for cam_idx in range(N):
                        if cam_mask_cpu is not None and not bool(
                            cam_mask_cpu[b_idx, cam_idx]
                        ):
                            continue

                        if cam_idx >= len(sample_img_shapes):
                            raise ValueError(
                                "img_metas img_shape is shorter than the "
                                "number of valid cameras in this batch. "
                                f"sample={b_idx}, camera={cam_idx}, "
                                f"num_img_shape={len(sample_img_shapes)}"
                            )

                        ih, iw = sample_img_shapes[cam_idx][:2]
                        full_mask[b_idx, cam_idx, :ih, :iw] = False
                # Resize to feature map resolution
                pad_mask_flat = F.interpolate(
                    full_mask.float().flatten(0, 1).unsqueeze(1),
                    size=(H, W), mode='nearest'
                ).squeeze(1).bool().view(B, N, H, W)
                masks = masks | pad_mask_flat

        x_flat = x.flatten(0, 1)    # (B*N, C, H, W)
        x_proj = self.input_proj(x_flat)
        # (B, N, C, H, W)
        x_proj = x_proj.view(B, N, *x_proj.shape[-3:]).contiguous()

        pos_embed, masks = self.position_embeding(
            x, img_metas, masks, imu2img=imu2img,
            depth=depth, ray=ray,
            intrinsics=intrinsics, extrinsics=extrinsics,
        )
        # Multiview position adaptation
        # (B, N, 2*num_feats, H, W) -> (B, N, 256, H, W)
        sin_embed = self.positional_encoding(masks)
        sin_embed = self.adapt_pos3d(
            sin_embed.flatten(0, 1)
        ).view(x.size()).contiguous()
        pos_embed = pos_embed + sin_embed

        # Query Embedding
        reference_points = self.reference_points.weight
        query_embeds = self.query_embedding(pos2posemb3d(reference_points))

        # Transformer
        # x_proj: (B, N, C, H, W)
        outs_dec, _ = self.transformer(
            x_proj, masks, query_embeds, pos_embed, self.reg_branches,
        )

        # Sanitize transformer output
        if self.debug_nan_checks and torch.isnan(outs_dec).any():
            print("\n!!! WARNING: NaN detected in transformer output !!!")
            print(
                "  outs_dec NaN count: "
                f"{torch.isnan(outs_dec).sum().item()} / {outs_dec.numel()}"
            )
        outs_dec = torch.nan_to_num(outs_dec)

        # Process outputs
        outputs_classes = []
        outputs_coords = []

        reference_points_input = reference_points.unsqueeze(0).repeat(B, 1, 1)

        for lvl in range(outs_dec.shape[0]):
            reference = inverse_sigmoid(
                reference_points_input.clone(), eps=eps
            )

            outputs_class = self.cls_branches[lvl](outs_dec[lvl])
            tmp = self.reg_branches[lvl](outs_dec[lvl]).clone()

            # The regression head can output extreme values that cause NaN
            # Apply reference point offsets for x, y, z
            # Remove clamps
            tmp[..., 0:2] = (tmp[..., 0:2] + reference[..., 0:2])
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            tmp[..., 4:5] = (tmp[..., 4:5] + reference[..., 2:3])
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()

            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        all_cls_scores = torch.stack(outputs_classes)
        all_bbox_preds = torch.stack(outputs_coords)

        # Denormalize x, y, z from sigmoid [0,1] to world coordinates
        # (matches original PETR)
        # This ensures L1 loss operates on meter-scale values, not [0,1]
        all_bbox_preds[..., 0:1] = (
            all_bbox_preds[..., 0:1] *
            (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
        )
        all_bbox_preds[..., 1:2] = (
            all_bbox_preds[..., 1:2] *
            (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
        )
        all_bbox_preds[..., 4:5] = (
            all_bbox_preds[..., 4:5] *
            (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
        )

        result = {
            'all_cls_scores': all_cls_scores,
            'all_bbox_preds': all_bbox_preds,
        }
        return result

    def loss(
        self,
        gt_bboxes_list,
        gt_labels_list,
        preds_dicts,
        sync=True,
        num_boxes_synced=None,
    ):
        # Prepare targets
        targets = [
            {'boxes_3d': boxes, 'labels': labels}
            for boxes, labels in zip(gt_bboxes_list, gt_labels_list)
        ]
        loss_dict = self.criterion(
            preds_dicts,
            targets,
            sync=sync,
            num_boxes_synced=num_boxes_synced,
        )
        # Phase 4 — fold in the auxiliary rig-raymap loss if produced in
        # ``forward``. Weight is applied here, so PETR.training_step's
        # ``sum(loss_dict.values())`` already accounts for it. SetCriterion
        # tolerates the extra key in preds_dicts (only reads cls/bbox).
        return loss_dict

    def get_bboxes(self, preds_dicts, img_metas=None, rescale=False):
        """Convert predictions to bboxes in world coordinates."""
        preds = preds_dicts['all_bbox_preds'][-1]
        scores = preds_dicts['all_cls_scores'][-1].sigmoid()

        B, N_query, _ = preds.shape
        results = []
        for i in range(B):
            p = preds[i]
            s = scores[i]
            # Denormalize from model output format to world coordinates
            boxes = denormalize_bbox(p, self.pc_range)
            results.append((boxes, s))
        return results

