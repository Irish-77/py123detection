"""WIP - Not tested yet!

TODO:
- Add docstrings
"""

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torchvision.ops import sigmoid_focal_loss

from py123detection.torch.models.common.util import normalize_bbox


class HungarianMatcher(nn.Module):
    def __init__(
        self,
        cost_class: float = 2.0,
        cost_bbox: float = 0.25,
        pc_range=None,
        debug_nan_checks: bool = False,
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.pc_range = pc_range
        self.debug_nan_checks = debug_nan_checks

    @torch.no_grad()
    def forward(self, outputs, targets, precomputed_norm_bboxes=None):
        bs, num_queries = outputs["all_cls_scores"].shape[1:3]
        # Use the last layer outputs for matching
        # outputs["all_cls_scores"] shape:
        # [num_layers, bs, num_queries, num_classes]

        # DETACH and CAST to float32 for stability
        # [bs, num_queries, num_classes]
        out_prob = outputs["all_cls_scores"][-1].float().sigmoid()
        # [bs, num_queries, 9]
        out_bbox = outputs["all_bbox_preds"][-1].float()

        # Optional NaN diagnostics. Keep disabled by default to avoid
        # per-step synchronization in hot training paths.
        if self.debug_nan_checks and (
            torch.isnan(out_bbox).any() or torch.isinf(out_bbox).any()
        ):
            nan_count = torch.isnan(out_bbox).sum().item()
            inf_count = torch.isinf(out_bbox).sum().item()
            valid_mask = ~(torch.isnan(out_bbox) | torch.isinf(out_bbox))
            print(
                "\n!!! CRITICAL: NaN/Inf detected in model output"
                "BEFORE cost computation !!!"
            )
            print(f"  out_bbox NaN count: {nan_count}")
            print(f"  out_bbox Inf count: {inf_count}")
            if valid_mask.any():
                valid_vals = out_bbox[valid_mask]
                print(
                    f"  out_bbox valid range: [{valid_vals.min().item():.4f},"
                    f"{valid_vals.max().item():.4f}]"
                )
            else:
                print("  out_bbox: ALL values are NaN/Inf!")
        # Always sanitize before matching for numerical safety.
        out_bbox = torch.nan_to_num(
            out_bbox, nan=0.0, posinf=100.0, neginf=-100.0
        )

        # Compute the classification cost.
        alpha = 0.25
        gamma = 2.0
        eps = 1e-8

        # Vectorized pos/neg class cost components
        neg_cost_class = (1 - alpha) * (out_prob ** gamma) *\
            (-(1 - out_prob + eps).log())   # [bs, num_queries, num_classes]
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) *\
            (-(out_prob + eps).log())  # [bs, num_queries, num_classes]

        indices = []
        for i in range(bs):
            tgt_ids = targets[i]["labels"]
            tgt_bbox = targets[i]["boxes_3d"].float()

            # If no ground truth in this sample,
            # cost matrix is effectively empty
            if len(tgt_ids) == 0:
                indices.append((
                    torch.as_tensor([], dtype=torch.int64),
                    torch.as_tensor([], dtype=torch.int64)
                ))
                continue

            # Class cost for this sample
            cost_class = pos_cost_class[i][:, tgt_ids] - \
                neg_cost_class[i][:, tgt_ids]  # [num_queries, num_tgt_in_i]

            # BBox L1 cost for this sample — use pre-computed norm if available
            if precomputed_norm_bboxes is not None:
                tgt_bbox_norm = precomputed_norm_bboxes[i]
            else:
                tgt_bbox_norm = normalize_bbox(tgt_bbox, self.pc_range)
            cost_bbox = torch.cdist(
                out_bbox[i][:, :8], tgt_bbox_norm[:, :8], p=1
            )  # [num_queries, num_tgt_in_i]

            # Final cost matrix for this sample
            C = self.cost_bbox * cost_bbox + self.cost_class * \
                cost_class  # [num_queries, num_tgt_in_i]

            # Optional diagnostics only (sync-heavy).
            if self.debug_nan_checks and (
                torch.isnan(C).any() or torch.isinf(C).any()
            ):
                print(
                    "Warning: Cost matrix C contains NaNs or Infs!"
                    "replacing with 0."
                )
                if torch.isnan(cost_class).any():
                    print("  -> NaNs detected in cost_class")
                if torch.isnan(cost_bbox).any():
                    print("  -> NaNs detected in cost_bbox")
                    if torch.isnan(out_bbox).any():
                        print("     -> NaNs in out_bbox (Model output)")
                    if torch.isnan(tgt_bbox_norm).any():
                        print("     -> NaNs in tgt_bbox_norm (Ground Truth)")

            C = torch.nan_to_num(C, nan=0.0, posinf=1e6, neginf=-1e6).cpu()

            row_ind, col_ind = linear_sum_assignment(C)
            indices.append((
                torch.as_tensor(row_ind, dtype=torch.int64),
                torch.as_tensor(col_ind, dtype=torch.int64)
            ))

        return indices


class SetCriterion(nn.Module):
    def __init__(
        self,
        num_classes,
        matcher,
        weight_dict,
        losses,
        pc_range,
        code_weights=None
    ):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.pc_range = pc_range
        self.code_weights = code_weights

    def get_layer_loss(
        self,
        cls_scores,
        bbox_preds,
        targets,
        indices,
        num_boxes,
        precomputed_norm_bboxes=None,
    ):
        # cls_scores: [bs, num_queries, num_classes]
        # bbox_preds: [bs, num_queries, code_size]

        # If no ground truth boxes
        num_matches = sum(len(src) for src, _ in indices)
        if num_matches == 0:
            target_classes = torch.full(
                cls_scores.shape[:2], self.num_classes,
                dtype=torch.int64, device=cls_scores.device,
            )
            target_classes_onehot = F.one_hot(
                target_classes, num_classes=self.num_classes + 1,
            )[..., :-1].float()
            loss_ce = sigmoid_focal_loss(
                cls_scores, target_classes_onehot,
                alpha=0.25, gamma=2.0, reduction="none",
            ).sum() / num_boxes
            loss_bbox = bbox_preds.sum() * 0  # zero loss, preserves grad graph
            return {'loss_ce': loss_ce, 'loss_bbox': loss_bbox}

        # Labels Loss
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )
        target_classes = torch.full(
            cls_scores.shape[:2], self.num_classes,
            dtype=torch.int64, device=cls_scores.device
        )
        target_classes[idx] = target_classes_o

        target_classes_onehot = F.one_hot(
            target_classes,
            num_classes=self.num_classes + 1
        )[..., :-1].float()

        loss_ce = sigmoid_focal_loss(
            cls_scores,
            target_classes_onehot,
            alpha=0.25,
            gamma=2.0,
            reduction="none"
        )
        loss_ce = loss_ce.sum() / num_boxes

        # BBox Loss — use pre-normalized targets if available to avoid
        # redundant normalize_bbox (log/sin/cos) calls per decoder layer.
        src_boxes = bbox_preds[idx]
        if precomputed_norm_bboxes is not None:
            target_boxes = torch.cat(
                [precomputed_norm_bboxes[i][j]
                 for i, (_, j) in enumerate(indices)
                 if len(j) > 0],
                dim=0,
            )
        else:
            raw_target_boxes = torch.cat(
                [t['boxes_3d'][i] for t, (_, i) in zip(targets, indices)],
                dim=0,
            )
            target_boxes = normalize_bbox(raw_target_boxes, self.pc_range)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        if self.code_weights is not None:
            loss_bbox = loss_bbox * self.code_weights.to(loss_bbox.device)

        loss_bbox = loss_bbox.sum() / num_boxes

        return {'loss_ce': loss_ce, 'loss_bbox': loss_bbox}

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full((len(src),), i, dtype=torch.int64)
             for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def forward(self, outputs, targets, sync=True, num_boxes_synced=None):
        # outputs: dict with all_cls_scores, all_bbox_preds (stacked layers)
        # targets: list of dicts with boxes_3d, labels

        # [num_layers, bs, num_queries, num_classes]
        all_cls_scores = outputs['all_cls_scores']
        all_bbox_preds = outputs['all_bbox_preds']

        num_layers = len(all_cls_scores)

        if num_boxes_synced is not None:
            num_boxes = num_boxes_synced
        else:
            num_boxes = sum(len(t["labels"]) for t in targets)
            num_boxes = torch.as_tensor(
                num_boxes, dtype=torch.float, device=all_cls_scores.device
            )
            if sync and dist.is_available() and dist.is_initialized():
                torch.distributed.all_reduce(num_boxes)
                world_size = dist.get_world_size()
                num_boxes = torch.clamp(num_boxes / world_size, min=1.0)
            else:
                num_boxes = torch.clamp(num_boxes, min=1.0)

        # Pre-compute normalized GT boxes once for all layers.
        # normalize_bbox involves log/sin/cos — cheap to do once, not per-layer
        precomputed_norm_bboxes = [
            normalize_bbox(t['boxes_3d'].float(), self.pc_range)
            if len(t['labels']) > 0 else None
            for t in targets
        ]

        losses = {}
        for layer_idx in range(num_layers):
            cls_scores = all_cls_scores[layer_idx]
            bbox_preds = all_bbox_preds[layer_idx]

            # Run matching independently per layer (matches original PETR
            # behavior)
            # Passing [l:l+1] so the matcher's [-1] indexing picks this layer
            layer_outputs = {
                'all_cls_scores': all_cls_scores[layer_idx:layer_idx+1],
                'all_bbox_preds': all_bbox_preds[layer_idx:layer_idx+1],
            }
            indices = self.matcher(
                layer_outputs, targets,
                precomputed_norm_bboxes=precomputed_norm_bboxes
            )

            layer_losses = self.get_layer_loss(
                cls_scores, bbox_preds, targets, indices, num_boxes,
                precomputed_norm_bboxes=precomputed_norm_bboxes,
            )

            # Weighting: last layer has full weight, others usually same or
            # aux weight. Reference doesn't specify aux weights usually it's
            # same.

            key_prefix = f'd{layer_idx}.' if layer_idx < num_layers - 1 else ''

            for k, v in layer_losses.items():
                losses[key_prefix + k] = v * self.weight_dict.get(k, 1.0)

        return losses
