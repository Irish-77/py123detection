"""
Shared PETR-style 3D frustum position embedding.

Provides:
- ``compute_frustum_coords3d``: pure-math helper that turns per-camera
  ``imu2img`` projection matrices into a per-pixel 3D coordinate volume
  ready for an ``inverse_sigmoid`` + MLP embedding. Used by both PETRHead's
  frustum branch and by :class:`PETRFrustumPositionEmbedding` (below).
- :class:`PETRFrustumPositionEmbedding`: standalone ``nn.Module`` wrapping
  the full PETR 3D position embedding pipeline (frustum math + MLP +
  optional sine-PE + ``adapt_pos3d`` fusion). Used by BEVFormer's
  ``"petr"`` camera-embedding option.

Factoring the math into a standalone helper keeps PETRHead's module
ownership (and therefore its ``state_dict`` keys) unchanged while letting
BEVFormer reuse the exact same geometry!
Othwerwise, error for already trained weights as some layer weight keys not
found!!!
"""

from typing import Optional, Tuple

import torch

# Same wider-than-pc_range default as PETRHead. Ensures points inside the
# detection volume map to the interior of [0, 1] after normalization, so the
# subsequent ``inverse_sigmoid`` does not saturate at the boundaries.
DEFAULT_POSITION_RANGE = [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0]


def compute_frustum_coords3d(
    B: int,
    N: int,
    H: int,
    W: int,
    pad_h: int,
    pad_w: int,
    imu2img: torch.Tensor,
    depth_num: int,
    depth_start: float,
    LID: bool,
    position_range_t: torch.Tensor,
    mask_strict: bool,
    device: torch.device,
    masks: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build the 3D frustum coordinate volume and its validity mask.

    Mirrors the frustum branch of ``PETRHead.position_embeding``
    (petr/head.py:297-451). Returns tensors that are ready to be passed
    through ``inverse_sigmoid`` and the per-model ``position_encoder`` MLP.

    Args:
        B, N: Batch size and number of cameras.
        H, W: Feature-grid spatial size.
        pad_h, pad_w: Padded image dimensions (the feature grid is
            downsampled from the padded image).
        imu2img: ``(B, N, 4, 4)`` camera projection matrices
            ``intrinsics @ extrinsics``. Singular matrices (padded cams,
            ``|det| < 1e-6``) are replaced with the identity.
        depth_num: Number of discrete depth bins along each ray (PETR
            default: 64).
        depth_start: Near-plane depth in meters (PETR default: 1.0).
        LID: If ``True`` use log-increasing discretization (finer bins
            near the camera); else uniform.
        position_range_t: ``(6,)`` FP32 buffer with
            ``[x_min, y_min, z_min, x_max, y_max, z_max]`` used for
            normalization (should be **wider** than the detection
            ``pc_range`` to avoid saturation at the inverse_sigmoid edges).
        mask_strict: If ``True`` match official PETR's sum-of-violations
            logic; else use the legacy "any component out" logic.
        device: Device for the constructed tensors.
        masks: Optional ``(B, N, H, W)`` bool tensor where ``True`` marks
            invalid pixels (padded cameras, padded image borders). Used
            to zero out those pixels' coordinates before inverse_sigmoid.
        eps: Numerical floor for perspective division.

    Returns:
        coords3d: ``(B*N, depth_num * 3, H, W)`` flattened coordinate
            volume. Still in ``[0, 1]`` (NOT yet inverse-sigmoid'd) so
            the caller is free to pick its own FP32 autocast scope.
        coords_mask: ``(B, N, H, W)`` bool tensor where ``True`` marks
            pixels that should be masked in downstream attention. The
            center pixel of every batch element is forced valid to
            prevent all-masked rows.
    """
    D = depth_num

    # Pixel centres at the feature grid mapped back to padded-image space.
    coords_h = torch.arange(H, device=device).float() * pad_h / H
    coords_w = torch.arange(W, device=device).float() * pad_w / W

    # Depth bin centres (LID: finer near-camera; uniform otherwise).
    if LID:
        index = torch.arange(start=0, end=D, step=1, device=device).float()
        index_1 = index + 1
        bin_size = (position_range_t[3] - depth_start) / (D * (1 + D))
        coords_d = depth_start + bin_size * index * index_1
    else:
        index = torch.arange(start=0, end=D, step=1, device=device).float()
        bin_size = (position_range_t[3] - depth_start) / D
        coords_d = depth_start + bin_size * index

    # (W, H, D, 3) then homogenize to (..., 4) with x/y pre-multiplied by d.
    coords = torch.stack(
        torch.meshgrid([coords_w, coords_h, coords_d], indexing='ij')
    )
    coords = coords.permute(1, 2, 3, 0)  # (W, H, D, 3)
    coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1)
    coords[..., :2] = coords[..., :2] * torch.maximum(
        coords[..., 2:3], torch.ones_like(coords[..., 2:3]) * eps
    )

    # Invert imu2img; replace singular matrices (padded cams) with identity.
    if imu2img.dim() == 3:
        imu2img = imu2img.unsqueeze(1)
    imu2img = imu2img.to(device=device, dtype=torch.float32)
    singular = (torch.linalg.det(imu2img).abs() < 1e-6)
    eye = torch.eye(4, device=device, dtype=imu2img.dtype)
    imu2img = torch.where(
        singular.unsqueeze(-1).unsqueeze(-1),
        eye.view(1, 1, 4, 4),
        imu2img,
    )
    img2imus = torch.linalg.inv(imu2img)  # (B, N, 4, 4)

    coords = coords.view(1, 1, W, H, D, 4, 1).expand(B, N, -1, -1, -1, -1, -1)
    img2imus = img2imus.view(B, N, 1, 1, 1, 4, 4).expand(
        -1, -1, W, H, D, -1, -1
    )

    # Broadcast matmul → world-frame coords (B, N, W, H, D, 3).
    coords3d = torch.matmul(img2imus, coords).squeeze(-1)[..., :3]

    # Normalize to [0, 1] using position_range (no clamping — matches PETR).
    coords3d[..., 0:1] = (coords3d[..., 0:1] - position_range_t[0]) \
        / (position_range_t[3] - position_range_t[0])
    coords3d[..., 1:2] = (coords3d[..., 1:2] - position_range_t[1]) \
        / (position_range_t[4] - position_range_t[1])
    coords3d[..., 2:3] = (coords3d[..., 2:3] - position_range_t[2]) \
        / (position_range_t[5] - position_range_t[2])

    # Out-of-bounds detection.
    oob = (coords3d > 1.0) | (coords3d < 0.0)
    if mask_strict:
        # Sum of violations across (x, y, z) for each depth bin.
        # (B, N, W, H, D, 3) → flatten(-2) → (B, N, W, H, D*3) → sum(-1)
        violations = oob.flatten(-2).sum(-1)
        coords_mask = violations > (D * 0.5)
    else:
        # Legacy: any component out → ray invalid at that depth.
        is_point_out = oob.any(dim=-1)                 # (B, N, W, H, D)
        num_out = is_point_out.float().sum(dim=-1)     # (B, N, W, H)
        coords_mask = num_out > (D * 0.5)

    # (B, N, W, H) → (B, N, H, W) and merge with caller-provided mask.
    coords_mask = coords_mask.permute(0, 1, 3, 2)
    if masks is not None:
        coords_mask = masks | coords_mask

    # Prevent all-masked rows (would NaN-out the cross-attention).
    flat_mask = coords_mask.flatten(1)
    all_masked = flat_mask.all(dim=1)
    coords_mask[all_masked, 0, H // 2, W // 2] = False

    # Permute to (B*N, D*3, H, W) for the subsequent 1×1 Conv2d MLP.
    # Source dims: 0:B, 1:N, 2:W, 3:H, 4:D, 5:3
    # permute(0, 1, 4, 5, 3, 2) → (B, N, D, 3, H, W)
    coords3d = coords3d.permute(0, 1, 4, 5, 3, 2).contiguous().view(
        B * N, -1, H, W
    )

    # Zero out padded-camera coords BEFORE inverse_sigmoid. Padded cams have
    # identity imu2img which yields bogus coords; inverse_sigmoid(0.5) = 0,
    # which keeps their contribution neutral.
    if masks is not None:
        flat_mask_bn = masks.view(B * N, 1, H, W).expand_as(coords3d)
        coords3d = coords3d.masked_fill(flat_mask_bn, 0.5)

    return coords3d, coords_mask
