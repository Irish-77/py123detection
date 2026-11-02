from typing import List, Tuple

import torch


def create_bev_grid(bev_h: int, bev_w: int) -> torch.Tensor:
    """Create a normalized [0, 1] 2D BEV grid of cell centers.

    Args:
        bev_h: Number of BEV cells along the y-axis (height).
        bev_w: Number of BEV cells along the x-axis (width).

    Returns:
        grid: (bev_h * bev_w, 2) tensor of (x_norm, y_norm) in [0, 1].
    """
    xs = torch.linspace(0.5, bev_w - 0.5, bev_w) / bev_w
    ys = torch.linspace(0.5, bev_h - 0.5, bev_h) / bev_h
    # meshgrid with 'ij' indexing: grid_y[i, j] = ys[i], grid_x[i, j] = xs[j]
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    return torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)


def create_3d_reference_points(
    bev_grid: torch.Tensor,
    num_heights: int,
    pc_range: List[float],
) -> torch.Tensor:
    """Create 3D reference points for Spatial Cross-Attention.

    For each BEV cell, creates a vertical pillar of 3D points at uniformly
    spaced heights within the detection range.

    Args:
        bev_grid: (Q, 2) normalized [0, 1] BEV grid from create_bev_grid().
        num_heights: Number of height samples per pillar.
        pc_range: [x_min, y_min, z_min, x_max, y_max, z_max] detection range.

    Returns:
        ref_3d: (Q, num_heights, 3) 3D reference points in world coordinates.
    """
    Q = bev_grid.shape[0]
    x_min, y_min, z_min, x_max, y_max, z_max = pc_range

    # Denormalize BEV grid from [0, 1] to world coordinates
    x_world = bev_grid[:, 0] * (x_max - x_min) + x_min  # (Q,)
    y_world = bev_grid[:, 1] * (y_max - y_min) + y_min  # (Q,)

    # Uniformly spaced height samples
    z_vals = torch.linspace(
        z_min, z_max, num_heights, device=bev_grid.device
    )  # (num_heights,)

    # Expand: (Q, num_heights, 3)
    x_exp = x_world[:, None].expand(-1, num_heights)
    y_exp = y_world[:, None].expand(-1, num_heights)
    z_exp = z_vals[None, :].expand(Q, -1)

    return torch.stack([x_exp, y_exp, z_exp], dim=-1)


def project_3d_to_cameras(
    points_3d: torch.Tensor,
    imu2img: torch.Tensor,
    pad_h: int,
    pad_w: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Project 3D world-frame points to all camera image planes.

    Uses the imu2img (= intrinsics @ extrinsics) projection matrices
    from the dataset to map 3D reference points to 2D image coordinates.

    Args:
        points_3d: (B, Q, P, 3) 3D points in world/IMU frame.
            P = num_heights (pillar samples).
        imu2img: (B, N, 4, 4) projection matrices (intrinsics @ extrinsics).
        pad_h: Padded image height (from img_metas['pad_shape']).
        pad_w: Padded image width.

    Returns:
        coords_2d: (B, N, Q, P, 2) normalized [0, 1] image coordinates.
        valid_mask: (B, N, Q, P) bool tensor. True where projection is valid
            (positive depth and within image bounds).
    """
    B, Q, P, _ = points_3d.shape
    eps = 1e-5

    # To homogeneous: (B, Q, P, 4)
    ones = torch.ones(
        *points_3d.shape[:-1], 1,
        device=points_3d.device, dtype=points_3d.dtype,
    )
    pts_homo = torch.cat([points_3d, ones], dim=-1)

    # Project: (B, N, 4, 4) @ (B, Q, P, 4) -> (B, N, Q, P, 4)
    # Use einsum for clarity: i=4(out), j=4(contract)
    pts_cam = torch.einsum('bnij,bqpj->bnqpi', imu2img, pts_homo)
    # pts_cam[..., :3] = [u*d, v*d, d] in camera image space

    depth = pts_cam[..., 2]  # (B, N, Q, P)

    # Perspective divide
    u = pts_cam[..., 0] / depth.clamp(min=eps)
    v = pts_cam[..., 1] / depth.clamp(min=eps)

    # Normalize to [0, 1] using padded image dimensions
    u_norm = u / pad_w
    v_norm = v / pad_h

    # Validity check: positive depth AND within image bounds
    valid = (
        (depth > eps)
        & (u_norm >= 0) & (u_norm < 1)
        & (v_norm >= 0) & (v_norm < 1)
    )

    # Clamp to [0, 1] so invalid projections (behind camera, out of frame)
    # don't produce extreme float values that crash F.grid_sample in
    # deformable attention.  The valid mask already excludes these points.
    u_norm = u_norm.clamp(0.0, 1.0)
    v_norm = v_norm.clamp(0.0, 1.0)

    coords_2d = torch.stack([u_norm, v_norm], dim=-1)  # (B, N, Q, P, 2)

    return coords_2d, valid
