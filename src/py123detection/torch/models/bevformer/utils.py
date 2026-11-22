"""Geometry for BEVFormer's bird's-eye grid.

Three pure functions: lay out the grid, lift each cell into a vertical pillar of 3D points, and
project those points into every camera. No parameters, no modules, which is why they are tested
directly and reused by the camera renderer rather than reimplemented there.
"""

from typing import List, Tuple

import torch


def create_bev_grid(bev_h: int, bev_w: int) -> torch.Tensor:
    """Cell centres of the BEV grid, normalised to ``[0, 1]``.

    :param bev_h: Cells along y.
    :param bev_w: Cells along x.
    :returns: ``(bev_h * bev_w, 2)`` of ``(x, y)``, row-major in y.
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
    """Lift each BEV cell into a vertical pillar of 3D points, for spatial cross-attention.

    A cell has no height, but the cameras that see it do. Sampling a few evenly spaced heights
    inside the detection range is how a cell finds the pixels that look at it.

    :param bev_grid: ``(Q, 2)``, from :func:`create_bev_grid`.
    :param num_heights: Points per pillar.
    :param pc_range: ``[x_min, y_min, z_min, x_max, y_max, z_max]``.
    :returns: ``(Q, num_heights, 3)`` in world coordinates.
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
    """Project world-frame points into every camera.

    The projections come from the dataset as ``imu2img``, which is already
    ``intrinsics @ extrinsics``, so nothing here has to know how a rig is calibrated.

    :param points_3d: ``(B, Q, P, 3)`` in the world (IMU) frame; ``P`` is the pillar height count.
    :param imu2img: ``(B, N, 4, 4)``.
    :param pad_h: Padded image height, as ``img_metas['pad_shape']`` reports it.
    :param pad_w: Padded image width.
    :returns: ``(coords_2d, valid_mask)``. Coordinates are ``(B, N, Q, P, 2)`` normalised to
        ``[0, 1]``; the mask is ``(B, N, Q, P)`` and ``True`` where the point is in front of the
        camera and inside the image.
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
