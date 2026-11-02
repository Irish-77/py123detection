import math

import torch


def normalize_bbox(bboxes, pc_range=None):
    """Normalize bboxes to match model output format.

    Input bbox format: [cx, cy, cz, w, l, h, rot, vx, vy]
    (9 values, world coords)
    Output format: [
        x_norm, y_norm, log(w), log(l), z_norm, log(h),
        sin(rot), cos(rot),
        vx, vy
    ] (10 values)

    x_norm, y_norm, z_norm are in [0, 1] if pc_range is provided.
    """
    cx = bboxes[..., 0:1]
    cy = bboxes[..., 1:2]
    cz = bboxes[..., 2:3]

    # NOTE: cx, cy, cz are kept in world coordinates (NOT normalized to [0,1]).
    # This matches original PETR where the head forward pass denormalizes
    # sigmoid outputs to world coords, and normalize_bbox does not re-normalize

    w = bboxes[..., 3:4].clamp(min=1e-5).log()
    l = bboxes[..., 4:5].clamp(min=1e-5).log()  # noqa: E741
    h = bboxes[..., 5:6].clamp(min=1e-5).log()

    rot = bboxes[..., 6:7]
    if bboxes.size(-1) > 7:
        vx = bboxes[..., 7:8]
        vy = bboxes[..., 8:9]
        normalized_bboxes = torch.cat(
            (cx, cy, w, l, cz, h, rot.sin(), rot.cos(), vx, vy), dim=-1
        )
    else:
        normalized_bboxes = torch.cat(
            (cx, cy, w, l, cz, h, rot.sin(), rot.cos()), dim=-1
        )
    return normalized_bboxes


def denormalize_bbox(normalized_bboxes, pc_range=None):
    """Denormalize bboxes from model output format to world coordinates.

    Input format: [
        x_norm, y_norm, log(w), log(l), z_norm, log(h),
        sin(rot), cos(rot),
        vx, vy
    ] (10 values)
    Output format: [cx, cy, cz, w, l, h, rot, vx, vy] (9 values, world coords)
    """
    # rotation
    rot_sine = normalized_bboxes[..., 6:7]
    rot_cosine = normalized_bboxes[..., 7:8]
    rot = torch.atan2(rot_sine, rot_cosine)

    # center in the bev (already in world coordinates — matches original PETR)
    cx = normalized_bboxes[..., 0:1]
    cy = normalized_bboxes[..., 1:2]
    cz = normalized_bboxes[..., 4:5]

    # size (log space -> actual size)
    w = normalized_bboxes[..., 2:3]
    l = normalized_bboxes[..., 3:4]  # noqa: E741
    h = normalized_bboxes[..., 5:6]

    w = w.exp()
    l = l.exp()  # noqa: E741
    h = h.exp()
    if normalized_bboxes.size(-1) > 8:
        # velocity
        vx = normalized_bboxes[..., 8:9]
        vy = normalized_bboxes[..., 9:10]
        denormalized_bboxes = torch.cat(
            [cx, cy, cz, w, l, h, rot, vx, vy],
            dim=-1
        )
    else:
        denormalized_bboxes = torch.cat([cx, cy, cz, w, l, h, rot], dim=-1)
    return denormalized_bboxes


def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.

    Args:
        x (Tensor): The tensor to do the inverse.
        eps (float): EPS avoid numerical overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse function of sigmoid, has same
        shape with input.
    """
    # Match official PETR implementation
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def pos2posemb3d(pos, num_pos_feats=128, temperature=10000):
    """
    Convert 3D positions to sinusoidal positional embeddings.

    Args:
        pos (torch.Tensor): Tensor of shape (B, N, 3) containing 3D positions.
        num_pos_feats (int): Number of positional encoding features.
        temperature (int): Temperature for positional encoding.

    Returns:
        torch.Tensor: Tensor of shape (B, N, num_pos_feats) containing
        positional embeddings.
    """
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_z = pos[..., 2, None] / dim_t
    pos_x = torch.stack(
        (pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1
    ).flatten(-2)
    pos_y = torch.stack(
        (pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1
    ).flatten(-2)
    pos_z = torch.stack(
        (pos_z[..., 0::2].sin(), pos_z[..., 1::2].cos()), dim=-1
    ).flatten(-2)
    posemb = torch.cat((pos_y, pos_x, pos_z), dim=-1)
    return posemb
