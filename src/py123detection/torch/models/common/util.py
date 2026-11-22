"""Box encoding and the small numerical helpers both heads use.

A detection head does not regress a box in metres and radians. Sizes are learned in log space, so
a 20 cm error on a pedestrian and on a bus cost the same relatively, and yaw is learned as a
``(sin, cos)`` pair, so the discontinuity at ±pi does not sit in the middle of the loss.
:func:`normalize_bbox` and :func:`denormalize_bbox` move between that encoding and world units.
"""

import math

import torch


def normalize_bbox(bboxes, pc_range=None):
    """World-unit boxes into the head's encoding.

    ``[cx, cy, cz, w, l, h, rot, vx, vy]`` in, and
    ``[cx, cy, log w, log l, cz, log h, sin rot, cos rot, vx, vy]`` out. Note the column order:
    ``cz`` sits between ``log l`` and ``log h`` rather than with ``cx`` and ``cy``. That is
    official PETR's layout, and the code weights are indexed by it.

    Boxes without velocity (7 columns) come back without the last two.

    Centres stay in world coordinates. ``pc_range`` is accepted so that call sites can pass it
    uniformly, but nothing here reads it: the head's forward pass already denormalises its own
    sigmoid outputs into world units, which is what official PETR does.
    """
    cx = bboxes[..., 0:1]
    cy = bboxes[..., 1:2]
    cz = bboxes[..., 2:3]

    # Centres stay in world coordinates: the head denormalises its own sigmoid outputs, so
    # normalising again here would halve every coordinate the loss sees.

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
    """The head's encoding back into world units, the inverse of :func:`normalize_bbox`.

    ``[cx, cy, log w, log l, cz, log h, sin rot, cos rot, vx, vy]`` in,
    ``[cx, cy, cz, w, l, h, rot, vx, vy]`` out.
    """
    # rotation
    rot_sine = normalized_bboxes[..., 6:7]
    rot_cosine = normalized_bboxes[..., 7:8]
    rot = torch.atan2(rot_sine, rot_cosine)

    # Centres are already in world coordinates, as in official PETR.
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
    """The inverse of ``sigmoid``, clamped so that 0 and 1 do not become infinite.

    :param eps: How far from the open interval the input may sit before it is pulled back in.
    """
    # Match official PETR implementation
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def pos2posemb3d(pos, num_pos_feats=128, temperature=10000):
    """Sinusoidal embedding of 3D positions: ``(B, N, 3)`` in, ``(B, N, 3 * num_pos_feats)`` out.

    The three axes are concatenated in the order y, x, z, which is what official PETR does. The
    order is arbitrary but not free to change: a checkpoint's query embedding learned one of them.

    :param num_pos_feats: Width of the embedding per axis, before concatenation.
    :param temperature: Wavelength spread of the sinusoids, as in the original transformer.
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
