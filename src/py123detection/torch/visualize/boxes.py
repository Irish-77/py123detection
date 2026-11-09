"""Box corners, and the edges that join them.

Both renderers need the same thing from a box — where its eight corners are — so the geometry lives
here once rather than in each of them. Pure numpy: no torch, no matplotlib, no py123d, which is what
makes it cheap to test exhaustively.
"""

from __future__ import annotations

import numpy as np

#: Columns of a box: ``[x, y, z, w, l, h, yaw, vx, vy]``; only the first seven are needed here.
YAW = 6

#: The twelve edges of a box, as index pairs into :func:`corners`' output. Bottom face, top face,
#: then the four uprights.
EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


def corners(boxes: np.ndarray) -> np.ndarray:
    """``(N, 7+)`` boxes as ``(N, 8, 3)`` corners, in the frame the boxes are given in.

    Corners 0-3 are the bottom face and 4-7 the top, each going round in the same direction, so
    :data:`EDGES` joins them without a special case. ``w`` spans y and ``l`` spans x, matching the
    layout the dataset and the metric both use.
    """
    boxes = np.asarray(boxes, dtype=np.float64)
    if boxes.size == 0:
        return np.zeros((0, 8, 3), dtype=np.float64)

    centres = boxes[:, 0:3]
    width, length, height = boxes[:, 3], boxes[:, 4], boxes[:, 5]
    yaw = boxes[:, YAW]

    # Unit box corners, scaled per box: x by length, y by width, z by height.
    signs = np.array(
        [
            [+1, +1, -1], [+1, -1, -1], [-1, -1, -1], [-1, +1, -1],
            [+1, +1, +1], [+1, -1, +1], [-1, -1, +1], [-1, +1, +1],
        ],
        dtype=np.float64,
    )
    extents = np.stack([length, width, height], axis=-1) / 2.0
    local = signs[None, :, :] * extents[:, None, :]

    cos, sin = np.cos(yaw), np.sin(yaw)
    rotated_x = local[..., 0] * cos[:, None] - local[..., 1] * sin[:, None]
    rotated_y = local[..., 0] * sin[:, None] + local[..., 1] * cos[:, None]
    rotated = np.stack([rotated_x, rotated_y, local[..., 2]], axis=-1)

    return rotated + centres[:, None, :]


def footprints(boxes: np.ndarray) -> np.ndarray:
    """``(N, 4, 2)`` ground-plane outlines, for a bird's-eye view."""
    return corners(boxes)[:, :4, :2]


def to_ego(boxes: np.ndarray, ego_pose: np.ndarray) -> np.ndarray:
    """Global-frame boxes back into the ego frame, so a view can be drawn vehicle-forward.

    The inverse of the conversion :mod:`py123detection.torch.evaluate` applies on the way out.
    Without it a bird's-eye view is aligned to global north, which is readable but tells you nothing
    about where the vehicle was pointing.
    """
    boxes = np.asarray(boxes, dtype=np.float64)
    if boxes.size == 0:
        return boxes

    rotation = np.asarray(ego_pose, dtype=np.float64)[:3, :3]
    translation = np.asarray(ego_pose, dtype=np.float64)[:3, 3]
    inverse = rotation.T

    moved = boxes.copy()
    moved[:, :3] = (boxes[:, :3] - translation) @ inverse.T
    moved[:, YAW] = boxes[:, YAW] - float(np.arctan2(rotation[1, 0], rotation[0, 0]))
    if boxes.shape[1] > 8:
        moved[:, 7:9] = boxes[:, 7:9] @ inverse[:2, :2].T
    return moved
