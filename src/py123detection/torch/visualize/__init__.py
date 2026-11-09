"""Rendering a run's predictions: from above, and on the images they came from.

Both views read a predictions file written by ``py123det-eval``, so neither needs a checkpoint or a
GPU. The bird's-eye view needs nothing else at all; drawing on camera images needs the dataset the
run was scored on, because the file holds boxes and not pixels.

There is no viewer here: py123d ships one, and :mod:`py123detection.torch.visualize.py123d_interop`
converts a predictions file into the detection type it draws, which is the piece that was missing.

Needs the ``viz`` extra (matplotlib; ``cv2`` comes with ``torch`` and ``images``).
"""

from py123detection.torch.visualize.bev import render_bev, save_bev
from py123detection.torch.visualize.boxes import EDGES, corners, footprints, to_ego
from py123detection.torch.visualize.py123d_interop import (
    add_to_viser,
    frames_by_token,
    label_lookup,
    to_box_array,
    to_detections,
    tokens_in_order,
    unmapped_classes,
    wireframe,
)
from py123detection.torch.visualize.cameras import (
    draw_boxes,
    project_corners,
    render_camera,
    save_cameras,
    tile,
)

__all__ = [
    "EDGES",
    "add_to_viser",
    "corners",
    "draw_boxes",
    "footprints",
    "project_corners",
    "render_bev",
    "render_camera",
    "save_bev",
    "save_cameras",
    "frames_by_token",
    "label_lookup",
    "tile",
    "to_box_array",
    "to_detections",
    "to_ego",
    "tokens_in_order",
    "unmapped_classes",
    "wireframe",
]
