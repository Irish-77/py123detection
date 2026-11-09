"""Rendering a run's predictions: from above, and on the images they came from.

Both views read a predictions file written by ``py123det-eval``, so neither needs a checkpoint or a
GPU. The bird's-eye view needs nothing else at all; drawing on camera images needs the dataset the
run was scored on, because the file holds boxes and not pixels.

Needs the ``viz`` extra (matplotlib; ``cv2`` comes with ``torch`` and ``images``).
"""

from py123detection.torch.visualize.bev import render_bev, save_bev
from py123detection.torch.visualize.boxes import EDGES, corners, footprints, to_ego
from py123detection.torch.visualize.cameras import (
    draw_boxes,
    project_corners,
    render_camera,
    save_cameras,
    tile,
)

__all__ = [
    "EDGES",
    "corners",
    "draw_boxes",
    "footprints",
    "project_corners",
    "render_bev",
    "render_camera",
    "save_bev",
    "save_cameras",
    "tile",
    "to_ego",
]
