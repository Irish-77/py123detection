"""Boxes drawn onto the camera images they were predicted from.

A bird's-eye view tells you where the model put a box; this tells you what it was looking at when it
did. Between them they separate the two failure modes that look identical in a metric: the object
was invisible, or the object was visible and the model missed it.

Projection is :func:`py123detection.torch.models.bevformer.utils.project_3d_to_cameras`, the same
function the BEVFormer encoder uses to decide which cameras see a BEV cell, rather than a second
implementation of the same arithmetic. It also returns the validity mask, which is what lets an edge
leaving the frame be dropped instead of drawn to a clamped corner.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from py123detection.torch.visualize.boxes import EDGES, corners

#: BGR, since cv2 draws in BGR. The same two colours the bird's-eye view uses, so the two panels
#: of a frame read as one picture: ground truth orange, predictions blue shaded by confidence.
TRUTH_COLOUR = (0, 140, 255)  # darkorange
PREDICTION_COLOUR = (255, 144, 30)  # dodgerblue
LOW_CONFIDENCE_COLOUR = (204, 204, 204)
SCORE_THRESHOLD = 0.3

#: Stroke width in pixels. Predictions get the full width and ground truth half of it, which keeps
#: them apart where a 3D box drawn on a 2D image cannot use fill.
THICKNESS = 3


def shade(colour: Tuple[int, int, int], score: float) -> Tuple[int, int, int]:
    """A prediction's colour: grey at score 0, the full colour at 1."""
    weight = float(np.clip(score, 0.0, 1.0))
    return tuple(
        int(round(LOW_CONFIDENCE_COLOUR[channel] * (1.0 - weight) + colour[channel] * weight))
        for channel in range(3)
    )


def project_corners(
    corners_3d: np.ndarray,
    imu2img: np.ndarray,
    image_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """``(N, 8, 3)`` corners to ``(N, 8, 2)`` pixels, with a ``(N, 8)`` validity mask.

    :param imu2img: ``(4, 4)`` projection for one camera, as the dataset supplies it.
    :param image_size: ``(height, width)`` of the image being drawn on.
    """
    import torch

    from py123detection.torch.models.bevformer.utils import project_3d_to_cameras

    height, width = image_size
    if len(corners_3d) == 0:
        return np.zeros((0, 8, 2)), np.zeros((0, 8), dtype=bool)

    points = torch.as_tensor(np.asarray(corners_3d, dtype=np.float64).reshape(1, -1, 1, 3))
    matrix = torch.as_tensor(np.asarray(imu2img, dtype=np.float64)).view(1, 1, 4, 4)
    normalised, valid = project_3d_to_cameras(points, matrix, pad_h=height, pad_w=width)

    pixels = normalised[0, 0, :, 0, :].numpy() * np.array([width, height], dtype=np.float64)
    return pixels.reshape(-1, 8, 2), valid[0, 0, :, 0].numpy().reshape(-1, 8)


def draw_boxes(
    image: np.ndarray,
    boxes: np.ndarray,
    imu2img: np.ndarray,
    colour: Tuple[int, int, int] = PREDICTION_COLOUR,
    thickness: int = THICKNESS,
    scores: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Draw each box's twelve edges onto a copy of ``image``.

    An edge is drawn only where both of its corners project in front of the camera and inside the
    frame. A box partly out of view therefore appears partly drawn, which is honest — joining it to
    a clamped corner would invent a shape the model never predicted.
    """
    import cv2

    canvas = np.ascontiguousarray(image.copy())
    height, width = canvas.shape[:2]
    pixels, valid = project_corners(corners(boxes), imu2img, (height, width))

    shades = [shade(colour, score) for score in scores] if scores is not None else None
    for index, (box_pixels, box_valid) in enumerate(zip(pixels, valid)):
        edge_colour = shades[index] if shades is not None else colour
        for start, end in EDGES:
            if not (box_valid[start] and box_valid[end]):
                continue
            cv2.line(
                canvas,
                (int(round(box_pixels[start, 0])), int(round(box_pixels[start, 1]))),
                (int(round(box_pixels[end, 0])), int(round(box_pixels[end, 1]))),
                edge_colour,
                thickness,
                lineType=cv2.LINE_AA,
            )
    return canvas


def render_camera(
    image: np.ndarray,
    imu2img: np.ndarray,
    predictions: np.ndarray,
    ground_truth: Optional[np.ndarray] = None,
    scores: Optional[Sequence[float]] = None,
    score_threshold: float = SCORE_THRESHOLD,
    thickness: int = THICKNESS,
) -> np.ndarray:
    """One camera image with ground truth under the predictions.

    Ground truth goes down first, in orange and at half the stroke width; predictions go on top in
    blue, shaded towards grey by their score. Boxes are in the frame ``imu2img`` projects from,
    which for this dataset is the ego frame.
    """
    canvas = image
    if ground_truth is not None and len(ground_truth):
        canvas = draw_boxes(canvas, ground_truth, imu2img, TRUTH_COLOUR, max(1, thickness // 2))

    kept, kept_scores = predictions, None
    if scores is not None and len(predictions):
        keep = np.asarray(scores, dtype=np.float64) >= score_threshold
        kept = predictions[keep]
        kept_scores = np.asarray(scores, dtype=np.float64)[keep]
    return draw_boxes(canvas, kept, imu2img, PREDICTION_COLOUR, thickness, scores=kept_scores)


def tile(images: Sequence[np.ndarray], columns: int = 3) -> np.ndarray:
    """Lay a rig's cameras out in a grid, so one picture covers the whole frame.

    Cameras of a rig need not share a resolution — Argoverse 2's do not — so each row is padded to
    its widest member rather than assuming they match.
    """
    if not images:
        raise ValueError("nothing to tile")

    rows = []
    for start in range(0, len(images), columns):
        row = list(images[start : start + columns])
        height = max(image.shape[0] for image in row)
        padded = [
            np.pad(image, ((0, height - image.shape[0]), (0, 0), (0, 0)), mode="constant")
            for image in row
        ]
        rows.append(np.concatenate(padded, axis=1))

    width = max(row.shape[1] for row in rows)
    return np.concatenate(
        [np.pad(row, ((0, 0), (0, width - row.shape[1]), (0, 0)), mode="constant") for row in rows],
        axis=0,
    )


def save_cameras(path, images: Sequence[np.ndarray], columns: int = 3) -> None:
    """Write a tiled rig view to a file."""
    import cv2

    cv2.imwrite(str(path), tile(images, columns=columns))
