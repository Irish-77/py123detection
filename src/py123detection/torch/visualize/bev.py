"""A bird's-eye view of one frame: what the model found, over what was there.

This reads nothing but a line of a predictions file, so it needs no checkpoint, no GPU and no 123D
logs — which is the point. After an evaluation you have a number per dataset; this is how you find
out *why* it is that number, one frame at a time.

Predictions and ground truth are drawn in the same frame by the same code. The boxes in the file are
in the global frame, so they come back into the ego frame here when the file records the pose,
leaving the vehicle at the origin facing +x. Without the pose the view falls back to centring on the
vehicle without rotating, which is still readable but aligned to global north.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np

from py123detection.torch.predictions import FramePrediction
from py123detection.torch.visualize.boxes import footprints, to_ego

#: Ground truth in grey, predictions in colour: the eye should go to what the model claimed.
TRUTH_COLOUR = "#555555"
PREDICTION_COLOUR = "#d62728"

#: Default score below which a prediction is not drawn. A detector emits one box per query — 900 of
#: them for the shipped configs — so an unfiltered view is a solid block of boxes and shows nothing.
SCORE_THRESHOLD = 0.3


def _ego_frame(frame: FramePrediction) -> Dict[str, np.ndarray]:
    """Boxes in the ego frame where the pose allows it, otherwise centred but unrotated."""
    if frame.ego_pose is not None:
        return {
            "boxes": to_ego(frame.boxes, frame.ego_pose),
            "gt_boxes": to_ego(frame.gt_boxes, frame.ego_pose),
            "aligned": True,
        }

    centred = {}
    for name, boxes in (("boxes", frame.boxes), ("gt_boxes", frame.gt_boxes)):
        moved = np.asarray(boxes, dtype=np.float64).copy()
        if moved.size:
            moved[:, :3] = moved[:, :3] - np.asarray(frame.ego_translation, dtype=np.float64)
        centred[name] = moved
    centred["aligned"] = False
    return centred


def render_bev(
    frame: FramePrediction,
    class_names: Optional[Sequence[str]] = None,
    score_threshold: float = SCORE_THRESHOLD,
    radius: float = 60.0,
    ax: Any = None,
):
    """Draw one frame's ground truth and predictions from above.

    :param class_names: Used only for the legend; indices are shown when it is omitted.
    :param score_threshold: Predictions scoring below this are left out, and the title says how
        many that was — a count of what you are not seeing is part of reading the picture.
    :param radius: Half-width of the view in metres.
    :param ax: Draw into this axis instead of making a figure, so several frames can share one.
    :returns: the matplotlib ``Figure`` the axis belongs to.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Polygon

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))

    placed = _ego_frame(frame)
    keep = np.asarray(frame.scores, dtype=np.float64) >= score_threshold
    predictions = placed["boxes"][keep] if len(placed["boxes"]) else placed["boxes"]
    dropped = int((~keep).sum())

    for outline in footprints(placed["gt_boxes"]):
        ax.add_patch(Polygon(outline, closed=True, fill=False, edgecolor=TRUTH_COLOUR, linewidth=1.8))

    scores = np.asarray(frame.scores, dtype=np.float64)[keep] if len(frame.scores) else np.empty(0)
    for outline, score in zip(footprints(predictions), scores):
        # Confidence as opacity, so a dense frame still reads: the solid boxes are the sure ones.
        ax.add_patch(
            Polygon(
                outline,
                closed=True,
                fill=False,
                edgecolor=PREDICTION_COLOUR,
                linewidth=1.4,
                alpha=float(np.clip(score, 0.25, 1.0)),
            )
        )

    # The ego vehicle, pointing the way the view is oriented.
    ax.add_patch(Polygon([[-2.3, -0.9], [2.3, -0.9], [2.3, 0.9], [-2.3, 0.9]], closed=True,
                         fill=True, facecolor="#1f77b4", alpha=0.7))
    ax.arrow(0, 0, radius * 0.12, 0, head_width=radius * 0.03, color="#1f77b4", alpha=0.7)

    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m), vehicle forward" if placed["aligned"] else "x (m), global east")
    ax.set_ylabel("y (m), vehicle left" if placed["aligned"] else "y (m), global north")
    ax.grid(True, linewidth=0.3, alpha=0.4)

    classes = ""
    if class_names is not None and len(frame.gt_labels):
        present = sorted({class_names[label] for label in frame.gt_labels})
        classes = f"  ·  {', '.join(present)}"
    ax.set_title(
        f"{frame.dataset}  {frame.token}\n"
        f"{len(placed['gt_boxes'])} ground truth  ·  {len(predictions)} predictions "
        f"at score ≥ {score_threshold:g}  ({dropped} below){classes}",
        fontsize=9,
    )
    ax.legend(
        handles=[
            Line2D([], [], color=TRUTH_COLOUR, label="ground truth"),
            Line2D([], [], color=PREDICTION_COLOUR, label="prediction"),
        ],
        loc="upper right",
        fontsize=8,
    )
    return ax.figure


def save_bev(frame: FramePrediction, path, **kwargs) -> None:
    """Render one frame straight to a file, closing the figure so a long loop does not leak."""
    import matplotlib.pyplot as plt

    figure = render_bev(frame, **kwargs)
    figure.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(figure)
