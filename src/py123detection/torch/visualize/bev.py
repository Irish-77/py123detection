"""A bird's-eye view of one frame: what the model found, over what was there.

Reads one line of a predictions file. No checkpoint, no GPU, no 123D logs.

The drawing follows the convention the thesis used, so figures from the two are comparable:

* **Forward is up, left is left.** The plot axes are ``(ego y, ego x)`` with x inverted.
* **Ground truth is filled orange**, predictions are **hollow blue outlines** on top. Filled
  against hollow separates them at a glance; colour alone does not, at a thumbnail size.
* **Confidence shades the outline**, from light grey at 0 to full blue at 1. A faint box is one
  the model was unsure of.
* Range rings at 10, 25 and 50 m, and a tick from each box's centre to its leading edge for
  heading.

Boxes whose centre falls outside the window are dropped rather than clipped, so a box at the edge
is a box that is really there.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np

from py123detection.torch.predictions import FramePrediction
from py123detection.torch.visualize.boxes import footprints, to_ego

#: Ground truth: filled, warm. Predictions: hollow, cool, shaded by confidence towards grey.
TRUTH_COLOUR = "darkorange"
PREDICTION_COLOUR = "dodgerblue"
LOW_CONFIDENCE_COLOUR = "#cccccc"
TRUTH_FILL_ALPHA = 0.35

#: Default score below which a prediction is not drawn. A detector emits one box per query — 900
#: for the shipped configs — so an unfiltered view is a solid block and shows nothing.
SCORE_THRESHOLD = 0.3

#: Half-extent of the view in metres, forward and sideways.
RANGE = 50.0

#: The ego vehicle's drawn footprint.
EGO_LENGTH = 4.5
EGO_WIDTH = 2.0


def _ego_frame(frame: FramePrediction) -> Dict[str, Any]:
    """Boxes in the ego frame where the pose allows it, otherwise centred but unrotated."""
    if frame.ego_pose is not None:
        return {
            "boxes": to_ego(frame.boxes, frame.ego_pose),
            "gt_boxes": to_ego(frame.gt_boxes, frame.ego_pose),
            "aligned": True,
        }

    centred: Dict[str, Any] = {}
    for name, boxes in (("boxes", frame.boxes), ("gt_boxes", frame.gt_boxes)):
        moved = np.asarray(boxes, dtype=np.float64).copy()
        if moved.size:
            moved[:, :3] = moved[:, :3] - np.asarray(frame.ego_translation, dtype=np.float64)
        centred[name] = moved
    centred["aligned"] = False
    return centred


def _plot_xy(ego_xy: np.ndarray) -> np.ndarray:
    """``(ego x, ego y)`` to figure coordinates ``(ego y, ego x)``.

    With the x-axis inverted at the figure level this gives forward up, left left.
    """
    return np.stack([ego_xy[..., 1], ego_xy[..., 0]], axis=-1)


def _shade(colour, score: float):
    """A prediction's colour: grey at score 0, the full colour at 1."""
    import matplotlib.colors as mcolors

    low = np.array(mcolors.to_rgb(LOW_CONFIDENCE_COLOUR))
    high = np.array(mcolors.to_rgb(colour))
    weight = float(np.clip(score, 0.0, 1.0))
    return tuple(low * (1.0 - weight) + high * weight)


def _in_range(outlines: np.ndarray, extent: float) -> np.ndarray:
    """Which boxes have their centre inside the window."""
    if len(outlines) == 0:
        return np.zeros(0, dtype=bool)
    centres = outlines.mean(axis=1)
    return (np.abs(centres[:, 0]) <= extent) & (np.abs(centres[:, 1]) <= extent)


def _draw_box(ax, outline: np.ndarray, colour, line_width: float, fill_alpha: float, zorder: int) -> None:
    """One oriented rectangle plus its heading tick."""
    from matplotlib.patches import Polygon

    ax.add_patch(
        Polygon(
            _plot_xy(outline),
            closed=True,
            facecolor=(*colour, fill_alpha) if fill_alpha > 0 else "none",
            edgecolor=colour,
            linewidth=line_width,
            zorder=zorder,
        )
    )
    # Corners 0 and 1 are the leading edge, so this tick says which way the box points.
    tick = _plot_xy(np.stack([outline.mean(axis=0), outline[:2].mean(axis=0)]))
    ax.plot(tick[:, 0], tick[:, 1], color=colour, linewidth=line_width, zorder=zorder)


def render_bev(
    frame: FramePrediction,
    class_names: Optional[Sequence[str]] = None,
    score_threshold: float = SCORE_THRESHOLD,
    radius: float = RANGE,
    line_width: float = 1.5,
    ax: Any = None,
    legend: bool = True,
    title: bool = True,
):
    """Draw one frame's ground truth and predictions from above.

    :param class_names: Named in the title when given; otherwise the title omits classes.
    :param score_threshold: Predictions below this are left out, and the title says how many.
    :param radius: Half-extent of the view in metres, forward and sideways.
    :param ax: Draw into this axis instead of making a figure.
    :returns: the matplotlib ``Figure`` the axis belongs to.
    """
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle, Patch, Polygon

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    placed = _ego_frame(frame)
    scores = np.asarray(frame.scores, dtype=np.float64)
    keep = scores >= score_threshold if len(scores) else np.zeros(0, dtype=bool)
    predictions = placed["boxes"][keep] if len(placed["boxes"]) else placed["boxes"]

    ax.set_xlim(radius, -radius)  # inverted, so the vehicle's left is the figure's left
    ax.set_ylim(-radius, radius)
    ax.set_aspect("equal")
    ax.set_facecolor("white")

    for ring in (10.0, 25.0, 50.0):
        if ring <= radius:
            ax.add_patch(
                Circle((0, 0), ring, fill=False, edgecolor="#cccccc", linewidth=0.5, linestyle="--", zorder=0)
            )
    ax.axhline(0, color="#dddddd", linewidth=0.4, zorder=0)
    ax.axvline(0, color="#dddddd", linewidth=0.4, zorder=0)

    half = np.array([EGO_LENGTH, EGO_WIDTH]) / 2.0
    ego = np.array([[+half[0], +half[1]], [+half[0], -half[1]], [-half[0], -half[1]], [-half[0], +half[1]]])
    ax.add_patch(
        Polygon(_plot_xy(ego), closed=True, facecolor="#5a5a5a", edgecolor="black",
                linewidth=line_width, alpha=0.9, zorder=2)
    )

    truth = mcolors.to_rgb(TRUTH_COLOUR)
    outlines = footprints(placed["gt_boxes"])
    for outline in outlines[_in_range(outlines, radius)]:
        _draw_box(ax, outline, truth, line_width, TRUTH_FILL_ALPHA, zorder=3)

    outlines = footprints(predictions)
    visible = _in_range(outlines, radius)
    for outline, score in zip(outlines[visible], scores[keep][visible]):
        _draw_box(ax, outline, _shade(PREDICTION_COLOUR, score), line_width * 1.5, 0.0, zorder=4)

    ax.set_xlabel("ego y (m)" if placed["aligned"] else "global east (m)", fontsize=7, labelpad=4)
    ax.set_ylabel("ego x (m)" if placed["aligned"] else "global north (m)", fontsize=7, labelpad=4)
    ax.tick_params(labelsize=6, pad=3)

    if legend:
        ax.legend(
            handles=[
                Patch(facecolor=(*truth, TRUTH_FILL_ALPHA), edgecolor=truth, label="ground truth"),
                Line2D([], [], color=PREDICTION_COLOUR, linewidth=line_width * 1.5, label="prediction"),
                Line2D([], [], color=LOW_CONFIDENCE_COLOUR, linewidth=line_width * 1.5, label="low confidence"),
            ],
            loc="upper right",
            fontsize=6,
            framealpha=0.85,
        )

    if title:
        classes = ""
        if class_names is not None and len(frame.gt_labels):
            classes = "  ·  " + ", ".join(sorted({class_names[label] for label in frame.gt_labels}))
        ax.set_title(
            f"{frame.dataset}  {frame.token}\n"
            f"{len(placed['gt_boxes'])} ground truth  ·  {len(predictions)} of {len(scores)} "
            f"predictions at score ≥ {score_threshold:g}{classes}",
            fontsize=8,
            pad=4,
        )
    return ax.figure


def save_bev(frame: FramePrediction, path, **kwargs) -> None:
    """Render one frame straight to a file, closing the figure so a long loop does not leak."""
    import matplotlib.pyplot as plt

    figure = render_bev(frame, **kwargs)
    figure.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(figure)
