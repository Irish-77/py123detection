"""One frame as two panels: the front camera, and the bird's-eye view beside it.

This is the figure the thesis used to report qualitative results, and the most useful single
picture of a detector. The camera panel says what the model was looking at; the BEV panel says
where it put the box. A miss that is invisible in the image and a miss that is plainly visible
look identical in a metric and nothing alike here.

Both panels share one colour scheme: ground truth orange, predictions blue shaded towards grey by
confidence.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from py123detection.torch.predictions import FramePrediction
from py123detection.torch.visualize.bev import render_bev
from py123detection.torch.visualize.cameras import SCORE_THRESHOLD, render_camera


def render_frame(
    image: np.ndarray,
    imu2img: np.ndarray,
    frame: FramePrediction,
    predictions: np.ndarray,
    ground_truth: Optional[np.ndarray] = None,
    class_names: Optional[Sequence[str]] = None,
    score_threshold: float = SCORE_THRESHOLD,
    radius: float = 50.0,
    camera_name: str = "front",
    height: float = 4.5,
) -> Any:
    """Camera on the left, bird's-eye view on the right.

    :param image: The camera image, RGB.
    :param imu2img: That camera's projection, as the dataset supplies it.
    :param frame: The record the bird's-eye view is drawn from; it holds global-frame boxes.
    :param predictions: The same predictions in the **ego** frame, which is what projects onto
        the image. Passed separately rather than derived here because the caller already has them.
    :param ground_truth: Ego-frame ground truth, drawn on the camera panel when given.
    :returns: the matplotlib ``Figure``.
    """
    import matplotlib.pyplot as plt

    if len(predictions) != len(frame.scores):
        raise ValueError(
            f"{len(predictions)} ego-frame prediction(s) for {len(frame.scores)} score(s): they "
            f"are the same boxes in two frames, so a mismatch means the panels would disagree."
        )

    # Both panels get the same height and exactly the width they need, so neither is boxed in by
    # the other's aspect ratio: the BEV is square, the image is not, and camera rigs disagree.
    aspect = image.shape[1] / image.shape[0]
    figure, (camera_ax, bev_ax) = plt.subplots(
        1, 2,
        figsize=(height * aspect + height + 0.8, height),
        dpi=110,
        gridspec_kw={"width_ratios": [aspect, 1.0]},
    )

    drawn = render_camera(
        image[..., ::-1],  # to BGR, which is what cv2 draws in
        imu2img,
        predictions,
        ground_truth=ground_truth,
        scores=frame.scores,
        score_threshold=score_threshold,
    )
    camera_ax.imshow(drawn[..., ::-1])
    camera_ax.set_title(camera_name, fontsize=8, pad=4)
    camera_ax.set_xticks([])
    camera_ax.set_yticks([])

    render_bev(
        frame,
        class_names=class_names,
        score_threshold=score_threshold,
        radius=radius,
        ax=bev_ax,
        title=False,
    )
    bev_ax.set_title("bird's-eye view", fontsize=8, pad=4)

    scores = np.asarray(frame.scores, dtype=np.float64)
    kept = int((scores >= score_threshold).sum()) if len(scores) else 0
    figure.suptitle(
        f"{frame.dataset}  {frame.token}  ·  {len(frame.gt_labels)} ground truth  ·  "
        f"{kept} of {len(scores)} predictions at score ≥ {score_threshold:g}",
        fontsize=8,
    )
    figure.subplots_adjust(left=0.04, right=0.98, bottom=0.08, top=0.90, wspace=0.10)
    return figure


def save_frame(path, *args, **kwargs) -> None:
    """Render one frame straight to a file, closing the figure so a long loop does not leak."""
    import matplotlib.pyplot as plt

    figure = render_frame(*args, **kwargs)
    figure.savefig(path, dpi=110)
    plt.close(figure)
