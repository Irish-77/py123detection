"""BEVFormer: 3D detection through an explicit bird's-eye-view feature map.

Where PETR leaves the features in the images and teaches the queries where to look, BEVFormer
builds a top-down grid first. Each cell of the grid attends back into whichever cameras can see
the ground it covers, and the detection decoder then works entirely in that grid.

Both produce the same predictions in the same format, so the data, the loss, the metric and the
training loop are shared and only the head differs. That is why this file is short.
"""

from __future__ import annotations

from typing import Any, List

from py123detection.torch.models.bevformer.head import BEVFormerHead
from py123detection.torch.models.multiview import MultiViewDetector


class BEVFormer(MultiViewDetector):
    """BEVFormer's head on the shared backbone and training loop.

    :param bev_h: Rows of the bird's-eye grid. The grid spans ``pc_range``, so 50 rows over
        102.4 m is about two metres a cell.
    :param bev_w: Columns of the grid.
    :param num_encoder_layers: Rounds of BEV self-attention and camera cross-attention.
    :param camera_embedding: ``"bevformer"`` adds the paper's learned per-camera token, which ties
        the model to one rig. ``"none"`` is what allows training across datasets whose camera
        counts differ, which is the reason this package exists.
    """

    def __init__(
        self,
        bev_h: int = 50,
        bev_w: int = 50,
        num_encoder_layers: int = 3,
        num_heights: int = 4,
        num_points: int = 4,
        feedforward_channels_encoder: int = 512,
        feedforward_channels_decoder: int = 2048,
        with_box_refine: bool = True,
        camera_embedding: str = "none",
        num_cameras: int = 6,
        **kwargs: Any,
    ) -> None:
        self._head_options = dict(
            bev_h=bev_h,
            bev_w=bev_w,
            num_encoder_layers=num_encoder_layers,
            num_heights=num_heights,
            num_points=num_points,
            feedforward_channels_encoder=feedforward_channels_encoder,
            feedforward_channels_decoder=feedforward_channels_decoder,
            with_box_refine=with_box_refine,
            cam_embed_type=camera_embedding,
            num_cameras=num_cameras,
        )
        super().__init__(**kwargs)
        self.save_hyperparameters()

    def build_head(
        self,
        num_classes: int,
        in_channels: int,
        num_query: int,
        num_decoder_layers: int,
        pc_range: List[float],
        with_cp: bool,
        **unused: Any,
    ) -> BEVFormerHead:
        """BEVFormer's head.

        The patch-geometry arguments describe PETR's positional embedding, which a bird's-eye grid
        does not have. They are accepted and dropped so both detectors share one constructor.
        """
        return BEVFormerHead(
            num_classes=num_classes,
            in_channels=in_channels,
            num_query=num_query,
            num_decoder_layers=num_decoder_layers,
            pc_range=pc_range,
            with_cp=with_cp,
            **self._head_options,
        )
