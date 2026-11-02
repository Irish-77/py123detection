"""PETR: 3D detection from multi-view images, with position-aware queries.

The idea the paper is named for lives in the head. Image features get a positional embedding that
encodes *where in 3D space each pixel could be looking*, derived from the camera's own
calibration, and learned queries then attend across all cameras with ordinary attention. No view
transformation, no bird's-eye grid, no depth prediction.
"""

from __future__ import annotations

from typing import Any, List, Optional

from py123detection.torch.models.multiview import MultiViewDetector
from py123detection.torch.models.petr.head import PETRHead


class PETR(MultiViewDetector):
    """PETR as published, trained with Lightning.

    :param position_range: The volume the 3D positional embedding spans. Deliberately wider than
        ``pc_range``, so an object at the edge of the detection range still gets a meaningful
        embedding rather than a saturated one.
    """

    def __init__(self, position_range: Optional[List[float]] = None, **kwargs: Any) -> None:
        self.position_range = position_range or [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0]
        super().__init__(**kwargs)
        self.save_hyperparameters()

    def build_head(self, **kwargs: Any) -> PETRHead:
        return PETRHead(position_range=self.position_range, **kwargs)
