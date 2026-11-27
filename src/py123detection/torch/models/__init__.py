"""Camera-based 3D detectors, sharing one training loop.

:class:`~py123detection.torch.models.base.Detector3D` owns the Lightning machinery;
:class:`~py123detection.torch.models.petr.detector.PETR` and
:class:`~py123detection.torch.models.bevformer.detector.BEVFormer` supply a head each.
"""

from py123detection.torch.models.base import Detector3D
from py123detection.torch.models.multiview import MultiViewDetector
from py123detection.torch.models.bevformer.detector import BEVFormer
from py123detection.torch.models.petr.detector import PETR

__all__ = ["BEVFormer", "Detector3D", "MultiViewDetector", "PETR"]
