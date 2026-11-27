"""The nuScenes camera order, and why this package has two of them.

A model with per-camera parameters learns position by index, so the order a rig presents its
cameras in is part of the trained weights. The two branches of this package inherited different
orders for the same six nuScenes cameras, and both are load-bearing:

* :data:`MMDET3D_NUSCENES_ORDER` is what mmdetection3d's own nuScenes converter writes, and what
  pretrained PETR, StreamPETR and BEVDet checkpoints expect. The export branch has to match it,
  or an exported pickle is no longer interchangeable with one from that converter, which is the
  claim ``docs/validation.md`` measures.
* :data:`THESIS_NUSCENES_ORDER` is what the thesis trained with, so it is the order the torch
  branch's checkpoints and golden fixtures were produced under.

Changing either one invalidates something that already exists, so neither is changed. Naming both
here is the next best thing: the difference becomes a documented fact in one file instead of two
tables that happen to disagree, three directories apart. What follows from it is that a checkpoint
from one branch has its camera table permuted relative to the other, and cannot be loaded into the
other branch's model without reordering.
"""

from typing import Tuple

#: ``CAM_FRONT``, ``CAM_FRONT_RIGHT``, ``CAM_FRONT_LEFT``, ``CAM_BACK``, ``CAM_BACK_LEFT``,
#: ``CAM_BACK_RIGHT`` as 123D camera ids. The order mmdetection3d's nuScenes converter uses.
MMDET3D_NUSCENES_ORDER: Tuple[str, ...] = ("PCAM_F0", "PCAM_R0", "PCAM_L0", "PCAM_B0", "PCAM_L1", "PCAM_R1")

#: ``CAM_FRONT``, ``CAM_FRONT_LEFT``, ``CAM_FRONT_RIGHT``, ``CAM_BACK_LEFT``, ``CAM_BACK_RIGHT``,
#: ``CAM_BACK``. Front to back, left before right, which is how the thesis's rigs were written.
THESIS_NUSCENES_ORDER: Tuple[str, ...] = ("PCAM_F0", "PCAM_L0", "PCAM_R0", "PCAM_L1", "PCAM_R1", "PCAM_B0")
