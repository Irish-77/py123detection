"""Which cameras a dataset has, and which part of each image is worth keeping.

Rigs differ in ways a detector cannot absorb on its own. Waymo's cameras are portrait and point
forward; Argoverse 2's front camera is 2048 px tall with most of the sky in it; nuScenes' six
cameras already ring the vehicle. Cutting the unusable rows away and bringing every camera to a
common shape is what makes one model trainable on several of them at once.

The legacy code had one module per dataset for this, five of which were the same forty lines with
a different table of row ranges and a different output size. Here there is one class and a table
per dataset, so adding a dataset is one entry rather than one file.

A :class:`CameraPreprocessing` does not touch pixels. It is the first factor of the affine map
from sensor pixels to network input (see :mod:`~py123detection.torch.data.images`), so the crop,
the resize to the rig's shape and the augmentation are resampled in one pass, and the intrinsics
are left-multiplied by the same matrix rather than re-derived.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class CameraPreprocessing:
    """Cut fixed rows out of each camera, then scale every camera to one shape.

    :param row_ranges: ``{camera: (first_row, last_row)}`` in original pixels. A camera with no
        entry is only scaled. Ranges are clamped to the image, so a dataset whose resolution
        changed does not crash, it just crops less.
    :param target_size: ``(width, height)`` every camera is brought to. The augmentation's zoom is
        relative to the covering scale of this shape, so it also sets each dataset's pixel scale.
    """

    row_ranges: Mapping[str, Tuple[int, int]]
    target_size: Tuple[int, int]

    def matrix(self, camera: str, height: int, width: int) -> np.ndarray:
        """The 3x3 map from a ``height`` by ``width`` image's pixels to the rig's shape."""
        row_start, row_end = self.row_ranges.get(camera, (0, height))
        row_start = max(0, min(int(row_start), height))
        row_end = max(row_start, min(int(row_end), height))

        scale_x = self.target_size[0] / width
        scale_y = self.target_size[1] / max(1, row_end - row_start)
        return np.array(
            [[scale_x, 0.0, 0.0], [0.0, scale_y, -scale_y * row_start], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class CameraRig:
    """The cameras to read, in the order the model expects them.

    Order matters and is not alphabetical: a model with per-camera parameters learns position by
    index, so the same dataset has to present its cameras the same way every time.

    :param preprocessing: The fixed crop and shape in front of the per-sample transform. ``None``
        leaves each camera at its own size, which is right when the rig needs no cropping; cameras
        of different sizes are then each transformed against their own.
    """

    cameras: Tuple[str, ...]
    preprocessing: Optional[CameraPreprocessing] = None

    def __len__(self) -> int:
        return len(self.cameras)

    def camera_matrix(self, camera: str, height: int, width: int) -> Tuple[np.ndarray, Tuple[int, int]]:
        """The rig's part of the affine map for one camera, and the ``(height, width)`` it yields."""
        if self.preprocessing is None:
            return np.eye(3, dtype=np.float64), (height, width)
        target_width, target_height = self.preprocessing.target_size
        return self.preprocessing.matrix(camera, height, width), (target_height, target_width)


#: Cropping presets, keyed the way the 123D splits are named.
#:
#: The row ranges come from the master thesis, where they were tuned per dataset by looking at
#: what the cameras actually see: sky and bonnet on Argoverse, the bottom band of the Waymo
#: portrait frames, a different dead strip per nuPlan camera.
PREPROCESSING: Dict[str, CameraPreprocessing] = {
    "nuscenes": CameraPreprocessing(row_ranges={}, target_size=(1640, 920)),
    "av2": CameraPreprocessing(
        row_ranges={
            "PCAM_F0": (750, 1750),
            "PCAM_L0": (250, 1500),
            "PCAM_R0": (250, 1500),
            "PCAM_L1": (250, 1500),
            "PCAM_R1": (250, 1500),
            "PCAM_L2": (60, 1310),
            "PCAM_R2": (60, 1310),
        },
        target_size=(1600, 960),
    ),
    "wod": CameraPreprocessing(
        row_ranges={
            "PCAM_F0": (0, 1100),
            "PCAM_L0": (0, 1100),
            "PCAM_R0": (0, 1100),
            "PCAM_L1": (0, 886),
            "PCAM_R1": (0, 886),
        },
        target_size=(1720, 880),
    ),
    "pandaset": CameraPreprocessing(
        row_ranges={
            "PCAM_F0": (0, 1080),
            "PCAM_B0": (0, 825),
            "PCAM_L0": (0, 1080),
            "PCAM_R0": (0, 1080),
            "PCAM_L1": (0, 1080),
            "PCAM_R1": (0, 1080),
        },
        target_size=(1800, 880),
    ),
    "nuplan": CameraPreprocessing(
        row_ranges={
            "PCAM_F0": (0, 1015),
            "PCAM_R0": (0, 983),
            "PCAM_R1": (0, 1080),
            "PCAM_R2": (0, 1010),
            "PCAM_L0": (0, 1010),
            "PCAM_L1": (0, 1080),
            "PCAM_L2": (0, 1025),
            "PCAM_B0": (0, 1080),
        },
        target_size=(1600, 900),
    ),
    "carla": CameraPreprocessing(
        row_ranges={
            "PCAM_F0": (0, 450),
            "PCAM_L0": (0, 450),
            "PCAM_R0": (0, 450),
            "PCAM_L1": (0, 450),
            "PCAM_R1": (0, 450),
            "PCAM_B0": (0, 420),
        },
        target_size=(1640, 920),
    ),
}

#: Ready-made rigs, one per dataset the thesis trained on.
RIGS: Dict[str, CameraRig] = {
    "nuscenes": CameraRig(
        cameras=("PCAM_F0", "PCAM_L0", "PCAM_R0", "PCAM_L1", "PCAM_R1", "PCAM_B0"),
        preprocessing=PREPROCESSING["nuscenes"],
    ),
    "av2": CameraRig(
        cameras=("PCAM_F0", "PCAM_L0", "PCAM_R0", "PCAM_L1", "PCAM_R1", "PCAM_L2", "PCAM_R2"),
        preprocessing=PREPROCESSING["av2"],
    ),
    "wod": CameraRig(
        cameras=("PCAM_F0", "PCAM_L0", "PCAM_R0", "PCAM_L1", "PCAM_R1"),
        preprocessing=PREPROCESSING["wod"],
    ),
    "pandaset": CameraRig(
        cameras=("PCAM_F0", "PCAM_L0", "PCAM_R0", "PCAM_L1", "PCAM_R1", "PCAM_B0"),
        preprocessing=PREPROCESSING["pandaset"],
    ),
    "nuplan": CameraRig(
        cameras=("PCAM_F0", "PCAM_L0", "PCAM_R0", "PCAM_L1", "PCAM_R1", "PCAM_L2", "PCAM_R2", "PCAM_B0"),
        preprocessing=PREPROCESSING["nuplan"],
    ),
    "carla": CameraRig(
        cameras=("PCAM_F0", "PCAM_L0", "PCAM_R0", "PCAM_L1", "PCAM_R1", "PCAM_B0"),
        preprocessing=PREPROCESSING["carla"],
    ),
}


def get_rig(name: str) -> CameraRig:
    if name not in RIGS:
        raise KeyError(f"Unknown rig '{name}'. Available: {sorted(RIGS)}")
    return RIGS[name]
