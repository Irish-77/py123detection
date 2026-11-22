"""The two branches present a nuScenes rig's cameras in different orders, and must keep doing so.

A model with per-camera parameters learns position by index, so this is not cosmetic. The export
branch has to match mmdetection3d's converter for its pickles to stay interchangeable with that
converter's; the torch branch has to match the thesis for its checkpoints and golden fixtures to
keep loading. Changing either breaks something that already exists, so both orders are named in
:mod:`py123detection.core.cameras` and these tests pin which branch uses which.
"""

from __future__ import annotations

import pytest

from py123detection.core.cameras import MMDET3D_NUSCENES_ORDER, THESIS_NUSCENES_ORDER
from py123detection.mmcv.export.converter import DEFAULT_CAMERA_ORDER


def test_the_two_orders_are_the_same_cameras_in_a_different_sequence() -> None:
    assert set(MMDET3D_NUSCENES_ORDER) == set(THESIS_NUSCENES_ORDER)
    assert len(MMDET3D_NUSCENES_ORDER) == len(THESIS_NUSCENES_ORDER) == 6
    assert MMDET3D_NUSCENES_ORDER != THESIS_NUSCENES_ORDER, (
        "if these ever agree, say so in core.cameras and drop one of them"
    )


def test_both_start_at_the_front_camera() -> None:
    """Index 0 is the front camera either way, which is what ``py123det-viz --camera 0`` assumes."""
    assert MMDET3D_NUSCENES_ORDER[0] == THESIS_NUSCENES_ORDER[0] == "PCAM_F0"


def test_the_export_writes_mmdetection3ds_order() -> None:
    assert DEFAULT_CAMERA_ORDER == MMDET3D_NUSCENES_ORDER


@pytest.mark.torch
def test_the_torch_rig_uses_the_thesis_order() -> None:
    from py123detection.torch.data.rigs import RIGS

    assert tuple(RIGS["nuscenes"].cameras) == THESIS_NUSCENES_ORDER
