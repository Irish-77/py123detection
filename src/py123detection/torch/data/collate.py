"""Batching samples whose rigs differ.

Mixing datasets means mixing camera counts: nuScenes has six, Argoverse seven, Waymo five. A
batch pads every sample up to the largest and carries ``cam_mask`` so a model can ignore what was
added. Ground truth stays a list, because the number of objects varies per frame and padding it
would invent boxes.

Padded projection matrices are the identity, never zeros. A zero 4x4 is singular, and a model that
inverts or chains it gets NaN out of a camera that does not exist.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import torch

#: Kept as lists: their length varies per sample, so stacking would need padding that means
#: something. A padded box is a false object.
PER_SAMPLE_KEYS = ("gt_bboxes_3d", "gt_labels_3d", "img_metas")

#: Stacked along a camera dimension that is padded up to the batch's widest rig.
PER_CAMERA_KEYS = ("img", "intrinsics", "extrinsics", "imu2img", "cam_valid_mask")

#: Entries of ``img_metas`` holding one value per camera, extended alongside the tensors.
PER_CAMERA_METADATA = ("ori_shape", "img_shape", "pad_shape")


def _pad_cameras(value: torch.Tensor, target: int, key: str) -> torch.Tensor:
    missing = target - value.shape[0]
    if missing <= 0:
        return value

    if key in ("intrinsics", "extrinsics", "imu2img"):
        padding = torch.eye(4, dtype=value.dtype).unsqueeze(0).repeat(missing, 1, 1)
    else:
        padding = torch.zeros(missing, *value.shape[1:], dtype=value.dtype)
    return torch.cat([value, padding], dim=0)


def _extend_metadata(metadata: Dict[str, Any], target: int) -> Dict[str, Any]:
    """Pad the per-camera entries of one sample's metadata, without touching the original."""
    extended = dict(metadata)
    for key in PER_CAMERA_METADATA:
        values = extended.get(key)
        if isinstance(values, list) and 0 < len(values) < target:
            extended[key] = list(values) + [values[-1]] * (target - len(values))
    return extended


def collate_rigs(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate samples from rigs of different sizes.

    Adds two keys the samples do not have:

    ``cam_mask`` ``(B, N)``
        True where a camera is both present in that sample's rig and had an image. This is the
        one a model should mask attention with.
    ``num_cams`` ``(B,)``
        How many cameras each sample's own rig has, before padding.
    """
    if not batch:
        return {}

    widest = max(sample["img"].shape[0] for sample in batch)
    collated: Dict[str, Any] = {}

    for key in batch[0]:
        values = [sample[key] for sample in batch]

        if key == "img_metas":
            collated[key] = [_extend_metadata(metadata, widest) for metadata in values]
        elif key in PER_SAMPLE_KEYS:
            collated[key] = values
        elif key in PER_CAMERA_KEYS:
            collated[key] = torch.stack([_pad_cameras(value, widest, key) for value in values])
        elif isinstance(values[0], torch.Tensor) and all(v.shape == values[0].shape for v in values):
            collated[key] = torch.stack(values)
        else:
            collated[key] = values

    counts: List[int] = [sample["img"].shape[0] for sample in batch]
    mask = torch.zeros(len(batch), widest, dtype=torch.bool)
    for position, (sample, count) in enumerate(zip(batch, counts)):
        mask[position, :count] = True
        valid = sample.get("cam_valid_mask")
        if valid is not None:
            mask[position, : len(valid)] &= valid

    collated["cam_mask"] = mask
    collated["num_cams"] = torch.tensor(counts, dtype=torch.long)
    return collated
