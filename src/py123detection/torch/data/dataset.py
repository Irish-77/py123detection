"""A PyTorch dataset reading 123D scenes straight into camera-based 3D detection batches.

One 123D frame becomes one sample: the rig's images, the matrices that project the ego frame into
each of them, and the boxes in the ego frame. No pickle, no mmdetection3d, no conversion step —
:class:`~py123detection.core.sources.Source` names the logs and the dataset reads them.

The work is split by what it is about, and this class only puts the parts in order:

* :class:`~py123detection.torch.data.frames.FrameIndex` — which frames, and how many;
* :class:`~py123detection.torch.data.boxes.BoxLoader` — a frame's boxes, filtered, in the ego frame;
* :class:`~py123detection.torch.data.augment.Augmentation` — the random draw for a training sample;
* :class:`~py123detection.torch.data.cameras.CameraLoader` — the rig's images in network geometry.

The sample is the layout the PETR family expects::

    img             (N_cam, 3, H, W)  uint8, zero-padded to the backbone's stride; the model normalises
    img_shape       (2,)              (H, W) before padding
    intrinsics      (N_cam, 4, 4)     including every crop, resize and flip applied to the image
    extrinsics      (N_cam, 4, 4)     ego -> camera
    imu2img         (N_cam, 4, 4)     ego -> pixels, i.e. intrinsics @ extrinsics
    ego_pose        (4, 4)            ego -> world, and ego_pose_inv its inverse
    cam_valid_mask  (N_cam,)          False where a camera had no image this frame
    gt_bboxes_3d    (M, 9)            [x, y, z, w, l, h, yaw, vx, vy] in the ego frame
    gt_labels_3d    (M,)              indices into the taxonomy's class_names

Coordinates are the ego (IMU) frame throughout, not the lidar frame: 123D defines the rig against
the ego body, and a camera-only detector never needs the lidar's. Yaw is measured from the ego
x axis towards y, and velocity is the box's ground-plane motion in that same frame.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from py123detection.core.geometry import invert_rigid
from py123detection.core.sources import Source
from py123detection.core.taxonomy import Taxonomy
from py123detection.torch.data.augment import Augmentation, Draw
from py123detection.torch.data.boxes import Boxes, BoxFilters, BoxLoader
from py123detection.torch.data.cameras import CameraLoader, Cameras
from py123detection.torch.data.frames import FrameIndex
from py123detection.torch.data.images import ImagePipeline
from py123detection.torch.data.rigs import CameraRig

logger = logging.getLogger(__name__)


class Py123DDetectionDataset(Dataset):
    """Camera-based 3D detection samples from 123D logs.

    :param source: Which logs and at what frame rate. See :class:`~py123detection.core.sources.Source`.
    :param taxonomy: Turns 123D labels into class names and ids, and merges riders onto their
        vehicles where a dataset boxes them separately.
    :param rig: Which cameras to read and which part of each image to keep.
    :param images: The network input size and how images are brought to it.
    :param augment: The random draw for training samples. ``None`` is the validation dataset.
    :param filters: Which boxes count. See :class:`~py123detection.torch.data.boxes.BoxFilters`.
    :param compute_velocity: Derive box velocity from track positions over time. Needed for
        datasets that store none, such as Argoverse 2.
    :param temporal: Add ``prev_exists`` to each sample, telling a temporal model whether the
        previous index is the same scene.
    :param name: Label carried in ``img_metas`` and used when several datasets are mixed.
    :param missing_camera: What to do when a frame has no image for a camera — ``"skip"`` rejects
        the frame, ``"zero"`` substitutes a blank image and clears its ``cam_valid_mask`` entry.
    :param num_frames: Serve this many frames, spread evenly over all logs. See
        :class:`~py123detection.torch.data.frames.FrameIndex`.
    :param seed: Seeds the augmentation generator. The data module reseeds it per worker and rank.
    """

    def __init__(
        self,
        source: Source,
        taxonomy: Taxonomy,
        rig: CameraRig,
        images: Optional[ImagePipeline] = None,
        augment: Optional[Augmentation] = None,
        filters: Optional[BoxFilters] = None,
        compute_velocity: bool = False,
        temporal: bool = False,
        name: Optional[str] = None,
        missing_camera: str = "skip",
        num_frames: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.source = source
        self.taxonomy = taxonomy
        self.rig = rig
        self.images = images if images is not None else ImagePipeline()
        self.augment = augment
        self.filters = filters if filters is not None else BoxFilters()
        self.temporal = temporal
        self.name = name or source.label

        self.scenes = source.load_scenes()
        self.frames = FrameIndex(self.scenes, num_frames=num_frames, name=self.name)
        self.boxes = BoxLoader(source, self.scenes, taxonomy, self.filters, rig.cameras, compute_velocity)
        self.cameras = CameraLoader(rig, self.images, missing_camera=missing_camera)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        logger.info(
            "%s: %d of %d frame(s) over %d log(s), %d camera(s), taxonomy '%s'",
            self.name, len(self.frames), self.frames.available, len(self.scenes), len(rig), taxonomy.name,
        )

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.get(index)

    def get(self, index: int, draw: Optional[Draw] = None) -> Dict[str, Any]:
        """One sample, or the next usable one.

        A frame can be unusable — a camera dropped out, or every box was filtered away. Rather
        than returning nothing and breaking the batch, the next frame is tried; scanning the whole
        dataset without success is an error worth raising.

        :param draw: Use this augmentation instead of drawing one, to reproduce a sample exactly.
        """
        if len(self.frames) == 0:
            raise IndexError("Dataset is empty.")

        for offset in range(len(self.frames)):
            sample = self.load((index + offset) % len(self.frames), draw)
            if sample is not None:
                return sample

        raise RuntimeError(
            f"{self.name}: no usable frame in {len(self.frames)}. Every one was rejected by "
            f"missing_camera={self.cameras.missing_camera!r} or by the box filters."
        )

    def reseed(self, *entropy: int) -> None:
        """Restart the augmentation generator, e.g. from a worker's seed and the trainer's rank."""
        self.rng = np.random.default_rng(np.random.SeedSequence([int(value) % 2**64 for value in entropy]))

    def draw(self) -> Draw:
        """A training sample's random draw, or the identity for validation."""
        return self.augment.draw(self.rng) if self.augment is not None else Draw()

    # -- the pipeline ---------------------------------------------------------------------------

    def load(self, index: int, draw: Optional[Draw] = None) -> Optional[Dict[str, Any]]:
        """Build the sample at exactly ``index``, or ``None`` when that frame cannot produce one."""
        scene_index, iteration = self.frames[index]
        scene = self.scenes[scene_index]
        ego_to_world = np.asarray(
            scene.get_ego_state_se3_at_iteration(iteration).imu_se3.transformation_matrix, dtype=np.float64
        )

        boxes = self.boxes.load(scene, iteration, invert_rigid(ego_to_world))
        if len(boxes) == 0 and self.filters.drop_frames_without_boxes:
            return None

        draw = draw if draw is not None else self.draw()
        cameras = self.cameras.load(scene, iteration, draw.image)
        if cameras is None:
            return None

        timestamp = float(scene.get_timestamp_at_iteration(iteration).time_s)
        sample = assemble(cameras, boxes, ego_to_world, draw)
        sample["timestamp"] = timestamp
        sample["img_metas"] = {
            "scene_token": scene.scene_uuid,
            "scene_uuid": scene.scene_uuid,
            "frame_idx": iteration,
            "sample_idx": str(index),
            "dataset_name": self.name,
            "box_mode_3d": "ego",
            "class_names": list(self.taxonomy.class_names),
            "gt_class_names": boxes.class_names,
            "ori_shape": list(cameras.source_sizes),
            "img_shape": [cameras.size] * len(self.rig),
            "pad_shape": [tuple(cameras.images.shape[-2:])] * len(self.rig),
            "timestamp": timestamp,
            **draw.metadata(),
        }
        if self.temporal:
            sample["prev_exists"] = self.frames.continues_scene(index)
        return sample


def assemble(cameras: Cameras, boxes: Boxes, ego_to_world: np.ndarray, draw: Draw) -> Dict[str, Any]:
    """The tensors of one sample, with the bird's-eye augmentation applied to boxes and matrices."""
    bev = draw.bev
    intrinsics = torch.from_numpy(cameras.intrinsics)
    extrinsics = torch.from_numpy(cameras.extrinsics)
    ego_to_image = torch.matmul(intrinsics, extrinsics)
    ego_pose = torch.from_numpy(ego_to_world.astype(np.float32))
    ego_pose_inv = torch.from_numpy(invert_rigid(ego_to_world).astype(np.float32))

    if not bev.is_identity:
        # The boxes moved, so every matrix reading *from* the ego frame has to undo that move
        # before it is applied, and the one reading *into* it has to redo it.
        inverse = torch.from_numpy(bev.inverse_matrix())
        ego_to_image = torch.matmul(ego_to_image, inverse)
        extrinsics = torch.matmul(extrinsics, inverse)
        ego_pose = torch.matmul(ego_pose, inverse)
        ego_pose_inv = torch.matmul(torch.from_numpy(bev.forward_matrix()), ego_pose_inv)

    return {
        "img": torch.from_numpy(cameras.images),
        "img_shape": torch.tensor(cameras.size, dtype=torch.int32),
        "intrinsics": intrinsics,
        "extrinsics": extrinsics,
        "imu2img": ego_to_image,
        "ego_pose": ego_pose,
        "ego_pose_inv": ego_pose_inv,
        "cam_valid_mask": torch.from_numpy(cameras.valid),
        "gt_bboxes_3d": torch.from_numpy(bev.apply_to_boxes(boxes.boxes).astype(np.float32)),
        "gt_labels_3d": torch.from_numpy(boxes.labels),
    }
