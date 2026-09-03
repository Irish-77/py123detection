"""The exported ``info`` schema, and what each field means for 123D-sourced data.

The payload written by :func:`~py123detection.mmcv_export.export.export_to_mmdet3d` is::

    {"infos": [info, ...], "metadata": {...}}

which is exactly what ``mmdet3d.datasets.NuScenesDataset.load_annotations`` expects: it reads
``data['infos']``, sorts them by ``timestamp``, and reads ``data['metadata']['version']``.

Reference frame
---------------
Everything geometric is expressed in the **lidar frame of the frame's own keyframe** — the same
convention as mmdetection3d's nuScenes converter. 123D stores poses and boxes in the global
frame, so the export composes ``lidar -> ego -> global`` and inverts it.

Fields
------
Top level, per frame:

``lidar_path`` (str)
    Path to the point cloud. Either a reference into the original dataset or a file written by
    the export — see :mod:`py123detection.mmcv_export.sensors`.
``token`` (str)
    Frame identifier. The deterministic 123D UUID by default; the dataset's native token when a
    resolver is configured (needed for official nuScenes evaluation — see
    :mod:`py123detection.mmcv_export.tokens`).
``prev`` / ``next`` (str)
    Tokens of the neighbouring frames within the same log; empty at the log boundaries.
``scene_token`` (str)
    ``"{split}/{log_name}"`` — stable per log, used by StreamPETR to detect sequence changes.
``frame_idx`` (int)
    Index of the frame within its log, starting at 0.
``timestamp`` (int)
    Microseconds. Plus ``Source.timestamp_offset_us`` when sources were spaced apart.
``lidar2ego_translation`` / ``lidar2ego_rotation`` (list[float])
    Extrinsic of the reference lidar. Rotation is a scalar-first quaternion ``[w, x, y, z]``.
    The rig calibration by default; under ``lidar2ego_mode="dynamic"`` it also absorbs the offset
    between the frame's ego timestamp and the lidar's own.
``ego2global_translation`` / ``ego2global_rotation`` (list[float])
    Ego pose at the frame's reference timestamp. Never altered by the extrinsic modes.
``sweeps`` (list[dict])
    Preceding lidar frames of the same log, newest first, each carrying its own
    ``sensor2lidar_rotation`` / ``sensor2lidar_translation`` into the current frame. **Empty for
    the first frame of every log**, which is what StreamPETR's ``_set_sequence_group_flag`` keys
    on. Unlike nuScenes' native 20 Hz sweeps these are the log's exported frames, so a 2 Hz
    keyframe-only 123D log offers 2 Hz sweeps.
``cams`` (dict[str, dict])
    Per-camera payloads in model input order (see below).

Annotations, per frame:

``gt_boxes`` (float64 ``(N, 7)``)
    ``[x, y, z, dx, dy, dz, yaw]`` in the lidar frame, with ``(dx, dy, dz)`` the box's
    ``(length, width, height)`` and ``(x, y, z)`` its **center** — mmdet3d re-anchors it to the
    bottom face at load time via ``origin=(0.5, 0.5, 0.5)``.
``gt_names`` (``(N,)`` str)
    Taxonomy class names; must match the ``class_names`` of the mmdet3d config.
``gt_velocity`` (float64 ``(N, 2)``)
    Lidar-frame ground-plane velocity.
``num_lidar_pts`` (int64 ``(N,)``)
    Lidar points inside each box, or ``-1`` when the source log does not record it.
``num_radar_pts`` (int64 ``(N,)``)
    Always zero: 123D does not carry radar point counts.
``valid_flag`` (bool ``(N,)``)
    ``num_lidar_pts > 0``, with unknown counts (``-1``) treated as valid. mmdet3d computes this
    as ``num_lidar_pts + num_radar_pts > 0``, so exports differ for the rare box seen by radar
    only.
``instance_tokens`` (list[str])
    Per-box track identifiers, stable across frames. An export extra; mmdet3d ignores it.

Per-camera 2D annotations (present when ``with_2d_annotations`` is set), each a list with one
entry per camera, in ``cams`` order — StreamPETR's ``CustomNuScenesDataset`` reads all of them
during training:

``bboxes2d`` ``(M, 4)``, ``labels2d`` ``(M,)``, ``centers2d`` ``(M, 2)``, ``depths`` ``(M,)``,
``bboxes3d_cams`` ``(M, 7)``, ``bboxes_ignore`` ``(K, 4)``, ``visibilities`` (list[str])
    ``bboxes_ignore`` is always empty (123D has no crowd flag) and ``visibilities`` always holds
    empty strings (123D does not carry nuScenes visibility bins).

Camera payload (``info['cams'][key]``):

``data_path`` (str), ``type`` (str), ``timestamp`` (int)
    Image path, the camera key, and the camera's own capture time in microseconds — cameras fire
    slightly off the lidar keyframe, and the pose fields below reflect that.
``sensor2ego_translation`` / ``sensor2ego_rotation``
    Rig extrinsic of the camera. Under ``camera2ego_mode="dynamic"`` it instead absorbs the
    camera's timing offset, becoming a per-frame transform.
``ego2global_translation`` / ``ego2global_rotation``
    Ego pose at the *camera's* timestamp, recovered from 123D's motion-compensated
    camera-to-global pose — so it differs between cameras of the same frame. Under
    ``camera2ego_mode="dynamic"`` it is the frame's shared ego pose instead. Either way
    ``ego2global @ sensor2ego`` rebuilds the same camera-to-global transform.
``sensor2lidar_rotation`` ``(3, 3)`` / ``sensor2lidar_translation`` ``(3,)``
    Camera-to-lidar transform, composed through the global frame. This is the pair PETR and
    StreamPETR turn into ``lidar2img``.
``cam_intrinsic`` ``(3, 3)``
    Pinhole intrinsic matrix.
``sample_data_token`` (str)
    Synthesized as ``"{split}/{log}/{camera}/{timestamp_us}"``. 123D does not carry native
    sample-data tokens; nothing in the training path reads this field.
``width`` / ``height`` (int)
    Native image size, an export extra that saves a decode when computing image shapes.

123D provenance (export extras, ignored by mmdet3d): ``py123d_uuid``, ``py123d_dataset``,
``py123d_split``, ``py123d_log_name``, ``py123d_location``, ``py123d_iteration``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

# Fields mmdetection3d's NuScenesDataset and its derivatives read out of an info.
REQUIRED_INFO_KEYS: Tuple[str, ...] = (
    "lidar_path",
    "token",
    "sweeps",
    "cams",
    "lidar2ego_translation",
    "lidar2ego_rotation",
    "ego2global_translation",
    "ego2global_rotation",
    "timestamp",
    "gt_boxes",
    "gt_names",
    "gt_velocity",
    "num_lidar_pts",
    "num_radar_pts",
    "valid_flag",
)

# Additional fields StreamPETR's CustomNuScenesDataset.get_data_info requires during training.
STREAMPETR_INFO_KEYS: Tuple[str, ...] = (
    "prev",
    "next",
    "scene_token",
    "frame_idx",
    "bboxes2d",
    "labels2d",
    "centers2d",
    "depths",
    "bboxes_ignore",
)

REQUIRED_CAM_KEYS: Tuple[str, ...] = (
    "data_path",
    "type",
    "sensor2ego_translation",
    "sensor2ego_rotation",
    "ego2global_translation",
    "ego2global_rotation",
    "timestamp",
    "sensor2lidar_rotation",
    "sensor2lidar_translation",
    "cam_intrinsic",
)


def validate_info(info: Dict[str, Any], streampetr: bool = True) -> List[str]:
    """Check one exported info against the schema.

    :param info: The info to check.
    :param streampetr: Also require the StreamPETR-only training fields.
    :return: A list of problems; empty when the info is well-formed.
    """
    problems: List[str] = []

    expected_keys = REQUIRED_INFO_KEYS + (STREAMPETR_INFO_KEYS if streampetr else ())
    for key in expected_keys:
        if key not in info:
            problems.append(f"missing key '{key}'")

    cams = info.get("cams")
    if not isinstance(cams, dict) or not cams:
        problems.append("'cams' must be a non-empty dict")
    else:
        for camera_key, payload in cams.items():
            for key in REQUIRED_CAM_KEYS:
                if key not in payload:
                    problems.append(f"camera '{camera_key}' missing key '{key}'")

    boxes = info.get("gt_boxes")
    names = info.get("gt_names")
    velocity = info.get("gt_velocity")
    if boxes is not None and getattr(boxes, "ndim", None) == 2:
        count = boxes.shape[0]
        if boxes.shape[1] != 7:
            problems.append(f"'gt_boxes' must have 7 columns, got {boxes.shape[1]}")
        if names is not None and len(names) != count:
            problems.append(f"'gt_names' has {len(names)} entries for {count} boxes")
        if velocity is not None and getattr(velocity, "shape", (0, 0))[0] != count:
            problems.append("'gt_velocity' length does not match 'gt_boxes'")

    if streampetr and isinstance(cams, dict):
        for key in ("bboxes2d", "labels2d", "centers2d", "depths", "bboxes_ignore"):
            value = info.get(key)
            if isinstance(value, list) and len(value) != len(cams):
                problems.append(f"'{key}' has {len(value)} entries for {len(cams)} cameras")

    return problems
