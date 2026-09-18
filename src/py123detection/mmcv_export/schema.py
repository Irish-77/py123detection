"""The exported ``info`` schema and a checker for it.

The pickle is ``{"infos": [info, ...], "metadata": {...}}``, which is what mmdet3d's
``NuScenesDataset.load_annotations`` reads: it sorts ``infos`` by ``timestamp`` and reads
``metadata['version']``. Geometry is in the lidar frame of each frame's keyframe, as in
mmdetection3d's nuScenes converter.

Per frame:

``lidar_path``
    A reference into the original dataset or a file written by the export (see ``sensors.py``).
``token``
    The 123D frame UUID, or a native token when a resolver is configured (see ``tokens.py``).
``prev`` / ``next``
    Neighbouring tokens within the log; empty at log boundaries.
``scene_token``
    ``"{split}/{log_name}"``. StreamPETR uses it to detect sequence changes.
``frame_idx``
    Index within the log, from 0.
``timestamp``
    Microseconds, plus ``Source.timestamp_offset_us``.
``lidar2ego_translation`` / ``lidar2ego_rotation``
    Reference lidar extrinsic, quaternions scalar-first ``[w, x, y, z]``. Under
    ``lidar2ego_mode="dynamic"`` it also absorbs the ego/lidar timestamp offset.
``ego2global_translation`` / ``ego2global_rotation``
    Ego pose at the frame's timestamp.
``sweeps``
    Preceding lidar frames of the log, newest first, each with ``sensor2lidar_*`` into this
    frame. Empty for the first frame of a log, which StreamPETR's ``_set_sequence_group_flag``
    relies on. Sweeps come at the rate of the 123D log, so a 2 Hz keyframe log has 2 Hz sweeps.
``cams``
    Camera payloads in model input order.
``gt_boxes`` float64 ``(N, 7)``
    ``[x, y, z, l, w, h, yaw]`` around the box center (or the ``mmdet3d_0.17`` layout).
``gt_names``, ``gt_velocity``
    Shapes ``(N,)`` and float64 ``(N, 2)``.
``num_lidar_pts`` int64 ``(N,)``
    ``-1`` when the log does not record it.
``num_radar_pts`` int64 ``(N,)``
    Always zero, 123D has no radar counts.
``valid_flag`` bool ``(N,)``
    ``num_lidar_pts != 0``. mmdet3d also counts radar points, so boxes seen only by radar differ.
``instance_tokens``
    Per-box track ids. Ignored by mmdet3d.
``bboxes2d``, ``labels2d``, ``centers2d``, ``depths``, ``bboxes3d_cams``, ``bboxes_ignore``, ``visibilities``
    Shapes ``(M, 4)``, ``(M,)``, ``(M, 2)``, ``(M,)``, ``(M, 7)``, ``(K, 4)``, one entry per camera in ``cams`` order, when 2D annotations are enabled. StreamPETR reads
    them during training. ``bboxes_ignore`` is always empty and ``visibilities`` holds empty
    strings.
``py123d_uuid``, ``py123d_dataset``, ``py123d_split``, ``py123d_log_name``, ``py123d_location``, ``py123d_iteration``
    Provenance, ignored by mmdet3d.

Per camera (``info['cams'][key]``):

``data_path``, ``type``, ``timestamp``
    Image path, camera key, and the camera's own capture time. Cameras fire slightly off the
    lidar keyframe.
``sensor2ego_*``
    Rig extrinsic, or a per-frame transform absorbing the timing offset under
    ``camera2ego_mode="dynamic"``.
``ego2global_*``
    Ego pose at the camera's timestamp, or the frame's ego pose under ``camera2ego_mode="dynamic"``.
    Either way ``ego2global @ sensor2ego`` is the camera-to-global pose.
``sensor2lidar_rotation`` ``(3, 3)`` / ``sensor2lidar_translation`` ``(3,)``
    What PETR and StreamPETR turn into ``lidar2img``.
``cam_intrinsic``, ``width``, ``height``
    Intrinsic matrix ``(3, 3)`` and the image size in pixels.
``sample_data_token``
    ``"{split}/{log}/{camera}/{timestamp_us}"``. Not read during training.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

# Read by mmdetection3d's NuScenesDataset and its derivatives.
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

# Additionally required by StreamPETR's CustomNuScenesDataset.get_data_info.
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
    """Problems found in one info; empty when it is well-formed.

    :param streampetr: Also require the StreamPETR training fields.
    """
    problems: List[str] = []

    for key in REQUIRED_INFO_KEYS + (STREAMPETR_INFO_KEYS if streampetr else ()):
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
