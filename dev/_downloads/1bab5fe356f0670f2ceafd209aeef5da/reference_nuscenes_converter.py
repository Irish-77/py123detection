"""Reference nuScenes -> mmdetection3d converter, used as ground truth in the comparison test.

This is a faithful port of ``tools/data_converter/nuscenes_converter.py`` from StreamPETR (itself
derived from mmdetection3d), with the mmcv / mmdet3d imports removed so it runs in a plain
environment. The geometry, the field layout and the filtering rules are unchanged:

* ``obtain_sensor2top`` — the sweep -> ego -> global -> ego' -> lidar chain, verbatim;
* ``get_2d_boxes`` — corner projection and convex-hull clipping, verbatim;
* ``_fill_trainval_infos`` — the info assembly, verbatim.

Two substitutions were needed, neither of which changes a number:

* ``mmdet3d.core.bbox.points_cam2img`` is reimplemented below (six lines of projection);
* ``mmcv.imread(path).shape`` for image size is replaced by the ``width`` / ``height`` fields of
  the same ``sample_data`` record, which is where nuScenes stores them.

``mmdet3d.datasets.NuScenesDataset.NameMapping`` and the category / attribute tuples are inlined
verbatim from mmdetection3d.
"""

from __future__ import annotations

import argparse
import pickle
from collections import OrderedDict
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
from shapely.geometry import MultiPoint, box

nus_categories = (
    "car",
    "truck",
    "trailer",
    "bus",
    "construction_vehicle",
    "bicycle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "barrier",
)

nus_attributes = (
    "cycle.with_rider",
    "cycle.without_rider",
    "pedestrian.moving",
    "pedestrian.standing",
    "pedestrian.sitting_lying_down",
    "vehicle.moving",
    "vehicle.parked",
    "vehicle.stopped",
    "None",
)

# Verbatim from mmdet3d.datasets.NuScenesDataset.NameMapping.
NAME_MAPPING = {
    "movable_object.barrier": "barrier",
    "vehicle.bicycle": "bicycle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.car": "car",
    "vehicle.construction": "construction_vehicle",
    "vehicle.motorcycle": "motorcycle",
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
    "movable_object.trafficcone": "traffic_cone",
    "vehicle.trailer": "trailer",
    "vehicle.truck": "truck",
}

CAMERA_TYPES = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]


def points_cam2img(points_3d: np.ndarray, proj_mat: np.ndarray, with_depth: bool = False) -> np.ndarray:
    """Project camera-frame points to image coordinates (port of ``mmdet3d.core.bbox.points_cam2img``)."""
    points_num = points_3d.shape[0]
    points_4 = np.concatenate([points_3d, np.ones((points_num, 1))], axis=-1)
    proj_mat_4 = np.eye(4)
    proj_mat_4[: proj_mat.shape[0], : proj_mat.shape[1]] = proj_mat
    point_2d = points_4 @ proj_mat_4.T
    point_2d_res = point_2d[:, :2] / point_2d[:, 2:3]
    if with_depth:
        return np.concatenate([point_2d_res, point_2d[:, 2:3]], axis=-1)
    return point_2d_res


def obtain_sensor2top(nusc, sensor_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, sensor_type="lidar"):
    """Transform from a sensor to the top lidar frame (verbatim from the mmdet3d converter)."""
    sd_rec = nusc.get("sample_data", sensor_token)
    cs_record = nusc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
    pose_record = nusc.get("ego_pose", sd_rec["ego_pose_token"])
    data_path = str(nusc.get_sample_data_path(sd_rec["token"]))
    sweep = {
        "data_path": data_path,
        "type": sensor_type,
        "sample_data_token": sd_rec["token"],
        "sensor2ego_translation": cs_record["translation"],
        "sensor2ego_rotation": cs_record["rotation"],
        "ego2global_translation": pose_record["translation"],
        "ego2global_rotation": pose_record["rotation"],
        "timestamp": sd_rec["timestamp"],
    }
    l2e_r_s = sweep["sensor2ego_rotation"]
    l2e_t_s = sweep["sensor2ego_translation"]
    e2g_r_s = sweep["ego2global_rotation"]
    e2g_t_s = sweep["ego2global_translation"]

    # sweep -> ego -> global -> ego' -> lidar
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T) + l2e_t @ np.linalg.inv(l2e_r_mat).T
    sweep["sensor2lidar_rotation"] = R.T  # points @ R.T + T
    sweep["sensor2lidar_translation"] = T
    return sweep


def post_process_coords(
    corner_coords: List, imsize: Tuple[int, int] = (1600, 900)
) -> Union[Tuple[float, float, float, float], None]:
    """Intersect the convex hull of reprojected corners with the image canvas (verbatim)."""
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = box(0, 0, imsize[0], imsize[1])

    if polygon_from_2d_box.intersects(img_canvas):
        img_intersection = polygon_from_2d_box.intersection(img_canvas)
        intersection_coords = np.array([coord for coord in img_intersection.exterior.coords])
        return (
            min(intersection_coords[:, 0]),
            min(intersection_coords[:, 1]),
            max(intersection_coords[:, 0]),
            max(intersection_coords[:, 1]),
        )
    return None


def generate_record(ann_rec: dict, x1: float, y1: float, x2: float, y2: float, sample_data_token: str, filename: str):
    """Build one 2D annotation record (verbatim)."""
    repro_rec = OrderedDict()
    repro_rec["sample_data_token"] = sample_data_token
    coco_rec = dict()

    relevant_keys = [
        "attribute_tokens",
        "category_name",
        "instance_token",
        "next",
        "num_lidar_pts",
        "num_radar_pts",
        "prev",
        "sample_annotation_token",
        "sample_data_token",
        "visibility_token",
    ]
    for key, value in ann_rec.items():
        if key in relevant_keys:
            repro_rec[key] = value

    repro_rec["bbox_corners"] = [x1, y1, x2, y2]
    repro_rec["filename"] = filename

    coco_rec["file_name"] = filename
    coco_rec["image_id"] = sample_data_token
    coco_rec["area"] = (y2 - y1) * (x2 - x1)

    if repro_rec["category_name"] not in NAME_MAPPING:
        return None
    cat_name = NAME_MAPPING[repro_rec["category_name"]]
    coco_rec["category_name"] = cat_name
    coco_rec["category_id"] = nus_categories.index(cat_name)
    coco_rec["bbox"] = [x1, y1, x2 - x1, y2 - y1]
    coco_rec["iscrowd"] = 0
    coco_rec["visibility_token"] = repro_rec["visibility_token"]
    return coco_rec


def get_2d_boxes(nusc, sample_data_token: str, visibilities: List[str], mono3d: bool = True):
    """Get 2D annotation records for one camera keyframe (verbatim)."""
    sd_rec = nusc.get("sample_data", sample_data_token)
    assert sd_rec["sensor_modality"] == "camera", "get_2d_boxes only works for camera sample_data!"
    if not sd_rec["is_key_frame"]:
        raise ValueError("The 2D re-projections are available only for keyframes.")

    s_rec = nusc.get("sample", sd_rec["sample_token"])
    cs_rec = nusc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
    pose_rec = nusc.get("ego_pose", sd_rec["ego_pose_token"])
    camera_intrinsic = np.array(cs_rec["camera_intrinsic"])

    ann_recs = [nusc.get("sample_annotation", token) for token in s_rec["anns"]]
    ann_recs = [ann_rec for ann_rec in ann_recs if (ann_rec["visibility_token"] in visibilities)]

    repro_recs = []
    for ann_rec in ann_recs:
        ann_rec["sample_annotation_token"] = ann_rec["token"]
        ann_rec["sample_data_token"] = sample_data_token

        box3d = nusc.get_box(ann_rec["token"])
        box3d.translate(-np.array(pose_rec["translation"]))
        box3d.rotate(Quaternion(pose_rec["rotation"]).inverse)
        box3d.translate(-np.array(cs_rec["translation"]))
        box3d.rotate(Quaternion(cs_rec["rotation"]).inverse)

        corners_3d = box3d.corners()
        in_front = np.argwhere(corners_3d[2, :] > 0).flatten()
        corners_3d = corners_3d[:, in_front]

        corner_coords = view_points(corners_3d, camera_intrinsic, True).T[:, :2].tolist()
        final_coords = post_process_coords(corner_coords)
        if final_coords is None:
            continue
        min_x, min_y, max_x, max_y = final_coords

        repro_rec = generate_record(ann_rec, min_x, min_y, max_x, max_y, sample_data_token, sd_rec["filename"])

        if mono3d and (repro_rec is not None):
            loc = box3d.center.tolist()
            dim = box3d.wlh
            dim[[0, 1, 2]] = dim[[1, 2, 0]]  # wlh -> lhw
            dim = dim.tolist()
            rot = box3d.orientation.yaw_pitch_roll[0]
            rot = [-rot]  # into the camera convention

            global_velo2d = nusc.box_velocity(box3d.token)[:2]
            global_velo3d = np.array([*global_velo2d, 0.0])
            e2g_r_mat = Quaternion(pose_rec["rotation"]).rotation_matrix
            c2e_r_mat = Quaternion(cs_rec["rotation"]).rotation_matrix
            cam_velo3d = global_velo3d @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(c2e_r_mat).T
            velo = cam_velo3d[0::2].tolist()

            repro_rec["bbox_cam3d"] = loc + dim + rot
            repro_rec["velo_cam3d"] = velo

            center3d = np.array(loc).reshape([1, 3])
            center2d = points_cam2img(center3d, camera_intrinsic, with_depth=True)
            repro_rec["center2d"] = center2d.squeeze().tolist()
            if repro_rec["center2d"][2] <= 0:
                continue

            ann_token = nusc.get("sample_annotation", box3d.token)["attribute_tokens"]
            attr_name = "None" if len(ann_token) == 0 else nusc.get("attribute", ann_token[0])["name"]
            repro_rec["attribute_name"] = attr_name
            repro_rec["attribute_id"] = nus_attributes.index(attr_name)

        repro_recs.append(repro_rec)

    return repro_recs


def fill_trainval_infos(nusc, train_scenes, val_scenes, test=False, max_sweeps=10, with_2d=True):
    """Build the train/val infos (verbatim, minus the mmcv progress bar and image read)."""
    train_nusc_infos = []
    val_nusc_infos = []
    frame_idx = 0

    for sample in nusc.sample:
        lidar_token = sample["data"]["LIDAR_TOP"]
        sd_rec = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        cs_record = nusc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
        pose_record = nusc.get("ego_pose", sd_rec["ego_pose_token"])
        lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)

        info = {
            "lidar_path": lidar_path,
            "token": sample["token"],
            "prev": sample["prev"],
            "next": sample["next"],
            "sweeps": [],
            "frame_idx": frame_idx,
            "cams": dict(),
            "scene_token": sample["scene_token"],
            "lidar2ego_translation": cs_record["translation"],
            "lidar2ego_rotation": cs_record["rotation"],
            "ego2global_translation": pose_record["translation"],
            "ego2global_rotation": pose_record["rotation"],
            "timestamp": sample["timestamp"],
        }

        l2e_r = info["lidar2ego_rotation"]
        l2e_t = info["lidar2ego_translation"]
        e2g_r = info["ego2global_rotation"]
        e2g_t = info["ego2global_translation"]
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        frame_idx = 0 if sample["next"] == "" else frame_idx + 1

        for cam in CAMERA_TYPES:
            cam_token = sample["data"][cam]
            _, _, cam_intrinsic = nusc.get_sample_data(cam_token)
            cam_info = obtain_sensor2top(nusc, cam_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, cam)
            cam_info.update(cam_intrinsic=cam_intrinsic)
            info["cams"].update({cam: cam_info})

        sd_rec = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        sweeps = []
        while len(sweeps) < max_sweeps:
            if not sd_rec["prev"] == "":
                sweep = obtain_sensor2top(nusc, sd_rec["prev"], l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, "lidar")
                sweeps.append(sweep)
                sd_rec = nusc.get("sample_data", sd_rec["prev"])
            else:
                break
        info["sweeps"] = sweeps

        if not test:
            annotations = [nusc.get("sample_annotation", token) for token in sample["anns"]]
            locs = np.array([b.center for b in boxes]).reshape(-1, 3)
            dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
            rots = np.array([b.orientation.yaw_pitch_roll[0] for b in boxes]).reshape(-1, 1)
            velocity = np.array([nusc.box_velocity(token)[:2] for token in sample["anns"]])
            valid_flag = np.array(
                [(anno["num_lidar_pts"] + anno["num_radar_pts"]) > 0 for anno in annotations], dtype=bool
            ).reshape(-1)

            for i in range(len(boxes)):
                velo = np.array([*velocity[i], 0.0])
                velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                velocity[i] = velo[:2]

            names = [b.name for b in boxes]
            for i in range(len(names)):
                if names[i] in NAME_MAPPING:
                    names[i] = NAME_MAPPING[names[i]]
            names = np.array(names)

            gt_boxes = np.concatenate([locs, dims[:, [1, 0, 2]], rots], axis=1)
            assert len(gt_boxes) == len(annotations)
            info["gt_boxes"] = gt_boxes
            info["gt_names"] = names
            info["gt_velocity"] = velocity.reshape(-1, 2)
            info["num_lidar_pts"] = np.array([a["num_lidar_pts"] for a in annotations])
            info["num_radar_pts"] = np.array([a["num_radar_pts"] for a in annotations])
            info["valid_flag"] = valid_flag
            info["instance_tokens"] = [a["instance_token"] for a in annotations]

            if with_2d:
                info.update(_collect_2d_annotations(nusc, info))

        if sample["scene_token"] in train_scenes:
            train_nusc_infos.append(info)
        else:
            val_nusc_infos.append(info)

    return train_nusc_infos, val_nusc_infos


def _collect_2d_annotations(nusc, info) -> dict:
    """Collect the per-camera 2D annotation lists (verbatim from StreamPETR's converter)."""
    gt_2dbboxes_cams, gt_3dbboxes_cams, centers2d_cams = [], [], []
    gt_2dbboxes_ignore_cams, gt_2dlabels_cams, depths_cams, visibilities = [], [], [], []

    for _, cam_info in info["cams"].items():
        gt_3dbboxes, gt_2dbboxes, centers2d = [], [], []
        gt_2dbboxes_ignore, gt_2dlabels, depths, visibility = [], [], [], []

        sd_rec = nusc.get("sample_data", cam_info["sample_data_token"])
        height, width = sd_rec["height"], sd_rec["width"]

        annos_cam = get_2d_boxes(
            nusc, cam_info["sample_data_token"], visibilities=["", "1", "2", "3", "4"], mono3d=True
        )
        for ann in annos_cam:
            if ann is None or ann.get("ignore", False):
                continue
            x1, y1, w, h = ann["bbox"]
            inter_w = max(0, min(x1 + w, width) - max(x1, 0))
            inter_h = max(0, min(y1 + h, height) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann["area"] <= 0 or w < 1 or h < 1:
                continue
            if ann["category_name"] not in nus_categories:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get("iscrowd", False):
                gt_2dbboxes_ignore.append(bbox)
            else:
                gt_2dbboxes.append(bbox)
                gt_2dlabels.append(ann["category_id"])
                centers2d.append(ann["center2d"][:2])
                depths.append(ann["center2d"][2])
                visibility.append(ann["visibility_token"])
                gt_3dbboxes.append(ann["bbox_cam3d"])

        gt_2dbboxes_cams.append(np.array(gt_2dbboxes, dtype=np.float32))
        gt_2dlabels_cams.append(np.array(gt_2dlabels, dtype=np.int64))
        centers2d_cams.append(np.array(centers2d, dtype=np.float32))
        gt_3dbboxes_cams.append(np.array(gt_3dbboxes, dtype=np.float32))
        depths_cams.append(np.array(depths, dtype=np.float32))
        gt_2dbboxes_ignore_cams.append(np.array(gt_2dbboxes_ignore, dtype=np.float32))
        visibilities.append(visibility)

    return dict(
        bboxes2d=gt_2dbboxes_cams,
        bboxes3d_cams=gt_3dbboxes_cams,
        labels2d=gt_2dlabels_cams,
        centers2d=centers2d_cams,
        depths=depths_cams,
        bboxes_ignore=gt_2dbboxes_ignore_cams,
        visibilities=visibilities,
    )


def create_nuscenes_infos(
    root_path: str,
    out_dir: str,
    info_prefix: str = "reference",
    version: str = "v1.0-mini",
    max_sweeps: int = 10,
    with_2d: bool = True,
) -> Tuple[Path, Path]:
    """Write the reference train/val info pickles.

    :param root_path: nuScenes data root.
    :param out_dir: Directory to write the pickles into.
    :param info_prefix: Filename prefix.
    :param version: nuScenes database version.
    :param max_sweeps: Maximum lidar sweeps per info.
    :param with_2d: Also produce the per-camera 2D annotations.
    :return: ``(train_path, val_path)``.
    """
    from nuscenes.utils import splits

    nusc = NuScenes(version=version, dataroot=root_path, verbose=False)
    if version == "v1.0-trainval":
        train_scene_names, val_scene_names = splits.train, splits.val
    elif version == "v1.0-test":
        train_scene_names, val_scene_names = splits.test, []
    elif version == "v1.0-mini":
        train_scene_names, val_scene_names = splits.mini_train, splits.mini_val
    else:
        raise ValueError(f"Unknown version {version}")

    name_to_token = {scene["name"]: scene["token"] for scene in nusc.scene}
    train_scenes = {name_to_token[name] for name in train_scene_names if name in name_to_token}
    val_scenes = {name_to_token[name] for name in val_scene_names if name in name_to_token}

    train_infos, val_infos = fill_trainval_infos(
        nusc, train_scenes, val_scenes, test=False, max_sweeps=max_sweeps, with_2d=with_2d
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = dict(version=version)

    train_path = out_dir / f"{info_prefix}_infos_train.pkl"
    val_path = out_dir / f"{info_prefix}_infos_val.pkl"
    with open(train_path, "wb") as handle:
        pickle.dump(dict(infos=train_infos, metadata=metadata), handle, protocol=4)
    with open(val_path, "wb") as handle:
        pickle.dump(dict(infos=val_infos, metadata=metadata), handle, protocol=4)

    print(f"train: {len(train_infos)} frames -> {train_path}")
    print(f"val:   {len(val_infos)} frames -> {val_path}")
    return train_path, val_path


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point for the reference converter."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root-path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--info-prefix", default="reference")
    parser.add_argument("--max-sweeps", type=int, default=10)
    parser.add_argument("--no-2d", action="store_true")
    args = parser.parse_args(argv)

    create_nuscenes_infos(
        root_path=args.root_path,
        out_dir=args.out_dir,
        info_prefix=args.info_prefix,
        version=args.version,
        max_sweeps=args.max_sweeps,
        with_2d=not args.no_2d,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
