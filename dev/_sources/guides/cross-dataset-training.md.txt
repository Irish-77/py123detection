# Cross-dataset training

CoIn3D trains across datasets by converting each into the nuScenes data format specified by mmcv and concatenating them into one `ann_file` under a shared class list. That is one command here:

```bash
py123det-export-mmcv export \
  --split nuscenes_train \
  --merge "/data/py123d:wod-perception_train" \
  --merge "/data/py123d:av2-sensor_train" \
  --taxonomy general_3cls \
  --camera-key camera_id \
  --disjoint-timestamps \
  --out /data/mixed_infos_train.pkl
```

Three flags do the work:

- `--taxonomy general_3cls` picks a taxonomy defined over default labels, so every source maps into
  the same three classes.
- `--camera-key camera_id` keys `info['cams']` by the 123D camera id (`PCAM_F0`) instead of the
  dataset-native name (`CAM_FRONT`), so the camera dictionaries are consistent across datasets. The
  default camera order is already expressed in 123D ids and resolves to mmdet3d's nuScenes order on
  a nuScenes rig.
- `--disjoint-timestamps` spaces the sources apart in exported time.
  `NuScenesDataset.load_annotations` sorts every info by `timestamp`, and datasets recorded in
  overlapping wall-clock windows would interleave under that sort, shuffling logs together and
  breaking sequence samplers. Intra-log deltas are left untouched. The exporter checks for
  interleaving either way and warns if it finds any.

Sensor paths are absolute by default, so each source keeps pointing at its own dataset root
regardless of the config's `data_root`.
