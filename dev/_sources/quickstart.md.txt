# Quick start

```bash
export PY123D_DATA_ROOT=/data/py123d
export NUSCENES_DATA_ROOT=/data/nuscenes      # only needed while images are referenced, not copied

py123det-export-mmcv export \
  --split nuscenes-mini_train \
  --out /data/nuscenes/py123d_infos_train.pkl
```

```
Wrote 323 frames to /data/nuscenes/py123d_infos_train.pkl (7.3 MB, protocol 4, arrays=numpy)
  logs: 8   boxes: 13923
  classes: {'car': 5051, 'pedestrian': 3657, 'barrier': 2323, ...}
  sensor payloads: 2261 referenced, 0 extracted

Set class_names in your mmdet3d config to:
  ['car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier']
```

Same thing from Python:

```python
from py123detection import Source
from py123detection.mmcv_export import export_to_mmdet3d

report = export_to_mmdet3d(
    Source(data_root="/data/py123d", splits=["nuscenes-mini_train"], sample_rate_hz=2.0),
    output_path="/data/nuscenes/py123d_infos_train.pkl",
)
print(report.summary())
```

Then point an existing config at it:

```python
dataset_type = "NuScenesDataset"
class_names = ["car", "truck", "trailer", "bus", "construction_vehicle",
               "bicycle", "motorcycle", "pedestrian", "traffic_cone", "barrier"]

data = dict(
    train=dict(type=dataset_type, ann_file="data/nuscenes/py123d_infos_train.pkl",
               classes=class_names, data_root="data/nuscenes/", ...),
    val=dict(type=dataset_type, ann_file="data/nuscenes/py123d_infos_val.pkl", ...),
)
```

Nothing else changes. The pickle *is* the nuScenes info schema. See
{py:mod}`schema.py <py123detection.mmcv_export.schema>` for the field-by-field description.
