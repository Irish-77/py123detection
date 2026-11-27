# Examples

Commands for the common jobs. Each one links to the guide that explains the flags.

## Export one split

```bash
export PY123D_DATA_ROOT=/data/py123d
export NUSCENES_DATA_ROOT=/data/nuscenes

py123det-export-mmcv export \
  --split nuscenes-mini_train \
  --out /data/nuscenes/py123d_infos_train.pkl
```

```python
from py123detection import Source
from py123detection.mmcv.export import export_to_mmdet3d

report = export_to_mmdet3d(
    Source(data_root="/data/py123d", splits=["nuscenes-mini_train"]),
    output_path="/data/nuscenes/py123d_infos_train.pkl",
)
print(report.summary())
```

## Export train and val

Both files must be written with the same settings, so build the config once.

```python
from py123detection import Source
from py123detection.mmcv.export import ExportConfig, export_to_mmdet3d

config = ExportConfig()

for split, name in [("nuscenes-mini_train", "train"), ("nuscenes-mini_val", "val")]:
    export_to_mmdet3d(
        Source(splits=[split]),
        output_path=f"/data/nuscenes/py123d_infos_{name}.pkl",
        config=config,
        version="v1.0-mini",
    )
```

`version` becomes `metadata['version']`, which mmdet3d looks up in its nuScenes `eval_set_map`.
Keep a real nuScenes version string if you plan to evaluate.

## Export fewer frames

```bash
py123det-export-mmcv export \
  --split av2-sensor_train \
  --sample-rate-hz 2.0 \
  --out /data/av2/py123d_infos_train.pkl
```

`--sample-rate-hz 2.0` keeps every 5th frame of a 10 Hz log. Add `--max-logs 2` while trying a new
dataset, and it takes seconds instead of hours.

## Pick a different class set

```bash
py123det-export-mmcv export \
  --split av2-sensor_train \
  --taxonomy general_3cls \
  --out /data/av2/py123d_infos_train.pkl
```

The default, `nuscenes_detection_10cls`, maps native nuScenes labels only and drops every box of
any other dataset. The eight that ship, how to write your own, and how merge rules work are in
[Choosing what to export](guides/choosing-what-to-export.md#classes).

## Export a dataset that is not nuScenes

```bash
py123det-export-mmcv export \
  --split av2-sensor_train \
  --taxonomy general_3cls \
  --velocity-source tracks \
  --ego-pose-source nearest \
  --token-style log_timestamp \
  --out /data/av2/py123d_infos_train.pkl
```

Four decisions here. A taxonomy that maps on default labels. Velocity derived from tracks, since
AV2 annotates none. Ego pose looked up by timestamp rather than trusted from the sync table. And
frame tokens built from log name and timestamp, since only nuScenes has native ones.

[Exporting other 123D datasets](other-datasets.md) is the full checklist, including how to prove a
new export against a reference converter.

## Merge datasets into one pickle

```python
from py123detection import Source, get_taxonomy
from py123detection.mmcv.export import ExportConfig, export_to_mmdet3d

export_to_mmdet3d(
    [Source(splits=["nuscenes_train"]),
     Source(splits=["wod-perception_train"]),
     Source(splits=["av2-sensor_train"])],
    output_path="/data/mixed_infos_train.pkl",
    config=ExportConfig(taxonomy=get_taxonomy("general_3cls"), camera_key="camera_id"),
    disjoint_timestamps=True,
)
```

Why those two settings, and the command-line form, are in
[Cross-dataset training](guides/cross-dataset-training.md).

## Keep the native nuScenes tokens

Official nuScenes evaluation matches predictions by `sample_token`. Reading the nuScenes database
takes a while, so dump the map once and reuse it:

```bash
py123det-export-mmcv token-map --nuscenes-root /data/nuscenes --out /data/token_map.json

py123det-export-mmcv export \
  --split nuscenes_val \
  --token-map /data/token_map.json \
  --out /data/nuscenes/py123d_infos_val.pkl
```

`--nuscenes-root /data/nuscenes` on the export does the same in one step, and needs the `nuscenes`
extra. See [Frame identifiers and nuScenes evaluation](guides/tokens-and-evaluation.md).

## Copy the sensor files instead of pointing at them

```bash
py123det-export-mmcv export \
  --split nuscenes-mini_train \
  --camera-mode extract --lidar-mode extract \
  --extract-root /data/exported/sensors \
  --out /data/exported/py123d_infos_train.pkl
```

For when the training machine cannot see the original dataset. Logs that store images inside MP4
files need the `images` extra. See [Images and point clouds](guides/sensor-payloads.md).

## Export for a training environment on numpy 1

```bash
py123det-export-mmcv export \
  --split nuscenes_train \
  --array-format portable \
  --out /data/nuscenes/py123d_infos_train.pkl
```

A pickle written under numpy 2 fails to load under numpy 1. Installing `numpy<2` next to py123d is
the simpler fix; `portable` is for when that is not possible, and the file then has to be read
through `Py123DNuScenesDataset`. See
[Crossing the Python version boundary](guides/python-version-boundary.md).

## Check what came out

```bash
py123det-export-mmcv inspect /data/nuscenes/py123d_infos_train.pkl
```

```
/data/nuscenes/py123d_infos_train.pkl: 323 frames
metadata:
  version: v1.0-mini
  taxonomy: nuscenes_detection_10cls
  class_names: ['car', 'truck', 'trailer', 'bus', ...]
  camera_order: ['PCAM_F0', 'PCAM_R0', 'PCAM_L0', 'PCAM_B0', 'PCAM_L1', 'PCAM_R1']
  python_version: 3.11.13
  numpy_version: 1.26.4
  stats: {'num_logs': 8, 'num_frames': 323, 'num_boxes': 13923, ...}
```

Run it in the training environment too: if the file loads there, it survived the version hop.

To diff an export against a pickle from the native converter:

```bash
python tools/compare_infos.py nuscenes_infos_val.pkl py123d_infos_val.pkl
```

[Validation](validation.md) covers what that checks and which differences are expected.
