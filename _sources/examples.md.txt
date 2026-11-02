# Examples

Short recipes for the common jobs. Each section links to the guide that explains the flags in
depth.

## Export one split

```bash
export PY123D_DATA_ROOT=/data/py123d
export NUSCENES_DATA_ROOT=/data/nuscenes

py123det-export-mmcv export \
  --split nuscenes-mini_train \
  --out /data/nuscenes/py123d_infos_train.pkl
```

The same call from Python:

```python
from py123detection import Source
from py123detection.mmcv_export import export_to_mmdet3d

report = export_to_mmdet3d(
    Source(data_root="/data/py123d", splits=["nuscenes-mini_train"]),
    output_path="/data/nuscenes/py123d_infos_train.pkl",
)
print(report.summary())
```

## Export train and val

Both files must be written with the same settings, so build the config once and reuse it.

```python
from py123detection import Source
from py123detection.mmcv_export import ExportConfig, export_to_mmdet3d

config = ExportConfig()

for split, name in [("nuscenes-mini_train", "train"), ("nuscenes-mini_val", "val")]:
    report = export_to_mmdet3d(
        Source(splits=[split]),
        output_path=f"/data/nuscenes/py123d_infos_{name}.pkl",
        config=config,
        version="v1.0-mini",
    )
    print(report.summary())
```

`version` is stored as `metadata['version']`. mmdet3d looks it up in its nuScenes `eval_set_map`,
so keep a real nuScenes version string if you plan to evaluate.

## Export fewer frames

`--sample-rate-hz 2.0` keeps every 5th frame of a 10 Hz log.

```bash
py123det-export-mmcv export \
  --split av2-sensor_train \
  --sample-rate-hz 2.0 \
  --out /data/av2/py123d_infos_train.pkl
```

While trying out a new dataset, add `--max-logs 2`. The run then takes seconds instead of hours.

## Pick a different class set

Five taxonomies ship with the package:

| name | classes | maps on |
| --- | --- | --- |
| `nuscenes_detection_10cls` | the 10 nuScenes benchmark classes | native nuScenes labels |
| `general_3cls` | vehicle, pedestrian, cyclist | default labels |
| `coin3d_3cls` | car, pedestrian, motorcycle | default labels |
| `vehicle_1cls` | vehicle | default labels |
| `default_123d` | the 11 default 123D labels | default labels |

```bash
py123det-export-mmcv export \
  --split av2-sensor_train \
  --taxonomy general_3cls \
  --out /data/av2/py123d_infos_train.pkl
```

The default is `nuscenes_detection_10cls`, which only maps native nuScenes labels. Exporting any
other dataset with it drops every box. Taxonomies that map on default labels work for all 123D
datasets, because every one of them implements `to_default()`. See
[Classes](guides/choosing-what-to-export.md#classes).

## Write your own taxonomy

A taxonomy is a class list plus the rules that map 123D labels onto it. `default_map` keys are
`DefaultBoxDetectionLabel` members and cover every dataset. `native_map` keys are dataset native
enum members and override single labels.

```python
from py123detection import Source, Taxonomy
from py123detection.taxonomy import VOID
from py123detection.mmcv_export import ExportConfig, export_to_mmdet3d

ROAD_USERS = Taxonomy(
    name="road_users_2cls",
    class_names=("vehicle", "person"),
    default_map={
        "VEHICLE": "vehicle",
        "TRAIN": "vehicle",
        "TWO_WHEELER": "vehicle",
        "PERSON": "person",
        "EGO": VOID,
        "ANIMAL": VOID,
    },
)

export_to_mmdet3d(
    Source(splits=["av2-sensor_train"]),
    output_path="/data/av2/py123d_infos_train.pkl",
    config=ExportConfig(taxonomy=ROAD_USERS),
)
```

The index of a class name in `class_names` is its label id in the mmdet3d config.

Labels mapped to `VOID` are dropped, and so are labels with no entry at all. Listing the ones you
drop on purpose is optional, but it keeps the mapping readable.

The constructor checks both maps against `class_names`, so a misspelled target raises instead of
quietly dropping boxes:

```python
Taxonomy(name="typo", class_names=("vehicle",), default_map={"VEHICLE": "vehicles"})
# ValueError: Taxonomy 'typo' default_map targets unknown class names ['vehicles']
```

To change one dataset without touching the rest, add a `native_map` entry. Argoverse 2 boxes the
rider of a bicycle separately, and nuScenes has no rider class:

```python
native_map={"AV2SensorBoxDetectionLabel.BICYCLIST": VOID}
```

Lookups go from specific to general: qualified enum member, then bare member name, then the
default label.

## Add a dataset that is not nuScenes

Most of the export already works on any 123D dataset. Four flags usually need a decision:

```bash
py123det-export-mmcv export \
  --split av2-sensor_train \
  --taxonomy general_3cls \
  --velocity-source tracks \
  --ego-pose-source nearest \
  --token-style log_timestamp \
  --out /data/av2/py123d_infos_train.pkl
```

`--taxonomy general_3cls` maps on default labels. `--velocity-source tracks` derives velocity as a
central difference over each track, which Argoverse 2 needs because it stores none.
`--ego-pose-source nearest` looks the ego pose up by timestamp instead of trusting the sync table.
`--token-style log_timestamp` builds frame tokens from the log name and timestamp, since only
nuScenes has native sample tokens.

[Exporting other 123D datasets](other-datasets.md) goes through the full checklist and shows how
to prove a new export against a reference converter.

## Merge datasets into one pickle

Pass several sources and they land in one file under one class list.

```python
from py123detection import Source, get_taxonomy
from py123detection.mmcv_export import ExportConfig, export_to_mmdet3d

sources = [
    Source(data_root="/data/py123d", splits=["nuscenes_train"]),
    Source(data_root="/data/py123d", splits=["wod-perception_train"]),
    Source(data_root="/data/py123d", splits=["av2-sensor_train"]),
]

export_to_mmdet3d(
    sources,
    output_path="/data/mixed_infos_train.pkl",
    config=ExportConfig(taxonomy=get_taxonomy("general_3cls"), camera_key="camera_id"),
    disjoint_timestamps=True,
)
```

`camera_key="camera_id"` keys `info['cams']` by the 123D camera id, so the camera dictionaries
match across datasets. `disjoint_timestamps=True` spaces the sources apart in exported time,
because `NuScenesDataset.load_annotations` sorts every info by timestamp and would otherwise
interleave logs from different datasets. See
[Cross-dataset training](guides/cross-dataset-training.md) for the command line version.

## Keep the native nuScenes tokens

The official nuScenes evaluation matches predictions by `sample_token`, so an export used for
evaluation needs the native ones. This needs the `nuscenes` extra.

```bash
py123det-export-mmcv export \
  --split nuscenes_val \
  --nuscenes-root /data/nuscenes \
  --out /data/nuscenes/py123d_infos_val.pkl
```

Reading the nuScenes database takes a while. Dump the map once and reuse it:

```bash
py123det-export-mmcv token-map --nuscenes-root /data/nuscenes --out /data/token_map.json

py123det-export-mmcv export \
  --split nuscenes_val \
  --token-map /data/token_map.json \
  --out /data/nuscenes/py123d_infos_val.pkl
```

See [Frame identifiers and nuScenes evaluation](guides/tokens-and-evaluation.md).

## Copy the sensor files instead of pointing at them

By default the pickle holds paths into the original dataset. Extract the payloads when the
training machine cannot see that dataset.

```bash
py123det-export-mmcv export \
  --split nuscenes-mini_train \
  --camera-mode extract \
  --lidar-mode extract \
  --extract-root /data/exported/sensors \
  --out /data/exported/py123d_infos_train.pkl
```

Logs that store images inside MP4 files need the `images` extra. See
[Images and point clouds](guides/sensor-payloads.md).

## Export for a training environment on numpy 1

A pickle written under numpy 2 fails to load under numpy 1 with
`ModuleNotFoundError: No module named 'numpy._core'`. Installing `numpy<2` next to py123d is the
simple fix. When that is not possible, write the arrays as nested lists:

```bash
py123det-export-mmcv export \
  --split nuscenes_train \
  --array-format portable \
  --out /data/nuscenes/py123d_infos_train.pkl
```

The training config then reads it through the plugin dataset, which rebuilds the arrays:

```python
plugin = True
plugin_dir = "py123detection/mmcv_plugin/"

data = dict(train=dict(type="Py123DNuScenesDataset", ann_file="py123d_infos_train.pkl"))
```

See [Crossing the Python version boundary](guides/python-version-boundary.md).

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

Run it in the training environment as well. If the file loads there, it survived the version hop.

To compare an export against a pickle from the native converter:

```bash
python tools/compare_infos.py nuscenes_infos_val.pkl py123d_infos_val.pkl
```

[Validation](validation.md) covers what the comparison checks and which differences are expected.
