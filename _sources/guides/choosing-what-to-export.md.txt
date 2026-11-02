# Choosing what to export

## Pass a folder, not a scene list

The 123D `SceneFilter` is expressive, but only one corner of it produces what a frame-indexed
pickle needs: **one scene per log, spanning the whole log**. `Source` encodes that corner and
exposes only the knobs that matter for an export:

```python
Source(
    data_root="/data/py123d",          # or $PY123D_DATA_ROOT
    splits=["nuscenes_train"],          # or datasets=/split_types=/log_names=
    sample_rate_hz=2.0,                 # or frame_stride=5
    required_modalities=["camera:all"], # 123D modality requirements
    max_logs=None,
)
```

If you want the full filter, `Source(scene_filter=...)` takes one verbatim, and
`Source.from_scenes(scenes)` takes scenes you built yourself. Both stay compatible with the rest
of the pipeline as long as the filter yields whole logs.

## Sample rate

`sample_rate_hz` (or `frame_stride`) thins the exported frames. **Sweeps are not thinned with
them.** The exporter loads a second, unsampled view of each log and draws `info['sweeps']` from
that, so a 2 Hz export off a 10 Hz log still gets 10 Hz sweeps. Verified on the mini kit: a 1 Hz
export produced frames 1.0 s apart with sweeps 0.5 s apart.

## Classes

`--taxonomy` picks how 123D labels become the strings in `gt_names`:

| Taxonomy | Classes | Works on |
| --- | --- | --- |
| `nuscenes_detection_10cls` (default) | the standard 10, in mmdet3d's order | nuScenes-derived logs |
| `general_3cls` | vehicle, pedestrian, cyclist | **every 123D dataset** |
| `coin3d_3cls` | car, pedestrian, motorcycle (CoIn3D's vehicle / pedestrian / two-wheel, spelled with nuScenes names so the nuScenes devkit accepts them) | **every 123D dataset** |
| `vehicle_1cls` | vehicle | every 123D dataset |
| `default_123d` | the 11 `DefaultBoxDetectionLabel` classes | every 123D dataset |

The cross-dataset ones are expressed over `DefaultBoxDetectionLabel`, which every 123D dataset enum
implements via `to_default()`. That is what makes one taxonomy apply to nuScenes, Waymo and AV2
without per-dataset code. Custom taxonomies are a dataclass:

```python
from py123detection import Taxonomy
from py123detection.taxonomy import VOID

MY_TAXONOMY = Taxonomy(
    name="cars_and_people",
    class_names=("car", "person"),
    native_map={"VEHICLE_CAR": "car", "ANIMAL": VOID},   # dataset-native names, full granularity
    default_map={"PERSON": "person"},                     # applies to any 123D dataset
)
```

A `native_map` key can also be qualified with the label enum
(`"AV2SensorBoxDetectionLabel.BICYCLIST"`) to override one dataset without touching a same-named
member of another. Lookup order is qualified native, then bare native, then default.
