# Choosing what to export

Which logs, how many frames, and which classes. Everything else about an export has a defensible
default.

## Which logs

A frame-indexed pickle needs one scene per log, spanning the whole log. `Source` is 123D's
`SceneFilter` restricted to that case, with only the knobs an export uses:

```python
Source(
    data_root="/data/py123d",           # or $PY123D_DATA_ROOT
    splits=["nuscenes_train"],          # or datasets= / split_types= / log_names=
    sample_rate_hz=2.0,                 # or frame_stride=5
    required_modalities=["camera:all"],
    max_logs=None,
)
```

`Source(scene_filter=...)` takes a full `SceneFilter` instead, and `Source.from_scenes(scenes)`
takes scenes you built yourself. Both work as long as the filter yields whole logs.

## How many frames

`sample_rate_hz` (or `frame_stride`) thins the exported frames. **Sweeps are not thinned with
them**: the exporter loads a second, unsampled view of each log and draws `info['sweeps']` from
that, so a 2 Hz export off a 10 Hz log still gets 10 Hz sweeps. Measured on the mini kit, a 1 Hz
export produced frames 1.0 s apart with sweeps 0.5 s apart.

## Classes

`--taxonomy` decides how 123D labels become the strings in `gt_names`. Eight ship:

| name | classes | applies to |
| --- | --- | --- |
| `nuscenes_detection_10cls` (default) | the standard 10, in mmdet3d's order | **nuScenes-derived logs only** |
| `general_3cls` | vehicle, pedestrian, cyclist | every 123D dataset |
| `coin3d_3cls` | car, pedestrian, motorcycle | every 123D dataset |
| `mixed_8cls` | car, truck, bus, trailer, traffic_cone, pedestrian, motorcycle, bicycle | every 123D dataset |
| `mixed2_3cls` | vehicle, pedestrian, bicycle | every 123D dataset |
| `mixed3_1cls` | vehicle | every 123D dataset |
| `vehicle_1cls` | vehicle | every 123D dataset |
| `default_123d` | the 11 `DefaultBoxDetectionLabel` classes | every 123D dataset |

The default maps native nuScenes labels only. **Exporting any other dataset with it drops every
box**; pick one of the others.

`coin3d_3cls` spells CoIn3D's vehicle / pedestrian / two-wheeler classes with nuScenes names
because the nuScenes devkit's `DetectionBox` rejects anything else — see
[the evaluation constraint](../other-datasets.md#9-training-side).

`vehicle_1cls` and `mixed3_1cls` both have one `vehicle` class and are not the same: `vehicle_1cls`
is the default label `VEHICLE` plus `TRAIN`, while `mixed3_1cls` adds the per-dataset rules that
make nuScenes, AV2 and Waymo agree on what a vehicle is — a motorcycle counts, a tram does not.

### The two levels

A taxonomy maps a label at either level, and the first hit wins:

1. **qualified native** — `"AV2SensorBoxDetectionLabel.BICYCLIST"`, which changes one dataset
   without touching a same-named member of another;
2. **bare native** — `"VEHICLE_CAR"`, the dataset's own enum member, at full granularity;
3. **default** — `"PERSON"`, a `DefaultBoxDetectionLabel` member.

A taxonomy written only at the default level applies to every 123D dataset, because every parser
implements `to_default()`. That is what makes one taxonomy work across nuScenes, Waymo and AV2
with no per-dataset code.

Labels mapped to `VOID`, and labels with no entry at all, are dropped. Listing the ones you drop on
purpose is optional and keeps the mapping readable.

### Merge rules

Some datasets box a rider separately from the vehicle. A merge rule folds them back into one box
after mapping:

```python
merge_rules=(MergeRule(target="vehicle", sources=("av2_motorcycle", "av2_motorcyclist")),)
```

The sources are intermediate class names the native map produces and `class_names` does not
contain, so they cannot leak into `gt_names`. Secondaries are paired to primaries by bird's-eye
IoU, greedily and highest first; a pair becomes the tight box around both sets of corners, keeping
the primary's orientation and velocity, and a secondary that pairs with nothing is dropped.
`mixed2_3cls` and `mixed3_1cls` use this for Argoverse 2, which boxes riders separately.

### Writing one

```python
from py123detection import Taxonomy
from py123detection.core.taxonomy import VOID

ROAD_USERS = Taxonomy(
    name="road_users_2cls",
    class_names=("vehicle", "person"),
    default_map={"VEHICLE": "vehicle", "TRAIN": "vehicle", "TWO_WHEELER": "vehicle",
                 "PERSON": "person", "EGO": VOID, "ANIMAL": VOID},
    native_map={"AV2SensorBoxDetectionLabel.BICYCLIST": VOID},
)
```

A class name's index in `class_names` is its label id in the mmdet3d config.

The constructor checks both maps against `class_names`, so a typo raises instead of silently
dropping boxes:

```python
Taxonomy(name="typo", class_names=("vehicle",), default_map={"VEHICLE": "vehicles"})
# ValueError: Taxonomy 'typo' default_map targets unknown class names ['vehicles'];
# declared class_names are ('vehicle',) and merge sources are [].
```

To ship it as a `--taxonomy` choice, add it to `TAXONOMIES` in
`py123detection/core/taxonomy.py`, with the semantics in its docstring.
