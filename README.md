# py123detection

A 123D toolkit for 3D multi-view object detection.

Two branches share the 123D data model:

| Branch | Module | Status |
| --- | --- | --- |
| **mmcv export** — turn any 123D dataset into the `info` pickle the mmcv / mmdetection3d ecosystem reads, so existing PETR / StreamPETR / BEVDet configs train on 123D data unchanged | `py123detection.mmcv_export` (+ `py123detection.mmcv_plugin` on the training side) | implemented |
| **modern stack** — the mmcv-free path from the master thesis | — | not started |

This README covers the first branch. The notes this work started from are kept verbatim in
[`docs/original-brief.md`](docs/original-brief.md).

---

## Why this exists

123D already normalizes a dozen driving datasets into one schema. mmdetection3d, meanwhile,
speaks exactly one dialect fluently: the nuScenes `info` pickle. Every camera-based detector
built on it — PETR, StreamPETR, BEVDet, CoIn3D — reads that dialect, and CoIn3D's cross-dataset
training works by converting Waymo and Lyft *into* it and concatenating the results.

So: one converter from 123D to that dialect, and every 123D dataset becomes trainable with
off-the-shelf configs, individually or mixed.

---

## Install

```bash
pip install -e py123detection
```

Optional extras:

```bash
pip install -e "py123detection[nuscenes]"   # restore native nuScenes sample tokens
pip install -e "py123detection[images]"     # re-encode images when extracting MP4-backed logs
```

The exporter needs **py123d and numpy — not mmcv, not mmdet3d, not torch**. It is meant to run in
your 123D environment and write a file your training environment reads. See
[Crossing the Python version boundary](#crossing-the-python-version-boundary).

---

## Quick start

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

Nothing else changes. The pickle *is* the nuScenes info schema — see
[`schema.py`](src/py123detection/mmcv_export/schema.py) for the field-by-field description.

---

## Does it actually match?

The one question worth answering up front: does `nuScenes → 123D → pkl` land where
`nuScenes → pkl` lands? [`tools/reference_nuscenes_converter.py`](tools/reference_nuscenes_converter.py)
is a dependency-light port of the converter StreamPETR and mmdetection3d ship, and
[`tools/compare_infos.py`](tools/compare_infos.py) diffs the two pickles field by field —
frames matched by token, boxes matched by instance token, rotations compared as matrices so the
`q`/`-q` sign ambiguity cannot hide a difference.

On the nuScenes v1.0-mini kit (404 frames, 10 logs, both splits):

| Quantity | max abs deviation | values compared |
| --- | --- | --- |
| frame `timestamp` | **0** | 404 |
| camera `timestamp` | **0** | 2 424 |
| `lidar2ego` translation | **0** | 1 212 |
| `lidar2ego` rotation | 3.3e-16 | 3 636 |
| `ego2global` translation | **0** | 1 212 |
| `ego2global` rotation | 4.3e-15 | 3 636 |
| `cam_intrinsic` | **0** | 21 816 |
| camera `sensor2ego` translation / rotation | **0** / 7.6e-16 | 7 272 / 21 816 |
| camera `ego2global` translation / rotation | 2.3e-13 m / 1.3e-15 | 7 272 / 21 816 |
| camera `sensor2lidar` translation / rotation | 9.1e-13 m / 5.4e-15 | 7 272 / 21 816 |
| `gt_boxes` center | 8.5e-13 m | 55 092 |
| `gt_boxes` size (l, w, h) | **0** | 55 092 |
| `gt_boxes` yaw | 5.6e-15 rad | 18 364 |
| `num_lidar_pts` | **0** | 18 364 |
| 2D `bboxes2d` | **0** | 87 600 |
| 2D `centers2d` / `depths` | **0** | 43 800 / 21 900 |
| mono-3D `bboxes3d_cams` | **0** | 153 300 |
| `gt_velocity` | 9.7e-6 m/s | 36 668 |

Frame counts, box counts, class names, camera order, and every sensor path are identical.
`gt_velocity` is the one number that is not bit-exact, and it is explained
[below](#known-differences).

Reproduce it:

```bash
python tools/reference_nuscenes_converter.py --root-path /data/nuscenes --out-dir /tmp/ref
py123det-export-mmcv export --split nuscenes-mini_val --nuscenes-root /data/nuscenes --out /tmp/py123d_val.pkl
python tools/compare_infos.py /tmp/ref/reference_infos_val.pkl /tmp/py123d_val.pkl
```

or as a test:

```bash
NUSCENES_DATA_ROOT=/data/nuscenes PY123D_DATA_ROOT=/data/py123d pytest -m integration
```

---

## Choosing what to export

### Pass a folder, not a scene list

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

### Sample rate

`sample_rate_hz` (or `frame_stride`) thins the exported frames. **Sweeps are not thinned with
them**: the exporter loads a second, unsampled view of each log and draws `info['sweeps']` from
that, so a 2 Hz export off a 10 Hz log still gets 10 Hz sweeps. Verified on the mini kit — a 1 Hz
export produced frames 1.0 s apart with sweeps 0.5 s apart.

### Classes

`--taxonomy` picks how 123D labels become the strings in `gt_names`:

| Taxonomy | Classes | Works on |
| --- | --- | --- |
| `nuscenes_detection_10cls` (default) | the standard 10, in mmdet3d's order | nuScenes-derived logs |
| `general_3cls` | vehicle, pedestrian, cyclist | **every 123D dataset** |
| `vehicle_1cls` | vehicle | every 123D dataset |
| `default_123d` | the 11 `DefaultBoxDetectionLabel` classes | every 123D dataset |

The cross-dataset ones are expressed over `DefaultBoxDetectionLabel`, which every 123D dataset
enum implements via `to_default()` — that is what makes one taxonomy apply to nuScenes, Waymo and
AV2 without per-dataset code. Custom taxonomies are a dataclass:

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

---

## Static vs. dynamic extrinsics

Sensors do not fire at the same instant, and where that offset gets booked is a real modelling
choice. 123D exposes it asymmetrically, which is worth knowing before you pick a mode:

- the **camera** Arrow table stores a per-row, motion-compensated `camera_to_global_se3`, so a
  camera's true pose at its own capture time is always available;
- the **lidar** table stores only timestamps and the payload — no per-row pose. The only lidar
  extrinsic 123D holds is the static rig calibration in the modality metadata.

Two options control what the export writes. **Both default to `static`, which is what
mmdetection3d's own converter emits** and what the bit-comparison above was run against.

### `--lidar2ego-mode {static,dynamic}`

`static` writes the rig calibration. `dynamic` looks up the ego pose at the *lidar's* own
timestamp and re-derives the extrinsic as
`inv(ego2global_at_frame) @ ego2global_at_lidar_time @ lidar2ego_static`, so
`ego2global @ lidar2ego` reconstructs where the sensor actually was when it fired. `ego2global`
itself is never rewritten.

It can only differ from `static` when the log's ego stream is denser than its frames, because the
correction is reconstructed from that stream (nearest sample — no interpolation, so the export
never invents poses the log does not contain). Measured on the mini kit:

| 123D conversion | ego stream | `lidar2ego` shift vs. static | box centre shift |
| --- | --- | --- | --- |
| keyframe (`sync`, 2 Hz) | 2 Hz — one state per frame | 1.5e-15 m (no-op) | 4.5e-13 m |
| async (native rates) | ~20 Hz | mean 0.13 m, **max 0.45 m** | mean 0.14 m, **max 0.60 m** |

So on a standard keyframe conversion the flag is inert — there is simply nothing closer to look
up — and on an async conversion it corrects up to half a metre.

### `--camera2ego-mode {static,dynamic}`

The camera offset is *always* accounted for; this only decides where it is reported.

| | `sensor2ego` | camera `ego2global` |
| --- | --- | --- |
| `static` (default, mmdet3d) | rig calibration | at the camera's own timestamp — differs per camera |
| `dynamic` (master-thesis style) | absorbs the offset | the frame's shared pose, identical for all cameras |

Measured on the mini kit: 0.40 m moves out of `ego2global` and into `sensor2ego`, and
**`sensor2lidar` is bit-identical (0.0)** between the two. So PETR and StreamPETR — which only
consume `sensor2lidar` — cannot tell the difference. It matters for BEVDet-family configs, which
read `sensor2ego` and `ego2global` separately.

A caveat on both: 123D reads rig calibration once per log (for nuScenes, from the scene's first
frame). A dataset that recalibrates mid-log would not be captured, in either mode.

---

## Images and point clouds: references or copies

123D can store a sensor stream either as a **reference** (a relative path into the original
dataset — `camera_store_option="path"`) or as an **embedded payload** (JPEG/PNG bytes, an MP4
frame index, a LAZ/Draco point cloud). mmdetection3d only reads files, so both have to end up as
paths.

| `--camera-mode` / `--lidar-mode` | Behaviour |
| --- | --- |
| `auto` (default) | reference when the log stores a path, extract otherwise |
| `reference` | always reference the original file; fail loudly if the log embeds the payload |
| `extract` | always decode and write into `--extract-root` |

Referencing costs nothing — no copies, no extra disk, the pickle points straight at the original
dataset files (verified byte-identical to what the native converter emits). It needs the dataset
root resolvable, either from 123D's usual environment variables or via
`--sensor-root nuscenes=/data/nuscenes`.

Extraction is byte-exact where it can be: a log holding whole JPEG or PNG payloads is copied out
verbatim rather than decoded and re-encoded (confirmed byte-identical to the originals on the mini
kit). Point clouds are written as flat float32 `.bin` in the nuScenes `(x, y, z, intensity, ring)`
layout that `LoadPointsFromFile(load_dim=5)` expects. Note that 123D's lossy lidar codecs are
lossy: a `laz`-stored log round-trips to ~5 mm of coordinate quantization. Use `ipc`/`ipc_zstd`
at 123D conversion time if that matters.

**One subtlety worth knowing.** 123D reframes point clouds to the **ego** frame on load, while the
original files it references are in the **sensor** frame. The exporter reconciles this: extracted
points are transformed into whichever frame the export's boxes use, so annotations and points
always agree. `--lidar-frame ego` switches the whole export to the ego frame instead (`lidar2ego`
becomes identity), which is the right choice when referencing files a 123D parser already
reframed.

---

## Cross-dataset training

CoIn3D trains across datasets by converting each into the nuScenes dialect and concatenating them
into one `ann_file` under a shared class list. That is one command here:

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

Three flags carry the weight:

- `--taxonomy general_3cls` — a taxonomy defined over default labels, so every source maps into
  the same three classes.
- `--camera-key camera_id` — keys `info['cams']` by the 123D camera id (`PCAM_F0`) instead of the
  dataset-native name (`CAM_FRONT`), so the camera dictionaries are consistent across datasets.
  The default camera order is already expressed in 123D ids and resolves to mmdet3d's nuScenes
  order on a nuScenes rig.
- `--disjoint-timestamps` — `NuScenesDataset.load_annotations` sorts every info by `timestamp`.
  Datasets recorded in overlapping wall-clock windows would interleave under that sort, shuffling
  logs together and breaking sequence samplers. This spaces the sources apart in exported time
  while leaving intra-log deltas untouched. The exporter checks for interleaving either way and
  warns if it finds any.

Sensor paths are absolute by default, so each source keeps pointing at its own dataset root
regardless of the config's `data_root`.

---

## Frame identifiers and nuScenes evaluation

123D does not carry a dataset's native frame token. It stamps each synchronized frame with a
deterministic UUIDv5 over `(split, log_name, timestamp_us)` — stable across conversions, but not
the nuScenes `sample_token`.

For training that is fine; `token` only has to be unique. For **official nuScenes evaluation it is
not** — `NuScenesEval` keys submissions by `sample_token`. Since the UUID is derived from
`(log_name, timestamp_us)`, the native tokens can be recovered by joining on the same pair:

```bash
py123det-export-mmcv export ... --nuscenes-root /data/nuscenes
```

or, without the devkit at export time:

```bash
py123det-export-mmcv token-map --nuscenes-root /data/nuscenes --out tokens.json
py123det-export-mmcv export ... --token-map tokens.json
```

Also set `--version v1.0-mini` (or `v1.0-trainval`) when you plan to run nuScenes evaluation:
mmdet3d reads `metadata['version']` and looks it up in its `eval_set_map`.

---

## Crossing the Python version boundary

> *Conversion needs py123d (Python ≥ 3.9) while mmcv/mmdet3d is usually pinned to Python 3.8 — is
> writing the pickle in one and reading it in the other a problem?*

Three things can break, and two of them are handled for you.

**1. Pickle protocol.** Protocol 5 is unreadable before Python 3.8. Exports default to protocol 4,
which every Python from 3.4 onwards reads. Raising it warns.

**2. Class identity.** A pickle referencing `pathlib.Path`, a 123D enum, or any custom class only
loads where that class is importable — and would drag py123d into your training environment. The
writer walks the whole payload before writing and refuses anything outside built-ins, numpy arrays,
lists, tuples and string-keyed dicts. Inspecting the produced files confirms it:

```
py123d_infos_val.pkl            globals referenced: ['numpy.core.multiarray._reconstruct', 'numpy.ndarray', 'ndarray.dtype']
py123d_infos_val_portable.pkl   globals referenced: NONE
```

**3. numpy's own pickle format.** This is the one that actually bites. numpy 2 renamed
`numpy.core` to `numpy._core`, so arrays pickled under numpy 2 fail to load under numpy 1 with
`ModuleNotFoundError: No module named 'numpy._core'`. The reverse direction is fine, and mmdet3d
environments are essentially always on numpy 1.x.

Two ways out:

- **Install `numpy<2` in the conversion environment.** Recommended; nothing else changes. The
  exporter detects numpy 2 and prints exactly this advice.
- **`--array-format portable`.** Stores every array as nested Python lists, so the pickle
  references *no* modules at all and loads under any Python and any (or no) numpy. Then use
  `Py123DNuScenesDataset` from `py123detection.mmcv_plugin`, which rebuilds the arrays — including
  the shapes of empty ones, so a frame with no boxes still yields `(0, 7)`:

  ```python
  plugin = True
  plugin_dir = "py123detection/mmcv_plugin/"

  data = dict(train=dict(type="Py123DNuScenesDataset", ann_file="py123d_infos_train.pkl", ...))
  ```

  The round trip is exact (verified bit-identical against the numpy-format export), at the cost of
  a larger file and a load-time conversion. That dataset class also adopts `class_names` from the
  pickle's own taxonomy metadata, so the class list lives in one place.

Every export records `python_version` and `numpy_version` in its metadata, and
`py123det-export-mmcv inspect <pkl>` — run it *in the training environment* — is the one-line
proof that the file survived the hop.

---

## Known differences

Everything below is a consequence of what 123D stores, not of the conversion. Nothing else
differs from the native converter.

**`gt_velocity` — agrees to ~1e-5 m/s, not exactly.** The nuScenes devkit computes box velocity as
a central difference over an annotation's `prev`/`next`; 123D recomputes the same central
difference over the same keyframes at conversion time (`infer_box_dynamics`). Same formula, same
inputs, so the residual is float noise in the timestamp division. Where the devkit returns `NaN`
(at track boundaries) 123D returns a one-sided difference or zero; mmdet3d zeroes NaN at load time
anyway.

**`sweeps` are as dense as the 123D log, not as dense as the dataset.** nuScenes' native sweeps
are 20 Hz; a keyframe-only 123D conversion contains 2 Hz lidar and simply has no intermediate
sweeps to offer. Convert with the interpolated (10 Hz) nuScenes profile if you need denser ones.
Camera-only models do not read sweeps — StreamPETR only uses the *empty* list at a log's first
frame to detect sequence starts, and those line up exactly (asserted in the integration test).

**`num_radar_pts` is always zero.** 123D's `BoxDetectionAttributes` carries lidar point counts but
not radar ones. mmdet3d computes `valid_flag = num_lidar_pts + num_radar_pts > 0`, so the export
differs only for the rare box seen by radar alone. Boxes whose count is unknown (`-1`) are kept
valid rather than silently dropped.

**`visibilities` is empty and `bboxes_ignore` is always empty.** nuScenes visibility bins and a
crowd flag have no 123D equivalent. Neither is read by StreamPETR's training path; both are
exported for schema parity.

**`lidar2ego` is the rig calibration by default**, not a per-frame pose — 123D stores no per-row
lidar extrinsic. See [Static vs. dynamic extrinsics](#static-vs-dynamic-extrinsics) for the
`--lidar2ego-mode dynamic` reconstruction and when it changes anything.

**`sample_data_token` is synthesized** as `"{split}/{log}/{camera}/{timestamp_us}"`. It is only
used at conversion time by the native converter; nothing in the training path reads it.

**Yaw convention.** mmdetection3d's converter reads yaw through `pyquaternion`'s `yaw_pitch_roll`,
which is *not* the textbook z-y-x yaw — it computes `atan2(2(wz − xy), 1 − 2(y² + z²))` where the
standard form has `+xy`. The two agree for an upright box and diverge as roll/pitch grows: a few
milliradians in a lidar frame, a large angle in a camera frame. The export reproduces mmdet3d's
read-out by default so pickles stay interchangeable. `--yaw-convention heading` switches to the
geometric heading the nuScenes *evaluation* devkit assumes.

**Velocity convention.** mmdet3d drops the global vertical velocity before rotating into the lidar
frame, and its evaluation path inverts the same assumption. The export does the same;
`--full-3d-velocity` keeps `vz`, at the cost of letting a tilted lidar mix it into `(vx, vy)`.

---

## Layout

```
src/py123detection/
  sources.py                 Source: a 123D data root + splits -> scenes, with sampling
  taxonomy.py                Taxonomy + presets: 123D labels -> mmdet3d class names
  annotations.py             global-frame records -> lidar-frame gt_boxes / gt_velocity
  geometry.py                SE(3) helpers and the two yaw conventions
  mmcv_export/
    export.py                export_to_mmdet3d: the one-call entry point
    converter.py             MMDet3DConverter: scenes -> info dicts
    annotations2d.py         per-camera 2D / mono-3D annotations (StreamPETR)
    sensors.py               reference vs extract; path resolution
    tokens.py                123D UUID vs native dataset tokens
    writer.py                pickle writing, portability validation, numpy compatibility
    schema.py                the info schema, documented and checkable
    cli.py                   py123det-export-mmcv
  mmcv_plugin/
    dataset.py               Py123DNuScenesDataset (training side; imports mmdet3d lazily)
tools/
  reference_nuscenes_converter.py   ground truth: the mmdet3d/StreamPETR converter, ported
  compare_infos.py                  field-by-field diff of two info pickles
tests/                       132 unit tests + 12 integration tests
```

## Tests

```bash
pytest                                  # 132 unit tests, no data needed
NUSCENES_DATA_ROOT=... PY123D_DATA_ROOT=... pytest -m integration    # the round-trip comparison
```
