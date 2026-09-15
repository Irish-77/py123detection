# Exporting other 123D datasets (and proving the export right)

`py123det-export-mmcv` was validated first on nuScenes, where the target format was born, and
then on Argoverse 2 (sensor), where nothing about the source looks like nuScenes: 7 cameras with
one portrait view, no native frame token, no annotated velocity, annotations in the ego frame,
a 10 Hz log rate, and 30 categories. This page records what it took, so that the next dataset
(Waymo, KITTI-360, PandaSet, ...) is a checklist rather than a research project.

The short version: **no converter code changes per dataset.** The export is driven entirely by
flags; what *is* dataset-specific is (1) the taxonomy, (2) a handful of convention choices, and
(3) if you want proof, a small reference converter written from the raw data.

---

## 1. What is already dataset-agnostic

| Concern | Mechanism | Why it generalises |
|---|---|---|
| Which logs | `--split`, `--dataset`, `--log-name`, `--max-logs` | 123D's own split registry |
| Which cameras, in which order | `--camera-order` with 123D `CameraID` names (`PCAM_F0,PCAM_R0,...`) | ids, not dataset channel names |
| Camera dictionary keys | `--camera-key native` (dataset names) or `camera_id` (123D ids, for merged exports) | |
| Where images/point clouds live | `--sensor-root <dataset>=<path>` | 123D logs store paths *relative* to the dataset root |
| Class mapping | `--taxonomy` over 123D's `DefaultBoxDetectionLabel` (see §3) | every parser implements `to_default()` |
| Frame rate | `--frame-stride N` / `--sample-rate-hz` | applied to the log's native sync table |
| Sweeps | `--max-sweeps` from the native-rate stream even when the export is subsampled | |
| Frame tokens | `--token-style log_timestamp` (§4) | no native token needed |
| Velocity | `--velocity-source tracks` (§5) | derived from tracks, not read from the dataset |
| Reference frame | `--lidar-frame sensor|ego` (§2) | |
| Box layout | `--box-layout streampetr|mmdet3d_0.17` (§6) | trainer-, not dataset-specific |
| Ego pose | `--ego-pose-source sync|nearest` (§7) | |

Everything the pickle needs is read through the 123D scene API, never from the raw dataset.

---

## 2. Reference frame: `--lidar-frame`

mmdetection3d's nuScenes convention expresses boxes and camera extrinsics in the frame of the
top lidar, because nuScenes point-cloud files are stored in that frame. Argoverse 2 (like Waymo)
stores sweeps and annotations in the **egovehicle frame**. Exporting AV2 with the default
`--lidar-frame sensor` would put the boxes in the `up_lidar` frame while the referenced point
clouds stay in the ego frame — harmless for a camera-only model, wrong for anything that reads
the points. Use `--lidar-frame ego`: `lidar2ego` becomes the identity, `sensor2lidar` becomes
camera-to-ego, and the boxes are the annotation rows verbatim.

Rule of thumb: match the frame the dataset's point clouds are stored in.

---

## 3. Taxonomy

A `Taxonomy` maps a 123D label to one of an ordered list of class names, or drops it. Two levels:

* **default level** — keys are `DefaultBoxDetectionLabel` names (`VEHICLE`, `PERSON`,
  `TWO_WHEELER`, ...). A taxonomy written only at this level applies to *every* 123D dataset,
  which is what cross-dataset training wants.
* **native level** — keys are dataset enum member names, either bare (`VEHICLE_CAR`) or
  **qualified** with the enum class (`AV2SensorBoxDetectionLabel.BICYCLIST`). Qualified keys
  let one taxonomy carve out per-dataset exceptions without touching same-named members of
  another dataset's enum. Lookup order: qualified native → bare native → default.

`COIN3D_3CLS` is the worked example: CoIn3D's vehicle / pedestrian / two-wheel classes, spelled
`car` / `pedestrian` / `motorcycle` because the nuScenes devkit's `DetectionBox` only accepts
nuScenes class names (§8). It is a default-level taxonomy plus qualified exceptions: AV2's
rider boxes (`BICYCLIST`, `MOTORCYCLIST`, `WHEELED_RIDER`) are dropped because nuScenes — and so
CoIn3D — has no rider class; the rider is part of the two-wheeler box.

To add a taxonomy: append a `Taxonomy(...)` to `py123detection/taxonomy.py` and register it in
`TAXONOMIES`; it becomes a `--taxonomy` choice. Keep the semantics written down in its docstring
— that is what a reviewer will read.

---

## 4. Tokens: `--token-style log_timestamp`

123D stamps frames with a deterministic UUID. nuScenes exports restore the native
`sample_token` (`--nuscenes-root` / `--token-map`) because the official evaluation keys on it.
Most datasets have no such token, and a UUID is impossible for an independent converter to
reproduce. `--token-style log_timestamp` writes `"{log_name}/{timestamp_us}"` instead: unique,
readable, and trivially reproducible, so `tools/compare_infos.py` can align frames.

---

## 5. Velocity: `--velocity-source tracks`

PETR-family models regress velocity and the nuScenes metric scores it (mAVE). Datasets that do
not annotate it (AV2, KITTI-360) leave zeros in the 123D log, which trains the head to predict
zero and makes mAVE meaningless. `--velocity-source tracks` derives velocity per box as the
central difference of the global box centre over the neighbouring annotations of the same track
at the log's native rate (one-sided at track ends, zero without a neighbour within
`--velocity-max-dt`, default 0.25 s). That is the nuScenes devkit's `box_velocity` rule, so the
result is what mmdetection3d would have written had the dataset been nuScenes. The rule is
specified in `py123detection/velocity.py` precisely so a reference converter can reproduce it.

---

## 6. Box layout: `--box-layout`

Two incompatible `gt_boxes` layouts exist in the mmdet3d ecosystem:

| layout | columns | who reads it |
|---|---|---|
| `streampetr` (default) | `[x, y, z, l, w, h, yaw]` | StreamPETR, mmdetection3d ≥ 1.0 |
| `mmdet3d_0.17` | `[x, y, z, w, l, h, -yaw - π/2]` | mmdetection3d 0.17's own converter and dataset — PETR v1's pin |

Choosing the wrong one does not crash; it trains on transposed, mis-rotated boxes. Match the
trainer, and if in doubt diff against the trainer's own converter output.

---

## 7. Ego pose: `--ego-pose-source nearest`

By default a frame's `ego2global` is the ego-state row the log's sync table associated with the
frame. The sync table is built as "first row at or after the reference timestamp", so it relies
on the ego stream carrying a row at *exactly* the frame timestamp. On py123d 0.6.0's AV2 logs it
does not: the parser round-tripped nanosecond stamps through float64 (`DataFrame.iterrows`),
~5% of ego rows came back 1 µs early, and for ~40% of frames the sync table skipped the exact row
and took the next one, 2–10 ms later — up to 9 cm of ego pose on a city log, propagated into
every box and camera extrinsic of the frame. `--ego-pose-source nearest` looks the ego state up
by the frame's timestamp and is immune. The parser is fixed in this tree
(`py123d/parser/av2/av2_sensor_parser.py`, `itertuples`), but already-converted logs keep the
old stamps, so use the flag.

The flag is cheap and never wrong when an exact row exists, so it is a sensible default for any
dataset whose ego stream is denser than its frames.

---

## 8. Proving an export: the reference-converter pattern

A diff needs two independent implementations. For nuScenes the second one was mmdetection3d's
own converter. For a dataset without one, write a **reference converter** from the raw files —
`tools/reference_av2_converter.py` is the template:

* read the raw tables directly (feather / JSON), no 123D, no py123detection imports;
* implement the same conventions the export is configured with (frame selection, sync rule,
  reference frame, velocity rule, class map, tokens, box layout) — each one documented in the
  module docstring, so a reviewer can check both sides against one description;
* write the same `metadata` keys (`class_names`, `box_layout`, ...).

Then `tools/compare_infos.py REF.pkl EXPORT.pkl --no-2d` reports the maximum absolute deviation
of every field. On AV2 val the export matches the reference to `1.8e-12 m` on box centres and
camera extrinsics, `0` on intrinsics, sizes, timestamps, paths and point counts, `9e-12 m/s` on
velocity — i.e. float round-off of the two implementations' matrix products. Anything at `1e-3`
or above is a convention mismatch, and the deviation table tells you which one.

Two 123D-side issues surfaced this way on AV2 and are worth knowing about:

1. **Camera sync convention.** py123d pairs each lidar sweep with the *first* image after the
   sweep start (`forward`, +40 ms on AV2); av2-api's `SensorDataloader` pairs the *nearest*
   image (−10 ms). The reference converter implements both (`--camera-matching`), so the
   provenance test (`forward` vs export) and the convention effect (`nearest` vs `forward`) are
   separable.
2. **The 1 µs ego timestamp error** of §7.

---

## 9. Training side

`py123detection.mmcv_plugin.Py123DNuScenesDataset` reads any export and adds three things to
mmdetection3d's `NuScenesDataset`:

* **PETR-family keys** — `get_data_info` also returns `intrinsics`, `extrinsics` (lidar-to-camera)
  and `img_timestamp`, the additions PETR's and StreamPETR's `CustomNuScenesDataset` make, so
  their configs can point at this dataset directly.
* **Self-contained evaluation** — `eval_mode="self_contained"` (default) builds the nuScenes
  devkit's GT boxes from the pickle itself and runs the devkit's metric code, so no nuScenes
  database and no official split are needed. GT goes through the same
  `output_to_nusc_box` → `lidar_nusc_box_to_global` path as predictions, so the box layout cannot
  be misread on one side only. Differences from the official protocol: no bike-rack filter, ego
  distance from the pickle's `ego2global_translation`, and — because exports carry no attributes —
  `mAAE = 1.0` by construction (CoIn3D's Lyft and Waymo tables show the same). `eval_class_range`
  overrides the per-class radius. Feeding the GT back as predictions must give mAP 1.0 and zero
  TP errors; the AV2 dry run does.
* `eval_mode="nuscenes"` restores the stock devkit path for nuScenes exports with native tokens.

**Constraint:** the devkit's `DetectionBox` asserts that class names are nuScenes detection names
(`car`, `truck`, `bus`, `trailer`, `construction_vehicle`, `pedestrian`, `motorcycle`,
`bicycle`, `traffic_cone`, `barrier`), and mmdet3d's `_format_bbox` looks attributes up by those
names. A taxonomy meant for this evaluator must spell its classes with them — which is exactly
why CoIn3D did.

The training environment does not need py123d: `import py123detection` degrades to
`HAS_EXPORT_SIDE = False` and `py123detection.mmcv_plugin` keeps working. Put
`py123detection/src` on `PYTHONPATH` and add to the config:

```python
custom_imports = dict(imports=['py123detection.mmcv_plugin'], allow_failed_imports=False)
dataset_type = 'Py123DNuScenesDataset'
```

### Mixed image sizes (AV2's portrait front camera)

mmdetection3d's `LoadMultiViewImageFromFiles` `np.stack`s the views and PETR's
`ResizeCropFlipImage` derives resize dims from the config's `H, W`, so a rig with one portrait
camera either crashes or gets that view squashed. PETR's plugin in this tree gains two classes
(`projects/mmdet3d_plugin/datasets/pipelines/`): `LoadMultiViewImageFromFilesMixedSize` keeps the
views as a list, and `ResizeCropFlipImagePerView` applies one shared scale to each view's true
size, crops bottom-aligned and right-pads the narrow view — what Far3D's AV2 pipeline does. On a
uniform rig they reproduce the stock behaviour, so they are safe defaults for any dataset. See
`PETR/projects/configs/petr/petr_r50dcn_gridmask_p4_av2.py` for the full recipe.

---

## 10. Worked example: Argoverse 2

```bash
export PY123D_DATA_ROOT=/data/py123d
py123det-export-mmcv export \
  --split av2-sensor_val --frame-stride 5 \
  --taxonomy coin3d_3cls \
  --camera-order PCAM_F0,PCAM_R0,PCAM_L0,PCAM_L1,PCAM_R1,PCAM_L2,PCAM_R2 \
  --lidar-frame ego --token-style log_timestamp --box-layout mmdet3d_0.17 \
  --velocity-source tracks --ego-pose-source nearest \
  --max-sweeps 10 --no-2d-annotations \
  --sensor-root av2-sensor=/data/av2/sensor --version av2-sensor \
  --out /data/av2/py123d_av2_infos_val.pkl

python tools/reference_av2_converter.py --av2-root /data/av2/sensor --split val \
  --camera-matching forward --out /data/av2/av2_infos_val.pkl
python tools/compare_infos.py /data/av2/av2_infos_val.pkl /data/av2/py123d_av2_infos_val.pkl --no-2d
```

Checklist for the next dataset:

1. One-log export with `--max-logs 1`, then `py123det-export-mmcv inspect` — check cameras,
   frame count, class histogram, velocity statistics.
2. Decide `--lidar-frame` from where the point clouds live (§2).
3. Pick or write the taxonomy (§3); spell classes with nuScenes names if you want the
   self-contained evaluator (§9).
4. `--token-style log_timestamp`, `--velocity-source tracks` if the dataset annotates none,
   `--ego-pose-source nearest`, the trainer's `--box-layout`.
5. Write the reference converter from the raw files, diff, and expect `≤1e-9` everywhere; chase
   anything larger to a convention.
6. Train with `Py123DNuScenesDataset`; if the rig mixes image sizes, use the two PETR pipeline
   classes.
