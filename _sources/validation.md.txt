# Validation

## Does the output match the native converter?

Does `nuScenes → 123D → pkl` land where `nuScenes → pkl` lands?
[`tools/reference_nuscenes_converter.py`](../tools/reference_nuscenes_converter.py) is a
dependency-light port of the converter StreamPETR and mmdetection3d ship, and
[`tools/compare_infos.py`](../tools/compare_infos.py) diffs the two pickles field by field. Frames are
matched by token, boxes by instance token, and rotations are compared as matrices, so the `q`/`-q`
sign ambiguity cannot hide a difference.

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

Frame counts, box counts, class names, camera order and every sensor path are identical.
`gt_velocity` is the one number that is not bit-exact, for the reason given under
[Known differences](#known-differences).

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

The same comparison has been run at full scale on nuScenes trainval and on Argoverse 2, in both
cases against a converter written independently from the raw files. See
[Other datasets](#other-datasets).

## Other datasets

Nothing in the converter is nuScenes-specific, but a new dataset does force a few choices, and each
one is a flag: reference frame, taxonomy, tokens, velocity, box layout, ego-pose lookup.
[`docs/other-datasets.md`](other-datasets.md) walks through them with Argoverse 2 as the worked
example.

It also covers the reference-converter pattern used to prove an export right
([`tools/reference_av2_converter.py`](../tools/reference_av2_converter.py) plus `compare_infos.py`,
which agree to 3.6e-12 m over 22 473 train and 4 806 val frames), the self-contained evaluator that
scores any export with the nuScenes metric code and no nuScenes database, and the two py123d issues
the comparison surfaced: a camera-sync convention that differs from av2-api, and a 1 µs
ego-timestamp round-trip error.

## Known differences

Everything below is a consequence of what 123D stores, not of the conversion. Nothing else differs
from the native converter.

**`gt_velocity` agrees to ~1e-5 m/s, not exactly.** The nuScenes devkit computes box velocity as a
central difference over an annotation's `prev`/`next`; 123D recomputes the same central difference
over the same keyframes at conversion time (`infer_box_dynamics`). Same formula, same inputs, so
the residual is float noise in the timestamp division. Where the devkit returns `NaN` (at track
boundaries) 123D returns a one-sided difference or zero, and mmdet3d zeroes NaN at load time
anyway.

**`sweeps` are as dense as the 123D log, not as dense as the dataset.** nuScenes' native sweeps are
20 Hz; a keyframe-only 123D conversion contains 2 Hz lidar and has no intermediate sweeps to offer.
Convert with the interpolated (10 Hz) nuScenes profile if you need denser ones. Camera-only models
do not read sweeps. StreamPETR only uses the *empty* list at a log's first frame to detect sequence
starts, and those line up exactly (asserted in the integration test).

**`num_radar_pts` is always zero.** 123D's `BoxDetectionAttributes` carries lidar point counts but
not radar ones. mmdet3d computes `valid_flag = num_lidar_pts + num_radar_pts > 0`, so the export
differs only for the rare box seen by radar alone. Boxes whose count is unknown (`-1`) are kept
valid rather than silently dropped.

**`visibilities` is empty and `bboxes_ignore` is always empty.** nuScenes visibility bins and a
crowd flag have no 123D equivalent. Neither is read by StreamPETR's training path; both are
exported for schema parity.

**`lidar2ego` is the rig calibration by default**, not a per-frame pose, because 123D stores no
per-row lidar extrinsic. See [Static vs. dynamic extrinsics](guides/extrinsics.md) for the
`--lidar2ego-mode dynamic` reconstruction and when it changes anything.

**`sample_data_token` is synthesized** as `"{split}/{log}/{camera}/{timestamp_us}"`. It is only used
at conversion time by the native converter, and nothing in the training path reads it.

**Yaw convention.** mmdetection3d's converter reads yaw through `pyquaternion`'s `yaw_pitch_roll`,
which is *not* the textbook z-y-x yaw: it computes `atan2(2(wz − xy), 1 − 2(y² + z²))` where the
standard form has `+xy`. The two agree for an upright box and diverge as roll/pitch grows, by a few
milliradians in a lidar frame and by a large angle in a camera frame. The export reproduces
mmdet3d's read-out by default so pickles stay interchangeable. `--yaw-convention heading` switches
to the geometric heading the nuScenes *evaluation* devkit assumes.

**Velocity convention.** mmdet3d drops the global vertical velocity before rotating into the lidar
frame, and its evaluation path inverts the same assumption. The export does the same;
`--full-3d-velocity` keeps `vz`, at the cost of letting a tilted lidar mix it into `(vx, vy)`.
