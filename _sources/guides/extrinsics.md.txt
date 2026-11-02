# Static vs. dynamic extrinsics

Sensors do not fire at the same instant, and where that offset gets booked is a real modelling
choice. 123D exposes it asymmetrically, which is worth knowing before you pick a mode:

- the **camera** Arrow table stores a per-row, motion-compensated `camera_to_global_se3`, so a
  camera's true pose at its own capture time is always available;
- the **lidar** table stores only timestamps and the payload, with no per-row pose. The only lidar
  extrinsic 123D holds is the static rig calibration in the modality metadata.

Two options control what the export writes. **Both default to `static`, which is what
mmdetection3d's own converter emits** and what the
[native-converter comparison](../validation.md#does-the-output-match-the-native-converter) was run against.

## `--lidar2ego-mode {static,dynamic}`

`static` writes the rig calibration. `dynamic` looks up the ego pose at the *lidar's* own timestamp
and re-derives the extrinsic as
`inv(ego2global_at_frame) @ ego2global_at_lidar_time @ lidar2ego_static`, so
`ego2global @ lidar2ego` reconstructs where the sensor actually was when it fired. `ego2global`
itself is never rewritten.

It can only differ from `static` when the log's ego stream is denser than its frames, because the
correction is reconstructed from that stream. The lookup takes the nearest sample and does not
interpolate, so the export never invents poses the log does not contain. Measured on the mini kit:

| 123D conversion | ego stream | `lidar2ego` shift vs. static | box centre shift |
| --- | --- | --- | --- |
| keyframe (`sync`, 2 Hz) | 2 Hz, one state per frame | 1.5e-15 m (no-op) | 4.5e-13 m |
| async (native rates) | ~20 Hz | mean 0.13 m, **max 0.45 m** | mean 0.14 m, **max 0.60 m** |

On a standard keyframe conversion the flag is inert, because there is nothing closer to look up. On
an async conversion it corrects up to half a metre.

## `--camera2ego-mode {static,dynamic}`

The camera offset is *always* accounted for. This only decides where it is reported.

| | `sensor2ego` | camera `ego2global` |
| --- | --- | --- |
| `static` (default, mmdet3d) | rig calibration | at the camera's own timestamp, so it differs per camera |
| `dynamic` (master-thesis style) | absorbs the offset | the frame's shared pose, identical for all cameras |

Measured on the mini kit: 0.40 m moves out of `ego2global` and into `sensor2ego`, and
**`sensor2lidar` is bit-identical (0.0)** between the two. PETR and StreamPETR only consume
`sensor2lidar`, so they cannot tell the difference. It matters for BEVDet-family configs, which
read `sensor2ego` and `ego2global` separately.

A caveat on both: 123D reads rig calibration once per log (for nuScenes, from the scene's first
frame). A dataset that recalibrates mid-log would not be captured, in either mode.
