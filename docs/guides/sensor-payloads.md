# Images and point clouds: references or copies

123D can store a sensor stream either as a **reference** (a relative path into the original
dataset, `camera_store_option="path"`) or as an **embedded payload** (JPEG/PNG bytes, an MP4 frame
index, a LAZ/Draco point cloud). mmdetection3d only reads files, so both have to end up as paths.

| `--camera-mode` / `--lidar-mode` | Behaviour |
| --- | --- |
| `auto` (default) | reference when the log stores a path, extract otherwise |
| `reference` | always reference the original file; fail loudly if the log embeds the payload |
| `extract` | always decode and write into `--extract-root` |

Referencing costs nothing. No copies, no extra disk, and the pickle points straight at the original
dataset files, verified byte-identical to what the native converter emits. It needs the dataset
root resolvable, either from 123D's usual environment variables or via
`--sensor-root nuscenes=/data/nuscenes`.

Extraction is byte-exact where it can be. A log holding whole JPEG or PNG payloads is copied out
verbatim rather than decoded and re-encoded, confirmed byte-identical to the originals on the mini
kit. Point clouds are written as flat float32 `.bin` in the nuScenes `(x, y, z, intensity, ring)`
layout that `LoadPointsFromFile(load_dim=5)` expects. 123D's lossy lidar codecs stay lossy: a
`laz`-stored log round-trips to about 5 mm of coordinate quantization. Use `ipc`/`ipc_zstd` at 123D
conversion time if that matters.

One subtlety worth knowing. 123D reframes point clouds to the **ego** frame on load, while the
original files it references are in the **sensor** frame. The exporter reconciles this: extracted
points are transformed into whichever frame the export's boxes use, so annotations and points
always agree. `--lidar-frame ego` switches the whole export to the ego frame instead (`lidar2ego`
becomes identity), which is the right choice when referencing files a 123D parser already reframed.
