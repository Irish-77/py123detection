# Training without mmcv

The mmcv branch exports 123D into the pickle mmdetection3d reads, so existing configs train
unchanged. This branch skips that step: it reads 123D logs directly and trains PETR or BEVFormer
on them with PyTorch and Lightning.

The trade is simple. The mmcv branch buys you every detector already written against mmdetection3d.
This branch buys you a stack you can read: no registries, no config inheritance chains, and a
dataset that hands you tensors instead of a pickle whose provenance you have to reconstruct.

```bash
export PY123D_DATA_ROOT=/data/py123d
py123det-train experiment=nuscenes
```

That is the whole command. Everything it did is in
[`src/py123detection/torch/conf/`](https://github.com/gwb092/py123detection/tree/main/src/py123detection/torch/conf).

## What you can change

Five groups, each a directory of files you can list:

| Group | Choices | What it decides |
| --- | --- | --- |
| `model` | `petr`, `bevformer` | Which detector |
| `taxonomy` | `nuscenes_10cls`, `mixed_8cls`, `mixed2_3cls`, `general_3cls`, `coin3d_3cls`, `vehicle_1cls`, `default_123d` | Which classes, and how each dataset's labels map onto them |
| `images` | `petr_704`, `petr_320` | Input resolution |
| `trainer` | `default`, `smoke`, `ddp` | Epochs, precision, devices |
| `experiment` | `nuscenes`, `nuscenes_mini`, `nuscenes_av2`, `nuscenes_av2_wod`, `carla_nuscenes`, `carla_multirig` | Which datasets, and how many frames of each |

They compose:

```bash
py123det-train experiment=nuscenes_av2 model=bevformer taxonomy=mixed2_3cls trainer=ddp
```

and any single value can be overridden without editing a file:

```bash
py123det-train experiment=nuscenes model.learning_rate=1e-4 dataloader.batch_size=2
```

Each run gets its own directory, named after the model, the dataset mix and the taxonomy —
`bevformer_nuscenes_av2_3cls` — so two runs never overwrite each other's checkpoints.

## Training across datasets

This is what the package is for. Camera rigs differ: nuScenes has six cameras, Argoverse 2 seven,
Waymo five and no rear camera at all. A model that has learned "camera 3 looks right" has learned
something that is false on the next dataset.

```bash
py123det-train experiment=nuscenes_av2_wod taxonomy=mixed2_3cls
```

Three things make that one command rather than a project:

**One taxonomy over every source.** `mixed2_3cls` is defined over 123D's default labels, so each
dataset maps into the same three classes. Pick a taxonomy all your datasets can express — the
exporter guide has [more on choosing one](choosing-what-to-export.md).

**Geometry instead of camera order.** Both detectors work from each camera's calibration, so a rig
with seven cameras is not a special case. For BEVFormer, keep `camera_embedding=none`: the
alternative is the paper's learned per-camera token, which is indexed by camera position and
therefore locks the model to one rig.

**A loss you can read.** A batch may hold frames from several datasets. The loss is computed once
per dataset and weighted by that dataset's share of the batch, which gives the same gradient as
computing it over the whole batch but logs `train_loss_nuscenes` and `train_loss_av2` separately.
When a mixed run goes wrong, that is usually the first place it shows.

:::{note}
Under DDP, the loss normalises by a ground-truth box count reduced across all ranks. That reduction
happens once, before the per-dataset split. Doing it inside the split would have rank 0 reducing
over three groups while rank 1 reduces over two, and the job would hang rather than fail.
:::

## Choosing frames

Each entry under `data.train` and `data.val` picks its frames in two steps. First the logs are
thinned as they are read — `sample_rate_hz: 2.0` (nuScenes' keyframe rate, and the default in
every experiment) or `frame_stride: N`, every Nth frame of the native log; one or the other, not
both. Then `num_frames` optionally takes a fixed number of those, spread evenly over all logs:

```yaml
data:
  train:
    - dataset: av2
      splits: [av2-sensor_train]
      sample_rate_hz: 2.0     # 22477 frames
      num_frames: 10000       # of which every 2.25th, across all 700 logs
```

or from the command line, `+data.train.1.num_frames=10000`. Asking for more than exist keeps them
all and logs a warning.

`num_frames` is also how to set a mix. Training draws uniformly from the concatenation, so each
dataset's share of the batches is its share of the frames: at 2 Hz, nuScenes, Argoverse 2 and
Waymo contribute 28130, 22477 and 31880 frames. Giving each the same `num_frames` weighs them
equally. The thesis's mixed3 strides (3, 10 and 14 frames) came to about 9400, 11000 and 11300.

A frame left without any box after filtering is replaced by the next usable one, so a dataset's
length counts frames, not guaranteed-distinct samples; with vehicle classes on these datasets, the
difference is negligible.

Boxes with no lidar return are dropped as unseen — but only where the log records a count.
Waymo's 123D logs record none (every box reads `-1`), which keeps about 9% of its boxes that no
sensor saw. `filters.lidar_point_cache` points at a cache of counts computed from the lidar and
laid out like `logs/`, and takes the counts from there:

```bash
py123det-train experiment=nuscenes_av2_wod 'filters.lidar_point_cache=${oc.env:PY123D_DATA_ROOT}/num_points_cache'
```

## Learning rate

PETR's and BEVFormer's published schedule, which the thesis also used: 500 steps of linear warmup
from a third of `model.learning_rate`, then a cosine over epochs — constant within an epoch — down
to a thousandth of it. The backbone trains at a tenth of the head's rate throughout.

## Images and augmentation

Every step from sensor pixels to network input is an affine map of pixel coordinates: the rig's
fixed row crop and resize (`rigs.py`), the scale at which the image just covers `images.size`, the
crop window, the horizontal flip. So each camera gets one 3x3 matrix, and it is used twice — once
to warp the image, in a single resampling, and once to update the intrinsics. Pixels and
projection cannot drift apart, which a test checks to a tenth of a pixel through every
augmentation. When the network input is at most half the sensor's resolution, the image is also
decoded at 1/2, 1/4 or 1/8 size, which is several times faster and filters instead of aliasing.

`images` holds only what the network sees: `size`, `mode` (`crop`, PETR's cover-and-cut, or
`stretch`) and `pad_divisor`. Everything random is under `augment`, drawn once per sample and
shared by its cameras:

| Key | Default | Meaning |
| --- | --- | --- |
| `zoom_range` | `[1.0, 1.25]` | Scale relative to the covering one. Validation uses 1. |
| `flip` | `true` | Mirror half the samples |
| `rotation_range` | `[-0.3925, 0.3925]` | Bird's-eye yaw of the world about the ego vehicle, radians |
| `scale_range` | `[0.95, 1.05]` | Bird's-eye scale of the world |

The zoom is relative on purpose. PETR's `resize_lim` is an absolute scale, right for one source
resolution and one output size only — the same `(0.8, 1.0)` at 320x800 trains on objects twice the
size validation shows. A relative zoom means the same thing at every size and for every rig.

Each dataloader worker reseeds its dataset's generator from its own seed and the trainer's rank, so
every epoch sees new augmentations and a repeated run sees the same ones. To look at one exact
sample, pass its draw: `dataset.get(index, Draw(image=ImageDraw(zoom=1.2, flip=True)))`.

Images leave the dataset as `uint8`. The model subtracts the ImageNet mean and divides by the
standard deviation on the GPU (`MultiViewDetector.normalize_images`), which moves a quarter of the
bytes through the dataloader and saves the CPU the costliest step of the old pipeline.

## A controlled comparison

`carla_multirig` is the cleanest test available: the same CARLA scenes recorded through the
nuScenes, Argoverse 2 and Waymo rigs. Because the driving is identical, a difference in the score
is a difference the rig caused.

```bash
py123det-train experiment=carla_multirig taxonomy=mixed2_3cls
```

## Evaluating

```bash
py123det-eval experiment=nuscenes evaluation.checkpoint=work_dirs/petr_nuscenes_10cls/checkpoints/last.ckpt
```

This runs the nuScenes detection metric — the real one, from the devkit's own `algo.py`. The model
predicts in the ego frame and the metric measures in the global frame, so predictions and ground
truth are converted together by the same function.

**You get one score per dataset.** A mixed run writes `<output_dir>/<dataset>/metrics.json` for
each, plus an index at `<output_dir>/metrics.json` and a table on stdout. A single number averaged
over nuScenes, Argoverse 2 and Waymo cannot tell you which of them regressed, which is the same
reason the training loop keeps its validation losses apart. `evaluation.combined=true` additionally
scores every frame together, the number older runs reported.

**Predictions are kept.** Each dataset streams to `<output_dir>/<dataset>/predictions.jsonl` as it
is predicted, one line per frame, with ground truth alongside. Inference therefore holds one frame
rather than the whole split, and a run can be scored again without the model:

```bash
py123det-eval experiment=nuscenes evaluation.rescore_from=work_dirs/petr_nuscenes_10cls/eval \
    evaluation.tp_metrics=nuscenes
```

That is the way to try a different `tp_metrics`, `class_range` or box cap on a finished run; it
needs no checkpoint and no 123D logs.

**Inference shards across GPUs.** `evaluation.devices=4` runs it under a Lightning `Trainer`, each
rank writing `predictions_rank<N>.jsonl` and rank zero merging them into `predictions.jsonl` before
scoring. The merge drops duplicates, which is correctness rather than tidiness: `DistributedSampler`
pads the last batch so every rank sees the same number of samples, handing the first few frames of a
split to two ranks, and scoring a frame twice inflates its hits and its misses together. The number
dropped is logged and recorded in `metrics.json` as `duplicate_frames_dropped`.

```bash
py123det-eval experiment=nuscenes_av2_wod evaluation.checkpoint=... evaluation.devices=4
```

Note `evaluation.precision`, which defaults to `32-true` and is **not** inherited from
`trainer.precision`. Every shipped trainer config trains at `16-mixed`, evaluation has always run in
fp32, and quietly adopting the training precision would move every score.

Two settings usually need attention on non-nuScenes data:

`evaluation.tp_metrics` selects which true-positive errors count. The default here is
`no_attributes`, which drops `attr_err`; only nuScenes stores attributes, and scoring a dataset
without them on a metric that expects them gives every class a `nan` that then propagates into the
NDS. `nuscenes` gives you the official five.

`taxonomy.class_range` is how far out each class is scored. The devkit knows radii for its own ten
classes and nothing else, so a custom taxonomy declares its own — and is refused rather than
guessed at, because that radius silently decides how much of the scene counts.

Two further settings change the score, so both default to leaving it as it was:

`evaluation.score_empty_frames` decides whether frames with no ground truth are scored at all. It
defaults to `false`, because `filters.drop_frames_without_boxes` is on and every number published
from this branch so far was produced that way. Be aware of what the default costs: a frame with no
ground truth that the model puts boxes on is pure false positive, so dropping those frames cannot
help the score and in general flatters it. On the synthetic fixture in `tests/test_evaluate.py` the
difference is mAP 0.2489 against 0.2300. Every metrics file records which setting produced it.

`evaluation.max_boxes_per_sample` caps each frame at its highest-scoring boxes. The devkit caps a
submission at 500; this branch never has, so `null` keeps all 900 queries. Setting it saves memory
and time — the cap is applied before the metric's box objects are built — but it changes scores.

## Checking a change

`trainer=smoke` runs two short epochs, and `experiment=carla_nuscenes` needs no dataset download:

```bash
py123det-train experiment=carla_nuscenes trainer=smoke images=petr_320
```

If that finishes, the data, the model, the loss, the optimiser and the checkpointing all work.

## Memory

Multi-camera detection is memory-hungry: seven cameras at 704×1280 will not fit a batch of two on
an 11 GB card. In rough order of what to try:

```bash
py123det-train experiment=nuscenes images=petr_320                        # fewer pixels
py123det-train experiment=nuscenes model.backbone_checkpointing=true      # recompute the backbone
py123det-train experiment=nuscenes model.head_checkpointing=true          # recompute the decoder
py123det-train experiment=nuscenes trainer.accumulate_grad_batches=4      # a larger effective batch
```

Gradient checkpointing trades roughly a third of the step time for a large part of the activation
memory, which is usually the right trade here.

## Using the pieces directly

Nothing above is required. The dataset is an ordinary `torch.utils.data.Dataset`:

```python
from py123detection import Source
from py123detection.core.taxonomy import get_taxonomy
from py123detection.torch.data import Augmentation, ImagePipeline, Py123DDetectionDataset
from py123detection.torch.data.rigs import get_rig

dataset = Py123DDetectionDataset(
    source=Source(splits=["nuscenes_train"], sample_rate_hz=2.0),
    taxonomy=get_taxonomy("mixed2_3cls"),
    rig=get_rig("nuscenes"),
    images=ImagePipeline(size=(704, 1280)),
    augment=Augmentation(),   # leave out for validation data
    num_frames=10000,
)
sample = dataset[0]
```

and the models are ordinary `LightningModule`s. See the [API reference](../api/index.rst) for
what each one takes.
