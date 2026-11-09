# Looking at what the model found

An evaluation gives you one number per dataset. These views tell you why it is that number.

Both read the `predictions.jsonl` that `py123det-eval` writes next to each dataset's metrics, so
neither needs a checkpoint or a GPU. Install the extra first:

```bash
pip install -e "py123detection[viz]"
```

## From above

```bash
py123det-viz bev work_dirs/petr_nuscenes_10cls/eval/nuscenes/predictions.jsonl -o frames/
```

Ground truth in grey, predictions in red, and each prediction's opacity is its confidence — so a
dense frame still reads, and the boxes you can barely see are the ones the model was not sure
about. A grey box with nothing on it is a miss; a red box with nothing under it is a false
positive. The title says how many predictions fell below the threshold, because a count of what
you are *not* seeing is part of reading the picture.

This needs nothing but the file. The boxes in it are in the global frame, and the view puts them
back into the ego frame so the vehicle sits at the origin facing +x — a file written before the ego
pose was recorded still renders, with the axes aligned to global north instead, and says so.

Useful switches:

| | |
| --- | --- |
| `--score-threshold` | default 0.3. A detector emits one box per query — 900 for the shipped configs — so an unfiltered view is a solid block and shows nothing. |
| `--limit` | how many frames to render, default 20. |
| `--dataset` | only frames of one dataset, for a file covering several. |
| `--radius` | half-width of the view in metres, default 60. |
| `--taxonomy` | name classes in the legend. Either spelling works: the config group (`nuscenes_10cls`) or the registry name (`nuscenes_detection_10cls`). |

## On the camera images

```bash
py123det-viz cameras .../predictions.jsonl -o frames/ --experiment nuscenes_av2_wod
```

The same boxes projected onto the images they were predicted from, one tiled picture per frame.
Between the two views you can separate the two failures that look identical in a metric: the object
was invisible, or it was visible and the model missed it.

This one additionally needs the 123D logs the run was scored on, because the file holds boxes and
not pixels — hence `--experiment`, which composes the same dataset the evaluation used so the
frames line up with the ones that were scored. Frames in the file that the dataset does not
contain are reported and skipped rather than guessed at.

An edge is drawn only where both of its corners project in front of the camera and inside the
frame, so a box leaving the view appears partly drawn. That is deliberate: joining it to a clamped
corner would draw a shape the model never predicted.

## From Python

The rendering is ordinary functions, so a notebook or a script can use them directly:

```python
from py123detection.torch.predictions import read_predictions
from py123detection.torch.visualize import render_bev, render_camera

for frame in read_predictions("eval/nuscenes/predictions.jsonl"):
    figure = render_bev(frame, score_threshold=0.4)
    ...
```

`render_camera(image, imu2img, predictions, ground_truth=..., scores=...)` draws onto one camera
image and returns a new array; `tile` lays a rig's cameras out in a grid, padding rather than
assuming they share a resolution, because Argoverse 2's do not.

## In py123d's viewer

There is no viewer in this package, deliberately: py123d already ships one
(`py123d.visualization.viser`), and a second would be a copy of several thousand lines of somebody
else's maintained code. What this package provides is the piece that was missing — the conversion
from a predictions file to the detection type that viewer draws.

```python
from py123detection.core.taxonomy import get_taxonomy
from py123detection.torch.visualize import add_to_viser, frames_by_token, tokens_in_order, to_detections

taxonomy = get_taxonomy("general_3cls")
frames = frames_by_token("eval/nuscenes/predictions.jsonl")

# As py123d's own type, for anything in its visualisation stack:
detections = to_detections(frames["<scene uuid>:118"], taxonomy, score_threshold=0.4)

# Or straight onto a running viser server, as a wireframe overlay:
handle = add_to_viser(server, frames["<scene uuid>:118"], score_threshold=0.4)
```

`tokens_in_order` sorts tokens by frame number rather than as text, so stepping through a scene
plays in order instead of jumping from frame 1 to frame 10.

One limitation worth knowing before you read the colours. py123d's label vocabulary is coarse on
purpose — `VEHICLE` and `TWO_WHEELER`, not car, truck, bus, trailer, bicycle, motorcycle — so a
fine-grained taxonomy cannot round-trip through it. `mixed_8cls` loses four of its eight classes
and `nuscenes_detection_10cls` loses all ten, because it is built from `native_map`, which does not
name py123d labels at all. Those boxes still draw, as `GENERIC_OBJECT`; what they lose is being
told apart by colour. The conversion logs which classes that applies to, once per taxonomy, rather
than letting you discover it from a picture. `unmapped_classes(taxonomy)` answers the same question
up front.

## A note on matplotlib

The `viz` extra needs matplotlib 3.8 or newer. Older versions raise on `np.Inf`, which NumPy 2.0
removed, inside matplotlib's own title positioning — `envs/modern` on the cluster pairs matplotlib
3.5.3 with numpy 2.4.6 and cannot save a titled figure. If a render fails with that error, the
matplotlib in your environment is too old for its numpy.
