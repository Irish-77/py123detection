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

Ground truth is filled orange, predictions are hollow blue outlines on top, and a prediction's
outline fades to grey as its confidence drops. Filled against hollow is what separates them at a
thumbnail size; colour alone does not. An orange box with nothing on it is a miss, a blue box with
nothing under it is a false positive.

The layout is the thesis's, so figures from the two are comparable: forward is up, left is left,
range rings at 10, 25 and 50 m, and a tick from each box's centre to its leading edge for heading.
Boxes whose centre falls outside the window are dropped rather than clipped.

This needs nothing but the file. The boxes in it are global, and the view puts them back into the
ego frame — a file written before the ego pose was recorded still renders, aligned to global north
instead, and says so on the axes.

Useful switches:

| | |
| --- | --- |
| `--score-threshold` | default 0.3. A detector emits one box per query — 900 for the shipped configs — so an unfiltered view is a solid block and shows nothing. |
| `--limit` | how many frames to render, default 20. |
| `--dataset` | only frames of one dataset, for a file covering several. |
| `--radius` | half-width of the view in metres, default 50. |
| `--taxonomy` | name classes in the legend. Either spelling works: the config group (`nuscenes_10cls`) or the registry name (`nuscenes_detection_10cls`). |

## The front camera beside it

```bash
py123det-viz frame .../predictions.jsonl -o frames/ --experiment nuscenes_av2_wod
```

One figure per frame: the front camera on the left, the bird's-eye view on the right, same colours
in both. This is the thesis's qualitative figure and the most useful single picture of a detector —
the camera panel says what the model was looking at, the BEV panel says where it put the box. A
miss that is invisible in the image and a miss that is plainly visible look identical in a metric
and nothing alike here.

`--camera` picks a different camera of the rig; 0, the default, is the front one in every rig.
Like `cameras` below, it needs the 123D logs, so it takes the same `--experiment` and `-O`.

## On every camera image

```bash
py123det-viz cameras .../predictions.jsonl -o frames/ --experiment nuscenes_av2_wod
```

The same boxes on every camera of the rig, tiled into one picture per frame. Use this instead of
`frame` when the interesting object is not in front.

This one additionally needs the 123D logs the run was scored on, because the file holds boxes and
not pixels — hence `--experiment`, which composes the same dataset the evaluation used so the
frames line up with the ones that were scored. Frames in the file that the dataset does not
contain are reported and skipped rather than guessed at.

An edge is drawn only where both of its corners project in front of the camera and inside the
frame, so a box leaving the view appears partly drawn. That is deliberate: joining it to a clamped
corner would draw a shape the model never predicted.

`-O` passes any further Hydra override through to that composition, which a run scored on anything
but a stock config needs: `-O 'hydra.searchpath=[file:///path/to/conf]'` to reach an experiment
that lives outside the package, and `-O 'data.val=[...]'` to rebuild the exact frames it scored.
A frame whose own filters reject it when rebuilt is skipped and named, rather than drawn on the
images of whichever frame the dataset serves in its place.

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

## In the interactive viewer

```bash
py123det-view work_dirs/petr_nuscenes_10cls/eval/nuscenes/predictions.jsonl --split nuscenes_val
```

Opens py123d's Viser viewer — lidar, map, camera frustums, the log's own detections, playback —
with one element added: the model's boxes for the frame on screen, with a score slider and an
optional overlay of the run's *filtered* ground truth. Serve it on `localhost` and reach it through
an SSH tunnel when the data is on a cluster; `--host 0.0.0.0` opens it to the network instead.

There is no second viewer here, deliberately. py123d's is several thousand lines of maintained
code, and `PredictionsElement` extends it through the `_build_elements` hook py123d provides for
exactly this, so everything else in the picture stays py123d's and keeps working as py123d
improves it.

Two things have to line up, and both are checked rather than assumed. The scenes are looked up by
the uuid in each token, and one the data root does not hold is reported. The frame *numbering* is
the subtle one: a token is `<scene uuid>:<iteration>`, and the iteration counts frames of the
scene **as the evaluation built it** — a run scored at `frame_stride=12` numbered every twelfth
frame 0, 1, 2, so viewing it at the native rate would put frame 1's boxes on the log's second
frame. Pass the same `--frame-stride` the run used. Where the numbering cannot fit the scene the
command says so and stops; where it merely might not, nothing can tell, which is why the flag is
documented as the evaluation's and not the log's.

## From py123d's own types

The conversion underneath is a public function, so a notebook or another part of py123d's
visualisation stack can use it directly:

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
plays in order instead of jumping from frame 1 to frame 10. `add_to_viser` takes an `origin`:
py123d draws a scene relative to its centre, because a 123D scene sits at global coordinates large
enough to cost visible precision, and an overlay that skips this lands hundreds of metres from
everything else.

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
