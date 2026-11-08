# Development

## Layout

```
src/py123detection/
  core/                      shared by both branches; needs py123d and numpy, nothing else
    sources.py               Source: a 123D data root + splits -> scenes, with sampling
    taxonomy.py              Taxonomy + presets: 123D labels -> class names
    annotations.py           global-frame records -> ego/lidar-frame boxes and velocity
    velocity.py              track central differences, for datasets with no annotated velocity
    geometry.py              SE(3) helpers and the two yaw conventions
    visibility.py            which cameras can see a box, for visibility filtering
    evaluation.py            the nuScenes detection metric, run without the devkit's database
  mmcv/                      branch one: export to the mmdetection3d info pickle
    export/
      export.py              export_to_mmdet3d: the one-call entry point
      converter.py           ExportConfig and the scenes -> info dicts conversion
      annotations2d.py       per-camera 2D / mono-3D annotations (StreamPETR)
      sensors.py             reference vs extract; path resolution
      tokens.py              123D UUID, native dataset tokens, log/timestamp tokens
      writer.py              pickle writing, portability validation, numpy compatibility
      schema.py              the info schema, documented and checkable
      cli.py                 py123det-export-mmcv
    plugin/
      dataset.py             Py123DNuScenesDataset (training side; imports mmdet3d lazily)
      evaluation.py          nuScenes-metric evaluation from the pickle's own ground truth
  torch/                     branch two: train on 123D logs directly, no mmcv
    conf/                    Hydra config groups: model, taxonomy, images, trainer, experiment
    train.py                 py123det-train
    evaluate.py              py123det-eval; ego -> global conversion and the metric call
    build.py                 config -> datasets and model
    data/
      dataset.py             Py123DDetectionDataset: puts the parts below in order
      frames.py              FrameIndex: which frames, and how many (num_frames)
      boxes.py               BoxFilters, BoxLoader: a frame's boxes in the ego frame
      cameras.py             CameraLoader: a rig's images through one affine warp each
      images.py              ImagePipeline: the network input, and the per-camera affine map
      augment.py             Augmentation, Draw: everything random about a training sample
      rigs.py                which cameras each dataset has, and which rows of each to keep
      datamodule.py          concatenate for training, one loader per dataset; worker seeding
      collate.py             batching rigs whose camera counts differ
    models/
      base.py                Detector3D: the Lightning loop, per-dataset loss, optimiser
      multiview.py           MultiViewDetector: the shared backbone over a camera rig
      petr/                  PETR: 3D position embedding on image features
      bevformer/             BEVFormer: an explicit bird's-eye grid
      common/                backbone, losses, grid mask, shared helpers
tools/
  reference_nuscenes_converter.py   ground truth: the mmdet3d/StreamPETR converter, ported
  reference_av2_converter.py        the same idea for Argoverse 2, written from the raw files
  compare_infos.py                  field-by-field diff of two info pickles
  freeze_model_golden.py            freezes the model fixture the equivalence test compares against
  freeze_bevformer_golden.py        the same for BEVFormer, in both camera-embedding modes
  freeze_dataset_golden.py          freezes the dataset fixture, including expected differences
  freeze_evaluation_golden.py       freezes the metric, so an evaluation change proves it moved nothing
docs/
  conf.py                    Sphinx configuration for this site
  guides/                    task-oriented pages: what to export, extrinsics, training, ...
  api/                       API reference, generated from the docstrings
  other-datasets.md          exporting a dataset that is not nuScenes
tests/                       363 unit tests + 28 integration tests
```

## Tests

```bash
pytest                                  # 363 unit tests, no data needed
NUSCENES_DATA_ROOT=... PY123D_DATA_ROOT=... pytest -m integration    # the round-trip comparison
```

`test_model_equivalence.py`, `test_bevformer_equivalence.py` and `test_dataset_equivalence.py`
are the ones to watch when you touch a model or the data pipeline. Each replays fixed input at a
fixed seed and compares against a fixture frozen from the thesis implementation (`tests/golden/`):
the models by parameter count, forward outputs and every loss term, the dataset by every array in
every sample.

`test_evaluate.py` does the same for the metric. Its fixture was frozen from the implementation
that scored every dataset together in one process, so the per-dataset rework had to prove it
changed no score — and two of its checks compare against the previous implementation inlined in the
test file rather than a stored number, which makes them bit-exact rather than rounded.

Regenerate a fixture only when the numbers themselves are meant to change; otherwise a failure
there means the code moved, which is exactly what the fixtures exist to catch.

## Building the docs

```bash
pip install -e "py123detection[docs]"
sphinx-build -b dirhtml docs docs/_build              # one-off build, same builder as CI
sphinx-autobuild --builder dirhtml docs docs/_build --port 8000   # live-reloading preview
```

The API pages import the package, so build in an environment that has py123d. mmcv, mmdet3d and
torch are mocked (`autodoc_mock_imports` in `docs/conf.py`), so the training-side classes are
documented without them. The [CLI reference](cli.md) is generated from
`py123detection.mmcv.export.cli.build_parser`, so new flags show up without editing the docs.

### Serving a build locally

`sphinx-autobuild` is the usual way to look at the docs while writing them. To serve a finished
build instead -- any static file server will do, since the site is plain HTML:

```bash
python -m http.server 8000 --directory docs/_build     # -> http://localhost:8000
```

The `dirhtml` builder writes `validation/index.html` rather than `validation.html`, so pages live
at `http://localhost:8000/validation/`, exactly as they do on the published site.

To reproduce the published layout including the branch prefix (below), build into a per-branch
directory and serve the parent:

```bash
slug="$(.github/scripts/branch-slug.sh "$(git rev-parse --abbrev-ref HEAD)")"
sphinx-build -b dirhtml docs "docs/_site/$slug"
python -m http.server 8000 --directory docs/_site      # -> http://localhost:8000/$slug/
```

On a cluster without a browser, run the server on the login node and forward the port:
`ssh -L 8000:localhost:8000 <login-node>`, then open `http://localhost:8000`.

## Publishing

`.github/workflows/docs.yaml` publishes into the `gh-pages` branch, which GitHub Pages serves
(Settings -> Pages -> *Deploy from a branch* -> `gh-pages` / `(root)`):

| what | when | directory | URL |
| --- | --- | --- | --- |
| `main` | every push touching `docs/`, `src/`, `pyproject.toml` or the workflow files | `/` | <https://irish-77.github.io/py123detection/validation/> |
| any other branch | on demand: Actions -> *docs* -> *Run workflow* -> pick the branch | `/<branch>/` | `https://irish-77.github.io/py123detection/<branch>/validation/` |

Previews are opt-in because a published preview is only worth having when someone else needs to
read it -- for your own iteration the local server above beats a CI round-trip. A preview is a
snapshot of the commit it was built from and says so in a banner naming the branch and short SHA,
so it is mistaken neither for the released docs nor for the branch's current state; re-run the
workflow to refresh it.

The branch directory is the branch name lowercased with URL-unfriendly characters replaced by
`-`. `.github/scripts/branch-slug.sh` is the single definition of that mapping, used by both
workflows and by the local command above.

A preview is removed when its branch is deleted (`docs-cleanup.yaml`), and any that outlive that
event are collected the next time `main` deploys: a root deploy keeps only the directories of
branches that still exist on the remote.

One thing to keep in mind when naming a branch: previews sit next to `main`'s own top-level
pages, so a branch named `guides` or `api` would publish over that page's URL. Anything with a
`feat/`, `fix/` or similar prefix is clear of them.
