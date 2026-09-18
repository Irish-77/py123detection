# Development

## Layout

```
src/py123detection/
  sources.py                 Source: a 123D data root + splits -> scenes, with sampling
  taxonomy.py                Taxonomy + presets: 123D labels -> mmdet3d class names
  annotations.py             global-frame records -> lidar-frame gt_boxes / gt_velocity
  velocity.py                track central differences, for datasets with no annotated velocity
  geometry.py                SE(3) helpers and the two yaw conventions
  mmcv_export/
    export.py                export_to_mmdet3d: the one-call entry point
    converter.py             ExportConfig and the scenes -> info dicts conversion
    annotations2d.py         per-camera 2D / mono-3D annotations (StreamPETR)
    sensors.py               reference vs extract; path resolution
    tokens.py                123D UUID, native dataset tokens, log/timestamp tokens
    writer.py                pickle writing, portability validation, numpy compatibility
    schema.py                the info schema, documented and checkable
    cli.py                   py123det-export-mmcv
  mmcv_plugin/
    dataset.py               Py123DNuScenesDataset (training side; imports mmdet3d lazily)
    evaluation.py            nuScenes-metric evaluation from the pickle's own ground truth
tools/
  reference_nuscenes_converter.py   ground truth: the mmdet3d/StreamPETR converter, ported
  reference_av2_converter.py        the same idea for Argoverse 2, written from the raw files
  compare_infos.py                  field-by-field diff of two info pickles
docs/
  conf.py                    Sphinx configuration for this site
  guides/                    task-oriented pages: what to export, extrinsics, sensors, ...
  api/                       API reference, generated from the docstrings
  other-datasets.md          exporting a dataset that is not nuScenes
tests/                       179 unit tests + 12 integration tests
```

## Tests

```bash
pytest                                  # 179 unit tests, no data needed
NUSCENES_DATA_ROOT=... PY123D_DATA_ROOT=... pytest -m integration    # the round-trip comparison
```

## Building the docs

```bash
pip install -e "py123detection[docs]"
sphinx-build -b dirhtml docs docs/_build              # one-off build, same builder as CI
sphinx-autobuild --builder dirhtml docs docs/_build --port 8000   # live-reloading preview
```

The API pages import the package, so build in an environment that has py123d. mmcv, mmdet3d and
torch are mocked (`autodoc_mock_imports` in `docs/conf.py`), so the training-side classes are
documented without them. The [CLI reference](cli.md) is generated from
`py123detection.mmcv_export.cli.build_parser`, so new flags show up without editing the docs.

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
