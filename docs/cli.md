# CLI reference

`py123det-export-mmcv` has three subcommands: `export` writes the pickle, `inspect` prints its
metadata and first frame, and `token-map` dumps nuScenes sample tokens. This page is generated from
the argument parser, so it always matches the installed version.

```{argparse}
:module: py123detection.mmcv.export.cli
:func: build_parser
:prog: py123det-export-mmcv
```

## py123det-viz

Rendering a run's predictions. See [Looking at what the model found](guides/visualisation.md).

```{argparse}
:module: py123detection.torch.visualize.cli
:func: build_parser
:prog: py123det-viz
```
