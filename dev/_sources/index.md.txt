# py123detection

A [123D](https://kesai.eu/py123d/) toolkit for 3D multi-view object detection.

Two branches share the 123D data model:

| Branch | Module | Status |
| --- | --- | --- |
| **mmcv export**: turn any 123D dataset into the `info` pickle the mmcv / mmdetection3d ecosystem reads, so existing PETR / StreamPETR / BEVDet configs train on 123D data unchanged | `py123detection.mmcv_export`, plus `py123detection.mmcv_plugin` on the training side | implemented |
| **modern stack**: the mmcv-free path from the master thesis | | not started |

## Why this exists

123D already normalizes a dozen driving datasets into one schema. mmdetection3d natively supports exactly one data format: the nuScenes `info` pickle. Every camera-based detector built on it (PETR,
StreamPETR, BEVDet, CoIn3D) reads that specific data format, and for example CoIn3D's cross-dataset training works by converting Waymo and Lyft into it and concatenating the results.

One converter from 123D to that exact data format makes every 123D dataset trainable with off-the-shelf
configs, individually or mixed.

## How it fits together

There are two routes from a raw dataset to the pickle a detector trains on. The first needs a
hand-written converter for every dataset. The second needs none, because the per-dataset work
already happened when 123D was built.

```mermaid
flowchart TD
    RAW["<b>Raw dataset</b><br/>nuScenes, Argoverse 2, Waymo, ..."]

    RAW --> NATIVE["<b>mmdetection3d converter</b><br/><code>tools/create_data.py</code><br/><i>one per dataset, hand-written</i>"]
    RAW --> CONV["<b>123D conversion</b><br/><code>py123d-conversion</code>"]

    NATIVE --> PKLA["<b>info pickle</b>"]
    CONV --> LOGS["<b>123D logs</b><br/>Arrow tables, one schema<br/>for every dataset"]
    LOGS --> EXPORT["<b>py123detection exporter</b><br/><code>py123det-export-mmcv</code><br/><i>one converter, every dataset</i>"]
    EXPORT --> PKLB["<b>info pickle</b>"]

    PKLA --> CFG["mmdet3d config<br/><code>ann_file=...</code>"]
    PKLB --> CFG
    CFG --> MODEL["PETR / StreamPETR /<br/>BEVDet / CoIn3D"]

    style PKLA fill:#eef4fb,stroke:#2a78d6
    style PKLB fill:#eef4fb,stroke:#2a78d6
    style LOGS fill:#fdf1ea,stroke:#eb6834
    style EXPORT fill:#fdf1ea,stroke:#eb6834
```

The two pickles are interchangeable. That claim is measured, not asserted: see
[Does the output match the native converter?](validation.md#does-the-output-match-the-native-converter).

Inside the 123D route, these are the pieces you actually name in your own code or on the command
line. Everything else is internal.

```mermaid
flowchart TD
    LOGS["<b>123D logs</b>"] --> SRC

    subgraph EXPORTENV ["conversion environment: py123d + numpy"]
        SRC["<b>Source</b><br/><code>py123detection</code><br/>which logs, which splits,<br/>what frame rate"]
        TAX["<b>Taxonomy</b><br/><code>py123detection</code><br/>123D labels to class names<br/><i>presets: nuscenes_detection_10cls,<br/>general_3cls, coin3d_3cls, ...</i>"]
        CONF["<b>ExportConfig</b><br/><code>py123detection.mmcv_export</code><br/>cameras, reference frame, box layout,<br/>sweeps, velocity, tokens, paths"]
        RUN["<b>export_to_mmdet3d()</b><br/><code>py123detection.mmcv_export</code>"]
        REP["<b>ExportReport</b><br/>frame / box / class counts,<br/>skips, warnings"]

        SRC --> RUN
        TAX --> CONF
        CONF --> RUN
        RUN --> REP
    end

    RUN --> PKL["<b>info pickle</b><br/>infos + metadata"]

    subgraph TRAINENV ["training environment: mmcv / mmdet3d"]
        STOCK["<b>NuScenesDataset</b><br/>stock mmdet3d,<br/>nothing to install"]
        PLUG["<b>Py123DNuScenesDataset</b><br/><code>py123detection.mmcv_plugin</code><br/>portable arrays, taxonomy classes,<br/>evaluation without a nuScenes DB"]
    end

    PKL --> STOCK
    PKL --> PLUG

    style PKL fill:#eef4fb,stroke:#2a78d6
    style LOGS fill:#fdf1ea,stroke:#eb6834
```

The CLI is the same three objects behind a flag parser: `py123det-export-mmcv export` builds a
`Source` and an `ExportConfig` from its arguments and calls `export_to_mmdet3d`.

```{toctree}
:hidden:
:caption: Getting started

installation
quickstart
examples
```

```{toctree}
:hidden:
:caption: Guides

guides/choosing-what-to-export
guides/extrinsics
guides/sensor-payloads
guides/cross-dataset-training
guides/tokens-and-evaluation
guides/python-version-boundary
other-datasets
```

```{toctree}
:hidden:
:caption: Reference

validation
cli
api/index
```

```{toctree}
:hidden:
:caption: Development

contributing
development
```
