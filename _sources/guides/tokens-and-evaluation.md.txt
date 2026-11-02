# Frame identifiers and nuScenes evaluation

123D does not carry a dataset's native frame token. It stamps each synchronized frame with a
deterministic UUIDv5 over `(split, log_name, timestamp_us)`, stable across conversions but not the
nuScenes `sample_token`.

For training that is fine, since `token` only has to be unique. For **official nuScenes evaluation
it is not**: `NuScenesEval` keys submissions by `sample_token`. Since the UUID is derived from
`(log_name, timestamp_us)`, the native tokens can be recovered by joining on the same pair:

```bash
py123det-export-mmcv export ... --nuscenes-root /data/nuscenes
```

or, without the devkit at export time:

```bash
py123det-export-mmcv token-map --nuscenes-root /data/nuscenes --out tokens.json
py123det-export-mmcv export ... --token-map tokens.json
```

Also set `--version v1.0-mini` (or `v1.0-trainval`) when you plan to run nuScenes evaluation:
mmdet3d reads `metadata['version']` and looks it up in its `eval_set_map`.

For datasets with no native token, `--token-style log_timestamp` writes
`"{log_name}/{timestamp_us}"`, which an independent converter can reproduce exactly. That is what
makes a field-by-field diff possible on datasets outside nuScenes.
