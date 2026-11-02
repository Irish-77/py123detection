# Crossing the Python version boundary

Conversion needs py123d (Python ≥ 3.9) while mmcv/mmdet3d is usually pinned to Python 3.8. Writing
the pickle in one and reading it in the other can break in three ways, two of which are handled for
you.

**1. Pickle protocol.** Protocol 5 is unreadable before Python 3.8. Exports default to protocol 4,
which every Python from 3.4 onwards reads. Raising it warns.

**2. Class identity.** A pickle referencing `pathlib.Path`, a 123D enum, or any custom class only
loads where that class is importable, and would drag py123d into your training environment. The
writer walks the whole payload before writing and refuses anything outside built-ins, numpy arrays,
lists, tuples and string-keyed dicts. Inspecting the produced files confirms it:

```
py123d_infos_val.pkl            globals referenced: ['numpy.core.multiarray._reconstruct', 'numpy.ndarray', 'ndarray.dtype']
py123d_infos_val_portable.pkl   globals referenced: NONE
```

**3. numpy's own pickle format.** This is the one that actually bites. numpy 2 renamed `numpy.core`
to `numpy._core`, so arrays pickled under numpy 2 fail to load under numpy 1 with
`ModuleNotFoundError: No module named 'numpy._core'`. The reverse direction is fine, and mmdet3d
environments are essentially always on numpy 1.x.

Two ways out:

- **Install `numpy<2` in the conversion environment.** Recommended, and nothing else changes. The
  exporter detects numpy 2 and prints exactly this advice.
- **`--array-format portable`.** Stores every array as nested Python lists, so the pickle references
  no modules at all and loads under any Python and any (or no) numpy. Then use
  `Py123DNuScenesDataset` from `py123detection.mmcv_plugin`, which rebuilds the arrays, including
  the shapes of empty ones, so a frame with no boxes still yields `(0, 7)`:

  ```python
  plugin = True
  plugin_dir = "py123detection/mmcv_plugin/"

  data = dict(train=dict(type="Py123DNuScenesDataset", ann_file="py123d_infos_train.pkl", ...))
  ```

  The round trip is exact, verified bit-identical against the numpy-format export, at the cost of a
  larger file and a load-time conversion. That dataset class also adopts `class_names` from the
  pickle's own taxonomy metadata, so the class list lives in one place.

Every export records `python_version` and `numpy_version` in its metadata. Running
`py123det-export-mmcv inspect <pkl>` *in the training environment* shows whether the file survived
the hop.
