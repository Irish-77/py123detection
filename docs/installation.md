# Install

```bash
pip install -e py123detection
```

Optional extras:

```bash
pip install -e "py123detection[nuscenes]"   # restore native nuScenes sample tokens
pip install -e "py123detection[images]"     # re-encode images when extracting MP4-backed logs
pip install -e "py123detection[torch]"      # py123det-train / py123det-eval, the mmcv-free branch
pip install -e "py123detection[viz]"        # py123det-viz, rendering a run's predictions
```

The exporter needs py123d and numpy. It does not need mmcv, mmdet3d or torch. It is meant to run
in your 123D environment and write a file your training environment reads. See
[Crossing the Python version boundary](guides/python-version-boundary.md).
