# Install

```bash
pip install -e py123detection
```

Optional extras:

```bash
pip install -e "py123detection[nuscenes]"   # restore native nuScenes sample tokens
pip install -e "py123detection[images]"     # re-encode images when extracting MP4-backed logs
```

The exporter needs py123d and numpy. It does not need mmcv, mmdet3d or torch. It is meant to run
in your 123D environment and write a file your training environment reads. See
[Crossing the Python version boundary](guides/python-version-boundary.md).
