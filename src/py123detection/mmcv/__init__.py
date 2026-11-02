"""The mmcv / mmdetection3d branch.

:mod:`py123detection.mmcv.export` turns 123D logs into the nuScenes ``info`` pickle that
mmdetection3d reads, so stock PETR / StreamPETR / BEVDet configs train on 123D data with nothing
changed but ``ann_file``. :mod:`py123detection.mmcv.plugin` is the training-side half: a
``NuScenesDataset`` subclass and an evaluation that needs no nuScenes database.

The two halves normally live in different Python environments — export needs py123d, the plugin
needs mmdetection3d — so this module imports neither. Import the half you want.
"""
