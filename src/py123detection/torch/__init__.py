"""The mmcv-free branch: 123D logs straight into PyTorch and Lightning.

Where :mod:`py123detection.mmcv` exports a pickle for the mmdetection3d ecosystem, this branch
skips that ecosystem entirely. :mod:`py123detection.torch.data` reads 123D scenes into batches,
:mod:`py123detection.torch.models` holds the detectors, and evaluation goes through the same
:mod:`py123detection.core.evaluation` the mmcv branch uses, so a number from either branch means
the same thing.

Importing this needs torch and lightning; :mod:`py123detection.core` does not.
"""
