API reference
=============

Public names are re-exported from three packages, which is how you normally import them:

.. code-block:: python

   from py123detection import Source, Taxonomy                                # needs py123d
   from py123detection.mmcv_export import ExportConfig, export_to_mmdet3d     # conversion environment
   from py123detection.mmcv_plugin import Py123DNuScenesDataset               # training environment

Each name is documented once, on the page of the module that defines it.

.. autosummary::
   :toctree: generated
   :recursive:

   py123detection
