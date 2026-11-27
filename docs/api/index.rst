API reference
=============

Public names are re-exported from the package each branch lives in, which is how you normally
import them:

.. code-block:: python

   from py123detection import Source, Taxonomy                                # needs py123d
   from py123detection.mmcv.export import ExportConfig, export_to_mmdet3d     # conversion environment
   from py123detection.mmcv.plugin import Py123DNuScenesDataset               # training environment
   from py123detection.torch.models import PETR, BEVFormer                    # modern stack
   from py123detection.torch.data import Py123DDetectionDataset               # modern stack

Each name is documented once, on the page of the module that defines it.

.. autosummary::
   :toctree: generated
   :recursive:

   py123detection
