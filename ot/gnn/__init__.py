# -*- coding: utf-8 -*-
"""
Layers and functions for optimal transport in Graph Neural Networks.

.. warning::
    Note that by default the module is not imported in :mod:`ot`. In order to
    use it you need to explicitly import :mod:`ot.gnn`. This module is PyTorch Geometric dependent.
    The layers are compatible with their API.

"""

# Author: Sonia Mazelet <sonia.mazelet@ens-paris-saclay.fr>
#         Rémi Flamary <remi.flamary@unice.fr>
#
# License: MIT License

# All submodules and packages


from ._utils import (FGW_distance_to_templates, wasserstein_distance_to_templates)

from ._layers import (TFGWPooling, TWPooling)

__all__ = ['FGW_distance_to_templates', 'wasserstein_distance_to_templates', 'TFGWPooling', 'TWPooling']
