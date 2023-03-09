# -*- coding: utf-8 -*-
"""
This module contains all solvers related to Gromov-Wasserstein problems

.. currentmodule:: ot.gromov
​
.. automodule:: ot.gromov
    :no-members:
    :no-inherited-members:
​
:py:mod:`ot.gromov`:
​
.. autosummary::
    :toctree: gen_modules/
    :template: function.rst
    ​
    gromov_wasserstein
    gromov_wasserstein2
    fused_gromov_wasserstein
    fused_gromov_wasserstein2
    gromov_barycenters
    fgw_barycenters
    entropic_gromov_wasserstein
    entropic_gromov_wasserstein2
    entropic_gromov_barycenters
    GW_distance_estimation
    pointwise_gromov_wasserstein
    sampled_gromov_wasserstein
    gromov_wasserstein_dictionary_learning
    gromov_wasserstein_linear_unmixing
    fused_gromov_wasserstein_dictionary_learning
    fused_gromov_wasserstein_linear_unmixing
    semirelaxed_gromov_wasserstein
    semirelaxed_gromov_wasserstein2
    semirelaxed_fused_gromov_wasserstein
    semirelaxed_fused_gromov_wasserstein2



"""


# Author: Remi Flamary <remi.flamary@unice.fr>
#         Cedric Vincent-Cuaz <cedvincentcuaz@gmail.com>
#
# License: MIT License

# All submodules and packages
from ._utils import (init_matrix, tensor_product, gwloss, gwggrad,
                     update_square_loss, update_kl_loss,
                     init_matrix_semirelaxed)
from ._gw import (gromov_wasserstein, gromov_wasserstein2,
                  fused_gromov_wasserstein, fused_gromov_wasserstein2,
                  solve_gromov_linesearch, gromov_barycenters, fgw_barycenters,
                  update_structure_matrix, update_feature_matrix)
from ._bregman import (entropic_gromov_wasserstein,
                       entropic_gromov_wasserstein2,
                       entropic_gromov_barycenters)
from ._estimators import (GW_distance_estimation, pointwise_gromov_wasserstein,
                          sampled_gromov_wasserstein)
from ._semirelaxed import (semirelaxed_gromov_wasserstein,
                           semirelaxed_gromov_wasserstein2,
                           semirelaxed_fused_gromov_wasserstein,
                           semirelaxed_fused_gromov_wasserstein2,
                           solve_semirelaxed_gromov_linesearch)
from ._dictionary import (gromov_wasserstein_dictionary_learning,
                          gromov_wasserstein_linear_unmixing,
                          fused_gromov_wasserstein_dictionary_learning,
                          fused_gromov_wasserstein_linear_unmixing)


__all__ = ['init_matrix', 'tensor_product', 'gwloss', 'gwggrad',
           'update_square_loss', 'update_kl_loss', 'init_matrix_semirelaxed',
           'gromov_wasserstein', 'gromov_wasserstein2', 'fused_gromov_wasserstein',
           'fused_gromov_wasserstein2', 'solve_gromov_linesearch', 'gromov_barycenters',
           'fgw_barycenters', 'update_structure_matrix', 'update_feature_matrix',
           'entropic_gromov_wasserstein', 'entropic_gromov_wasserstein2',
           'entropic_gromov_barycenters', 'GW_distance_estimation',
           'pointwise_gromov_wasserstein', 'sampled_gromov_wasserstein',
           'semirelaxed_gromov_wasserstein', 'semirelaxed_gromov_wasserstein2',
           'semirelaxed_fused_gromov_wasserstein', 'semirelaxed_fused_gromov_wasserstein2',
           'solve_semirelaxed_gromov_linesearch', 'gromov_wasserstein_dictionary_learning',
           'gromov_wasserstein_linear_unmixing', 'fused_gromov_wasserstein_dictionary_learning',
           'fused_gromov_wasserstein_linear_unmixing']

