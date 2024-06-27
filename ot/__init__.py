"""
.. warning::
    The list of automatically imported sub-modules is as follows:
    :py:mod:`ot.lp`, :py:mod:`ot.bregman`, :py:mod:`ot.optim`
    :py:mod:`ot.utils`, :py:mod:`ot.datasets`,
    :py:mod:`ot.gromov`, :py:mod:`ot.smooth`
    :py:mod:`ot.stochastic`, :py:mod:`ot.partial`, :py:mod:`ot.regpath`
    , :py:mod:`ot.unbalanced`, :py:mod:`ot.mapping` .
    The following sub-modules are not imported due to additional dependencies:
    - :any:`ot.dr` : depends on :code:`pymanopt` and :code:`autograd`.
    - :any:`ot.plot` : depends on :code:`matplotlib`
"""

# Author: Remi Flamary <remi.flamary@unice.fr>
#         Nicolas Courty <ncourty@irisa.fr>
#
# License: MIT License


# All submodules and packages
from . import lp
from . import bregman
from . import optim
from . import utils
from . import datasets
from . import da
from . import gromov
from . import smooth
from . import stochastic
from . import unbalanced
from . import partial
from . import backend
from . import regpath
from . import weak
from . import factored
from . import solvers
from . import gaussian
from . import lowrank


# OT functions
from .lp import (emd, emd2, emd_1d, emd2_1d, wasserstein_1d,
                 binary_search_circle, wasserstein_circle,
                 semidiscrete_wasserstein2_unif_circle)
from .bregman import sinkhorn, sinkhorn2, barycenter
from .unbalanced import (sinkhorn_unbalanced, barycenter_unbalanced,
                         sinkhorn_unbalanced2)
from .da import sinkhorn_lpl1_mm
from .sliced import (sliced_wasserstein_distance, max_sliced_wasserstein_distance,
                     sliced_wasserstein_sphere, sliced_wasserstein_sphere_unif)
from .gromov import (gromov_wasserstein, gromov_wasserstein2,
                     gromov_barycenters, fused_gromov_wasserstein, fused_gromov_wasserstein2,
                     lowrank_gromov_wasserstein_samples)
from .weak import weak_optimal_transport
from .factored import factored_optimal_transport
from .solvers import solve, solve_gromov, solve_sample
from .lowrank import lowrank_sinkhorn

# utils functions
from .utils import dist, unif, tic, toc, toq

__version__ = "0.9.4"

__all__ = ['emd', 'emd2', 'emd_1d', 'sinkhorn', 'sinkhorn2', 'utils',
           'datasets', 'bregman', 'lp', 'tic', 'toc', 'toq', 'gromov',
           'emd2_1d', 'wasserstein_1d', 'backend', 'gaussian',
           'dist', 'unif', 'barycenter', 'sinkhorn_lpl1_mm', 'da', 'optim',
           'sinkhorn_unbalanced', 'barycenter_unbalanced',
           'sinkhorn_unbalanced2', 'sliced_wasserstein_distance', 'sliced_wasserstein_sphere',
           'gromov_wasserstein', 'gromov_wasserstein2', 'gromov_barycenters', 'fused_gromov_wasserstein',
           'fused_gromov_wasserstein2', 'max_sliced_wasserstein_distance', 'weak_optimal_transport',
           'factored_optimal_transport', 'solve', 'solve_gromov', 'solve_sample',
           'smooth', 'stochastic', 'unbalanced', 'partial', 'regpath', 'solvers',
           'binary_search_circle', 'wasserstein_circle',
           'semidiscrete_wasserstein2_unif_circle', 'sliced_wasserstein_sphere_unif', 'lowrank_sinkhorn', 
           'lowrank_gromov_wasserstein_samples']
