"""

This is the main module of the POT toolbox. It provides easy access to
a number of sub-modules and functions described below.

.. note::


    Here is a list of the submodules and short description of what they contain.

    - :any:`ot.lp` contains OT solvers for the exact (Linear Program) OT problems.
    - :any:`ot.bregman` contains OT solvers for the entropic OT problems using
      Bregman projections.
    - :any:`ot.lp` contains OT solvers for the exact (Linear Program) OT problems.
    - :any:`ot.smooth` contains OT solvers for the regularized (l2 and kl) smooth OT
      problems.
    - :any:`ot.gromov` contains solvers for Gromov-Wasserstein and Fused Gromov
      Wasserstein problems.
    - :any:`ot.optim` contains generic solvers OT based optimization problems
    - :any:`ot.da` contains classes and function related to Monge mapping
      estimation and Domain Adaptation (DA).
    - :any:`ot.gpu` contains GPU (cupy) implementation of some OT solvers
    - :any:`ot.dr` contains Dimension Reduction (DR) methods such as Wasserstein
      Discriminant Analysis.
    - :any:`ot.utils` contains utility functions such as distance computation and
      timing.
    - :any:`ot.datasets` contains toy dataset generation functions.
    - :any:`ot.plot` contains visualization functions
    - :any:`ot.stochastic` contains stochastic solvers for regularized OT.
    - :any:`ot.unbalanced` contains solvers for regularized unbalanced OT.

.. warning::
    The list of automatically imported sub-modules is as follows:
    :py:mod:`ot.lp`, :py:mod:`ot.bregman`, :py:mod:`ot.optim`
    :py:mod:`ot.utils`, :py:mod:`ot.datasets`,
    :py:mod:`ot.gromov`, :py:mod:`ot.smooth`
    :py:mod:`ot.stochastic`

    The following sub-modules are not imported due to additional dependencies:

    - :any:`ot.dr` : depends on :code:`pymanopt` and :code:`autograd`.
    - :any:`ot.gpu` : depends on :code:`cupy` and a CUDA GPU.
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

# OT functions
from .lp import emd, emd2, emd_1d, emd2_1d, wasserstein_1d
from .bregman import sinkhorn, sinkhorn2, barycenter
from .unbalanced import sinkhorn_unbalanced, barycenter_unbalanced, sinkhorn_unbalanced2
from .da import sinkhorn_lpl1_mm

# utils functions
from .utils import dist, unif, tic, toc, toq

__version__ = "0.6.0"

__all__ = ['emd', 'emd2', 'emd_1d', 'sinkhorn', 'sinkhorn2', 'utils', 'datasets',
           'bregman', 'lp', 'tic', 'toc', 'toq', 'gromov',
           'emd_1d', 'emd2_1d', 'wasserstein_1d',
           'dist', 'unif', 'barycenter', 'sinkhorn_lpl1_mm', 'da', 'optim',
           'sinkhorn_unbalanced', 'barycenter_unbalanced',
           'sinkhorn_unbalanced2']
