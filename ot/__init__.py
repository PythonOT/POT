"""Python Optimal Transport toolbox



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

# OT functions
from .lp import emd, emd2
from .bregman import sinkhorn, sinkhorn2, barycenter
from .da import sinkhorn_lpl1_mm

# utils functions
from .utils import dist, unif, tic, toc, toq

__version__ = "0.4.0"

__all__ = ["emd", "emd2", "sinkhorn", "sinkhorn2", "utils", 'datasets',
           'bregman', 'lp', 'tic', 'toc', 'toq', 'gromov',
           'dist', 'unif', 'barycenter', 'sinkhorn_lpl1_mm', 'da', 'optim']
