"""Python Optimal Transport toolbox"""


# All submodules and packages
from . import lp
from . import bregman
from . import optim
from . import utils
from . import datasets
from . import plot
from . import da

# OT functions
from .lp import emd, emd2
from .bregman import sinkhorn, barycenter
from .da import sinkhorn_lpl1_mm

# utils functions
from .utils import dist, unif, tic, toc, toq

__version__ = "0.1.12"

__all__ = ["emd", "emd2", "sinkhorn", "utils", 'datasets', 'bregman', 'lp', 
           'plot', 'tic', 'toc', 'toq',
           'dist', 'unif', 'barycenter', 'sinkhorn_lpl1_mm', 'da', 'optim']
