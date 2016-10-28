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
from .lp import emd
from .bregman import sinkhorn, barycenter
from .da import sinkhorn_lpl1_mm

# utils functions
from .utils import dist, unif

__all__ = ["emd", "sinkhorn", "utils", 'datasets', 'bregman', 'lp', 'plot',
           'dist', 'unif', 'barycenter', 'sinkhorn_lpl1_mm', 'da', 'optim']
