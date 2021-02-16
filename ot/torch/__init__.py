"""

This module contains several implementations of OT solvers that can be used with Pytorch tensors. They all provide gradients and /or sub-gradients w.r.t. all the tensor arguments.


"""

# Author: Remi Flamary <remi.flamary@unice.fr>
#
# License: MIT License

from . import lp
from . import sliced
from . import utils
from .lp import ot_loss, ot_solve, OptimalTransportLossFunction
from .sliced import ot_loss_sliced
from .utils import dist, unif, proj_simplex

__all__ = ['dist', 'unif', 'proj_simplex', 'ot_loss', "ot_solve", "ot_loss_sliced"]
