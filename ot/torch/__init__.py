"""

This module contains several implementations of OT solvers that can be used with Pytorch tensors. They all provide gradients and /or sub-gradients w.r.t. all the tensor arguments.


"""

# Author: Remi Flamary <remi.flamary@unice.fr>
#
# License: MIT License

from . import utils
from . import lp

from .utils import dist, unif, proj_simplex
from .lp import ot_loss, ot_solve, OptimalTransportLossFunction

__all__ = ['dist', 'unif', 'proj_simplex', 'ot_loss', "ot_solve"]

