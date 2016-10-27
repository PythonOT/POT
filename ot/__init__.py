# Python Optimal Transport toolbox

# All submodules and packages
from . import utils
from . import datasets
from . import plot
from . import bregman
from . import lp 
from . import da
from . import optim 


# OT functions
from ot.lp import emd
from ot.bregman import sinkhorn,barycenter
from ot.da import sinkhorn_lpl1_mm

# utils functions
from utils import dist,unif

__all__ = ["emd","sinkhorn","utils",'datasets','bregman','lp','plot','dist','unif','barycenter','sinkhorn_lpl1_mm','da','optim']
