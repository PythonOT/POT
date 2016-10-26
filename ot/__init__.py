# Python Optimal Transport toolbox

# All submodules and packages
import utils
import datasets
import plot
import bregman
import da
import optim 

# OT functions
from emd import emd
from bregman import sinkhorn,barycenter
from da import sinkhorn_lpl1_mm

# utils functions
from utils import dist,unif

__all__ = ["emd","sinkhorn","utils",'datasets','bregman','plot','dist','unif','barycenter','sinkhorn_lpl1_mm','da','optim']
