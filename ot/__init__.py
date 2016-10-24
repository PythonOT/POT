# Python Optimal Transport toolbox

# All submodules and packages
import utils
import datasets
import plot
import bregman

# OT functions
from emd import emd
from bregman import sinkhorn

# utils functions
from utils import dist,unif

__all__ = ["emd","sinkhorn","utils",'datasets','bregman','plot','dist','unif']
