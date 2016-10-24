
# Python Optimal Transport toolbox


import utils
import datasets
import plot


# Ot functions
from emd import emd
from bregman import sinkhorn



from utils import dist,dots

__all__ = ["emd","sinkhorn","utils",'datasets','plot','dist','dots']
