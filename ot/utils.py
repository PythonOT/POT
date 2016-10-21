
import numpy as np
from scipy.spatial.distance import cdist, pdist


def dist(x1,x2=None,metric='sqeuclidean'):
    """Compute distance between samples in x1 and x2"""
    if x2 is None:
        return pdist(x1,metric=metric)
    else:
        return cdist(x1,x2,metric=metric)  

def dots(*args):
    """ Stupid but nice dots function for multiple matrix multiply """
    return reduce(np.dot,args)