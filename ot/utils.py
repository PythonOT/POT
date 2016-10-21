
import numpy as np
from scipy.spatial.distance import cdist


def dist(x1,x2=None,metric='sqeuclidean'):
    """Compute distance between samples in x1 and x2"""
    if x2 is None:
        return cdist(x1,x1,metric=metric)
    else:
        return cdist(x1,x2,metric=metric)  
        
def dist0(n,method='linear'):
    """Compute stardard cos matrices for OT problems"""
    res=0
    if method=='linear':
        x=np.arange(n,dtype=np.float64).reshape((n,1))
        res=dist(x,x)
    return res
    

def dots(*args):
    """ Stupid but nice dots function for multiple matrix multiply """
    return reduce(np.dot,args)