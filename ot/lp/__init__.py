

from .emd import emd_c
import numpy as np

def emd(a,b,M):
    """
        Solves the Earth Movers distance problem and returns the optimal transport matrix
        
        gamm=emd(a,b,M)
    
    .. math::
        \gamma = arg\min_\gamma <\gamma,M>_F 
        
        s.t. \gamma 1 = a
        
             \gamma^T 1= b 
             
             \gamma\geq 0
    where :
    
    - M is the metric cost matrix
    - a and b are the sample weights
             
    Parameters
    ----------
    a : (ns,) ndarray, float64
        Source histogram (uniform weigth if empty list)
    b : (nt,) ndarray, float64
        Target histogram (uniform weigth if empty list)
    M : (ns,nt) ndarray, float64
        loss matrix        
        
    Examples
    --------
    
    Simple example with obvious solution. The function :func:emd accepts lists and
    perform automatic conversion tu numpy arrays 
    
    >>> a=[.5,.5]
    >>> b=[.5,.5]
    >>> M=[[0.,1.],[1.,0.]]
    >>> ot.emd(a,b,M)
    array([[ 0.5,  0. ],
           [ 0. ,  0.5]])
    
    Returns
    -------
    gamma: (ns x nt) ndarray
        Optimal transportation matrix for the given parameters
 
    """
    a=np.asarray(a,dtype=np.float64)
    b=np.asarray(b,dtype=np.float64)
    
    if len(a)==0:
        a=np.ones((M.shape[0],),dtype=np.float64)/M.shape[0]
    if len(b)==0:
        b=np.ones((M.shape[1],),dtype=np.float64)/M.shape[1]
    
    return emd_c(a,b,M)

