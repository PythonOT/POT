# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 15:08:19 2016

@author: rflamary
"""

import numpy as np
import scipy as sp
from scipy.optimize.linesearch import scalar_search_armijo
from emd import emd

# The corresponding scipy function does not work for matrices
def line_search_armijo(f,xk,pk,gfk,old_fval,args=(),c1=1e-4,alpha0=0.99):
    xk = np.atleast_1d(xk)
    fc = [0]
    
    def phi(alpha1):
        fc[0] += 1
        return f(xk + alpha1*pk, *args)
    
    if old_fval is None:
        phi0 = phi(0.)
    else:
        phi0 = old_fval
    
    derphi0 = np.sum(pk*gfk) # Quickfix for matrices
    alpha,phi1 = scalar_search_armijo(phi,phi0,derphi0,c1=c1,alpha0=alpha0)
    
    return alpha,fc[0],phi1


def cg(a,b,M,reg,f,df,G0=None,numItermax = 200,stopThr=1e-9,verbose=False,log=False):
    """
    Solve the general regularized OT problem with conditional gradient
    
        The function solves the following optimization problem:
    
    .. math::
        \gamma = arg\min_\gamma <\gamma,M>_F + reg*f(\gamma)
        
        s.t. \gamma 1 = a
        
             \gamma^T 1= b 
             
             \gamma\geq 0
    where :
    
    - M is the (ns,nt) metric cost matrix
    - :math:`f` is the regularization term ( and df is its gradient)
    - a and b are source and target weights (sum to 1)
    
    The algorithm used for solving the problem is conditional gradient as discussed in  [1]_
    
             
    Parameters
    ----------
    a : np.ndarray (ns,)
        samples weights in the source domain
    b : np.ndarray (nt,)
        samples in the target domain
    M : np.ndarray (ns,nt)
        loss matrix        
    reg: float()
        Regularization term >0
  
    
    Returns
    -------
    gamma: (ns x nt) ndarray
        Optimal transportation matrix for the given parameters
        
    References
    ----------
    
    .. [1] Ferradans, S., Papadakis, N., Peyré, G., & Aujol, J. F. (2014). Regularized discrete optimal transport. SIAM Journal on Imaging Sciences, 7(3), 1853-1882.
        
    See Also
    --------
    ot.emd.emd : Unregularized optimal ransport
    ot.bregman.sinkhorn : Entropic regularized optimal transport
           
    """
    
    loop=1
    
    if log:
        log={'loss':[]}
    
    if G0 is None:
        G=np.outer(a,b)
    else:
        G=G0
    
    def cost(G):
        return np.sum(M*G)+reg*f(G)
        
    f_val=cost(G)
    if log:
        log['loss'].append(f_val)
    
    it=0
    
    if verbose:
        print('{:5s}|{:12s}|{:8s}'.format('It.','Loss','Delta loss')+'\n'+'-'*32)
        print('{:5d}|{:8e}|{:8e}'.format(it,f_val,0))
    
    while loop:
        
        it+=1
        old_fval=f_val
        
        
        # problem linearization
        Mi=M+reg*df(G)
        
        # solve linear program
        Gc=emd(a,b,Mi)
        
        deltaG=Gc-G
        
        # line search
        alpha,fc,f_val = line_search_armijo(f,G,deltaG,Mi,f_val)
        
        
        G=G+alpha*deltaG
        
        # test convergence
        if it>=numItermax:
            loop=0
        
        delta_fval=(f_val-old_fval)/abs(f_val)
        if abs(delta_fval)<stopThr:
            loop=0
            
        
        if log:
            log['loss'].append(f_val)        
        
        if verbose:
            if it%20 ==0:
                print('{:5s}|{:12s}|{:8s}'.format('It.','Loss','Delta loss')+'\n'+'-'*32)
            print('{:5d}|{:8e}|{:8e}'.format(it,f_val,delta_fval))

    
    if log:
        return G,log
    else:
        return G

