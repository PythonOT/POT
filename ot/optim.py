# -*- coding: utf-8 -*-
"""
Optimization algorithms for OT
"""

import numpy as np
import scipy as sp
from scipy.optimize.linesearch import scalar_search_armijo
from .lp import emd

# The corresponding scipy function does not work for matrices
def line_search_armijo(f,xk,pk,gfk,old_fval,args=(),c1=1e-4,alpha0=0.99):
    """
    Armijo linesearch function that works with matrices
    
    find an approximate minimum of f(xk+alpha*pk) that satifies the 
    armijo conditions. 
    
    Parameters
    ----------

    f : function
        loss function
    xk : np.ndarray 
        initial position
    pk : np.ndarray 
        descent direction
    gfk : np.ndarray
        gradient of f at xk        
    old_fval: float
        loss value at xk
    args : tuple, optional
        arguments given to f
    c1 : float, optional
        c1 const in armijo rule (>0)
    alpha0 : float, optional
        initial step (>0)
  
    Returns
    -------
    alpha : float
        step that satisfy armijo conditions
    fc : int
        nb of function call
    fa : float
        loss value at step alpha
    
    """
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
    reg : float
        Regularization term >0
    G0 :  np.ndarray (ns,nt), optional
        initial guess (default is indep joint density)
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshol on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    
    Returns
    -------
    gamma: (ns x nt) ndarray
        Optimal transportation matrix for the given parameters
    log: dict
        log dictionary return only if log==True in parameters 
    
        
    References
    ----------
    
    .. [1] Ferradans, S., Papadakis, N., Peyré, G., & Aujol, J. F. (2014). Regularized discrete optimal transport. SIAM Journal on Imaging Sciences, 7(3), 1853-1882.
        
    See Also
    --------
    ot.lp.emd : Unregularized optimal ransport
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
        alpha,fc,f_val = line_search_armijo(cost,G,deltaG,Mi,f_val)
        
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

