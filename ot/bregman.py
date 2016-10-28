# -*- coding: utf-8 -*-
"""
Bregman projection for regularized Otimal transport
"""

import numpy as np


def sinkhorn(a,b, M, reg,numItermax = 1000,stopThr=1e-9):
    """
    Solve the entropic regularization optimal transport problem and return the OT matrix
    
    The function solves the following optimization problem:
    
    .. math::
        \gamma = arg\min_\gamma <\gamma,M>_F + reg\cdot\Omega(\gamma)
        
        s.t. \gamma 1 = a
        
             \gamma^T 1= b 
             
             \gamma\geq 0
    where :
    
    - M is the (ns,nt) metric cost matrix
    - :math:`\Omega` is the entropic regularization term :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - a and b are source and target weights (sum to 1)
    
    The algorithm used for solving the problem is the Sinkhorn-Knopp matrix scaling algorithm as proposed in [2]_
    
             
    Parameters
    ----------
    a : np.ndarray (ns,)
        samples weights in the source domain
    b : np.ndarray (nt,)
        samples in the target domain
    M : np.ndarray (ns,nt)
        loss matrix        
    reg: float
        Regularization term >0
    numItermax: int, optional
        Max number of iterations
    stopThr: float, optional
        Stop threshol on error (>0)
    
  
    
    Returns
    -------
    gamma: (ns x nt) ndarray
        Optimal transportation matrix for the given parameters
        

    Examples
    --------
        
    >>> a=[.5,.5]
    >>> b=[.5,.5]
    >>> M=[[0.,1.],[1.,0.]]
    >>> ot.sinkhorn(a,b,M,1)
    array([[ 0.36552929,  0.13447071],
           [ 0.13447071,  0.36552929]])
      
        
    References
    ----------
    
    .. [2] M. Cuturi, Sinkhorn Distances : Lightspeed Computation of Optimal Transport, Advances in Neural Information Processing Systems (NIPS) 26, 2013
        
        
    See Also
    --------
    ot.lp.emd : Unregularized OT
    ot.optim.cg : General regularized OT
        
    """    
    
    a=np.asarray(a,dtype=np.float64)
    b=np.asarray(b,dtype=np.float64)
    M=np.asarray(M,dtype=np.float64)
    
    if len(a)==0:
        a=np.ones((M.shape[0],),dtype=np.float64)/M.shape[0]
    if len(b)==0:
        b=np.ones((M.shape[1],),dtype=np.float64)/M.shape[1]    
    
    # init data
    Nini = len(a)
    Nfin = len(b)
    
    
    cpt = 0
    
    # we assume that no distances are null except those of the diagonal of distances
    u = np.ones(Nini)/Nini
    v = np.ones(Nfin)/Nfin 
    uprev=np.zeros(Nini)
    vprev=np.zeros(Nini)

    #print reg
 
    K = np.exp(-M/reg)
    #print np.min(K)
      
    Kp = np.dot(np.diag(1/a),K)
    transp = K
    cpt = 0
    err=1
    while (err>stopThr and cpt<numItermax):
        if np.any(np.dot(K.T,u)==0) or np.any(np.isnan(u)) or np.any(np.isnan(v)):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            print('Warning: numerical errrors')
            if cpt!=0:
                u = uprev
                v = vprev     
            break
        uprev = u
        vprev = v  
        v = np.divide(b,np.dot(K.T,u))
        u = 1./np.dot(Kp,v)
        if cpt%10==0:
            # we can speed up the process by checking for the error only all the 10th iterations
            transp = np.dot(np.diag(u),np.dot(K,np.diag(v)))
            err = np.linalg.norm((np.sum(transp,axis=0)-b))**2
        cpt = cpt +1
    #print 'err=',err,' cpt=',cpt  

    return np.dot(np.diag(u),np.dot(K,np.diag(v)))


def geometricBar(weights,alldistribT):
    assert(len(weights)==alldistribT.shape[1])       
    return np.exp(np.dot(np.log(alldistribT),weights.T))    

def geometricMean(alldistribT):     
    return np.exp(np.mean(np.log(alldistribT),axis=1))    

def projR(gamma,p):
    #return np.dot(np.diag(p/np.maximum(np.sum(gamma,axis=1),1e-10)),gamma)
    return np.multiply(gamma.T,p/np.maximum(np.sum(gamma,axis=1),1e-10)).T

def projC(gamma,q):
    #return (np.dot(np.diag(q/np.maximum(np.sum(gamma,axis=0),1e-10)),gamma.T)).T
    return np.multiply(gamma,q/np.maximum(np.sum(gamma,axis=0),1e-10))
    

def barycenter(A,M,reg, weights=None, numItermax = 1000, tol_error=1e-4,log=dict()):
    """Compute the Regularizzed wassersteien barycenter of distributions A"""
    
    
    if weights is None:
        weights=np.ones(A.shape[1])/A.shape[1]
    else:
        assert(len(weights)==A.shape[1])

    #compute Mmax once for all
    #M = M/np.median(M) # suggested by G. Peyre
    K = np.exp(-M/reg)

    cpt = 0
    err=1

    UKv=np.dot(K,np.divide(A.T,np.sum(K,axis=0)).T)
    u = (geometricMean(UKv)/UKv.T).T
    
    log['niter']=0
    log['all_err']=[]
    
    while (err>tol_error and cpt<numItermax):
        cpt = cpt +1
        UKv=u*np.dot(K,np.divide(A,np.dot(K,u)))
        u = (u.T*geometricBar(weights,UKv)).T/UKv
        if cpt%10==1:
            err=np.sum(np.std(UKv,axis=1))
            log['all_err'].append(err)
        
    log['niter']=cpt
    return geometricBar(weights,UKv),log
    

def unmix(distrib,D,M,M0,h0,reg,reg0,alpha,numItermax = 1000, tol_error=1e-3,log=dict()):
    """
        distrib : distribution to unmix
        D : Dictionnary 
        M : Metric matrix in the space of the distributions to unmix
        M0 : Metric matrix in the space of the 'abundance values' to solve for
        h0 : prior on solution (generally uniform distribution)
        reg,reg0 : transport regularizations
        alpha : how much should we trust the prior ? ([0,1])
    """
        
    #M = M/np.median(M) 
    K = np.exp(-M/reg)
 
    #M0 = M0/np.median(M0)  
    K0 = np.exp(-M0/reg0)
    old = h0

    err=1
    cpt=0    
    #log = {'niter':0, 'all_err':[]}
    log['niter']=0
    log['all_err']=[]
    
    
    while (err>tol_error and cpt<numItermax):
        K = projC(K,distrib)  
        K0 = projC(K0,h0)  
        new  = np.sum(K0,axis=1)
        inv_new = np.dot(D,new) # we recombine the current selection from dictionnary
        other = np.sum(K,axis=1)
        delta = np.exp(alpha*np.log(other)+(1-alpha)*np.log(inv_new)) # geometric interpolation
        K = projR(K,delta)
        K0 =  np.dot(np.diag(np.dot(D.T,delta/inv_new)),K0)
        
        err=np.linalg.norm(np.sum(K0,axis=1)-old)
        old = new
        log['all_err'].append(err)
        cpt = cpt+1
        
        
    log['niter']=cpt
    return np.sum(K0,axis=1),log
