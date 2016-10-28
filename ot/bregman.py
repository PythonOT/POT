# -*- coding: utf-8 -*-
"""
Bregman projections for regularized OT
"""

import numpy as np


def sinkhorn(a,b, M, reg, numItermax = 1000, stopThr=1e-9, verbose=False, log=False):
    """
    Solve the entropic regularization optimal transport problem

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
    if log:
        log={'err':[]}
    
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
            if log:
                log['err'].append(err)        
        
            if verbose:
                if cpt%200 ==0:
                    print('{:5s}|{:12s}'.format('It.','Err')+'\n'+'-'*19)
                print('{:5d}|{:8e}|'.format(cpt,err))
        cpt = cpt +1
    #print 'err=',err,' cpt=',cpt  
    if log:
        return np.dot(np.diag(u),np.dot(K,np.diag(v))),log
    else:
        return np.dot(np.diag(u),np.dot(K,np.diag(v)))


def geometricBar(weights,alldistribT):
    """return the weighted geometric mean of distributions"""
    assert(len(weights)==alldistribT.shape[1])       
    return np.exp(np.dot(np.log(alldistribT),weights.T))    

def geometricMean(alldistribT):     
    """return the  geometric mean of distributions"""
    return np.exp(np.mean(np.log(alldistribT),axis=1))    

def projR(gamma,p):
    #return np.dot(np.diag(p/np.maximum(np.sum(gamma,axis=1),1e-10)),gamma)
    return np.multiply(gamma.T,p/np.maximum(np.sum(gamma,axis=1),1e-10)).T

def projC(gamma,q):
    #return (np.dot(np.diag(q/np.maximum(np.sum(gamma,axis=0),1e-10)),gamma.T)).T
    return np.multiply(gamma,q/np.maximum(np.sum(gamma,axis=0),1e-10))
    

def barycenter(A,M,reg, weights=None, numItermax = 1000, stopThr=1e-4,verbose=False,log=False):
    """Compute the entropic regularized wasserstein barycenter of distributions A

     The function solves the following optimization problem:
    
    .. math::
       \mathbf{a} = arg\min_\mathbf{a} \sum_i W_{reg}(\mathbf{a},\mathbf{a}_i)
        
    where :
    
    - :math:`W_{reg}(\cdot,\cdot)` is the entropic regularized Wasserstein distance (see ot.bregman.sinkhorn)
    - :math:`\mathbf{a}_i` are training distributions in the columns of matrix :math:`\mathbf{A}`
    - reg and :math:`\mathbf{M}` are respectively the regularization term and the cost matrix for OT
    
    The algorithm used for solving the problem is the Sinkhorn-Knopp matrix scaling algorithm as proposed in [3]_
        
    Parameters
    ----------
    A : np.ndarray (d,n)
        n training distributions of size d 
    M : np.ndarray (d,d)
        loss matrix   for OT      
    reg: float
        Regularization term >0
    numItermax: int, optional
        Max number of iterations
    stopThr: float, optional
        Stop threshol on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True    
  
    
    Returns
    -------
    a: (d,) ndarray
        Wasserstein barycenter
    log: dict
        log dictionary return only if log==True in parameters     

        
    References
    ----------
    
    .. [3] Benamou, J. D., Carlier, G., Cuturi, M., Nenna, L., & Peyré, G. (2015). Iterative Bregman projections for regularized transportation problems. SIAM Journal on Scientific Computing, 37(2), A1111-A1138.
  
        
          
    """
    
    
    if weights is None:
        weights=np.ones(A.shape[1])/A.shape[1]
    else:
        assert(len(weights)==A.shape[1])
        
    if log:
        log={'err':[]}

    #M = M/np.median(M) # suggested by G. Peyre
    K = np.exp(-M/reg)

    cpt = 0
    err=1

    UKv=np.dot(K,np.divide(A.T,np.sum(K,axis=0)).T)
    u = (geometricMean(UKv)/UKv.T).T
    
    while (err>stopThr and cpt<numItermax):
        cpt = cpt +1
        UKv=u*np.dot(K,np.divide(A,np.dot(K,u)))
        u = (u.T*geometricBar(weights,UKv)).T/UKv
        
        if cpt%10==1:
            err=np.sum(np.std(UKv,axis=1))
            
            # log and verbose print
            if log:
                log['err'].append(err)
            
            if verbose:
                if cpt%200 ==0:
                    print('{:5s}|{:12s}'.format('It.','Err')+'\n'+'-'*19)
                print('{:5d}|{:8e}|'.format(cpt,err))            

    if log:
        log['niter']=cpt
        return geometricBar(weights,UKv),log
    else:
        return geometricBar(weights,UKv)
    

def unmix(a,D,M,M0,h0,reg,reg0,alpha,numItermax = 1000, stopThr=1e-3,verbose=False,log=False):
    """
    Compute the unmixing of an observation with a given dictionary using Wasserstein distance
    
    The function solve the following optimization problem:
    
    .. math::
       \mathbf{h} = arg\min_\mathbf{h}  (1- \\alpha) W_{M,reg}(\mathbf{a},\mathbf{Dh})+\\alpha W_{M0,reg0}(\mathbf{h}_0,\mathbf{h})


    where :
    
    - :math:`W_{M,reg}(\cdot,\cdot)` is the entropic regularized Wasserstein distance with M loss matrix (see ot.bregman.sinkhorn)
    - :math:`\mathbf{a}` is an observed distribution,  :math:`\mathbf{h}_0` is aprior on unmixing 
    - reg and :math:`\mathbf{M}` are respectively the regularization term and the cost matrix for OT data fitting
    - reg0 and :math:`\mathbf{M0}` are respectively the regularization term and the cost matrix for regularization
    - :math:`\\alpha`weight data fitting and regularization    
    
    The optimization problem is solved suing the algorithm described in [4]
  
        
    Parameters
    ----------
    a : np.ndarray (d)
        observed distribution
    D : np.ndarray (d,n)
        dictionary matrix    
    M : np.ndarray (d,d)
        loss matrix        
    M0 : np.ndarray (n,n)
        loss matrix        
    h0 : np.ndarray (n,)
        prior on h            
    reg: float
        Regularization term >0 (Wasserstein data fitting)
    reg0: float
        Regularization term >0 (Wasserstein reg with h0)   
    alpha: float
        How much should we trust the prior ([0,1])
    numItermax: int, optional
        Max number of iterations
    stopThr: float, optional
        Stop threshol on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True    
  
    
    Returns
    -------
    a: (d,) ndarray
        Wasserstein barycenter
    log: dict
        log dictionary return only if log==True in parameters     
  
    References
    ----------
    
    .. [4] S. Nakhostin, N. Courty, R. Flamary, D. Tuia, T. Corpetti, Supervised planetary unmixing with optimal transport, Whorkshop on Hyperspectral Image and Signal Processing : Evolution in Remote Sensing (WHISPERS), 2016.

    """
        
    #M = M/np.median(M) 
    K = np.exp(-M/reg)
 
    #M0 = M0/np.median(M0)  
    K0 = np.exp(-M0/reg0)
    old = h0

    err=1
    cpt=0    
    #log = {'niter':0, 'all_err':[]}
    if log:
        log={'err':[]}
    
    
    while (err>stopThr and cpt<numItermax):
        K = projC(K,a)  
        K0 = projC(K0,h0)  
        new  = np.sum(K0,axis=1)
        inv_new = np.dot(D,new) # we recombine the current selection from dictionnary
        other = np.sum(K,axis=1)
        delta = np.exp(alpha*np.log(other)+(1-alpha)*np.log(inv_new)) # geometric interpolation
        K = projR(K,delta)
        K0 =  np.dot(np.diag(np.dot(D.T,delta/inv_new)),K0)
        
        err=np.linalg.norm(np.sum(K0,axis=1)-old)
        old = new
        if log:
            log['err'].append(err)
            
        if verbose:
            if cpt%200 ==0:
                print('{:5s}|{:12s}'.format('It.','Err')+'\n'+'-'*19)
            print('{:5d}|{:8e}|'.format(cpt,err))      
            
        cpt = cpt+1
        
    if log:
        log['niter']=cpt
        return np.sum(K0,axis=1),log
    else:
        return np.sum(K0,axis=1)
