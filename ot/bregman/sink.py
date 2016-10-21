# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 09:40:21 2016

@author: rflamary
"""

import numpy as np


def sinkhorn(a,b, M, reg,numItermax = 1000,stopThr=1e-9):
    """
    Solve the optimal transport problem (OT)
    
    .. math::
        \gamma = arg\min_\gamma <\gamma,M>_F + reg\cdot\Omega(\gamma)
        
        s.t. \gamma 1 = a
        
             \gamma^T 1= b 
             
             \gamma\geq 0
    where :
    
    - M is the metric cost matrix
    - Omega is the entropic regularization term
    - a and b are the sample weights
             
    Parameters
    ----------
    a : (ns,) ndarray
        samples in the source domain
    b : (nt,) ndarray
        samples in the target domain
    M : (ns,nt) ndarray
        loss matrix        
    reg: float()
        Regularization term >0
  
    
    Returns
    -------
    gamma: (ns x nt) ndarray
        Optimal transportation matrix for the given parameters
        
    """    
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


