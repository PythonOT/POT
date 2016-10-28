# -*- coding: utf-8 -*-
"""
Domain adaptation with optimal transport
"""

import numpy as np
from .bregman import sinkhorn



def indices(a, func):
    return [i for (i, val) in enumerate(a) if func(val)]

def sinkhorn_lpl1_mm(a,labels_a, b, M, reg, eta=0.1,numItermax = 10,numInnerItermax = 200,stopInnerThr=1e-9,verbose=False,log=False):
    """
    Solve the entropic regularization optimal transport problem with nonconvex group lasso regularization
    
    The function solves the following optimization problem:
    
    .. math::
        \gamma = arg\min_\gamma <\gamma,M>_F + reg\cdot\Omega_e(\gamma)+ \eta \Omega_g(\gamma)
        
        s.t. \gamma 1 = a
        
             \gamma^T 1= b 
             
             \gamma\geq 0
    where :
    
    - M is the (ns,nt) metric cost matrix
    - :math:`\Omega_e` is the entropic regularization term :math:`\Omega_e(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - :math:`\Omega_g` is the group lasso  regulaization term :math:`\Omega_g(\gamma)=\sum_{i,c} \|\gamma_{i,\mathcal{I}_c}\|^{1/2}_1`   where  :math:`\mathcal{I}_c` are the index of samples from class c in the source domain.
    - a and b are source and target weights (sum to 1)
    
    The algorithm used for solving the problem is the generalised conditional gradient as proposed in  [5]_ [7]_
    
             
    Parameters
    ----------
    a : np.ndarray (ns,)
        samples weights in the source domain
    labels_a : np.ndarray (ns,)
        labels of samples in the source domain        
    b : np.ndarray (nt,)
        samples in the target domain
    M : np.ndarray (ns,nt)
        loss matrix        
    reg: float
        Regularization term for entropic regularization >0
    eta: float, optional
        Regularization term  for group lasso regularization >0        
    numItermax: int, optional
        Max number of iterations
    numInnerItermax: int, optional
        Max number of iterations (inner sinkhorn solver)
    stopInnerThr: float, optional
        Stop threshold on error (inner sinkhorn solver) (>0)        
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
    
    .. [5] N. Courty; R. Flamary; D. Tuia; A. Rakotomamonjy, "Optimal Transport for Domain Adaptation," in IEEE Transactions on Pattern Analysis and Machine Intelligence , vol.PP, no.99, pp.1-1
    .. [7] Rakotomamonjy, A., Flamary, R., & Courty, N. (2015). Generalized conditional gradient: analysis of convergence and applications. arXiv preprint arXiv:1510.06567.
    
    See Also
    --------
    ot.lp.emd : Unregularized OT
    ot.bregman.sinkhorn : Entropic regularized OT
    ot.optim.cg : General regularized OT
        
    """    
    p=0.5
    epsilon = 1e-3

    # init data
    Nini = len(a)
    Nfin = len(b)
     
    indices_labels = []
    idx_begin = np.min(labels_a)
    for c in range(idx_begin,np.max(labels_a)+1):
        idxc = indices(labels_a, lambda x: x==c)
        indices_labels.append(idxc)

    W=np.zeros(M.shape)

    for cpt in range(numItermax):
        Mreg = M + eta*W
        transp=sinkhorn(a,b,Mreg,reg,numItermax=numInnerItermax, stopThr=stopInnerThr)
        # the transport has been computed. Check if classes are really separated
        W = np.ones((Nini,Nfin))
        for t in range(Nfin):
            column = transp[:,t]
            all_maj = []
            for c in range(idx_begin,np.max(labels_a)+1):
                col_c = column[indices_labels[c-idx_begin]]
                if c!=-1:
                    maj = p*((sum(col_c)+epsilon)**(p-1))
                    W[indices_labels[c-idx_begin],t]=maj
                    all_maj.append(maj)

            # now we majorize the unlabelled by the min of the majorizations
            # do it only for unlabbled data
            if idx_begin==-1:
                W[indices_labels[0],t]=np.min(all_maj)
    
    return transp
    