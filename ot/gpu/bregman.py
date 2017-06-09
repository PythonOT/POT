# -*- coding: utf-8 -*-
"""
Bregman projections for regularized OT with GPU
"""

import numpy as np
import cudamat


def sinkhorn(a, b, M_GPU, reg, numItermax=1000, stopThr=1e-9, verbose=False,
                log=False, returnAsGPU=False):
    """
    Solve the entropic regularization optimal transport problem on GPU

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
    M_GPU : cudamat.CUDAMatrix (ns,nt)
        loss matrix
    reg : float
        Regularization term >0
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshol on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    returnAsGPU : bool, optional
        return the OT matrix as a cudamat.CUDAMatrix

    Returns
    -------
    gamma : (ns x nt) ndarray
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters

    Examples
    --------

    >>> import ot
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
    # init data
    Nini = len(a)
    Nfin = len(b)

    if log:
        log = {'err': []}

    # we assume that no distances are null except those of the diagonal of
    # distances
    u = (np.ones(Nini)/Nini).reshape((Nini, 1))
    u_GPU = cudamat.CUDAMatrix(u)
    a_GPU = cudamat.CUDAMatrix(a.reshape((Nini, 1)))
    ones_GPU = cudamat.empty(u_GPU.shape).assign(1)
    v = (np.ones(Nfin)/Nfin).reshape((Nfin, 1))
    v_GPU = cudamat.CUDAMatrix(v)
    b_GPU = cudamat.CUDAMatrix(b.reshape((Nfin, 1)))

    M_GPU.divide(-reg)

    K_GPU = cudamat.exp(M_GPU)

    ones_GPU.divide(a_GPU, target=a_GPU)
    Kp_GPU = cudamat.empty(K_GPU.shape)
    K_GPU.mult_by_col(a_GPU, target=Kp_GPU)

    tmp_GPU = cudamat.empty(K_GPU.shape)

    cpt = 0
    err = 1
    while (err > stopThr and cpt < numItermax):
        uprev_GPU = u_GPU.copy()
        vprev_GPU = v_GPU.copy()

        KtransposeU_GPU = K_GPU.transpose().dot(u_GPU)
        b_GPU.divide(KtransposeU_GPU, target=v_GPU)
        ones_GPU.divide(Kp_GPU.dot(v_GPU), target=u_GPU)

        if (np.any(KtransposeU_GPU.asarray() == 0) or
           not u_GPU.allfinite() or not v_GPU.allfinite()):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            print('Warning: numerical errors at iteration', cpt)
            u_GPU = uprev_GPU.copy()
            v_GPU = vprev_GPU.copy()
            break
        if cpt % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            K_GPU.mult_by_col(u_GPU, target=tmp_GPU)
            tmp_GPU.mult_by_row(v_GPU.transpose(), target=tmp_GPU)

            bcopy_GPU = b_GPU.copy().transpose()
            bcopy_GPU.add_sums(tmp_GPU, axis=0, beta=-1)
            err = bcopy_GPU.euclid_norm()**2
            if log:
                log['err'].append(err)

            if verbose:
                if cpt % 200 == 0:
                    print('{:5s}|{:12s}'.format('It.', 'Err')+'\n'+'-'*19)
                print('{:5d}|{:8e}|'.format(cpt, err))
        cpt += 1
    if log:
        log['u'] = u_GPU.asarray()
        log['v'] = v_GPU.asarray()

    K_GPU.mult_by_col(u_GPU, target=K_GPU)
    K_GPU.mult_by_row(v_GPU.transpose(), target=K_GPU)

    if returnAsGPU:
        res = K_GPU
    else:
        res = K_GPU.asarray()

    if log:
        return res, log
    else:
        return res
