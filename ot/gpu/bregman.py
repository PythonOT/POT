# -*- coding: utf-8 -*-
"""
Bregman projections for regularized OT with GPU
"""

# Author: Remi Flamary <remi.flamary@unice.fr>
#         Leo Gautheron <https://github.com/aje>
#
# License: MIT License

import numpy as np
import cupy as cp


def sinkhorn_knopp(a, b, M_GPU, reg, numItermax=1000, stopThr=1e-9,
                   verbose=False, log=False, returnAsGPU=False):
    r"""
    Solve the entropic regularization optimal transport problem on GPU

    The function solves the following optimization problem:

    .. math::
        \gamma = arg\min_\gamma <\gamma,M>_F + reg\cdot\Omega(\gamma)

        s.t. \gamma 1 = a

             \gamma^T 1= b

             \gamma\geq 0
    where :

    - M is the (ns,nt) metric cost matrix
    - :math:`\Omega` is the entropic regularization term
      :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - a and b are source and target weights (sum to 1)

    The algorithm used for solving the problem is the Sinkhorn-Knopp matrix
    scaling algorithm as proposed in [2]_


    Parameters
    ----------
    a : np.ndarray (ns,)
        samples weights in the source domain
    b : np.ndarray (nt,)
        samples in the target domain
    M_GPU : cupy.ndarray (ns,nt)
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
        return the OT matrix as a cupy.ndarray

    Returns
    -------
    gamma : (ns x nt) np.ndarray or cupy.ndarray
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

    .. [2] M. Cuturi, Sinkhorn Distances : Lightspeed Computation of Optimal
    Transport, Advances in Neural Information Processing Systems (NIPS) 26,
    2013.


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
    u = (np.ones(Nini) / Nini).reshape((Nini, 1))
    u_GPU = cp.array(u)
    a_GPU = cp.array(a.reshape((Nini, 1)))
    ones_GPU = cp.ones(u_GPU.shape)
    v = (np.ones(Nfin) / Nfin).reshape((Nfin, 1))
    v_GPU = cp.array(v)
    b_GPU = cp.array(b.reshape((Nfin, 1))).reshape(-1)
    M_GPU = cp.divide(M_GPU, -reg)

    K_GPU = cp.exp(M_GPU)

    cp.divide(ones_GPU, a_GPU, out=a_GPU)
    Kp_GPU = a_GPU.reshape(-1, 1) * K_GPU

    tmp_GPU = cp.empty(K_GPU.shape)
    tmp2_GPU = cp.empty(b_GPU.shape)
    KtransposeU_GPU = cp.empty(v_GPU.shape)

    cpt = 0
    err = 1

    while (err > stopThr and cpt < numItermax):
        uprev_GPU = u_GPU
        vprev_GPU = v_GPU
        K_GPU.T.dot(u_GPU, out=KtransposeU_GPU)
        cp.divide(b_GPU.reshape(-1, 1), KtransposeU_GPU, out=v_GPU)
        Kp_GPU.dot(v_GPU, out=u_GPU)
        cp.divide(ones_GPU, u_GPU, out=u_GPU)

        if (cp.any(KtransposeU_GPU == 0) or
                cp.any(cp.isnan(u_GPU)) or cp.any(cp.isnan(v_GPU)) or
                cp.any(cp.isinf(u_GPU)) or cp.any(cp.isinf(v_GPU))):

            # we have reached the machine precision
            # come back to previous solution and quit loop
            print('Warning: numerical errors at iteration', cpt)
            u_GPU = uprev_GPU
            v_GPU = vprev_GPU
            break
        if cpt % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations

            cp.multiply(u_GPU.reshape(-1, 1), K_GPU, out=tmp_GPU)
            cp.multiply(tmp_GPU, v_GPU.reshape(1, -1), out=tmp_GPU)
            cp.sum(tmp_GPU, axis=0, out=tmp2_GPU)
            tmp2_GPU -= b_GPU
            err = cp.linalg.norm(tmp2_GPU)**2
            if log:
                log['err'].append(err)

            if verbose:
                if cpt % 200 == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))
        cpt += 1
    if log:
        log['u'] = cp.asnumpy(u_GPU)
        log['v'] = cp.asnumpy(v_GPU)

    K_GPU = u_GPU.reshape(-1, 1) * K_GPU * v_GPU.reshape(1, -1)

    if returnAsGPU:
        res = K_GPU
    else:
        res = cp.asnumpy(K_GPU)

    if log:
        return res, log
    else:
        return res
