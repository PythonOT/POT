# -*- coding: utf-8 -*-
"""
Bregman projections for regularized OT with GPU
"""

import numpy as np


def sinkhorn(a, b, M_GPU, reg, numItermax=1000, stopThr=1e-9, verbose=False,
                log=False, cudamat=None):
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

    # print('err=',err,' cpt=',cpt)
    K_GPU.mult_by_col(u_GPU, target=K_GPU)
    K_GPU.mult_by_row(v_GPU.transpose(), target=K_GPU)
    if log:
        return K_GPU.asarray(), log
    else:
        return K_GPU.asarray()
