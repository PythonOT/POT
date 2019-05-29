# -*- coding: utf-8 -*-
"""
Domain adaptation with optimal transport with GPU implementation
"""

# Author: Remi Flamary <remi.flamary@unice.fr>
#         Nicolas Courty <ncourty@irisa.fr>
#         Michael Perrot <michael.perrot@univ-st-etienne.fr>
#         Leo Gautheron <https://github.com/aje>
#
# License: MIT License


import cupy as np  # np used for matrix computation
import cupy as cp  # cp used for cupy specific operations
import numpy as npp
from . import utils

from .bregman import sinkhorn


def sinkhorn_lpl1_mm(a, labels_a, b, M, reg, eta=0.1, numItermax=10,
                     numInnerItermax=200, stopInnerThr=1e-9, verbose=False,
                     log=False, to_numpy=True):
    """
    Solve the entropic regularization optimal transport problem with nonconvex
    group lasso regularization on GPU

    If the input matrix are in numpy format, they will be uploaded to the
    GPU first which can incur significant time overhead.


    The function solves the following optimization problem:

    .. math::
        \gamma = arg\min_\gamma <\gamma,M>_F + reg\cdot\Omega_e(\gamma)
        + \eta \Omega_g(\gamma)

        s.t. \gamma 1 = a

             \gamma^T 1= b

             \gamma\geq 0
    where :

    - M is the (ns,nt) metric cost matrix
    - :math:`\Omega_e` is the entropic regularization term
        :math:`\Omega_e(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - :math:`\Omega_g` is the group lasso  regulaization term
      :math:`\Omega_g(\gamma)=\sum_{i,c} \|\gamma_{i,\mathcal{I}_c}\|^{1/2}_1`
      where  :math:`\mathcal{I}_c` are the index of samples from class c
      in the source domain.
    - a and b are source and target weights (sum to 1)

    The algorithm used for solving the problem is the generalised conditional
    gradient as proposed in  [5]_ [7]_


    Parameters
    ----------
    a : np.ndarray (ns,)
        samples weights in the source domain
    labels_a : np.ndarray (ns,)
        labels of samples in the source domain
    b : np.ndarray (nt,)
        samples weights in the target domain
    M : np.ndarray (ns,nt)
        loss matrix
    reg : float
        Regularization term for entropic regularization >0
    eta : float, optional
        Regularization term  for group lasso regularization >0
    numItermax : int, optional
        Max number of iterations
    numInnerItermax : int, optional
        Max number of iterations (inner sinkhorn solver)
    stopInnerThr : float, optional
        Stop threshold on error (inner sinkhorn solver) (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    to_numpy : boolean, optional (default True)
        If true convert back the GPU array result to numpy format.


    Returns
    -------
    gamma : (ns x nt) ndarray
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters


    References
    ----------

    .. [5] N. Courty; R. Flamary; D. Tuia; A. Rakotomamonjy,
       "Optimal Transport for Domain Adaptation," in IEEE
       Transactions on Pattern Analysis and Machine Intelligence ,
       vol.PP, no.99, pp.1-1
    .. [7] Rakotomamonjy, A., Flamary, R., & Courty, N. (2015).
       Generalized conditional gradient: analysis of convergence
       and applications. arXiv preprint arXiv:1510.06567.

    See Also
    --------
    ot.lp.emd : Unregularized OT
    ot.bregman.sinkhorn : Entropic regularized OT
    ot.optim.cg : General regularized OT

    """

    a, labels_a, b, M = utils.to_gpu(a, labels_a, b, M)

    p = 0.5
    epsilon = 1e-3

    indices_labels = []
    labels_a2 = cp.asnumpy(labels_a)
    classes = npp.unique(labels_a2)
    for c in classes:
        idxc, = utils.to_gpu(npp.where(labels_a2 == c))
        indices_labels.append(idxc)

    W = np.zeros(M.shape)

    for cpt in range(numItermax):
        Mreg = M + eta * W
        transp = sinkhorn(a, b, Mreg, reg, numItermax=numInnerItermax,
                          stopThr=stopInnerThr, to_numpy=False)
        # the transport has been computed. Check if classes are really
        # separated
        W = np.ones(M.shape)
        for (i, c) in enumerate(classes):

            majs = np.sum(transp[indices_labels[i]], axis=0)
            majs = p * ((majs + epsilon)**(p - 1))
            W[indices_labels[i]] = majs

    if to_numpy:
        return utils.to_np(transp)
    else:
        return transp
