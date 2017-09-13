
# -*- coding: utf-8 -*-
"""
Gromov-Wasserstein transport method
===================================


"""

# Author: Erwan Vautier <erwan.vautier@gmail.com>
#         Nicolas Courty <ncourty@irisa.fr>
#
# License: MIT License

import numpy as np

from .bregman import sinkhorn
from .utils import dist


def square_loss(a, b):
    """
    Returns the value of L(a,b)=(1/2)*|a-b|^2
    """

    return 0.5 * (a - b)**2


def kl_loss(a, b):
    """
    Returns the value of L(a,b)=a*log(a/b)-a+b
    """

    return a * np.log(a / b) - a + b


def tensor_square_loss(C1, C2, T):
    """
    Returns the value of \mathcal{L}(C1,C2) \otimes T with the square loss
    function as the loss function of Gromow-Wasserstein discrepancy.

    Where :
        C1 : Metric cost matrix in the source space
        C2 : Metric cost matrix in the target space
        T : A coupling between those two spaces

    The square-loss function L(a,b)=(1/2)*|a-b|^2 is read as :
        L(a,b) = f1(a)+f2(b)-h1(a)*h2(b) with :
            f1(a)=(a^2)/2
            f2(b)=(b^2)/2
            h1(a)=a
            h2(b)=b

    Parameters
    ----------
    C1 : ndarray, shape (ns, ns)
         Metric cost matrix in the source space
    C2 : ndarray, shape (nt, nt)
         Metric costfr matrix in the target space
    T : ndarray, shape (ns, nt)
         Coupling between source and target spaces

    Returns
    -------
    tens : ndarray, shape (ns, nt)
           \mathcal{L}(C1,C2) \otimes T tensor-matrix multiplication result
    """

    C1 = np.asarray(C1, dtype=np.float64)
    C2 = np.asarray(C2, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)

    def f1(a):
        return (a**2) / 2

    def f2(b):
        return (b**2) / 2

    def h1(a):
        return a

    def h2(b):
        return b

    tens = -np.dot(h1(C1), T).dot(h2(C2).T)
    tens -= tens.min()

    return tens


def tensor_kl_loss(C1, C2, T):
    """
    Returns the value of \mathcal{L}(C1,C2) \otimes T with the square loss
    function as the loss function of Gromow-Wasserstein discrepancy.

    Where :

        C1 : Metric cost matrix in the source space
        C2 : Metric cost matrix in the target space
        T : A coupling between those two spaces

    The square-loss function L(a,b)=(1/2)*|a-b|^2 is read as :
        L(a,b) = f1(a)+f2(b)-h1(a)*h2(b) with :
            f1(a)=a*log(a)-a
            f2(b)=b
            h1(a)=a
            h2(b)=log(b)

    Parameters
    ----------
    C1 : ndarray, shape (ns, ns)
         Metric cost matrix in the source space
    C2 : ndarray, shape (nt, nt)
         Metric costfr matrix in the target space
    T :  ndarray, shape (ns, nt)
         Coupling between source and target spaces

    Returns
    -------
    tens : ndarray, shape (ns, nt)
           \mathcal{L}(C1,C2) \otimes T tensor-matrix multiplication result

    References
    ----------
    .. [12] Peyré, Gabriel, Marco Cuturi, and Justin Solomon,
    "Gromov-Wasserstein averaging of kernel and distance matrices."
    International Conference on Machine Learning (ICML). 2016.

    """

    C1 = np.asarray(C1, dtype=np.float64)
    C2 = np.asarray(C2, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)

    def f1(a):
        return a * np.log(a + 1e-15) - a

    def f2(b):
        return b

    def h1(a):
        return a

    def h2(b):
        return np.log(b + 1e-15)

    tens = -np.dot(h1(C1), T).dot(h2(C2).T)
    tens -= tens.min()

    return tens


def update_square_loss(p, lambdas, T, Cs):
    """
    Updates C according to the L2 Loss kernel with the S Ts couplings
    calculated at each iteration

    Parameters
    ----------
    p  : ndarray, shape (N,)
         masses in the targeted barycenter
    lambdas : list of float
              list of the S spaces' weights
    T : list of S np.ndarray(ns,N)
        the S Ts couplings calculated at each iteration
    Cs : list of S ndarray, shape(ns,ns)
         Metric cost matrices

    Returns
    ----------
    C : ndarray, shape (nt,nt)
        updated C matrix
    """
    tmpsum = sum([lambdas[s] * np.dot(T[s].T, Cs[s]).dot(T[s])
                  for s in range(len(T))])
    ppt = np.outer(p, p)

    return np.divide(tmpsum, ppt)


def update_kl_loss(p, lambdas, T, Cs):
    """
    Updates C according to the KL Loss kernel with the S Ts couplings calculated at each iteration


    Parameters
    ----------
    p  : ndarray, shape (N,)
         weights in the targeted barycenter
    lambdas : list of the S spaces' weights
    T : list of S np.ndarray(ns,N)
        the S Ts couplings calculated at each iteration
    Cs : list of S ndarray, shape(ns,ns)
         Metric cost matrices

    Returns
    ----------
    C : ndarray, shape (ns,ns)
        updated C matrix
    """
    tmpsum = sum([lambdas[s] * np.dot(T[s].T, Cs[s]).dot(T[s])
                  for s in range(len(T))])
    ppt = np.outer(p, p)

    return np.exp(np.divide(tmpsum, ppt))


def gromov_wasserstein(C1, C2, p, q, loss_fun, epsilon,
                       max_iter=1000, tol=1e-9, verbose=False, log=False):
    """
    Returns the gromov-wasserstein coupling between the two measured similarity matrices

    (C1,p) and (C2,q)

    The function solves the following optimization problem:

    .. math::
        \GW = arg\min_T \sum_{i,j,k,l} L(C1_{i,k},C2_{j,l})*T_{i,j}*T_{k,l}-\epsilon(H(T))

        s.t. \GW 1 = p

             \GW^T 1= q

             \GW\geq 0

    Where :
        C1 : Metric cost matrix in the source space
        C2 : Metric cost matrix in the target space
        p  : distribution in the source space
        q  : distribution in the target space
        L  : loss function to account for the misfit between the similarity matrices
        H  : entropy

    Parameters
    ----------
    C1 : ndarray, shape (ns, ns)
         Metric cost matrix in the source space
    C2 : ndarray, shape (nt, nt)
         Metric costfr matrix in the target space
    p :  ndarray, shape (ns,)
         distribution in the source space
    q :  ndarray, shape (nt,)
         distribution in the target space
    loss_fun :  string
        loss function used for the solver either 'square_loss' or 'kl_loss'
    epsilon : float
        Regularization term >0
    max_iter : int, optional
       Max number of iterations
    tol : float, optional
        Stop threshold on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True

    Returns
    -------
    T : ndarray, shape (ns, nt)
        coupling between the two spaces that minimizes :
            \sum_{i,j,k,l} L(C1_{i,k},C2_{j,l})*T_{i,j}*T_{k,l}-\epsilon(H(T))
    """

    C1 = np.asarray(C1, dtype=np.float64)
    C2 = np.asarray(C2, dtype=np.float64)

    T = np.outer(p, q)  # Initialization

    cpt = 0
    err = 1

    while (err > tol and cpt < max_iter):

        Tprev = T

        if loss_fun == 'square_loss':
            tens = tensor_square_loss(C1, C2, T)

        elif loss_fun == 'kl_loss':
            tens = tensor_kl_loss(C1, C2, T)

        T = sinkhorn(p, q, tens, epsilon)

        if cpt % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            err = np.linalg.norm(T - Tprev)

            if log:
                log['err'].append(err)

            if verbose:
                if cpt % 200 == 0:
                    print('{:5s}|{:12s}'.format(
                        'It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))

        cpt += 1

    if log:
        return T, log
    else:
        return T


def gromov_wasserstein2(C1, C2, p, q, loss_fun, epsilon,
                        max_iter=1000, tol=1e-9, verbose=False, log=False):
    """
    Returns the gromov-wasserstein discrepancy between the two measured similarity matrices

    (C1,p) and (C2,q)

    The function solves the following optimization problem:

    .. math::
        \GW_Dist = \min_T \sum_{i,j,k,l} L(C1_{i,k},C2_{j,l})*T_{i,j}*T_{k,l}-\epsilon(H(T))

    Where :
        C1 : Metric cost matrix in the source space
        C2 : Metric cost matrix in the target space
        p  : distribution in the source space
        q  : distribution in the target space
        L  : loss function to account for the misfit between the similarity matrices
        H  : entropy

    Parameters
    ----------
    C1 : ndarray, shape (ns, ns)
         Metric cost matrix in the source space
    C2 : ndarray, shape (nt, nt)
         Metric costfr matrix in the target space
    p :  ndarray, shape (ns,)
         distribution in the source space
    q :  ndarray, shape (nt,)
         distribution in the target space
    loss_fun :  string
        loss function used for the solver either 'square_loss' or 'kl_loss'
    epsilon : float
        Regularization term >0
    max_iter : int, optional
        Max number of iterations
    tol : float, optional
        Stop threshold on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True

    Returns
    -------
    gw_dist : float
        Gromov-Wasserstein distance
    """

    if log:
        gw, logv = gromov_wasserstein(
            C1, C2, p, q, loss_fun, epsilon, max_iter, tol, verbose, log)
    else:
        gw = gromov_wasserstein(C1, C2, p, q, loss_fun,
                                epsilon, max_iter, tol, verbose, log)

    if loss_fun == 'square_loss':
        gw_dist = np.sum(gw * tensor_square_loss(C1, C2, gw))

    elif loss_fun == 'kl_loss':
        gw_dist = np.sum(gw * tensor_kl_loss(C1, C2, gw))

    if log:
        return gw_dist, logv
    else:
        return gw_dist


def gromov_barycenters(N, Cs, ps, p, lambdas, loss_fun, epsilon,
                       max_iter=1000, tol=1e-9, verbose=False, log=False, init_C=None):
    """
    Returns the gromov-wasserstein barycenters of S measured similarity matrices

    (Cs)_{s=1}^{s=S}

    The function solves the following optimization problem:

    .. math::
        C = argmin_C\in R^NxN \sum_s \lambda_s GW(C,Cs,p,ps)


    Where :

        Cs : metric cost matrix
        ps  : distribution

    Parameters
    ----------
    N  : Integer
         Size of the targeted barycenter
    Cs : list of S np.ndarray(ns,ns)
         Metric cost matrices
    ps : list of S np.ndarray(ns,)
         sample weights in the S spaces
    p  : ndarray, shape(N,)
         weights in the targeted barycenter
    lambdas : list of float
              list of the S spaces' weights
    loss_fun :  tensor-matrix multiplication function based on specific loss function
    update : function(p,lambdas,T,Cs) that updates C according to a specific Kernel
             with the S Ts couplings calculated at each iteration
    epsilon : float
        Regularization term >0
    max_iter : int, optional
        Max number of iterations
    tol : float, optional
        Stop threshol on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    init_C : bool, ndarray, shape(N,N)
             random initial value for the C matrix provided by user

    Returns
    -------
    C : ndarray, shape (N, N)
        Similarity matrix in the barycenter space (permutated arbitrarily)
    """

    S = len(Cs)

    Cs = [np.asarray(Cs[s], dtype=np.float64) for s in range(S)]
    lambdas = np.asarray(lambdas, dtype=np.float64)

    # Initialization of C : random SPD matrix (if not provided by user)
    if init_C is None:
        xalea = np.random.randn(N, 2)
        C = dist(xalea, xalea)
        C /= C.max()
    else:
        C = init_C

    cpt = 0
    err = 1

    error = []

    while(err > tol and cpt < max_iter):
        Cprev = C

        T = [gromov_wasserstein(Cs[s], C, ps[s], p, loss_fun, epsilon,
                                max_iter, 1e-5, verbose, log) for s in range(S)]
        if loss_fun == 'square_loss':
            C = update_square_loss(p, lambdas, T, Cs)

        elif loss_fun == 'kl_loss':
            C = update_kl_loss(p, lambdas, T, Cs)

        if cpt % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            err = np.linalg.norm(C - Cprev)
            error.append(err)

            if log:
                log['err'].append(err)

            if verbose:
                if cpt % 200 == 0:
                    print('{:5s}|{:12s}'.format(
                        'It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))

        cpt += 1

    return C
