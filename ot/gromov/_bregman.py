# -*- coding: utf-8 -*-
"""
Bregman projections solvers for entropic Gromov-Wasserstein
"""

# Author: Erwan Vautier <erwan.vautier@gmail.com>
#         Nicolas Courty <ncourty@irisa.fr>
#         Rémi Flamary <remi.flamary@unice.fr>
#         Titouan Vayer <titouan.vayer@irisa.fr>
#         Cédric Vincent-Cuaz <cedvincentcuaz@gmail.com>
#
# License: MIT License

import numpy as np
import warnings

from ..bregman import sinkhorn
from ..utils import dist, list_to_array, check_random_state, unif
from ..backend import get_backend

from ._utils import init_matrix, gwloss, gwggrad
from ._utils import update_square_loss, update_kl_loss, update_feature_matrix


def entropic_gromov_wasserstein(
        C1, C2, p=None, q=None, loss_fun='square_loss', epsilon=0.1, symmetric=None, G0=None, max_iter=1000,
        tol=1e-9, solver='PGD', warmstart=False, verbose=False, log=False, **kwargs):
    r"""
    Returns the Gromov-Wasserstein transport between :math:`(\mathbf{C_1}, \mathbf{p})` and :math:`(\mathbf{C_2}, \mathbf{q})`
    estimated using Sinkhorn projections.

    If `solver="PGD"`, the function solves the following entropic-regularized
    Gromov-Wasserstein optimization problem using Projected Gradient Descent [12]:

    .. math::
        \mathbf{T}^* \in \mathop{\arg\min}_\mathbf{T} \quad \sum_{i,j,k,l} L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l}) \mathbf{T}_{i,j} \mathbf{T}_{k,l} - \epsilon H(\mathbf{T})

        s.t. \ \mathbf{T} \mathbf{1} &= \mathbf{p}

             \mathbf{T}^T \mathbf{1} &= \mathbf{q}

             \mathbf{T} &\geq 0

    Else if `solver="PPA"`, the function solves the following Gromov-Wasserstein
    optimization problem using Proximal Point Algorithm [51]:

    .. math::
        \mathbf{T}^* \in \mathop{\arg\min}_\mathbf{T} \quad \sum_{i,j,k,l} L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l}) \mathbf{T}_{i,j} \mathbf{T}_{k,l}

        s.t. \ \mathbf{T} \mathbf{1} &= \mathbf{p}

             \mathbf{T}^T \mathbf{1} &= \mathbf{q}

             \mathbf{T} &\geq 0
    Where :

    - :math:`\mathbf{C_1}`: Metric cost matrix in the source space
    - :math:`\mathbf{C_2}`: Metric cost matrix in the target space
    - :math:`\mathbf{p}`: distribution in the source space
    - :math:`\mathbf{q}`: distribution in the target space
    - `L`: loss function to account for the misfit between the similarity matrices
    - `H`: entropy

    .. note:: If the inner solver `ot.sinkhorn` did not convergence, the
        optimal coupling :math:`\mathbf{T}` returned by this function does not
        necessarily satisfy the marginal constraints
        :math:`\mathbf{T}\mathbf{1}=\mathbf{p}` and
        :math:`\mathbf{T}^T\mathbf{1}=\mathbf{q}`. So the returned
        Gromov-Wasserstein loss does not necessarily satisfy distance
        properties and may be negative.

    Parameters
    ----------
    C1 : array-like, shape (ns, ns)
        Metric cost matrix in the source space
    C2 : array-like, shape (nt, nt)
        Metric cost matrix in the target space
    p : array-like, shape (ns,), optional
        Distribution in the source space.
        If let to its default value None, uniform distribution is taken.
    q : array-like, shape (nt,), optional
        Distribution in the target space.
        If let to its default value None, uniform distribution is taken.
    loss_fun :  string, optional
        Loss function used for the solver either 'square_loss' or 'kl_loss'
    epsilon : float, optional
        Regularization term >0
    symmetric : bool, optional
        Either C1 and C2 are to be assumed symmetric or not.
        If let to its default None value, a symmetry test will be conducted.
        Else if set to True (resp. False), C1 and C2 will be assumed symmetric (resp. asymmetric).
    G0: array-like, shape (ns,nt), optional
        If None the initial transport plan of the solver is pq^T.
        Otherwise G0 will be used as initial transport of the solver. G0 is not
        required to satisfy marginal constraints but we strongly recommand it
        to correcly estimate the GW distance.
    max_iter : int, optional
        Max number of iterations
    tol : float, optional
        Stop threshold on error (>0)
    solver: string, optional
        Solver to use either 'PGD' for Projected Gradient Descent or 'PPA'
        for Proximal Point Algorithm.
        Default value is 'PGD'.
    warmstart: bool, optional
        Either to perform warmstart of dual potentials in the successive
        Sinkhorn projections.
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        Record log if True.
    **kwargs: dict
        parameters can be directly passed to the ot.sinkhorn solver.
        Such as `numItermax` and `stopThr` to control its estimation precision,
        e.g [51] suggests to use `numItermax=1`.
    Returns
    -------
    T : array-like, shape (`ns`, `nt`)
        Optimal coupling between the two spaces

    References
    ----------
    .. [12] Gabriel Peyré, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.

    .. [47] Chowdhury, S., & Mémoli, F. (2019). The gromov–wasserstein
        distance between networks and stable network invariants.
        Information and Inference: A Journal of the IMA, 8(4), 757-787.

    .. [51] Xu, H., Luo, D., Zha, H., & Duke, L. C. (2019). Gromov-wasserstein
        learning for graph matching and node embedding. In International
        Conference on Machine Learning (ICML), 2019.
    """
    if solver not in ['PGD', 'PPA']:
        raise ValueError("Unknown solver '%s'. Pick one in ['PGD', 'PPA']." % solver)

    C1, C2 = list_to_array(C1, C2)
    arr = [C1, C2]
    if p is not None:
        arr.append(list_to_array(p))
    else:
        p = unif(C1.shape[0], type_as=C1)
    if q is not None:
        arr.append(list_to_array(q))
    else:
        q = unif(C2.shape[0], type_as=C2)

    if G0 is not None:
        arr.append(G0)

    nx = get_backend(*arr)

    if G0 is None:
        G0 = nx.outer(p, q)

    T = G0
    constC, hC1, hC2 = init_matrix(C1, C2, p, q, loss_fun, nx)

    if symmetric is None:
        symmetric = nx.allclose(C1, C1.T, atol=1e-10) and nx.allclose(C2, C2.T, atol=1e-10)
    if not symmetric:
        constCt, hC1t, hC2t = init_matrix(C1.T, C2.T, p, q, loss_fun, nx)

    cpt = 0
    err = 1

    if warmstart:
        # initialize potentials to cope with ot.sinkhorn initialization
        N1, N2 = C1.shape[0], C2.shape[0]
        mu = nx.zeros(N1, type_as=C1) - np.log(N1)
        nu = nx.zeros(N2, type_as=C2) - np.log(N2)

    if log:
        log = {'err': []}

    while (err > tol and cpt < max_iter):

        Tprev = T

        # compute the gradient
        if symmetric:
            tens = gwggrad(constC, hC1, hC2, T, nx)
        else:
            tens = 0.5 * (gwggrad(constC, hC1, hC2, T, nx) + gwggrad(constCt, hC1t, hC2t, T, nx))

        if solver == 'PPA':
            tens = tens - epsilon * nx.log(T)

        if warmstart:
            T, loginn = sinkhorn(p, q, tens, epsilon, method='sinkhorn', log=True, warmstart=(mu, nu), **kwargs)
            mu = epsilon * nx.log(loginn['u'])
            nu = epsilon * nx.log(loginn['v'])

        else:
            T = sinkhorn(p, q, tens, epsilon, method='sinkhorn', **kwargs)

        if cpt % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            err = nx.norm(T - Tprev)

            if log:
                log['err'].append(err)

            if verbose:
                if cpt % 200 == 0:
                    print('{:5s}|{:12s}'.format(
                        'It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))

        cpt += 1

    if abs(nx.sum(T) - 1) > 1e-5:
        warnings.warn("Solver failed to produce a transport plan. You might "
                      "want to increase the regularization parameter `epsilon`.")
    if log:
        log['gw_dist'] = gwloss(constC, hC1, hC2, T, nx)
        return T, log
    else:
        return T


def entropic_gromov_wasserstein2(
        C1, C2, p=None, q=None, loss_fun='square_loss', epsilon=0.1, symmetric=None, G0=None, max_iter=1000,
        tol=1e-9, solver='PGD', warmstart=False, verbose=False, log=False, **kwargs):
    r"""
    Returns the Gromov-Wasserstein loss :math:`\mathbf{GW}` between :math:`(\mathbf{C_1}, \mathbf{p})` and :math:`(\mathbf{C_2}, \mathbf{q})`
    estimated using Sinkhorn projections. To recover the Gromov-Wasserstein distance as defined in [13] compute :math:`d_{GW} = \frac{1}{2} \sqrt{\mathbf{GW}}`.

    If `solver="PGD"`, the function solves the following entropic-regularized
    Gromov-Wasserstein optimization problem using Projected Gradient Descent [12]:

    .. math::
        \mathbf{GW} = \mathop{\min}_\mathbf{T} \quad \sum_{i,j,k,l} L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l}) \mathbf{T}_{i,j} \mathbf{T}_{k,l} - \epsilon H(\mathbf{T})

        s.t. \ \mathbf{T} \mathbf{1} &= \mathbf{p}

             \mathbf{T}^T \mathbf{1} &= \mathbf{q}

             \mathbf{T} &\geq 0

    Else if `solver="PPA"`, the function solves the following Gromov-Wasserstein
    optimization problem using Proximal Point Algorithm [51]:

    .. math::
        \mathbf{GW} = \mathop{\min}_\mathbf{T} \quad \sum_{i,j,k,l} L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l}) \mathbf{T}_{i,j} \mathbf{T}_{k,l}

        s.t. \ \mathbf{T} \mathbf{1} &= \mathbf{p}

             \mathbf{T}^T \mathbf{1} &= \mathbf{q}

             \mathbf{T} &\geq 0
    Where :

    - :math:`\mathbf{C_1}`: Metric cost matrix in the source space
    - :math:`\mathbf{C_2}`: Metric cost matrix in the target space
    - :math:`\mathbf{p}`: distribution in the source space
    - :math:`\mathbf{q}`: distribution in the target space
    - `L`: loss function to account for the misfit between the similarity matrices
    - `H`: entropy

    .. note:: If the inner solver `ot.sinkhorn` did not convergence, the
        optimal coupling :math:`\mathbf{T}` returned by this function does not
        necessarily satisfy the marginal constraints
        :math:`\mathbf{T}\mathbf{1}=\mathbf{p}` and
        :math:`\mathbf{T}^T\mathbf{1}=\mathbf{q}`. So the returned
        Gromov-Wasserstein loss does not necessarily satisfy distance
        properties and may be negative.

    Parameters
    ----------
    C1 : array-like, shape (ns, ns)
        Metric cost matrix in the source space
    C2 : array-like, shape (nt, nt)
        Metric cost matrix in the target space
    p : array-like, shape (ns,), optional
        Distribution in the source space.
        If let to its default value None, uniform distribution is taken.
    q : array-like, shape (nt,), optional
        Distribution in the target space.
        If let to its default value None, uniform distribution is taken.
    loss_fun :  string, optional
        Loss function used for the solver either 'square_loss' or 'kl_loss'
    epsilon : float, optional
        Regularization term >0
    symmetric : bool, optional
        Either C1 and C2 are to be assumed symmetric or not.
        If let to its default None value, a symmetry test will be conducted.
        Else if set to True (resp. False), C1 and C2 will be assumed symmetric (resp. asymmetric).
    G0: array-like, shape (ns,nt), optional
        If None the initial transport plan of the solver is pq^T.
        Otherwise G0 will be used as initial transport of the solver. G0 is not
        required to satisfy marginal constraints but we strongly recommand it
        to correcly estimate the GW distance.
    max_iter : int, optional
        Max number of iterations
    tol : float, optional
        Stop threshold on error (>0)
    solver: string, optional
        Solver to use either 'PGD' for Projected Gradient Descent or 'PPA'
        for Proximal Point Algorithm.
        Default value is 'PGD'.
    warmstart: bool, optional
        Either to perform warmstart of dual potentials in the successive
        Sinkhorn projections.
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        Record log if True.
    **kwargs: dict
        parameters can be directly passed to the ot.sinkhorn solver.
        Such as `numItermax` and `stopThr` to control its estimation precision,
        e.g [51] suggests to use `numItermax=1`.
    Returns
    -------
    gw_dist : float
        Gromov-Wasserstein distance

    References
    ----------
    .. [12] Gabriel Peyré, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.

    .. [51] Xu, H., Luo, D., Zha, H., & Duke, L. C. (2019). Gromov-wasserstein
        learning for graph matching and node embedding. In International
        Conference on Machine Learning (ICML), 2019.
    """
    T, logv = entropic_gromov_wasserstein(
        C1, C2, p, q, loss_fun, epsilon, symmetric, G0, max_iter,
        tol, solver, warmstart, verbose, log=True, **kwargs)

    logv['T'] = T

    if log:
        return logv['gw_dist'], logv
    else:
        return logv['gw_dist']


def entropic_gromov_barycenters(
        N, Cs, ps=None, p=None, lambdas=None, loss_fun='square_loss',
        epsilon=0.1, symmetric=True, max_iter=1000, tol=1e-9, warmstartT=False,
        verbose=False, log=False, init_C=None, random_state=None, **kwargs):
    r"""
    Returns the Gromov-Wasserstein barycenters of `S` measured similarity matrices :math:`(\mathbf{C}_s)_{1 \leq s \leq S}`
    estimated using Gromov-Wasserstein transports from Sinkhorn projections.

    The function solves the following optimization problem:

    .. math::

        \mathbf{C}^* = \mathop{\arg \min}_{\mathbf{C}\in \mathbb{R}^{N \times N}} \quad \sum_s \lambda_s \mathrm{GW}(\mathbf{C}, \mathbf{C}_s, \mathbf{p}, \mathbf{p}_s)

    Where :

    - :math:`\mathbf{C}_s`: metric cost matrix
    - :math:`\mathbf{p}_s`: distribution

    Parameters
    ----------
    N : int
        Size of the targeted barycenter
    Cs : list of S array-like of shape (ns,ns)
        Metric cost matrices
    ps : list of S array-like of shape (ns,), optional
        Sample weights in the `S` spaces.
        If let to its default value None, uniform distributions are taken.
    p : array-like, shape (N,), optional
        Weights in the targeted barycenter.
        If let to its default value None, uniform distribution is taken.
    lambdas : list of float, optional
        List of the `S` spaces' weights.
        If let to its default value None, uniform weights are taken.
    loss_fun : callable, optional
        tensor-matrix multiplication function based on specific loss function
    epsilon : float, optional
        Regularization term >0
    symmetric : bool, optional.
        Either structures are to be assumed symmetric or not. Default value is True.
        Else if set to True (resp. False), C1 and C2 will be assumed symmetric (resp. asymmetric).
    max_iter : int, optional
        Max number of iterations
    tol : float, optional
        Stop threshold on error (>0)
    warmstartT: bool, optional
        Either to perform warmstart of transport plans in the successive
        gromov-wasserstein transport problems.
    verbose : bool, optional
        Print information along iterations.
    log : bool, optional
        Record log if True.
    init_C : bool | array-like, shape (N, N)
        Random initial value for the :math:`\mathbf{C}` matrix provided by user.
    random_state : int or RandomState instance, optional
        Fix the seed for reproducibility
    **kwargs: dict
        parameters can be directly passed to the `ot.entropic_gromov_wasserstein` solver.

    Returns
    -------
    C : array-like, shape (`N`, `N`)
        Similarity matrix in the barycenter space (permutated arbitrarily)
    log : dict
        Log dictionary of error during iterations. Return only if `log=True` in parameters.

    References
    ----------
    .. [12] Gabriel Peyré, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.
    """
    Cs = list_to_array(*Cs)
    arr = [*Cs]
    if ps is not None:
        arr += list_to_array(*ps)
    else:
        ps = [unif(C.shape[0], type_as=C) for C in Cs]
    if p is not None:
        arr.append(list_to_array(p))
    else:
        p = unif(N, type_as=Cs[0])

    nx = get_backend(*arr)

    S = len(Cs)
    if lambdas is None:
        lambdas = [1. / S] * S

    # Initialization of C : random SPD matrix (if not provided by user)
    if init_C is None:
        generator = check_random_state(random_state)
        xalea = generator.randn(N, 2)
        C = dist(xalea, xalea)
        C /= C.max()
        C = nx.from_numpy(C, type_as=p)
    else:
        C = init_C

    cpt = 0
    err = 1

    error = []

    if warmstartT:
        T = [None] * S

    while (err > tol) and (cpt < max_iter):
        Cprev = C
        if warmstartT:
            T = [entropic_gromov_wasserstein(
                Cs[s], C, ps[s], p, loss_fun, epsilon, symmetric, T[s],
                max_iter, 1e-4, verbose=verbose, log=False, **kwargs) for s in range(S)]
        else:
            T = [entropic_gromov_wasserstein(
                Cs[s], C, ps[s], p, loss_fun, epsilon, symmetric, None,
                max_iter, 1e-4, verbose=verbose, log=False, **kwargs) for s in range(S)]

        if loss_fun == 'square_loss':
            C = update_square_loss(p, lambdas, T, Cs)

        elif loss_fun == 'kl_loss':
            C = update_kl_loss(p, lambdas, T, Cs)

        if cpt % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            err = nx.norm(C - Cprev)
            error.append(err)

            if verbose:
                if cpt % 200 == 0:
                    print('{:5s}|{:12s}'.format(
                        'It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))

        cpt += 1

    if log:
        return C, {"err": error}
    else:
        return C


def entropic_fused_gromov_wasserstein(
        M, C1, C2, p=None, q=None, loss_fun='square_loss', epsilon=0.1,
        symmetric=None, alpha=0.5, G0=None, max_iter=1000, tol=1e-9,
        solver='PGD', warmstart=False, verbose=False, log=False, **kwargs):
    r"""
    Returns the Fused Gromov-Wasserstein transport between :math:`(\mathbf{C_1}, \mathbf{Y_1}, \mathbf{p})` and :math:`(\mathbf{C_2}, \mathbf{Y_2}, \mathbf{q})`
    with pairwise distance matrix :math:`\mathbf{M}` between node feature matrices :math:`\mathbf{Y_1}` and :math:`\mathbf{Y_2}`,
    estimated using Sinkhorn projections.

    If `solver="PGD"`, the function solves the following entropic-regularized
    Fused Gromov-Wasserstein optimization problem using Projected Gradient Descent [12]:

    .. math::
        \mathbf{T}^* \in \mathop{\arg\min}_\mathbf{T} \quad (1 - \alpha) \langle \mathbf{T}, \mathbf{M} \rangle_F +
        \alpha \sum_{i,j,k,l} L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l}) \mathbf{T}_{i,j} \mathbf{T}_{k,l} - \epsilon H(\mathbf{T})

        s.t. \ \mathbf{T} \mathbf{1} &= \mathbf{p}

             \mathbf{T}^T \mathbf{1} &= \mathbf{q}

             \mathbf{T} &\geq 0

    Else if `solver="PPA"`, the function solves the following Fused Gromov-Wasserstein
    optimization problem using Proximal Point Algorithm [51]:

    .. math::
        \mathbf{T}^* \in\mathop{\arg\min}_\mathbf{T} \quad (1 - \alpha) \langle \mathbf{T}, \mathbf{M} \rangle_F +
        \alpha \sum_{i,j,k,l} L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l}) \mathbf{T}_{i,j} \mathbf{T}_{k,l}

        s.t. \ \mathbf{T} \mathbf{1} &= \mathbf{p}

             \mathbf{T}^T \mathbf{1} &= \mathbf{q}

             \mathbf{T} &\geq 0
    Where :

    - :math:`\mathbf{M}`: metric cost matrix between features across domains
    - :math:`\mathbf{C_1}`: Metric cost matrix in the source space
    - :math:`\mathbf{C_2}`: Metric cost matrix in the target space
    - :math:`\mathbf{p}`: distribution in the source space
    - :math:`\mathbf{q}`: distribution in the target space
    - `L`: loss function to account for the misfit between the similarity and feature matrices
    - `H`: entropy
    - :math:`\alpha`: trade-off parameter

    .. note:: If the inner solver `ot.sinkhorn` did not convergence, the
        optimal coupling :math:`\mathbf{T}` returned by this function does not
        necessarily satisfy the marginal constraints
        :math:`\mathbf{T}\mathbf{1}=\mathbf{p}` and
        :math:`\mathbf{T}^T\mathbf{1}=\mathbf{q}`. So the returned
        Fused Gromov-Wasserstein loss does not necessarily satisfy distance
        properties and may be negative.

    Parameters
    ----------
    M : array-like, shape (ns, nt)
        Metric cost matrix between features across domains
    C1 : array-like, shape (ns, ns)
        Metric cost matrix in the source space
    C2 : array-like, shape (nt, nt)
        Metric cost matrix in the target space
    p : array-like, shape (ns,), optional
        Distribution in the source space.
        If let to its default value None, uniform distribution is taken.
    q : array-like, shape (nt,), optional
        Distribution in the target space.
        If let to its default value None, uniform distribution is taken.
    loss_fun :  string, optional
        Loss function used for the solver either 'square_loss' or 'kl_loss'
    epsilon : float, optional
        Regularization term >0
    symmetric : bool, optional
        Either C1 and C2 are to be assumed symmetric or not.
        If let to its default None value, a symmetry test will be conducted.
        Else if set to True (resp. False), C1 and C2 will be assumed symmetric (resp. asymetric).
    alpha : float, optional
        Trade-off parameter (0 < alpha < 1)
    G0: array-like, shape (ns,nt), optional
        If None the initial transport plan of the solver is pq^T.
        Otherwise G0 will be used as initial transport of the solver. G0 is not
        required to satisfy marginal constraints but we strongly recommand it
        to correcly estimate the GW distance.
    max_iter : int, optional
        Max number of iterations
    tol : float, optional
        Stop threshold on error (>0)
    solver: string, optional
        Solver to use either 'PGD' for Projected Gradient Descent or 'PPA'
        for Proximal Point Algorithm.
        Default value is 'PGD'.
    warmstart: bool, optional
        Either to perform warmstart of dual potentials in the successive
        Sinkhorn projections.
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        Record log if True.
    **kwargs: dict
        parameters can be directly passed to the ot.sinkhorn solver.
        Such as `numItermax` and `stopThr` to control its estimation precision,
        e.g [51] suggests to use `numItermax=1`.
    Returns
    -------
    T : array-like, shape (`ns`, `nt`)
        Optimal coupling between the two joint spaces

    References
    ----------
    .. [12] Gabriel Peyré, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.

    .. [47] Chowdhury, S., & Mémoli, F. (2019). The gromov–wasserstein
        distance between networks and stable network invariants.
        Information and Inference: A Journal of the IMA, 8(4), 757-787.

    .. [51] Xu, H., Luo, D., Zha, H., & Duke, L. C. (2019). Gromov-wasserstein
        learning for graph matching and node embedding. In International
        Conference on Machine Learning (ICML), 2019.

    .. [24] Vayer Titouan, Chapel Laetitia, Flamary Rémi, Tavenard Romain
        and Courty Nicolas "Optimal Transport for structured data with
        application on graphs", International Conference on Machine Learning
        (ICML). 2019.
    """
    if solver not in ['PGD', 'PPA']:
        raise ValueError("Unknown solver '%s'. Pick one in ['PGD', 'PPA']." % solver)

    M, C1, C2 = list_to_array(M, C1, C2)
    arr = [M, C1, C2]
    if p is not None:
        arr.append(list_to_array(p))
    else:
        p = unif(C1.shape[0], type_as=C1)
    if q is not None:
        arr.append(list_to_array(q))
    else:
        q = unif(C2.shape[0], type_as=C2)

    if G0 is not None:
        arr.append(G0)

    nx = get_backend(*arr)

    if G0 is None:
        G0 = nx.outer(p, q)

    T = G0
    constC, hC1, hC2 = init_matrix(C1, C2, p, q, loss_fun, nx)
    if symmetric is None:
        symmetric = nx.allclose(C1, C1.T, atol=1e-10) and nx.allclose(C2, C2.T, atol=1e-10)
    if not symmetric:
        constCt, hC1t, hC2t = init_matrix(C1.T, C2.T, p, q, loss_fun, nx)
    cpt = 0
    err = 1

    if warmstart:
        # initialize potentials to cope with ot.sinkhorn initialization
        N1, N2 = C1.shape[0], C2.shape[0]
        mu = nx.zeros(N1, type_as=C1) - np.log(N1)
        nu = nx.zeros(N2, type_as=C2) - np.log(N2)

    if log:
        log = {'err': []}

    while (err > tol and cpt < max_iter):

        Tprev = T

        # compute the gradient
        if symmetric:
            tens = alpha * gwggrad(constC, hC1, hC2, T, nx) + (1 - alpha) * M
        else:
            tens = (alpha * 0.5) * (gwggrad(constC, hC1, hC2, T, nx) + gwggrad(constCt, hC1t, hC2t, T, nx)) + (1 - alpha) * M

        if solver == 'PPA':
            tens = tens - epsilon * nx.log(T)

        if warmstart:
            T, loginn = sinkhorn(p, q, tens, epsilon, method='sinkhorn', log=True, warmstart=(mu, nu), **kwargs)
            mu = epsilon * nx.log(loginn['u'])
            nu = epsilon * nx.log(loginn['v'])

        else:
            T = sinkhorn(p, q, tens, epsilon, method='sinkhorn', **kwargs)

        if cpt % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            err = nx.norm(T - Tprev)

            if log:
                log['err'].append(err)

            if verbose:
                if cpt % 200 == 0:
                    print('{:5s}|{:12s}'.format(
                        'It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))

        cpt += 1

    if abs(nx.sum(T) - 1) > 1e-5:
        warnings.warn("Solver failed to produce a transport plan. You might "
                      "want to increase the regularization parameter `epsilon`.")
    if log:
        log['fgw_dist'] = (1 - alpha) * nx.sum(M * T) + alpha * gwloss(constC, hC1, hC2, T, nx)
        return T, log
    else:
        return T


def entropic_fused_gromov_wasserstein2(
        M, C1, C2, p=None, q=None, loss_fun='square_loss', epsilon=0.1,
        symmetric=None, alpha=0.5, G0=None, max_iter=1000, tol=1e-9,
        solver='PGD', warmstart=False, verbose=False, log=False, **kwargs):
    r"""
    Returns the Fused Gromov-Wasserstein distance between :math:`(\mathbf{C_1}, \mathbf{Y_1}, \mathbf{p})` and :math:`(\mathbf{C_2}, \mathbf{Y_2}, \mathbf{q})`
    with pairwise distance matrix :math:`\mathbf{M}` between node feature matrices :math:`\mathbf{Y_1}` and :math:`\mathbf{Y_2}`,
    estimated using Sinkhorn projections.

    If `solver="PGD"`, the function solves the following entropic-regularized
    Fused Gromov-Wasserstein optimization problem using Projected Gradient Descent [12]:

    .. math::
        \mathbf{FGW} = \mathop{\min}_\mathbf{T} \quad (1 - \alpha) \langle \mathbf{T}, \mathbf{M} \rangle_F +
        \alpha \sum_{i,j,k,l} L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l}) \mathbf{T}_{i,j} \mathbf{T}_{k,l} - \epsilon H(\mathbf{T})

        s.t. \ \mathbf{T} \mathbf{1} &= \mathbf{p}

             \mathbf{T}^T \mathbf{1} &= \mathbf{q}

             \mathbf{T} &\geq 0

    Else if `solver="PPA"`, the function solves the following Fused Gromov-Wasserstein
    optimization problem using Proximal Point Algorithm [51]:

    .. math::
        \mathbf{FGW} = \mathop{\min}_\mathbf{T} \quad (1 - \alpha) \langle \mathbf{T}, \mathbf{M} \rangle_F +
        \alpha \sum_{i,j,k,l} L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l}) \mathbf{T}_{i,j} \mathbf{T}_{k,l}

        s.t. \ \mathbf{T} \mathbf{1} &= \mathbf{p}

             \mathbf{T}^T \mathbf{1} &= \mathbf{q}

             \mathbf{T} &\geq 0
    Where :

    - :math:`\mathbf{M}`: metric cost matrix between features across domains
    - :math:`\mathbf{C_1}`: Metric cost matrix in the source space
    - :math:`\mathbf{C_2}`: Metric cost matrix in the target space
    - :math:`\mathbf{p}`: distribution in the source space
    - :math:`\mathbf{q}`: distribution in the target space
    - `L`: loss function to account for the misfit between the similarity and feature matrices
    - `H`: entropy
    - :math:`\alpha`: trade-off parameter

    .. note:: If the inner solver `ot.sinkhorn` did not convergence, the
        optimal coupling :math:`\mathbf{T}` returned by this function does not
        necessarily satisfy the marginal constraints
        :math:`\mathbf{T}\mathbf{1}=\mathbf{p}` and
        :math:`\mathbf{T}^T\mathbf{1}=\mathbf{q}`. So the returned
        Fused Gromov-Wasserstein loss does not necessarily satisfy distance
        properties and may be negative.

    Parameters
    ----------
    M : array-like, shape (ns, nt)
        Metric cost matrix between features across domains
    C1 : array-like, shape (ns, ns)
        Metric cost matrix in the source space
    C2 : array-like, shape (nt, nt)
        Metric cost matrix in the target space
    p : array-like, shape (ns,), optional
        Distribution in the source space.
        If let to its default value None, uniform distribution is taken.
    q : array-like, shape (nt,), optional
        Distribution in the target space.
        If let to its default value None, uniform distribution is taken.
    loss_fun :  string, optional
        Loss function used for the solver either 'square_loss' or 'kl_loss'
    epsilon : float, optional
        Regularization term >0
    symmetric : bool, optional
        Either C1 and C2 are to be assumed symmetric or not.
        If let to its default None value, a symmetry test will be conducted.
        Else if set to True (resp. False), C1 and C2 will be assumed symmetric (resp. asymetric).
    alpha : float, optional
        Trade-off parameter (0 < alpha < 1)
    G0: array-like, shape (ns,nt), optional
        If None the initial transport plan of the solver is pq^T.
        Otherwise G0 will be used as initial transport of the solver. G0 is not
        required to satisfy marginal constraints but we strongly recommand it
        to correcly estimate the GW distance.
    max_iter : int, optional
        Max number of iterations
    tol : float, optional
        Stop threshold on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        Record log if True.

    Returns
    -------
    fgw_dist : float
        Fused Gromov-Wasserstein distance

    References
    ----------
    .. [12] Gabriel Peyré, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.

    .. [51] Xu, H., Luo, D., Zha, H., & Duke, L. C. (2019). Gromov-wasserstein
        learning for graph matching and node embedding. In International
        Conference on Machine Learning (ICML), 2019.

    .. [24] Vayer Titouan, Chapel Laetitia, Flamary Rémi, Tavenard Romain
        and Courty Nicolas "Optimal Transport for structured data with
        application on graphs", International Conference on Machine Learning
        (ICML). 2019.

    """
    T, logv = entropic_fused_gromov_wasserstein(
        M, C1, C2, p, q, loss_fun, epsilon, symmetric, alpha, G0, max_iter,
        tol, solver, warmstart, verbose, log=True, **kwargs)

    logv['T'] = T

    if log:
        return logv['fgw_dist'], logv
    else:
        return logv['fgw_dist']


def entropic_fused_gromov_barycenters(
        N, Ys, Cs, ps=None, p=None, lambdas=None, loss_fun='square_loss',
        epsilon=0.1, symmetric=True, alpha=0.5, max_iter=1000, tol=1e-9,
        warmstartT=False, verbose=False, log=False, init_C=None, init_Y=None,
        random_state=None, **kwargs):
    r"""
    Returns the Fused Gromov-Wasserstein barycenters of `S` measurable networks with node features :math:`(\mathbf{C}_s, \mathbf{Y}_s, \mathbf{p}_s)_{1 \leq s \leq S}`
    estimated using Fused Gromov-Wasserstein transports from Sinkhorn projections.

    The function solves the following optimization problem:

    .. math::

        \mathbf{C}^*, \mathbf{Y}^* = \mathop{\arg \min}_{\mathbf{C}\in \mathbb{R}^{N \times N}, \mathbf{Y}\in \mathbb{Y}^{N \times d}} \quad \sum_s \lambda_s \mathrm{FGW}_{\alpha}(\mathbf{C}, \mathbf{C}_s, \mathbf{Y}, \mathbf{Y}_s, \mathbf{p}, \mathbf{p}_s)

    Where :

    - :math:`\mathbf{Y}_s`: feature matrix
    - :math:`\mathbf{C}_s`: metric cost matrix
    - :math:`\mathbf{p}_s`: distribution

    Parameters
    ----------
    N : int
        Size of the targeted barycenter
    Ys: list of array-like, each element has shape (ns,d)
        Features of all samples
    Cs : list of S array-like of shape (ns,ns)
        Metric cost matrices
    ps : list of S array-like of shape (ns,), optional
        Sample weights in the `S` spaces.
        If let to its default value None, uniform distributions are taken.
    p : array-like, shape (N,), optional
        Weights in the targeted barycenter.
        If let to its default value None, uniform distribution is taken.
    lambdas : list of float, optional
        List of the `S` spaces' weights.
        If let to its default value None, uniform weights are taken.
    loss_fun : callable, optional
        tensor-matrix multiplication function based on specific loss function
    epsilon : float, optional
        Regularization term >0
    symmetric : bool, optional.
        Either structures are to be assumed symmetric or not. Default value is True.
        Else if set to True (resp. False), C1 and C2 will be assumed symmetric (resp. asymmetric).
    alpha : float, optional
        Trade-off parameter (0 < alpha < 1)
    max_iter : int, optional
        Max number of iterations
    tol : float, optional
        Stop threshold on error (>0)
    warmstartT: bool, optional
        Either to perform warmstart of transport plans in the successive
        fused gromov-wasserstein transport problems.
    verbose : bool, optional
        Print information along iterations.
    log : bool, optional
        Record log if True.
    init_C : bool | array-like, shape (N, N)
        Random initial value for the :math:`\mathbf{C}` matrix provided by user.
    init_Y : array-like, shape (N,d), optional
        Initialization for the barycenters' features. If not set a
        random init is used.
    random_state : int or RandomState instance, optional
        Fix the seed for reproducibility
    **kwargs: dict
        parameters can be directly passed to the `ot.entropic_fused_gromov_wasserstein` solver.

    Returns
    -------
    Y : array-like, shape (`N`, `d`)
        Feature matrix in the barycenter space (permutated arbitrarily)
    C : array-like, shape (`N`, `N`)
        Similarity matrix in the barycenter space (permutated as Y's rows)
    log : dict
        Log dictionary of error during iterations. Return only if `log=True` in parameters.

    References
    ----------
    .. [12] Gabriel Peyré, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.

    .. [24] Vayer Titouan, Chapel Laetitia, Flamary Rémi, Tavenard Romain
        and Courty Nicolas
        "Optimal Transport for structured data with application on graphs"
        International Conference on Machine Learning (ICML). 2019.
    """
    Cs = list_to_array(*Cs)
    Ys = list_to_array(*Ys)
    arr = [*Cs, *Ys]
    if ps is not None:
        arr += list_to_array(*ps)
    else:
        ps = [unif(C.shape[0], type_as=C) for C in Cs]
    if p is not None:
        arr.append(list_to_array(p))
    else:
        p = unif(N, type_as=Cs[0])

    nx = get_backend(*arr)
    S = len(Cs)
    if lambdas is None:
        lambdas = [1. / S] * S

    d = Ys[0].shape[1]  # dimension on the node features

    # Initialization of C : random SPD matrix (if not provided by user)
    if init_C is None:
        generator = check_random_state(random_state)
        xalea = generator.randn(N, 2)
        C = dist(xalea, xalea)
        C /= C.max()
        C = nx.from_numpy(C, type_as=p)
    else:
        C = init_C

    # Initialization of Y
    if init_Y is None:
        Y = nx.zeros((N, d), type_as=ps[0])
    else:
        Y = init_Y

    T = [nx.outer(p_, p) for p_ in ps]

    Ms = [dist(Ys[s], Y) for s in range(len(Ys))]

    cpt = 0
    err = 1

    err_feature = 1
    err_structure = 1

    if warmstartT:
        T = [None] * S

    if log:
        log_ = {}
        log_['err_feature'] = []
        log_['err_structure'] = []
        log_['Ts_iter'] = []

    while (err > tol) and (cpt < max_iter):
        Cprev = C
        Yprev = Y

        if warmstartT:
            T = [entropic_fused_gromov_wasserstein(
                Ms[s], Cs[s], C, ps[s], p, loss_fun, epsilon, symmetric, alpha,
                None, max_iter, 1e-4, verbose=verbose, log=False, **kwargs) for s in range(S)]

        else:
            T = [entropic_fused_gromov_wasserstein(
                Ms[s], Cs[s], C, ps[s], p, loss_fun, epsilon, symmetric, alpha,
                None, max_iter, 1e-4, verbose=verbose, log=False, **kwargs) for s in range(S)]

        if loss_fun == 'square_loss':
            C = update_square_loss(p, lambdas, T, Cs)

        elif loss_fun == 'kl_loss':
            C = update_kl_loss(p, lambdas, T, Cs)

        Ys_temp = [y.T for y in Ys]
        T_temp = [Ts.T for Ts in T]
        Y = update_feature_matrix(lambdas, Ys_temp, T_temp, p)
        Ms = [dist(Ys[s], Y) for s in range(len(Ys))]

        if cpt % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            err_feature = nx.norm(Y - nx.reshape(Yprev, (N, d)))
            err_structure = nx.norm(C - Cprev)
            if log:
                log_['err_feature'].append(err_feature)
                log_['err_structure'].append(err_structure)
                log_['Ts_iter'].append(T)

            if verbose:
                if cpt % 200 == 0:
                    print('{:5s}|{:12s}'.format(
                        'It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err_structure))
                print('{:5d}|{:8e}|'.format(cpt, err_feature))

        cpt += 1
        print('Y type:', type(Y))
    if log:
        log_['T'] = T  # from target to Ys
        log_['p'] = p
        log_['Ms'] = Ms

    if log:
        return Y, C, log_
    else:
        return Y, C
