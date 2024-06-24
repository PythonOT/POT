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
from ..utils import dist, UndefinedParameter, list_to_array, check_random_state, unif
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
    loss_fun : string, optional (default='square_loss')
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
        required to satisfy marginal constraints but we strongly recommend it
        to correctly estimate the GW distance.
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

    if loss_fun not in ('square_loss', 'kl_loss'):
        raise ValueError(f"Unknown `loss_fun='{loss_fun}'`. Use one of: {'square_loss', 'kl_loss'}.")

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
    loss_fun : string, optional (default='square_loss')
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


def BAPG_gromov_wasserstein(
        C1, C2, p=None, q=None, loss_fun='square_loss', epsilon=0.1,
        symmetric=None, G0=None, max_iter=1000, tol=1e-9, marginal_loss=False,
        verbose=False, log=False):
    r"""
    Returns the Gromov-Wasserstein transport between :math:`(\mathbf{C_1}, \mathbf{p})` and :math:`(\mathbf{C_2}, \mathbf{q})`
    estimated using Bregman Alternated Projected Gradient method.

    If `marginal_loss=True`, the function solves the following Gromov-Wasserstein
    optimization problem :

    .. math::
        \mathbf{T}^* \in \mathop{\arg\min}_\mathbf{T} \quad \sum_{i,j,k,l} L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l}) \mathbf{T}_{i,j} \mathbf{T}_{k,l}

        s.t. \ \mathbf{T} \mathbf{1} &= \mathbf{p}

             \mathbf{T}^T \mathbf{1} &= \mathbf{q}

             \mathbf{T} &\geq 0

    Else, the function solves an equivalent problem [63], where constant terms only
    depending on the marginals :math:`\mathbf{p}`: and :math:`\mathbf{q}`: are
    discarded while assuming that L decomposes as in Proposition 1 in [12]:

    .. math::
        \mathbf{T}^* \in\mathop{\arg\min}_\mathbf{T} \quad - \langle h_1(\mathbf{C}_1) \mathbf{T} h_2(\mathbf{C_2})^\top , \mathbf{T} \rangle_F

        s.t. \ \mathbf{T} \mathbf{1} &= \mathbf{p}

             \mathbf{T}^T \mathbf{1} &= \mathbf{q}

             \mathbf{T} &\geq 0
    Where :

    - :math:`\mathbf{C_1}`: Metric cost matrix in the source space
    - :math:`\mathbf{C_2}`: Metric cost matrix in the target space
    - :math:`\mathbf{p}`: distribution in the source space
    - :math:`\mathbf{q}`: distribution in the target space
    - `L`: loss function to account for the misfit between the similarity matrices
        satisfying :math:`L(a, b) = f_1(a) + f_2(b) - h_1(a) h_2(b)`

    .. note:: By algorithmic design the optimal coupling :math:`\mathbf{T}`
        returned by this function does not necessarily satisfy the marginal
        constraints :math:`\mathbf{T}\mathbf{1}=\mathbf{p}` and
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
    loss_fun : string, optional (default='square_loss')
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
        required to satisfy marginal constraints but we strongly recommend it
        to correctly estimate the GW distance.
    max_iter : int, optional
        Max number of iterations
    tol : float, optional
        Stop threshold on error (>0)
    marginal_loss: bool, optional. Default is False.
        Include constant marginal terms or not in the objective function.
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        Record log if True.
    Returns
    -------
    T : array-like, shape (`ns`, `nt`)
        Optimal coupling between the two spaces

    References
    ----------
    .. [63] Li, J., Tang, J., Kong, L., Liu, H., Li, J., So, A. M. C., & Blanchet, J.
        "A Convergent Single-Loop Algorithm for Relaxation of Gromov-Wasserstein
        in Graph Data". International Conference on Learning Representations (ICLR), 2022.

    """
    if loss_fun not in ('square_loss', 'kl_loss'):
        raise ValueError(f"Unknown `loss_fun='{loss_fun}'`. Use one of: {'square_loss', 'kl_loss'}.")

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

    if marginal_loss:
        if symmetric:
            def df(T):
                return gwggrad(constC, hC1, hC2, T, nx)
        else:
            def df(T):
                return 0.5 * (gwggrad(constC, hC1, hC2, T, nx) + gwggrad(constCt, hC1t, hC2t, T, nx))

    else:
        if symmetric:
            def df(T):
                A = - nx.dot(nx.dot(hC1, T), hC2.T)
                return 2 * A
        else:
            def df(T):
                A = - nx.dot(nx.dot(hC1, T), hC2t)
                At = - nx.dot(nx.dot(hC1t, T), hC2)
                return A + At

    cpt = 0
    err = 1e15

    if log:
        log = {'err': []}

    while (err > tol and cpt < max_iter):

        Tprev = T

        # rows update
        T = T * nx.exp(- df(T) / epsilon)
        row_scaling = p / nx.sum(T, 1)
        T = nx.reshape(row_scaling, (-1, 1)) * T

        # columns update
        T = T * nx.exp(- df(T) / epsilon)
        column_scaling = q / nx.sum(T, 0)
        T = nx.reshape(column_scaling, (1, -1)) * T

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

    if nx.any(nx.isnan(T)):
        warnings.warn("Solver failed to produce a transport plan. You might "
                      "want to increase the regularization parameter `epsilon`.",
                      UserWarning)
    if log:
        log['gw_dist'] = gwloss(constC, hC1, hC2, T, nx)

        if not marginal_loss:
            log['loss'] = log['gw_dist'] - nx.sum(constC * T)

        return T, log
    else:
        return T


def BAPG_gromov_wasserstein2(
        C1, C2, p=None, q=None, loss_fun='square_loss', epsilon=0.1, symmetric=None, G0=None, max_iter=1000,
        tol=1e-9, marginal_loss=False, verbose=False, log=False):
    r"""
    Returns the Gromov-Wasserstein loss :math:`\mathbf{GW}` between :math:`(\mathbf{C_1}, \mathbf{p})` and :math:`(\mathbf{C_2}, \mathbf{q})`
    estimated using Bregman Alternated Projected Gradient method.

    If `marginal_loss=True`, the function solves the following Gromov-Wasserstein
    optimization problem :


    .. math::
        \mathbf{GW} = \mathop{\min}_\mathbf{T} \quad \sum_{i,j,k,l} L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l}) \mathbf{T}_{i,j} \mathbf{T}_{k,l}

        s.t. \ \mathbf{T} \mathbf{1} &= \mathbf{p}

             \mathbf{T}^T \mathbf{1} &= \mathbf{q}

             \mathbf{T} &\geq 0

    Else, the function solves an equivalent problem [63, 64], where constant terms only
    depending on the marginals :math:`\mathbf{p}`: and :math:`\mathbf{q}`: are
    discarded while assuming that L decomposes as in Proposition 1 in [12]:

    .. math::
        \mathop{\min}_\mathbf{T}  \quad - \langle h_1(\mathbf{C}_1) \mathbf{T} h_2(\mathbf{C_2})^\top , \mathbf{T} \rangle_F

        s.t. \ \mathbf{T} \mathbf{1} &= \mathbf{p}

             \mathbf{T}^T \mathbf{1} &= \mathbf{q}

             \mathbf{T} &\geq 0
    Where :

    - :math:`\mathbf{C_1}`: Metric cost matrix in the source space
    - :math:`\mathbf{C_2}`: Metric cost matrix in the target space
    - :math:`\mathbf{p}`: distribution in the source space
    - :math:`\mathbf{q}`: distribution in the target space
    - `L`: loss function to account for the misfit between the similarity matrices
        satisfying :math:`L(a, b) = f_1(a) + f_2(b) - h_1(a) h_2(b)`

    .. note:: By algorithmic design the optimal coupling :math:`\mathbf{T}`
        returned by this function does not necessarily satisfy the marginal
        constraints :math:`\mathbf{T}\mathbf{1}=\mathbf{p}` and
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
    loss_fun : string, optional (default='square_loss')
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
    marginal_loss: bool, optional. Default is False.
        Include constant marginal terms or not in the objective function.
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        Record log if True.

    Returns
    -------
    gw_dist : float
        Gromov-Wasserstein distance

    References
    ----------
    .. [63] Li, J., Tang, J., Kong, L., Liu, H., Li, J., So, A. M. C., & Blanchet, J.
        "A Convergent Single-Loop Algorithm for Relaxation of Gromov-Wasserstein
        in Graph Data". International Conference on Learning Representations (ICLR), 2023.

    """

    T, logv = BAPG_gromov_wasserstein(
        C1, C2, p, q, loss_fun, epsilon, symmetric, G0, max_iter,
        tol, marginal_loss, verbose, log=True)

    logv['T'] = T

    if log:
        return logv['gw_dist'], logv
    else:
        return logv['gw_dist']


def entropic_gromov_barycenters(
        N, Cs, ps=None, p=None, lambdas=None, loss_fun='square_loss',
        epsilon=0.1, symmetric=True, max_iter=1000, tol=1e-9,
        stop_criterion='barycenter', warmstartT=False, verbose=False,
        log=False, init_C=None, random_state=None, **kwargs):
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
    loss_fun : string, optional (default='square_loss')
        Loss function used for the solver either 'square_loss' or 'kl_loss'
    epsilon : float, optional
        Regularization term >0
    symmetric : bool, optional.
        Either structures are to be assumed symmetric or not. Default value is True.
        Else if set to True (resp. False), C1 and C2 will be assumed symmetric (resp. asymmetric).
    max_iter : int, optional
        Max number of iterations
    tol : float, optional
        Stop threshold on error (>0)
    stop_criterion : str, optional. Default is 'barycenter'.
        Convergence criterion taking values in ['barycenter', 'loss']. If set to 'barycenter'
        uses absolute norm variations of estimated barycenters. Else if set to 'loss'
        uses the relative variations of the loss.
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
        Only returned when log=True. It contains the keys:

        - :math:`\mathbf{T}`: list of (`N`, `ns`) transport matrices
        - :math:`\mathbf{p}`: (`N`,) barycenter weights
        - values used in convergence evaluation.

    References
    ----------
    .. [12] Gabriel Peyré, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.
    """
    if loss_fun not in ('square_loss', 'kl_loss'):
        raise ValueError(f"Unknown `loss_fun='{loss_fun}'`. Use one of: {'square_loss', 'kl_loss'}.")

    if stop_criterion not in ['barycenter', 'loss']:
        raise ValueError(f"Unknown `stop_criterion='{stop_criterion}'`. Use one of: {'barycenter', 'loss'}.")

    if isinstance(Cs[0], list):
        raise ValueError("Deprecated feature in POT 0.9.4: structures Cs[i] are lists and should be arrays from a supported backend (e.g numpy).")

    arr = [*Cs]
    if ps is not None:
        if isinstance(ps[0], list):
            raise ValueError("Deprecated feature in POT 0.9.4: weights ps[i] are lists and should be arrays from a supported backend (e.g numpy).")

        arr += [*ps]
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
    err = 1e15  # either the error on 'barycenter' or 'loss'

    if warmstartT:
        T = [None] * S

    if stop_criterion == 'barycenter':
        inner_log = False
    else:
        inner_log = True
        curr_loss = 1e15

    if log:
        log_ = {}
        log_['err'] = []
        if stop_criterion == 'loss':
            log_['loss'] = []

    while (err > tol) and (cpt < max_iter):
        if stop_criterion == 'barycenter':
            Cprev = C
        else:
            prev_loss = curr_loss

        # get transport plans
        if warmstartT:
            res = [entropic_gromov_wasserstein(
                C, Cs[s], p, ps[s], loss_fun, epsilon, symmetric, T[s],
                max_iter, 1e-4, verbose=verbose, log=inner_log, **kwargs) for s in range(S)]
        else:
            res = [entropic_gromov_wasserstein(
                C, Cs[s], p, ps[s], loss_fun, epsilon, symmetric, None,
                max_iter, 1e-4, verbose=verbose, log=inner_log, **kwargs) for s in range(S)]
        if stop_criterion == 'barycenter':
            T = res
        else:
            T = [output[0] for output in res]
            curr_loss = np.sum([output[1]['gw_dist'] for output in res])

        # update barycenters
        if loss_fun == 'square_loss':
            C = update_square_loss(p, lambdas, T, Cs, nx)
        elif loss_fun == 'kl_loss':
            C = update_kl_loss(p, lambdas, T, Cs, nx)

        # update convergence criterion
        if stop_criterion == 'barycenter':
            err = nx.norm(C - Cprev)
            if log:
                log_['err'].append(err)

        else:
            err = abs(curr_loss - prev_loss) / prev_loss if prev_loss != 0. else np.nan
            if log:
                log_['loss'].append(curr_loss)
                log_['err'].append(err)

        if verbose:
            if cpt % 200 == 0:
                print('{:5s}|{:12s}'.format(
                    'It.', 'Err') + '\n' + '-' * 19)
            print('{:5d}|{:8e}|'.format(cpt, err))

        cpt += 1

    if log:
        log_['T'] = T
        log_['p'] = p

        return C, log_
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
    loss_fun : string, optional (default='square_loss')
        Loss function used for the solver either 'square_loss' or 'kl_loss'
    epsilon : float, optional
        Regularization term >0
    symmetric : bool, optional
        Either C1 and C2 are to be assumed symmetric or not.
        If let to its default None value, a symmetry test will be conducted.
        Else if set to True (resp. False), C1 and C2 will be assumed symmetric (resp. asymmetric).
    alpha : float, optional
        Trade-off parameter (0 < alpha < 1)
    G0: array-like, shape (ns,nt), optional
        If None the initial transport plan of the solver is pq^T.
        Otherwise G0 will be used as initial transport of the solver. G0 is not
        required to satisfy marginal constraints but we strongly recommend it
        to correctly estimate the GW distance.
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

    if loss_fun not in ('square_loss', 'kl_loss'):
        raise ValueError(f"Unknown `loss_fun='{loss_fun}'`. Use one of: {'square_loss', 'kl_loss'}.")

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
    loss_fun : string, optional (default='square_loss')
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
        required to satisfy marginal constraints but we strongly recommend it
        to correctly estimate the GW distance.
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

    nx = get_backend(M, C1, C2)

    T, logv = entropic_fused_gromov_wasserstein(
        M, C1, C2, p, q, loss_fun, epsilon, symmetric, alpha, G0, max_iter,
        tol, solver, warmstart, verbose, log=True, **kwargs)

    logv['T'] = T

    lin_term = nx.sum(T * M)
    logv['quad_loss'] = (logv['fgw_dist'] - (1 - alpha) * lin_term)
    logv['lin_loss'] = lin_term * (1 - alpha)

    if log:
        return logv['fgw_dist'], logv
    else:
        return logv['fgw_dist']


def BAPG_fused_gromov_wasserstein(
        M, C1, C2, p=None, q=None, loss_fun='square_loss', epsilon=0.1,
        symmetric=None, alpha=0.5, G0=None, max_iter=1000, tol=1e-9,
        marginal_loss=False, verbose=False, log=False):
    r"""
    Returns the Fused Gromov-Wasserstein transport between :math:`(\mathbf{C_1}, \mathbf{Y_1}, \mathbf{p})` and :math:`(\mathbf{C_2}, \mathbf{Y_2}, \mathbf{q})`
    with pairwise distance matrix :math:`\mathbf{M}` between node feature matrices :math:`\mathbf{Y_1}` and :math:`\mathbf{Y_2}`,
    estimated using Bregman Alternated Projected Gradient method.

    If `marginal_loss=True`, the function solves the following Fused Gromov-Wasserstein
    optimization problem :

    .. math::
        \mathbf{T}^* \in\mathop{\arg\min}_\mathbf{T} \quad (1 - \alpha) \langle \mathbf{T}, \mathbf{M} \rangle_F +
        \alpha \sum_{i,j,k,l} L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l}) \mathbf{T}_{i,j} \mathbf{T}_{k,l}

        s.t. \ \mathbf{T} \mathbf{1} &= \mathbf{p}

             \mathbf{T}^T \mathbf{1} &= \mathbf{q}

             \mathbf{T} &\geq 0

    Else, the function solves an equivalent problem [63, 64], where constant terms only
    depending on the marginals :math:`\mathbf{p}`: and :math:`\mathbf{q}`: are
    discarded while assuming that L decomposes as in Proposition 1 in [12]:

    .. math::
        \mathbf{T}^* \in\mathop{\arg\min}_\mathbf{T} \quad (1 - \alpha) \langle \mathbf{T}, \mathbf{M} \rangle_F -
        \alpha \langle h_1(\mathbf{C}_1) \mathbf{T} h_2(\mathbf{C_2})^\top , \mathbf{T} \rangle_F
        s.t. \ \mathbf{T} \mathbf{1} &= \mathbf{p}

             \mathbf{T}^T \mathbf{1} &= \mathbf{q}

             \mathbf{T} &\geq 0

    Where :

    - :math:`\mathbf{M}`: pairwise relation matrix between features across domains
    - :math:`\mathbf{C_1}`: Metric cost matrix in the source space
    - :math:`\mathbf{C_2}`: Metric cost matrix in the target space
    - :math:`\mathbf{p}`: distribution in the source space
    - :math:`\mathbf{q}`: distribution in the target space
    - `L`: loss function to account for the misfit between the similarity and feature matrices
        satisfying :math:`L(a, b) = f_1(a) + f_2(b) - h_1(a) h_2(b)`
    - :math:`\alpha`: trade-off parameter

    .. note:: By algorithmic design the optimal coupling :math:`\mathbf{T}`
        returned by this function does not necessarily satisfy the marginal
        constraints :math:`\mathbf{T}\mathbf{1}=\mathbf{p}` and
        :math:`\mathbf{T}^T\mathbf{1}=\mathbf{q}`. So the returned Fused
        Gromov-Wasserstein loss does not necessarily satisfy distance
        properties and may be negative.

    Parameters
    ----------
    M : array-like, shape (ns, nt)
        Pairwise relation matrix between features across domains
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
    loss_fun : string, optional (default='square_loss')
        Loss function used for the solver either 'square_loss' or 'kl_loss'
    epsilon : float, optional
        Regularization term >0
    symmetric : bool, optional
        Either C1 and C2 are to be assumed symmetric or not.
        If let to its default None value, a symmetry test will be conducted.
        Else if set to True (resp. False), C1 and C2 will be assumed symmetric (resp. asymmetric).
    alpha : float, optional
        Trade-off parameter (0 < alpha < 1)
    G0: array-like, shape (ns,nt), optional
        If None the initial transport plan of the solver is pq^T.
        Otherwise G0 will be used as initial transport of the solver. G0 is not
        required to satisfy marginal constraints but we strongly recommend it
        to correctly estimate the GW distance.
    max_iter : int, optional
        Max number of iterations
    tol : float, optional
        Stop threshold on error (>0)
    marginal_loss: bool, optional. Default is False.
        Include constant marginal terms or not in the objective function.
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        Record log if True.
    Returns
    -------
    T : array-like, shape (`ns`, `nt`)
        Optimal coupling between the two joint spaces

    References
    ----------
    .. [63] Li, J., Tang, J., Kong, L., Liu, H., Li, J., So, A. M. C., & Blanchet, J.
        "A Convergent Single-Loop Algorithm for Relaxation of Gromov-Wasserstein
        in Graph Data". International Conference on Learning Representations (ICLR), 2023.

    .. [64] Ma, X., Chu, X., Wang, Y., Lin, Y., Zhao, J., Ma, L., & Zhu, W.
        "Fused Gromov-Wasserstein Graph Mixup for Graph-level Classifications".
        In Thirty-seventh Conference on Neural Information Processing Systems.
    """
    if loss_fun not in ('square_loss', 'kl_loss'):
        raise ValueError(f"Unknown `loss_fun='{loss_fun}'`. Use one of: {'square_loss', 'kl_loss'}.")

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

    # Define gradients
    if marginal_loss:
        if symmetric:
            def df(T):
                return alpha * gwggrad(constC, hC1, hC2, T, nx) + (1 - alpha) * M
        else:
            def df(T):
                return (alpha * 0.5) * (gwggrad(constC, hC1, hC2, T, nx) + gwggrad(constCt, hC1t, hC2t, T, nx)) + (1 - alpha) * M

    else:
        if symmetric:
            def df(T):
                A = - nx.dot(nx.dot(hC1, T), hC2.T)
                return 2 * alpha * A + (1 - alpha) * M
        else:
            def df(T):
                A = - nx.dot(nx.dot(hC1, T), hC2t)
                At = - nx.dot(nx.dot(hC1t, T), hC2)
                return alpha * (A + At) + (1 - alpha) * M
    cpt = 0
    err = 1e15

    if log:
        log = {'err': []}

    while (err > tol and cpt < max_iter):

        Tprev = T

        # rows update
        T = T * nx.exp(- df(T) / epsilon)
        row_scaling = p / nx.sum(T, 1)
        T = nx.reshape(row_scaling, (-1, 1)) * T

        # columns update
        T = T * nx.exp(- df(T) / epsilon)
        column_scaling = q / nx.sum(T, 0)
        T = nx.reshape(column_scaling, (1, -1)) * T

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

    if nx.any(nx.isnan(T)):
        warnings.warn("Solver failed to produce a transport plan. You might "
                      "want to increase the regularization parameter `epsilon`.",
                      UserWarning)
    if log:
        log['fgw_dist'] = (1 - alpha) * nx.sum(M * T) + alpha * gwloss(constC, hC1, hC2, T, nx)

        if not marginal_loss:
            log['loss'] = log['fgw_dist'] - alpha * nx.sum(constC * T)

        return T, log
    else:
        return T


def BAPG_fused_gromov_wasserstein2(
        M, C1, C2, p=None, q=None, loss_fun='square_loss', epsilon=0.1,
        symmetric=None, alpha=0.5, G0=None, max_iter=1000, tol=1e-9,
        marginal_loss=False, verbose=False, log=False):
    r"""
    Returns the Fused Gromov-Wasserstein loss between :math:`(\mathbf{C_1}, \mathbf{Y_1}, \mathbf{p})` and :math:`(\mathbf{C_2}, \mathbf{Y_2}, \mathbf{q})`
    with pairwise distance matrix :math:`\mathbf{M}` between node feature matrices :math:`\mathbf{Y_1}` and :math:`\mathbf{Y_2}`,
    estimated using Bregman Alternated Projected Gradient method.

    If `marginal_loss=True`, the function solves the following Fused Gromov-Wasserstein
    optimization problem :

    .. math::
        \mathbf{FGW} = \mathop{\min}_\mathbf{T} \quad (1 - \alpha) \langle \mathbf{T}, \mathbf{M} \rangle_F +
        \alpha \sum_{i,j,k,l} L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l}) \mathbf{T}_{i,j} \mathbf{T}_{k,l}

        s.t. \ \mathbf{T} \mathbf{1} &= \mathbf{p}

             \mathbf{T}^T \mathbf{1} &= \mathbf{q}

             \mathbf{T} &\geq 0

    Else, the function solves an equivalent problem [63, 64], where constant terms only
    depending on the marginals :math:`\mathbf{p}`: and :math:`\mathbf{q}`: are
    discarded while assuming that L decomposes as in Proposition 1 in [12]:

    .. math::
        \mathop{\min}_\mathbf{T} \quad (1 - \alpha) \langle \mathbf{T}, \mathbf{M} \rangle_F -
        \alpha \langle h_1(\mathbf{C}_1) \mathbf{T} h_2(\mathbf{C_2})^\top , \mathbf{T} \rangle_F
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
        satisfying :math:`L(a, b) = f_1(a) + f_2(b) - h_1(a) h_2(b)`
    - :math:`\alpha`: trade-off parameter

    .. note:: By algorithmic design the optimal coupling :math:`\mathbf{T}`
        returned by this function does not necessarily satisfy the marginal
        constraints :math:`\mathbf{T}\mathbf{1}=\mathbf{p}` and
        :math:`\mathbf{T}^T\mathbf{1}=\mathbf{q}`. So the returned Fused
        Gromov-Wasserstein loss does not necessarily satisfy distance
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
    loss_fun : string, optional (default='square_loss')
        Loss function used for the solver either 'square_loss' or 'kl_loss'
    epsilon : float, optional
        Regularization term >0
    symmetric : bool, optional
        Either C1 and C2 are to be assumed symmetric or not.
        If let to its default None value, a symmetry test will be conducted.
        Else if set to True (resp. False), C1 and C2 will be assumed symmetric (resp. asymmetric).
    alpha : float, optional
        Trade-off parameter (0 < alpha < 1)
    G0: array-like, shape (ns,nt), optional
        If None the initial transport plan of the solver is pq^T.
        Otherwise G0 will be used as initial transport of the solver. G0 is not
        required to satisfy marginal constraints but we strongly recommend it
        to correctly estimate the GW distance.
    max_iter : int, optional
        Max number of iterations
    tol : float, optional
        Stop threshold on error (>0)
    marginal_loss: bool, optional. Default is False.
        Include constant marginal terms or not in the objective function.
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        Record log if True.
    Returns
    -------
    T : array-like, shape (`ns`, `nt`)
        Optimal coupling between the two joint spaces

    References
    ----------
    .. [63] Li, J., Tang, J., Kong, L., Liu, H., Li, J., So, A. M. C., & Blanchet, J.
        "A Convergent Single-Loop Algorithm for Relaxation of Gromov-Wasserstein
        in Graph Data". International Conference on Learning Representations (ICLR), 2023.

    .. [64] Ma, X., Chu, X., Wang, Y., Lin, Y., Zhao, J., Ma, L., & Zhu, W.
        "Fused Gromov-Wasserstein Graph Mixup for Graph-level Classifications".
        In Thirty-seventh Conference on Neural Information Processing Systems.
    """
    nx = get_backend(M, C1, C2)

    T, logv = BAPG_fused_gromov_wasserstein(
        M, C1, C2, p, q, loss_fun, epsilon, symmetric, alpha, G0, max_iter,
        tol, marginal_loss, verbose, log=True)

    logv['T'] = T

    lin_term = nx.sum(T * M)
    logv['quad_loss'] = (logv['fgw_dist'] - (1 - alpha) * lin_term)
    logv['lin_loss'] = lin_term * (1 - alpha)

    if log:
        return logv['fgw_dist'], logv
    else:
        return logv['fgw_dist']


def entropic_fused_gromov_barycenters(
        N, Ys, Cs, ps=None, p=None, lambdas=None, loss_fun='square_loss',
        epsilon=0.1, symmetric=True, alpha=0.5, max_iter=1000, tol=1e-9,
        stop_criterion='barycenter', warmstartT=False, verbose=False,
        log=False, init_C=None, init_Y=None, fixed_structure=False,
        fixed_features=False, random_state=None, **kwargs):
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
    loss_fun : string, optional (default='square_loss')
        Loss function used for the solver either 'square_loss' or 'kl_loss'
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
    stop_criterion : str, optional. Default is 'barycenter'.
        Stop criterion taking values in ['barycenter', 'loss']. If set to 'barycenter'
        uses absolute norm variations of estimated barycenters. Else if set to 'loss'
        uses the relative variations of the loss.
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
    fixed_structure : bool, optional
        Whether to fix the structure of the barycenter during the updates.
    fixed_features : bool, optional
        Whether to fix the feature of the barycenter during the updates
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
        Only returned when log=True. It contains the keys:

        - :math:`\mathbf{T}`: list of (`N`, `ns`) transport matrices
        - :math:`\mathbf{p}`: (`N`,) barycenter weights
        - :math:`(\mathbf{M}_s)_s`: all distance matrices between the feature of the barycenter and the other features :math:`(dist(\mathbf{X}, \mathbf{Y}_s))_s` shape (`N`, `ns`)
        - values used in convergence evaluation.

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
    if loss_fun not in ('square_loss', 'kl_loss'):
        raise ValueError(f"Unknown `loss_fun='{loss_fun}'`. Use one of: {'square_loss', 'kl_loss'}.")

    if stop_criterion not in ['barycenter', 'loss']:
        raise ValueError(f"Unknown `stop_criterion='{stop_criterion}'`. Use one of: {'barycenter', 'loss'}.")

    if isinstance(Cs[0], list) or isinstance(Ys[0], list):
        raise ValueError("Deprecated feature in POT 0.9.4: structures Cs[i] and/or features Ys[i] are lists and should be arrays from a supported backend (e.g numpy).")

    arr = [*Cs, *Ys]
    if ps is not None:
        if isinstance(ps[0], list):
            raise ValueError("Deprecated feature in POT 0.9.4: weights ps[i] are lists and should be arrays from a supported backend (e.g numpy).")

        arr += [*ps]
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

    # Initialization of C : random euclidean distance matrix (if not provided by user)
    if fixed_structure:
        if init_C is None:
            raise UndefinedParameter('If C is fixed it must be initialized')
        else:
            C = init_C
    else:
        if init_C is None:
            generator = check_random_state(random_state)
            xalea = generator.randn(N, 2)
            C = dist(xalea, xalea)
            C = nx.from_numpy(C, type_as=ps[0])
        else:
            C = init_C

    # Initialization of Y
    if fixed_features:
        if init_Y is None:
            raise UndefinedParameter('If Y is fixed it must be initialized')
        else:
            Y = init_Y
    else:
        if init_Y is None:
            Y = nx.zeros((N, d), type_as=ps[0])

        else:
            Y = init_Y

    Ms = [dist(Y, Ys[s]) for s in range(len(Ys))]

    if warmstartT:
        T = [None] * S

    cpt = 0

    if stop_criterion == 'barycenter':
        inner_log = False
        err_feature = 1e15
        err_structure = 1e15
        err_rel_loss = 0.

    else:
        inner_log = True
        err_feature = 0.
        err_structure = 0.
        curr_loss = 1e15
        err_rel_loss = 1e15

    if log:
        log_ = {}
        if stop_criterion == 'barycenter':
            log_['err_feature'] = []
            log_['err_structure'] = []
            log_['Ts_iter'] = []
        else:
            log_['loss'] = []
            log_['err_rel_loss'] = []

    while ((err_feature > tol or err_structure > tol or err_rel_loss > tol) and cpt < max_iter):
        if stop_criterion == 'barycenter':
            Cprev = C
            Yprev = Y
        else:
            prev_loss = curr_loss

        # get transport plans
        if warmstartT:
            res = [entropic_fused_gromov_wasserstein(
                Ms[s], C, Cs[s], p, ps[s], loss_fun, epsilon, symmetric, alpha,
                T[s], max_iter, 1e-4, verbose=verbose, log=inner_log, **kwargs) for s in range(S)]

        else:
            res = [entropic_fused_gromov_wasserstein(
                Ms[s], C, Cs[s], p, ps[s], loss_fun, epsilon, symmetric, alpha,
                None, max_iter, 1e-4, verbose=verbose, log=inner_log, **kwargs) for s in range(S)]

        if stop_criterion == 'barycenter':
            T = res
        else:
            T = [output[0] for output in res]
            curr_loss = np.sum([output[1]['fgw_dist'] for output in res])

        # update barycenters
        if not fixed_features:
            Ys_temp = [y.T for y in Ys]
            X = update_feature_matrix(lambdas, Ys_temp, T, p, nx).T
            Ms = [dist(X, Ys[s]) for s in range(len(Ys))]

        if not fixed_structure:
            if loss_fun == 'square_loss':
                C = update_square_loss(p, lambdas, T, Cs, nx)

            elif loss_fun == 'kl_loss':
                C = update_kl_loss(p, lambdas, T, Cs, nx)

        # update convergence criterion
        if stop_criterion == 'barycenter':
            err_feature, err_structure = 0., 0.
            if not fixed_features:
                err_feature = nx.norm(Y - Yprev)
            if not fixed_structure:
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
        else:
            err_rel_loss = abs(curr_loss - prev_loss) / prev_loss if prev_loss != 0. else np.nan
            if log:
                log_['loss'].append(curr_loss)
                log_['err_rel_loss'].append(err_rel_loss)

            if verbose:
                if cpt % 200 == 0:
                    print('{:5s}|{:12s}'.format(
                        'It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err_rel_loss))

        cpt += 1

    if log:
        log_['T'] = T
        log_['p'] = p
        log_['Ms'] = Ms

        return Y, C, log_
    else:
        return Y, C
