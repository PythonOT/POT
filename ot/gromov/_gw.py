# -*- coding: utf-8 -*-
"""
Gromov-Wasserstein and Fused-Gromov-Wasserstein conditional gradient solvers.
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


from ..utils import dist, UndefinedParameter, list_to_array
from ..optim import cg, line_search_armijo, solve_1d_linesearch_quad
from ..utils import check_random_state, unif
from ..backend import get_backend, NumpyBackend

from ._utils import init_matrix, gwloss, gwggrad
from ._utils import update_square_loss, update_kl_loss, update_feature_matrix


def gromov_wasserstein(C1, C2, p=None, q=None, loss_fun='square_loss', symmetric=None, log=False, armijo=False, G0=None,
                       max_iter=1e4, tol_rel=1e-9, tol_abs=1e-9, **kwargs):
    r"""
    Returns the Gromov-Wasserstein transport between :math:`(\mathbf{C_1}, \mathbf{p})` and :math:`(\mathbf{C_2}, \mathbf{q})`.

    The function solves the following optimization problem using Conditional Gradient:

    .. math::
        \mathbf{T}^* \in \mathop{\arg \min}_\mathbf{T} \quad \sum_{i,j,k,l}
        L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l}) \mathbf{T}_{i,j} \mathbf{T}_{k,l}

        s.t. \ \mathbf{\gamma} \mathbf{1} &= \mathbf{p}

             \mathbf{\gamma}^T \mathbf{1} &= \mathbf{q}

             \mathbf{\gamma} &\geq 0

    Where :

    - :math:`\mathbf{C_1}`: Metric cost matrix in the source space.
    - :math:`\mathbf{C_2}`: Metric cost matrix in the target space.
    - :math:`\mathbf{p}`: Distribution in the source space.
    - :math:`\mathbf{q}`: Distribution in the target space.
    - `L`: Loss function to account for the misfit between the similarity matrices.

    .. note:: This function is backend-compatible and will work on arrays
        from all compatible backends. But the algorithm uses the C++ CPU backend
        which can lead to copy overhead on GPU arrays.
    .. note:: All computations in the conjugate gradient solver are done with
        numpy to limit memory overhead.
    .. note:: This function will cast the computed transport plan to the data
        type of the provided input :math:`\mathbf{C}_1`. Casting to an integer
        tensor might result in a loss of precision. If this behaviour is
        unwanted, please make sure to provide a floating point input.

    Parameters
    ----------
    C1 : array-like, shape (ns, ns)
        Metric cost matrix in the source space.
    C2 : array-like, shape (nt, nt)
        Metric cost matrix in the target space.
    p : array-like, shape (ns,), optional
        Distribution in the source space.
        If let to its default value None, uniform distribution is taken.
    q : array-like, shape (nt,), optional
        Distribution in the target space.
        If let to its default value None, uniform distribution is taken.
    loss_fun : str, optional
        Loss function used for the solver either 'square_loss' or 'kl_loss'.
    symmetric : bool, optional
        Either C1 and C2 are to be assumed symmetric or not.
        If let to its default None value, a symmetry test will be conducted.
        Else if set to True (resp. False), C1 and C2 will be assumed symmetric (resp. asymmetric).
    verbose : bool, optional
        Print information along iterations.
    log : bool, optional
        Record log if True.
    armijo : bool, optional
        If True, the step of the line-search is found via an armijo search. Else closed form is used.
        If there are convergence issues, use False.
    G0: array-like, shape (ns,nt), optional
        If None, the initial transport plan of the solver is pq^T.
        Otherwise G0 must satisfy marginal constraints and will be used as initial transport of the solver.
    max_iter : int, optional
        Max number of iterations.
    tol_rel : float, optional
        Stop threshold on relative error (>0).
    tol_abs : float, optional
        Stop threshold on absolute error (>0).
    **kwargs : dict
        Parameters can be directly passed to the ot.optim.cg solver.

    Returns
    -------
    T : array-like, shape (`ns`, `nt`)
        Coupling between the two spaces that minimizes:

            :math:`\sum_{i,j,k,l} L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l}) \mathbf{T}_{i,j} \mathbf{T}_{k,l}`
    log : dict
        Convergence information and loss.

    References
    ----------
    .. [12] Gabriel Peyré, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.

    .. [13] Mémoli, Facundo. Gromov–Wasserstein distances and the
        metric approach to object matching. Foundations of computational
        mathematics 11.4 (2011): 417-487.

    .. [47] Chowdhury, S., & Mémoli, F. (2019). The gromov–wasserstein
        distance between networks and stable network invariants.
        Information and Inference: A Journal of the IMA, 8(4), 757-787.
    """
    arr = [C1, C2]
    if p is not None:
        arr.append(list_to_array(p))
    else:
        p = unif(C1.shape[0], type_as=C1)
    if q is not None:
        arr.append(list_to_array(q))
    else:
        q = unif(C2.shape[0], type_as=C1)
    if G0 is not None:
        G0_ = G0
        arr.append(G0)

    nx = get_backend(*arr)
    p0, q0, C10, C20 = p, q, C1, C2

    p = nx.to_numpy(p0)
    q = nx.to_numpy(q0)
    C1 = nx.to_numpy(C10)
    C2 = nx.to_numpy(C20)
    if symmetric is None:
        symmetric = np.allclose(C1, C1.T, atol=1e-10) and np.allclose(C2, C2.T, atol=1e-10)

    if G0 is None:
        G0 = p[:, None] * q[None, :]
    else:
        G0 = nx.to_numpy(G0_)
        # Check marginals of G0
        np.testing.assert_allclose(G0.sum(axis=1), p, atol=1e-08)
        np.testing.assert_allclose(G0.sum(axis=0), q, atol=1e-08)
    # cg for GW is implemented using numpy on CPU
    np_ = NumpyBackend()

    constC, hC1, hC2 = init_matrix(C1, C2, p, q, loss_fun, np_)

    def f(G):
        return gwloss(constC, hC1, hC2, G, np_)

    if symmetric:
        def df(G):
            return gwggrad(constC, hC1, hC2, G, np_)
    else:
        constCt, hC1t, hC2t = init_matrix(C1.T, C2.T, p, q, loss_fun, np_)

        def df(G):
            return 0.5 * (gwggrad(constC, hC1, hC2, G, np_) + gwggrad(constCt, hC1t, hC2t, G, np_))

    if armijo:
        def line_search(cost, G, deltaG, Mi, cost_G, **kwargs):
            return line_search_armijo(cost, G, deltaG, Mi, cost_G, nx=np_, **kwargs)
    else:
        def line_search(cost, G, deltaG, Mi, cost_G, **kwargs):
            return solve_gromov_linesearch(G, deltaG, cost_G, hC1, hC2, M=0., reg=1., nx=np_, symmetric=symmetric, **kwargs)

    if not nx.is_floating_point(C10):
        warnings.warn(
            "Input structure matrix consists of integers. The transport plan will be "
            "casted accordingly, possibly resulting in a loss of precision. "
            "If this behaviour is unwanted, please make sure your input "
            "structure matrix consists of floating point elements.",
            stacklevel=2
        )

    if log:
        res, log = cg(p, q, 0., 1., f, df, G0, line_search, log=True, numItermax=max_iter, stopThr=tol_rel, stopThr2=tol_abs, **kwargs)
        log['gw_dist'] = nx.from_numpy(log['loss'][-1], type_as=C10)
        log['u'] = nx.from_numpy(log['u'], type_as=C10)
        log['v'] = nx.from_numpy(log['v'], type_as=C10)
        return nx.from_numpy(res, type_as=C10), log
    else:
        return nx.from_numpy(cg(p, q, 0., 1., f, df, G0, line_search, log=False, numItermax=max_iter, stopThr=tol_rel, stopThr2=tol_abs, **kwargs), type_as=C10)


def gromov_wasserstein2(C1, C2, p=None, q=None, loss_fun='square_loss', symmetric=None, log=False, armijo=False, G0=None,
                        max_iter=1e4, tol_rel=1e-9, tol_abs=1e-9, **kwargs):
    r"""
    Returns the Gromov-Wasserstein loss :math:`\mathbf{GW}` between :math:`(\mathbf{C_1}, \mathbf{p})` and :math:`(\mathbf{C_2}, \mathbf{q})`.
    To recover the Gromov-Wasserstein distance as defined in [13] compute :math:`d_{GW} = \frac{1}{2} \sqrt{\mathbf{GW}}`.

    The function solves the following optimization problem using Conditional Gradient:

    .. math::
        \mathbf{GW} = \min_\mathbf{T} \quad \sum_{i,j,k,l}
        L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l}) \mathbf{T}_{i,j} \mathbf{T}_{k,l}

        s.t. \ \mathbf{\gamma} \mathbf{1} &= \mathbf{p}

             \mathbf{\gamma}^T \mathbf{1} &= \mathbf{q}

             \mathbf{\gamma} &\geq 0

    Where :

    - :math:`\mathbf{C_1}`: Metric cost matrix in the source space
    - :math:`\mathbf{C_2}`: Metric cost matrix in the target space
    - :math:`\mathbf{p}`: distribution in the source space
    - :math:`\mathbf{q}`: distribution in the target space
    - `L`: loss function to account for the misfit between the similarity
      matrices

    Note that when using backends, this loss function is differentiable wrt the
    matrices (C1, C2) and weights (p, q) for quadratic loss using the gradients from [38]_.

    .. note:: This function is backend-compatible and will work on arrays
        from all compatible backends. But the algorithm uses the C++ CPU backend
        which can lead to copy overhead on GPU arrays.
    .. note:: All computations in the conjugate gradient solver are done with
        numpy to limit memory overhead.
    .. note:: This function will cast the computed transport plan to the data
        type of the provided input :math:`\mathbf{C}_1`. Casting to an integer
        tensor might result in a loss of precision. If this behaviour is
        unwanted, please make sure to provide a floating point input.

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
    loss_fun :  str
        loss function used for the solver either 'square_loss' or 'kl_loss'
    symmetric : bool, optional
        Either C1 and C2 are to be assumed symmetric or not.
        If let to its default None value, a symmetry test will be conducted.
        Else if set to True (resp. False), C1 and C2 will be assumed symmetric (resp. asymmetric).
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    armijo : bool, optional
        If True the step of the line-search is found via an armijo research. Else closed form is used.
        If there are convergence issues use False.
    G0: array-like, shape (ns,nt), optional
        If None the initial transport plan of the solver is pq^T.
        Otherwise G0 must satisfy marginal constraints and will be used as initial transport of the solver.
    max_iter : int, optional
        Max number of iterations
    tol_rel : float, optional
        Stop threshold on relative error (>0)
    tol_abs : float, optional
        Stop threshold on absolute error (>0)
    **kwargs : dict
        parameters can be directly passed to the ot.optim.cg solver

    Returns
    -------
    gw_dist : float
        Gromov-Wasserstein distance
    log : dict
        convergence information and Coupling matrix

    References
    ----------
    .. [12] Gabriel Peyré, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.

    .. [13] Mémoli, Facundo. Gromov–Wasserstein distances and the
        metric approach to object matching. Foundations of computational
        mathematics 11.4 (2011): 417-487.

    .. [38] C. Vincent-Cuaz, T. Vayer, R. Flamary, M. Corneli, N. Courty, Online
        Graph Dictionary Learning, International Conference on Machine Learning
        (ICML), 2021.

    .. [47] Chowdhury, S., & Mémoli, F. (2019). The gromov–wasserstein
        distance between networks and stable network invariants.
        Information and Inference: A Journal of the IMA, 8(4), 757-787.
    """
    # simple get_backend as the full one will be handled in gromov_wasserstein
    nx = get_backend(C1, C2)

    # init marginals if set as None
    if p is None:
        p = unif(C1.shape[0], type_as=C1)
    if q is None:
        q = unif(C2.shape[0], type_as=C1)

    T, log_gw = gromov_wasserstein(
        C1, C2, p, q, loss_fun, symmetric, log=True, armijo=armijo, G0=G0,
        max_iter=max_iter, tol_rel=tol_rel, tol_abs=tol_abs, **kwargs)

    log_gw['T'] = T
    gw = log_gw['gw_dist']

    if loss_fun == 'square_loss':
        gC1 = 2 * C1 * nx.outer(p, p) - 2 * nx.dot(T, nx.dot(C2, T.T))
        gC2 = 2 * C2 * nx.outer(q, q) - 2 * nx.dot(T.T, nx.dot(C1, T))
    elif loss_fun == 'kl_loss':
        gC1 = nx.log(C1 + 1e-15) * nx.outer(p, p) - nx.dot(T, nx.dot(nx.log(C2 + 1e-15), T.T))
        gC2 = - nx.dot(T.T, nx.dot(C1, T)) / (C2 + 1e-15) + nx.outer(q, q)

    gw = nx.set_gradients(gw, (p, q, C1, C2),
                          (log_gw['u'] - nx.mean(log_gw['u']),
                           log_gw['v'] - nx.mean(log_gw['v']), gC1, gC2))

    if log:
        return gw, log_gw
    else:
        return gw


def fused_gromov_wasserstein(M, C1, C2, p=None, q=None, loss_fun='square_loss', symmetric=None, alpha=0.5,
                             armijo=False, G0=None, log=False, max_iter=1e4, tol_rel=1e-9, tol_abs=1e-9, **kwargs):
    r"""
    Returns the Fused Gromov-Wasserstein transport between :math:`(\mathbf{C_1}, \mathbf{Y_1}, \mathbf{p})` and :math:`(\mathbf{C_2}, \mathbf{Y_2}, \mathbf{q})`
    with pairwise distance matrix :math:`\mathbf{M}` between node feature matrices :math:`\mathbf{Y_1}` and :math:`\mathbf{Y_2}` (see :ref:`[24] <references-fused-gromov-wasserstein>`).

    The function solves the following optimization problem using Conditional Gradient:

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
    - :math:`\alpha`: trade-off parameter

    .. note:: This function is backend-compatible and will work on arrays
        from all compatible backends. But the algorithm uses the C++ CPU backend
        which can lead to copy overhead on GPU arrays.
    .. note:: All computations in the conjugate gradient solver are done with
        numpy to limit memory overhead.
    .. note:: This function will cast the computed transport plan to the data
        type of the provided input :math:`\mathbf{M}`. Casting to an integer
        tensor might result in a loss of precision. If this behaviour is
        unwanted, please make sure to provide a floating point input.


    Parameters
    ----------
    M : array-like, shape (ns, nt)
        Metric cost matrix between features across domains
    C1 : array-like, shape (ns, ns)
        Metric cost matrix representative of the structure in the source space
    C2 : array-like, shape (nt, nt)
        Metric cost matrix representative of the structure in the target space
    p : array-like, shape (ns,), optional
        Distribution in the source space.
        If let to its default value None, uniform distribution is taken.
    q : array-like, shape (nt,), optional
        Distribution in the target space.
        If let to its default value None, uniform distribution is taken.
    loss_fun : str, optional
        Loss function used for the solver
    symmetric : bool, optional
        Either C1 and C2 are to be assumed symmetric or not.
        If let to its default None value, a symmetry test will be conducted.
        Else if set to True (resp. False), C1 and C2 will be assumed symmetric (resp. asymmetric).
    alpha : float, optional
        Trade-off parameter (0 < alpha < 1)
    armijo : bool, optional
        If True the step of the line-search is found via an armijo research. Else closed form is used.
        If there are convergence issues use False.
    G0: array-like, shape (ns,nt), optional
        If None the initial transport plan of the solver is pq^T.
        Otherwise G0 must satisfy marginal constraints and will be used as initial transport of the solver.
    log : bool, optional
        record log if True
    max_iter : int, optional
        Max number of iterations
    tol_rel : float, optional
        Stop threshold on relative error (>0)
    tol_abs : float, optional
        Stop threshold on absolute error (>0)
    **kwargs : dict
        parameters can be directly passed to the ot.optim.cg solver

    Returns
    -------
    T : array-like, shape (`ns`, `nt`)
        Optimal transportation matrix for the given parameters.
    log : dict
        Log dictionary return only if log==True in parameters.


    .. _references-fused-gromov-wasserstein:
    References
    ----------
    .. [24] Vayer Titouan, Chapel Laetitia, Flamary Rémi, Tavenard Romain
        and Courty Nicolas "Optimal Transport for structured data with
        application on graphs", International Conference on Machine Learning
        (ICML). 2019.

    .. [47] Chowdhury, S., & Mémoli, F. (2019). The gromov–wasserstein
        distance between networks and stable network invariants.
        Information and Inference: A Journal of the IMA, 8(4), 757-787.
    """
    arr = [C1, C2, M]
    if p is not None:
        arr.append(list_to_array(p))
    else:
        p = unif(C1.shape[0], type_as=M)
    if q is not None:
        arr.append(list_to_array(q))
    else:
        q = unif(C2.shape[0], type_as=M)
    if G0 is not None:
        G0_ = G0
        arr.append(G0)

    nx = get_backend(*arr)
    p0, q0, C10, C20, M0, alpha0 = p, q, C1, C2, M, alpha

    p = nx.to_numpy(p0)
    q = nx.to_numpy(q0)
    C1 = nx.to_numpy(C10)
    C2 = nx.to_numpy(C20)
    M = nx.to_numpy(M0)
    alpha = nx.to_numpy(alpha0)

    if symmetric is None:
        symmetric = np.allclose(C1, C1.T, atol=1e-10) and np.allclose(C2, C2.T, atol=1e-10)

    if G0 is None:
        G0 = p[:, None] * q[None, :]
    else:
        G0 = nx.to_numpy(G0_)
        # Check marginals of G0
        np.testing.assert_allclose(G0.sum(axis=1), p, atol=1e-08)
        np.testing.assert_allclose(G0.sum(axis=0), q, atol=1e-08)
    # cg for GW is implemented using numpy on CPU
    np_ = NumpyBackend()

    constC, hC1, hC2 = init_matrix(C1, C2, p, q, loss_fun, np_)

    def f(G):
        return gwloss(constC, hC1, hC2, G, np_)

    if symmetric:
        def df(G):
            return gwggrad(constC, hC1, hC2, G, np_)
    else:
        constCt, hC1t, hC2t = init_matrix(C1.T, C2.T, p, q, loss_fun, np_)

        def df(G):
            return 0.5 * (gwggrad(constC, hC1, hC2, G, np_) + gwggrad(constCt, hC1t, hC2t, G, np_))

    if armijo:
        def line_search(cost, G, deltaG, Mi, cost_G, **kwargs):
            return line_search_armijo(cost, G, deltaG, Mi, cost_G, nx=np_, **kwargs)
    else:
        def line_search(cost, G, deltaG, Mi, cost_G, **kwargs):
            return solve_gromov_linesearch(G, deltaG, cost_G, hC1, hC2, M=(1 - alpha) * M, reg=alpha, nx=np_, symmetric=symmetric, **kwargs)
    if not nx.is_floating_point(M0):
        warnings.warn(
            "Input feature matrix consists of integer. The transport plan will be "
            "casted accordingly, possibly resulting in a loss of precision. "
            "If this behaviour is unwanted, please make sure your input "
            "feature matrix consists of floating point elements.",
            stacklevel=2
        )
    if log:
        res, log = cg(p, q, (1 - alpha) * M, alpha, f, df, G0, line_search, log=True, numItermax=max_iter, stopThr=tol_rel, stopThr2=tol_abs, **kwargs)
        log['fgw_dist'] = nx.from_numpy(log['loss'][-1], type_as=M0)
        log['u'] = nx.from_numpy(log['u'], type_as=M0)
        log['v'] = nx.from_numpy(log['v'], type_as=M0)
        return nx.from_numpy(res, type_as=M0), log
    else:
        return nx.from_numpy(cg(p, q, (1 - alpha) * M, alpha, f, df, G0, line_search, log=False, numItermax=max_iter, stopThr=tol_rel, stopThr2=tol_abs, **kwargs), type_as=M0)


def fused_gromov_wasserstein2(M, C1, C2, p=None, q=None, loss_fun='square_loss', symmetric=None, alpha=0.5,
                              armijo=False, G0=None, log=False, max_iter=1e4, tol_rel=1e-9, tol_abs=1e-9, **kwargs):
    r"""
    Returns the Fused Gromov-Wasserstein distance between :math:`(\mathbf{C_1}, \mathbf{Y_1}, \mathbf{p})` and :math:`(\mathbf{C_2}, \mathbf{Y_2}, \mathbf{q})`
    with pairwise distance matrix :math:`\mathbf{M}` between node feature matrices :math:`\mathbf{Y_1}` and :math:`\mathbf{Y_2}` (see :ref:`[24] <references-fused-gromov-wasserstein>`).

    The function solves the following optimization problem using Conditional Gradient:

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
    - :math:`\alpha`: trade-off parameter

    Note that when using backends, this loss function is differentiable wrt the
    matrices (C1, C2, M) and weights (p, q) for quadratic loss using the gradients from [38]_.

    .. note:: This function is backend-compatible and will work on arrays
        from all compatible backends. But the algorithm uses the C++ CPU backend
        which can lead to copy overhead on GPU arrays.
    .. note:: All computations in the conjugate gradient solver are done with
        numpy to limit memory overhead.
    .. note:: This function will cast the computed transport plan to the data
        type of the provided input :math:`\mathbf{M}`. Casting to an integer
        tensor might result in a loss of precision. If this behaviour is
        unwanted, please make sure to provide a floating point input.

    Parameters
    ----------
    M : array-like, shape (ns, nt)
        Metric cost matrix between features across domains
    C1 : array-like, shape (ns, ns)
        Metric cost matrix representative of the structure in the source space.
    C2 : array-like, shape (nt, nt)
        Metric cost matrix representative of the structure in the target space.
    p : array-like, shape (ns,), optional
        Distribution in the source space.
        If let to its default value None, uniform distribution is taken.
    q : array-like, shape (nt,), optional
        Distribution in the target space.
        If let to its default value None, uniform distribution is taken.
    loss_fun : str, optional
        Loss function used for the solver.
    symmetric : bool, optional
        Either C1 and C2 are to be assumed symmetric or not.
        If let to its default None value, a symmetry test will be conducted.
        Else if set to True (resp. False), C1 and C2 will be assumed symmetric (resp. asymmetric).
    alpha : float, optional
        Trade-off parameter (0 < alpha < 1)
    armijo : bool, optional
        If True the step of the line-search is found via an armijo research.
        Else closed form is used. If there are convergence issues use False.
    G0: array-like, shape (ns,nt), optional
        If None the initial transport plan of the solver is pq^T.
        Otherwise G0 must satisfy marginal constraints and will be used as initial transport of the solver.
    log : bool, optional
        Record log if True.
    max_iter : int, optional
        Max number of iterations
    tol_rel : float, optional
        Stop threshold on relative error (>0)
    tol_abs : float, optional
        Stop threshold on absolute error (>0)
    **kwargs : dict
        Parameters can be directly passed to the ot.optim.cg solver.

    Returns
    -------
    fgw-distance : float
        Fused Gromov-Wasserstein distance for the given parameters.
    log : dict
        Log dictionary return only if log==True in parameters.


    .. _references-fused-gromov-wasserstein2:
    References
    ----------
    .. [24] Vayer Titouan, Chapel Laetitia, Flamary Rémi, Tavenard Romain
        and Courty Nicolas
        "Optimal Transport for structured data with application on graphs"
        International Conference on Machine Learning (ICML). 2019.

    .. [38] C. Vincent-Cuaz, T. Vayer, R. Flamary, M. Corneli, N. Courty, Online
        Graph Dictionary Learning, International Conference on Machine Learning
        (ICML), 2021.

    .. [47] Chowdhury, S., & Mémoli, F. (2019). The gromov–wasserstein
        distance between networks and stable network invariants.
        Information and Inference: A Journal of the IMA, 8(4), 757-787.
    """
    nx = get_backend(C1, C2, M)

    # init marginals if set as None
    if p is None:
        p = unif(C1.shape[0], type_as=M)
    if q is None:
        q = unif(C2.shape[0], type_as=M)

    T, log_fgw = fused_gromov_wasserstein(
        M, C1, C2, p, q, loss_fun, symmetric, alpha, armijo, G0, log=True,
        max_iter=max_iter, tol_rel=tol_rel, tol_abs=tol_abs, **kwargs)

    fgw_dist = log_fgw['fgw_dist']
    log_fgw['T'] = T

    # compute separate terms for gradients and log
    lin_term = nx.sum(T * M)
    log_fgw['quad_loss'] = (fgw_dist - (1 - alpha) * lin_term)
    log_fgw['lin_loss'] = lin_term * (1 - alpha)
    gw_term = log_fgw['quad_loss'] / alpha

    if loss_fun == 'square_loss':
        gC1 = 2 * C1 * nx.outer(p, p) - 2 * nx.dot(T, nx.dot(C2, T.T))
        gC2 = 2 * C2 * nx.outer(q, q) - 2 * nx.dot(T.T, nx.dot(C1, T))
    elif loss_fun == 'kl_loss':
        gC1 = nx.log(C1 + 1e-15) * nx.outer(p, p) - nx.dot(T, nx.dot(nx.log(C2 + 1e-15), T.T))
        gC2 = - nx.dot(T.T, nx.dot(C1, T)) / (C2 + 1e-15) + nx.outer(q, q)
    if isinstance(alpha, int) or isinstance(alpha, float):
        fgw_dist = nx.set_gradients(fgw_dist, (p, q, C1, C2, M),
                                    (log_fgw['u'] - nx.mean(log_fgw['u']),
                                     log_fgw['v'] - nx.mean(log_fgw['v']),
                                     alpha * gC1, alpha * gC2, (1 - alpha) * T))
    else:
        fgw_dist = nx.set_gradients(fgw_dist, (p, q, C1, C2, M, alpha),
                                    (log_fgw['u'] - nx.mean(log_fgw['u']),
                                     log_fgw['v'] - nx.mean(log_fgw['v']),
                                     alpha * gC1, alpha * gC2, (1 - alpha) * T,
                                     gw_term - lin_term))

    if log:
        return fgw_dist, log_fgw
    else:
        return fgw_dist


def solve_gromov_linesearch(G, deltaG, cost_G, C1, C2, M, reg,
                            alpha_min=None, alpha_max=None, nx=None, symmetric=False, **kwargs):
    """
    Solve the linesearch in the FW iterations for any inner loss that decomposes as in Proposition 1 in :ref:`[12] <references-solve-linesearch>`.

    Parameters
    ----------

    G : array-like, shape(ns,nt)
        The transport map at a given iteration of the FW
    deltaG : array-like (ns,nt)
        Difference between the optimal map found by linearization in the FW algorithm and the value at a given iteration
    cost_G : float
        Value of the cost at `G`
    C1 : array-like (ns,ns), optional
        Transformed Structure matrix in the source domain.
        For the 'square_loss' and 'kl_loss', we provide hC1 from ot.gromov.init_matrix
    C2 : array-like (nt,nt), optional
        Transformed Structure matrix in the source domain.
        For the 'square_loss' and 'kl_loss', we provide hC2 from ot.gromov.init_matrix
    M : array-like (ns,nt)
        Cost matrix between the features.
    reg : float
        Regularization parameter.
    alpha_min : float, optional
        Minimum value for alpha
    alpha_max : float, optional
        Maximum value for alpha
    nx : backend, optional
        If let to its default value None, a backend test will be conducted.
    symmetric : bool, optional
        Either structures are to be assumed symmetric or not. Default value is False.
        Else if set to True (resp. False), C1 and C2 will be assumed symmetric (resp. asymmetric).

    Returns
    -------
    alpha : float
        The optimal step size of the FW
    fc : int
        nb of function call. Useless here
    cost_G : float
        The value of the cost for the next iteration


    .. _references-solve-linesearch:

    References
    ----------
    .. [24] Vayer Titouan, Chapel Laetitia, Flamary Rémi, Tavenard Romain and Courty Nicolas
        "Optimal Transport for structured data with application on graphs"
        International Conference on Machine Learning (ICML). 2019.
    .. [12] Gabriel Peyré, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.

    """
    if nx is None:
        if isinstance(M, int) or isinstance(M, float):
            nx = get_backend(G, deltaG, C1, C2)
        else:
            nx = get_backend(G, deltaG, C1, C2, M)

    dot = nx.dot(nx.dot(C1, deltaG), C2.T)
    a = - reg * nx.sum(dot * deltaG)
    if symmetric:
        b = nx.sum(M * deltaG) - 2 * reg * nx.sum(dot * G)
    else:
        b = nx.sum(M * deltaG) - reg * (nx.sum(dot * G) + nx.sum(nx.dot(nx.dot(C1, G), C2.T) * deltaG))

    alpha = solve_1d_linesearch_quad(a, b)
    if alpha_min is not None or alpha_max is not None:
        alpha = np.clip(alpha, alpha_min, alpha_max)

    # the new cost is deduced from the line search quadratic function
    cost_G = cost_G + a * (alpha ** 2) + b * alpha

    return alpha, 1, cost_G


def gromov_barycenters(
        N, Cs, ps=None, p=None, lambdas=None, loss_fun='square_loss',
        symmetric=True, armijo=False, max_iter=1000, tol=1e-9,
        stop_criterion='barycenter', warmstartT=False, verbose=False,
        log=False, init_C=None, random_state=None, **kwargs):
    r"""
    Returns the Gromov-Wasserstein barycenters of `S` measured similarity matrices :math:`(\mathbf{C}_s)_{1 \leq s \leq S}`

    The function solves the following optimization problem with block coordinate descent:

    .. math::

        \mathbf{C}^* = \mathop{\arg \min}_{\mathbf{C}\in \mathbb{R}^{N \times N}} \quad \sum_s \lambda_s \mathrm{GW}(\mathbf{C}, \mathbf{C}_s, \mathbf{p}, \mathbf{p}_s)

    Where :

    - :math:`\mathbf{C}_s`: metric cost matrix
    - :math:`\mathbf{p}_s`: distribution

    Parameters
    ----------
    N : int
        Size of the targeted barycenter
    Cs : list of S array-like of shape (ns, ns)
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
    symmetric : bool, optional.
        Either structures are to be assumed symmetric or not. Default value is True.
        Else if set to True (resp. False), C1 and C2 will be assumed symmetric (resp. asymmetric).
    armijo : bool, optional
        If True the step of the line-search is found via an armijo research.
        Else closed form is used. If there are convergence issues use False.
    max_iter : int, optional
        Max number of iterations
    tol : float, optional
        Stop threshold on relative error (>0)
    stop_criterion : str, optional. Default is 'barycenter'.
        Stop criterion taking values in ['barycenter', 'loss']. If set to 'barycenter'
        uses absolute norm variations of estimated barycenters. Else if set to 'loss'
        uses the relative variations of the loss.
    warmstartT: bool, optional
        Either to perform warmstart of transport plans in the successive
        fused gromov-wasserstein transport problems.s
    verbose : bool, optional
        Print information along iterations.
    log : bool, optional
        Record log if True.
    init_C : bool | array-like, shape(N,N)
        Random initial value for the :math:`\mathbf{C}` matrix provided by user.
    random_state : int or RandomState instance, optional
        Fix the seed for reproducibility

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

    while (err > tol and cpt < max_iter):
        if stop_criterion == 'barycenter':
            Cprev = C
        else:
            prev_loss = curr_loss

        # get transport plans
        if warmstartT:
            res = [gromov_wasserstein(
                C, Cs[s], p, ps[s], loss_fun, symmetric=symmetric, armijo=armijo, G0=T[s],
                max_iter=max_iter, tol_rel=1e-5, tol_abs=0., log=inner_log, verbose=verbose, **kwargs)
                for s in range(S)]
        else:
            res = [gromov_wasserstein(
                C, Cs[s], p, ps[s], loss_fun, symmetric=symmetric, armijo=armijo, G0=None,
                max_iter=max_iter, tol_rel=1e-5, tol_abs=0., log=inner_log, verbose=verbose, **kwargs)
                for s in range(S)]
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


def fgw_barycenters(
        N, Ys, Cs, ps=None, lambdas=None, alpha=0.5, fixed_structure=False,
        fixed_features=False, p=None, loss_fun='square_loss', armijo=False,
        symmetric=True, max_iter=100, tol=1e-9, stop_criterion='barycenter',
        warmstartT=False, verbose=False, log=False, init_C=None, init_X=None,
        random_state=None, **kwargs):
    r"""
    Returns the Fused Gromov-Wasserstein barycenters of `S` measurable networks with node features :math:`(\mathbf{C}_s, \mathbf{Y}_s, \mathbf{p}_s)_{1 \leq s \leq S}`
    (see eq (5) in :ref:`[24] <references-fgw-barycenters>`), estimated using Fused Gromov-Wasserstein transports from Conditional Gradient solvers.

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
        Desired number of samples of the target barycenter
    Ys: list of array-like, each element has shape (ns,d)
        Features of all samples
    Cs : list of array-like, each element has shape (ns,ns)
        Structure matrices of all samples
    ps : list of array-like, each element has shape (ns,), optional
        Masses of all samples.
        If let to its default value None, uniform distributions are taken.
    lambdas : list of float, optional
        List of the `S` spaces' weights.
        If let to its default value None, uniform weights are taken.
    alpha : float, optional
        Alpha parameter for the fgw distance.
    fixed_structure : bool, optional
        Whether to fix the structure of the barycenter during the updates.
    fixed_features : bool, optional
        Whether to fix the feature of the barycenter during the updates
    p : array-like, shape (N,), optional
        Weights in the targeted barycenter.
        If let to its default value None, uniform distribution is taken.
    loss_fun : str, optional
        Loss function used for the solver either 'square_loss' or 'kl_loss'
    armijo : bool, optional
        If True the step of the line-search is found via an armijo research.
        Else closed form is used. If there are convergence issues use False.
    symmetric : bool, optional
        Either structures are to be assumed symmetric or not. Default value is True.
        Else if set to True (resp. False), C1 and C2 will be assumed symmetric (resp. asymmetric).
    max_iter : int, optional
        Max number of iterations
    tol : float, optional
        Stop threshold on relative error (>0)
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
    init_C : array-like, shape (N,N), optional
        Initialization for the barycenters' structure matrix. If not set
        a random init is used.
    init_X : array-like, shape (N,d), optional
        Initialization for the barycenters' features. If not set a
        random init is used.
    random_state : int or RandomState instance, optional
        Fix the seed for reproducibility

    Returns
    -------
    X : array-like, shape (`N`, `d`)
        Barycenters' features
    C : array-like, shape (`N`, `N`)
        Barycenters' structure matrix
    log : dict
        Only returned when log=True. It contains the keys:

        - :math:`\mathbf{T}`: list of (`N`, `ns`) transport matrices
        - :math:`\mathbf{p}`: (`N`,) barycenter weights
        - :math:`(\mathbf{M}_s)_s`: all distance matrices between the feature of the barycenter and the other features :math:`(dist(\mathbf{X}, \mathbf{Y}_s))_s` shape (`N`, `ns`)
        - values used in convergence evaluation.

    .. _references-fgw-barycenters:
    References
    ----------
    .. [24] Vayer Titouan, Chapel Laetitia, Flamary Rémi, Tavenard Romain
        and Courty Nicolas
        "Optimal Transport for structured data with application on graphs"
        International Conference on Machine Learning (ICML). 2019.
    """
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

    if fixed_features:
        if init_X is None:
            raise UndefinedParameter('If X is fixed it must be initialized')
        else:
            X = init_X
    else:
        if init_X is None:
            X = nx.zeros((N, d), type_as=ps[0])

        else:
            X = init_X

    Ms = [dist(X, Ys[s]) for s in range(len(Ys))]

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
            Xprev = X
        else:
            prev_loss = curr_loss

        # get transport plans
        if warmstartT:
            res = [fused_gromov_wasserstein(
                Ms[s], C, Cs[s], p, ps[s], loss_fun=loss_fun, alpha=alpha, armijo=armijo, symmetric=symmetric,
                G0=T[s], max_iter=max_iter, tol_rel=1e-5, tol_abs=0., log=inner_log, verbose=verbose, **kwargs)
                for s in range(S)]
        else:
            res = [fused_gromov_wasserstein(
                Ms[s], C, Cs[s], p, ps[s], loss_fun=loss_fun, alpha=alpha, armijo=armijo, symmetric=symmetric,
                G0=None, max_iter=max_iter, tol_rel=1e-5, tol_abs=0., log=inner_log, verbose=verbose, **kwargs)
                for s in range(S)]
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
                err_feature = nx.norm(X - Xprev)
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

        return X, C, log_
    else:
        return X, C
