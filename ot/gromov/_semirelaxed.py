# -*- coding: utf-8 -*-
"""
Semi-relaxed Gromov-Wasserstein and Fused-Gromov-Wasserstein solvers.
"""

# Author: Rémi Flamary <remi.flamary@unice.fr>
#         Cédric Vincent-Cuaz <cedvincentcuaz@gmail.com>
#
# License: MIT License

import numpy as np


from ..utils import list_to_array, unif
from ..optim import semirelaxed_cg, solve_1d_linesearch_quad
from ..backend import get_backend

from ._utils import init_matrix_semirelaxed, gwloss, gwggrad


def semirelaxed_gromov_wasserstein(C1, C2, p, loss_fun='square_loss', symmetric=None, log=False, G0=None,
                                   max_iter=1e4, tol_rel=1e-9, tol_abs=1e-9, **kwargs):
    r"""
    Returns the semi-relaxed gromov-wasserstein divergence transport from :math:`(\mathbf{C_1}, \mathbf{p})` to :math:`\mathbf{C_2}`

    The function solves the following optimization problem:

    .. math::
        \mathbf{srGW} = \mathop{\arg \min}_\mathbf{T} \quad \sum_{i,j,k,l}
        L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l}) \mathbf{T}_{i,j} \mathbf{T}_{k,l}

        s.t. \ \mathbf{\gamma} \mathbf{1} &= \mathbf{p}

             \mathbf{\gamma} &\geq 0
    Where :

    - :math:`\mathbf{C_1}`: Metric cost matrix in the source space
    - :math:`\mathbf{C_2}`: Metric cost matrix in the target space
    - :math:`\mathbf{p}`: distribution in the source space

    - `L`: loss function to account for the misfit between the similarity matrices

    .. note:: This function is backend-compatible and will work on arrays
        from all compatible backends. However all the steps in the conditional
        gradient are not differentiable.

    Parameters
    ----------
    C1 : array-like, shape (ns, ns)
        Metric cost matrix in the source space
    C2 : array-like, shape (nt, nt)
        Metric cost matrix in the target space
    p : array-like, shape (ns,)
        Distribution in the source space
    loss_fun : str
        loss function used for the solver either 'square_loss' or 'kl_loss'.
        'kl_loss' is not implemented yet and will raise an error.
    symmetric : bool, optional
        Either C1 and C2 are to be assumed symmetric or not.
        If let to its default None value, a symmetry test will be conducted.
        Else if set to True (resp. False), C1 and C2 will be assumed symmetric (resp. asymetric).
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
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
    T : array-like, shape (`ns`, `nt`)
        Coupling between the two spaces that minimizes:

            :math:`\sum_{i,j,k,l} L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l}) \mathbf{T}_{i,j} \mathbf{T}_{k,l}`
    log : dict
        Convergence information and loss.

    References
    ----------
    .. [48]  Cédric Vincent-Cuaz, Rémi Flamary, Marco Corneli, Titouan Vayer, Nicolas Courty.
            "Semi-relaxed Gromov-Wasserstein divergence and applications on graphs"
            International Conference on Learning Representations (ICLR), 2022.
    """
    if loss_fun == 'kl_loss':
        raise NotImplementedError()
    p = list_to_array(p)
    if G0 is None:
        nx = get_backend(p, C1, C2)
    else:
        nx = get_backend(p, C1, C2, G0)

    if symmetric is None:
        symmetric = nx.allclose(C1, C1.T, atol=1e-10) and nx.allclose(C2, C2.T, atol=1e-10)
    if G0 is None:
        q = unif(C2.shape[0], type_as=p)
        G0 = nx.outer(p, q)
    else:
        q = nx.sum(G0, 0)
        # Check first marginal of G0
        np.testing.assert_allclose(nx.sum(G0, 1), p, atol=1e-08)

    constC, hC1, hC2, fC2t = init_matrix_semirelaxed(C1, C2, p, loss_fun, nx)

    ones_p = nx.ones(p.shape[0], type_as=p)

    def f(G):
        qG = nx.sum(G, 0)
        marginal_product = nx.outer(ones_p, nx.dot(qG, fC2t))
        return gwloss(constC + marginal_product, hC1, hC2, G, nx)

    if symmetric:
        def df(G):
            qG = nx.sum(G, 0)
            marginal_product = nx.outer(ones_p, nx.dot(qG, fC2t))
            return gwggrad(constC + marginal_product, hC1, hC2, G, nx)
    else:
        constCt, hC1t, hC2t, fC2 = init_matrix_semirelaxed(C1.T, C2.T, p, loss_fun, nx)

        def df(G):
            qG = nx.sum(G, 0)
            marginal_product_1 = nx.outer(ones_p, nx.dot(qG, fC2t))
            marginal_product_2 = nx.outer(ones_p, nx.dot(qG, fC2))
            return 0.5 * (gwggrad(constC + marginal_product_1, hC1, hC2, G, nx) + gwggrad(constCt + marginal_product_2, hC1t, hC2t, G, nx))

    def line_search(cost, G, deltaG, Mi, cost_G, **kwargs):
        return solve_semirelaxed_gromov_linesearch(G, deltaG, cost_G, C1, C2, ones_p, M=0., reg=1., nx=nx, **kwargs)

    if log:
        res, log = semirelaxed_cg(p, q, 0., 1., f, df, G0, line_search, log=True, numItermax=max_iter, stopThr=tol_rel, stopThr2=tol_abs, **kwargs)
        log['srgw_dist'] = log['loss'][-1]
        return res, log
    else:
        return semirelaxed_cg(p, q, 0., 1., f, df, G0, line_search, log=False, numItermax=max_iter, stopThr=tol_rel, stopThr2=tol_abs, **kwargs)


def semirelaxed_gromov_wasserstein2(C1, C2, p, loss_fun='square_loss', symmetric=None, log=False, G0=None,
                                    max_iter=1e4, tol_rel=1e-9, tol_abs=1e-9, **kwargs):
    r"""
    Returns the semi-relaxed gromov-wasserstein divergence from :math:`(\mathbf{C_1}, \mathbf{p})` to :math:`\mathbf{C_2}`

    The function solves the following optimization problem:

    .. math::
        srGW = \min_\mathbf{T} \quad \sum_{i,j,k,l}
        L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l}) \mathbf{T}_{i,j} \mathbf{T}_{k,l}

        s.t. \ \mathbf{\gamma} \mathbf{1} &= \mathbf{p}

             \mathbf{\gamma} &\geq 0
    Where :

    - :math:`\mathbf{C_1}`: Metric cost matrix in the source space
    - :math:`\mathbf{C_2}`: Metric cost matrix in the target space
    - :math:`\mathbf{p}`: distribution in the source space
    - `L`: loss function to account for the misfit between the similarity
      matrices

    Note that when using backends, this loss function is differentiable wrt the
    matrices (C1, C2) but not yet for the weights p.
    .. note:: This function is backend-compatible and will work on arrays
        from all compatible backends. However all the steps in the conditional
        gradient are not differentiable.

    Parameters
    ----------
    C1 : array-like, shape (ns, ns)
        Metric cost matrix in the source space
    C2 : array-like, shape (nt, nt)
        Metric cost matrix in the target space
    p : array-like, shape (ns,)
        Distribution in the source space.
    loss_fun : str
        loss function used for the solver either 'square_loss' or 'kl_loss'.
        'kl_loss' is not implemented yet and will raise an error.
    symmetric : bool, optional
        Either C1 and C2 are to be assumed symmetric or not.
        If let to its default None value, a symmetry test will be conducted.
        Else if set to True (resp. False), C1 and C2 will be assumed symmetric (resp. asymetric).
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
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
    srgw : float
        Semi-relaxed Gromov-Wasserstein divergence
    log : dict
        convergence information and Coupling matrix

    References
    ----------

    .. [48]  Cédric Vincent-Cuaz, Rémi Flamary, Marco Corneli, Titouan Vayer, Nicolas Courty.
            "Semi-relaxed Gromov-Wasserstein divergence and applications on graphs"
            International Conference on Learning Representations (ICLR), 2022.
    """
    nx = get_backend(p, C1, C2)

    T, log_srgw = semirelaxed_gromov_wasserstein(
        C1, C2, p, loss_fun, symmetric, log=True, G0=G0,
        max_iter=max_iter, tol_rel=tol_rel, tol_abs=tol_abs, **kwargs)

    q = nx.sum(T, 0)
    log_srgw['T'] = T
    srgw = log_srgw['srgw_dist']

    if loss_fun == 'square_loss':
        gC1 = 2 * C1 * nx.outer(p, p) - 2 * nx.dot(T, nx.dot(C2, T.T))
        gC2 = 2 * C2 * nx.outer(q, q) - 2 * nx.dot(T.T, nx.dot(C1, T))
        srgw = nx.set_gradients(srgw, (C1, C2), (gC1, gC2))

    if log:
        return srgw, log_srgw
    else:
        return srgw


def semirelaxed_fused_gromov_wasserstein(M, C1, C2, p, loss_fun='square_loss', symmetric=None, alpha=0.5, G0=None, log=False,
                                         max_iter=1e4, tol_rel=1e-9, tol_abs=1e-9, **kwargs):
    r"""
    Computes the semi-relaxed FGW transport between two graphs (see :ref:`[48] <references-semirelaxed-fused-gromov-wasserstein>`)

    .. math::
        \gamma = \mathop{\arg \min}_\gamma \quad (1 - \alpha) \langle \gamma, \mathbf{M} \rangle_F +
        \alpha \sum_{i,j,k,l} L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l}) \mathbf{T}_{i,j} \mathbf{T}_{k,l}

        s.t. \ \mathbf{\gamma} \mathbf{1} &= \mathbf{p}

             \mathbf{\gamma} &\geq 0

    where :

    - :math:`\mathbf{M}` is the (`ns`, `nt`) metric cost matrix
    - :math:`\mathbf{p}` source weights (sum to 1)
    - `L` is a loss function to account for the misfit between the similarity matrices


    .. note:: This function is backend-compatible and will work on arrays
        from all compatible backends. However all the steps in the conditional
        gradient are not differentiable.

    The algorithm used for solving the problem is conditional gradient as discussed in :ref:`[48] <references-semirelaxed-fused-gromov-wasserstein>`

    Parameters
    ----------
    M : array-like, shape (ns, nt)
        Metric cost matrix between features across domains
    C1 : array-like, shape (ns, ns)
        Metric cost matrix representative of the structure in the source space
    C2 : array-like, shape (nt, nt)
        Metric cost matrix representative of the structure in the target space
    p : array-like, shape (ns,)
        Distribution in the source space
    loss_fun : str
        loss function used for the solver either 'square_loss' or 'kl_loss'.
        'kl_loss' is not implemented yet and will raise an error.
    symmetric : bool, optional
        Either C1 and C2 are to be assumed symmetric or not.
        If let to its default None value, a symmetry test will be conducted.
        Else if set to True (resp. False), C1 and C2 will be assumed symmetric (resp. asymetric).
    alpha : float, optional
        Trade-off parameter (0 < alpha < 1)
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
    gamma : array-like, shape (`ns`, `nt`)
        Optimal transportation matrix for the given parameters.
    log : dict
        Log dictionary return only if log==True in parameters.


    .. _references-semirelaxed-fused-gromov-wasserstein:
    References
    ----------
    .. [24] Vayer Titouan, Chapel Laetitia, Flamary Rémi, Tavenard Romain
        and Courty Nicolas "Optimal Transport for structured data with
        application on graphs", International Conference on Machine Learning
        (ICML). 2019.

    .. [48] Cédric Vincent-Cuaz, Rémi Flamary, Marco Corneli, Titouan Vayer, Nicolas Courty.
            "Semi-relaxed Gromov-Wasserstein divergence and applications on graphs"
            International Conference on Learning Representations (ICLR), 2022.
    """
    if loss_fun == 'kl_loss':
        raise NotImplementedError()

    p = list_to_array(p)
    if G0 is None:
        nx = get_backend(p, C1, C2, M)
    else:
        nx = get_backend(p, C1, C2, M, G0)

    if symmetric is None:
        symmetric = nx.allclose(C1, C1.T, atol=1e-10) and nx.allclose(C2, C2.T, atol=1e-10)

    if G0 is None:
        q = unif(C2.shape[0], type_as=p)
        G0 = nx.outer(p, q)
    else:
        q = nx.sum(G0, 0)
        # Check marginals of G0
        np.testing.assert_allclose(nx.sum(G0, 1), p, atol=1e-08)

    constC, hC1, hC2, fC2t = init_matrix_semirelaxed(C1, C2, p, loss_fun, nx)

    ones_p = nx.ones(p.shape[0], type_as=p)

    def f(G):
        qG = nx.sum(G, 0)
        marginal_product = nx.outer(ones_p, nx.dot(qG, fC2t))
        return gwloss(constC + marginal_product, hC1, hC2, G, nx)

    if symmetric:
        def df(G):
            qG = nx.sum(G, 0)
            marginal_product = nx.outer(ones_p, nx.dot(qG, fC2t))
            return gwggrad(constC + marginal_product, hC1, hC2, G, nx)
    else:
        constCt, hC1t, hC2t, fC2 = init_matrix_semirelaxed(C1.T, C2.T, p, loss_fun, nx)

        def df(G):
            qG = nx.sum(G, 0)
            marginal_product_1 = nx.outer(ones_p, nx.dot(qG, fC2t))
            marginal_product_2 = nx.outer(ones_p, nx.dot(qG, fC2))
            return 0.5 * (gwggrad(constC + marginal_product_1, hC1, hC2, G, nx) + gwggrad(constCt + marginal_product_2, hC1t, hC2t, G, nx))

    def line_search(cost, G, deltaG, Mi, cost_G, **kwargs):
        return solve_semirelaxed_gromov_linesearch(
            G, deltaG, cost_G, C1, C2, ones_p, M=(1 - alpha) * M, reg=alpha, nx=nx, **kwargs)

    if log:
        res, log = semirelaxed_cg(p, q, (1 - alpha) * M, alpha, f, df, G0, line_search, log=True, numItermax=max_iter, stopThr=tol_rel, stopThr2=tol_abs, **kwargs)
        log['srfgw_dist'] = log['loss'][-1]
        return res, log
    else:
        return semirelaxed_cg(p, q, (1 - alpha) * M, alpha, f, df, G0, line_search, log=False, numItermax=max_iter, stopThr=tol_rel, stopThr2=tol_abs, **kwargs)


def semirelaxed_fused_gromov_wasserstein2(M, C1, C2, p, loss_fun='square_loss', symmetric=None, alpha=0.5, G0=None, log=False,
                                          max_iter=1e4, tol_rel=1e-9, tol_abs=1e-9, **kwargs):
    r"""
    Computes the semi-relaxed FGW divergence between two graphs (see :ref:`[48] <references-semirelaxed-fused-gromov-wasserstein2>`)

    .. math::
        \min_\gamma \quad (1 - \alpha) \langle \gamma, \mathbf{M} \rangle_F + \alpha \sum_{i,j,k,l}
        L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l}) \mathbf{T}_{i,j} \mathbf{T}_{k,l}

        s.t. \ \mathbf{\gamma} \mathbf{1} &= \mathbf{p}

             \mathbf{\gamma} &\geq 0

    where :

    - :math:`\mathbf{M}` is the (`ns`, `nt`) metric cost matrix
    - :math:`\mathbf{p}` source weights (sum to 1)
    - `L` is a loss function to account for the misfit between the similarity matrices

    The algorithm used for solving the problem is conditional gradient as
    discussed in :ref:`[48] <semirelaxed-fused-gromov-wasserstein2>`

    Note that when using backends, this loss function is differentiable wrt the
    matrices (C1, C2) but not yet for the weights p.

    .. note:: This function is backend-compatible and will work on arrays
        from all compatible backends. However all the steps in the conditional
        gradient are not differentiable.

    Parameters
    ----------
    M : array-like, shape (ns, nt)
        Metric cost matrix between features across domains
    C1 : array-like, shape (ns, ns)
        Metric cost matrix representative of the structure in the source space.
    C2 : array-like, shape (nt, nt)
        Metric cost matrix representative of the structure in the target space.
    p :  array-like, shape (ns,)
        Distribution in the source space.
    loss_fun : str, optional
        loss function used for the solver either 'square_loss' or 'kl_loss'.
        'kl_loss' is not implemented yet and will raise an error.
    symmetric : bool, optional
        Either C1 and C2 are to be assumed symmetric or not.
        If let to its default None value, a symmetry test will be conducted.
        Else if set to True (resp. False), C1 and C2 will be assumed symmetric (resp. asymetric).
    alpha : float, optional
        Trade-off parameter (0 < alpha < 1)
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
    srfgw-divergence : float
        Semi-relaxed Fused gromov wasserstein divergence for the given parameters.
    log : dict
        Log dictionary return only if log==True in parameters.


    .. _references-semirelaxed-fused-gromov-wasserstein2:
    References
    ----------
    .. [24] Vayer Titouan, Chapel Laetitia, Flamary Rémi, Tavenard Romain
        and Courty Nicolas "Optimal Transport for structured data with
        application on graphs", International Conference on Machine Learning
        (ICML). 2019.

    .. [48] Cédric Vincent-Cuaz, Rémi Flamary, Marco Corneli, Titouan Vayer, Nicolas Courty.
            "Semi-relaxed Gromov-Wasserstein divergence and applications on graphs"
            International Conference on Learning Representations (ICLR), 2022.
    """
    nx = get_backend(p, C1, C2, M)

    T, log_fgw = semirelaxed_fused_gromov_wasserstein(
        M, C1, C2, p, loss_fun, symmetric, alpha, G0, log=True,
        max_iter=max_iter, tol_rel=tol_rel, tol_abs=tol_abs, **kwargs)
    q = nx.sum(T, 0)
    srfgw_dist = log_fgw['srfgw_dist']
    log_fgw['T'] = T

    if loss_fun == 'square_loss':
        gC1 = 2 * C1 * nx.outer(p, p) - 2 * nx.dot(T, nx.dot(C2, T.T))
        gC2 = 2 * C2 * nx.outer(q, q) - 2 * nx.dot(T.T, nx.dot(C1, T))
        srfgw_dist = nx.set_gradients(srfgw_dist, (C1, C2, M),
                                      (alpha * gC1, alpha * gC2, (1 - alpha) * T))

    if log:
        return srfgw_dist, log_fgw
    else:
        return srfgw_dist


def solve_semirelaxed_gromov_linesearch(G, deltaG, cost_G, C1, C2, ones_p,
                                        M, reg, alpha_min=None, alpha_max=None, nx=None, **kwargs):
    """
    Solve the linesearch in the FW iterations

    Parameters
    ----------

    G : array-like, shape(ns,nt)
        The transport map at a given iteration of the FW
    deltaG : array-like (ns,nt)
        Difference between the optimal map found by linearization in the FW algorithm and the value at a given iteration
    cost_G : float
        Value of the cost at `G`
    C1 : array-like (ns,ns)
        Structure matrix in the source domain.
    C2 : array-like (nt,nt)
        Structure matrix in the target domain.
    ones_p: array-like (ns,1)
        Array of ones of size ns
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
    Returns
    -------
    alpha : float
        The optimal step size of the FW
    fc : int
        nb of function call. Useless here
    cost_G : float
        The value of the cost for the next iteration

    References
    ----------
    .. [48]  Cédric Vincent-Cuaz, Rémi Flamary, Marco Corneli, Titouan Vayer, Nicolas Courty.
            "Semi-relaxed Gromov-Wasserstein divergence and applications on graphs"
            International Conference on Learning Representations (ICLR), 2021.
    """
    if nx is None:
        G, deltaG, C1, C2, M = list_to_array(G, deltaG, C1, C2, M)

        if isinstance(M, int) or isinstance(M, float):
            nx = get_backend(G, deltaG, C1, C2)
        else:
            nx = get_backend(G, deltaG, C1, C2, M)

    qG, qdeltaG = nx.sum(G, 0), nx.sum(deltaG, 0)
    dot = nx.dot(nx.dot(C1, deltaG), C2.T)
    C2t_square = C2.T ** 2
    dot_qG = nx.dot(nx.outer(ones_p, qG), C2t_square)
    dot_qdeltaG = nx.dot(nx.outer(ones_p, qdeltaG), C2t_square)
    a = reg * nx.sum((dot_qdeltaG - 2 * dot) * deltaG)
    b = nx.sum(M * deltaG) + reg * (nx.sum((dot_qdeltaG - 2 * dot) * G) + nx.sum((dot_qG - 2 * nx.dot(nx.dot(C1, G), C2.T)) * deltaG))
    alpha = solve_1d_linesearch_quad(a, b)
    if alpha_min is not None or alpha_max is not None:
        alpha = np.clip(alpha, alpha_min, alpha_max)

    # the new cost can be deduced from the line search quadratic function
    cost_G = cost_G + a * (alpha ** 2) + b * alpha

    return alpha, 1, cost_G
