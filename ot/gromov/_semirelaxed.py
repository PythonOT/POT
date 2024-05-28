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


def semirelaxed_gromov_wasserstein(C1, C2, p=None, loss_fun='square_loss', symmetric=None, log=False, G0=None,
                                   max_iter=1e4, tol_rel=1e-9, tol_abs=1e-9, **kwargs):
    r"""
    Returns the semi-relaxed Gromov-Wasserstein divergence transport from :math:`(\mathbf{C_1}, \mathbf{p})` to :math:`\mathbf{C_2}` (see [48]).

    The function solves the following optimization problem using Conditional Gradient:

    .. math::
        \mathbf{T}^* \in \mathop{\arg \min}_{\mathbf{T}} \quad \sum_{i,j,k,l}
        L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l}) \mathbf{T}_{i,j} \mathbf{T}_{k,l}

        s.t. \ \mathbf{T} \mathbf{1} &= \mathbf{p}

             \mathbf{T} &\geq 0

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
    p : array-like, shape (ns,), optional
        Distribution in the source space.
        If let to its default value None, uniform distribution is taken.
    loss_fun : str
        loss function used for the solver either 'square_loss' or 'kl_loss'.
    symmetric : bool, optional
        Either C1 and C2 are to be assumed symmetric or not.
        If let to its default None value, a symmetry test will be conducted.
        Else if set to True (resp. False), C1 and C2 will be assumed symmetric (resp. asymmetric).
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
    .. [62] H. Van Assel, C. Vincent-Cuaz, T. Vayer, R. Flamary, N. Courty.
            "Interpolating between Clustering and Dimensionality Reduction with
            Gromov-Wasserstein". NeurIPS 2023 Workshop OTML.

    """
    arr = [C1, C2]
    if p is not None:
        arr.append(list_to_array(p))
    else:
        p = unif(C1.shape[0], type_as=C1)

    if G0 is not None:
        arr.append(G0)

    nx = get_backend(*arr)

    if symmetric is None:
        symmetric = nx.allclose(C1, C1.T, atol=1e-10) and nx.allclose(C2, C2.T, atol=1e-10)
    if G0 is None:
        q = unif(C2.shape[0], type_as=p)
        G0 = nx.outer(p, q)
    else:
        q = nx.sum(G0, 0)
        # Check first marginal of G0
        assert nx.allclose(nx.sum(G0, 1), p, atol=1e-08)

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
        return solve_semirelaxed_gromov_linesearch(G, deltaG, cost_G, hC1, hC2, ones_p, M=0., reg=1., fC2t=fC2t, nx=nx, **kwargs)

    if log:
        res, log = semirelaxed_cg(p, q, 0., 1., f, df, G0, line_search, log=True, numItermax=max_iter, stopThr=tol_rel, stopThr2=tol_abs, **kwargs)
        log['srgw_dist'] = log['loss'][-1]
        return res, log
    else:
        return semirelaxed_cg(p, q, 0., 1., f, df, G0, line_search, log=False, numItermax=max_iter, stopThr=tol_rel, stopThr2=tol_abs, **kwargs)


def semirelaxed_gromov_wasserstein2(C1, C2, p=None, loss_fun='square_loss', symmetric=None, log=False, G0=None,
                                    max_iter=1e4, tol_rel=1e-9, tol_abs=1e-9, **kwargs):
    r"""
    Returns the semi-relaxed Gromov-Wasserstein divergence from :math:`(\mathbf{C_1}, \mathbf{p})` to :math:`\mathbf{C_2}` (see [48]).

    The function solves the following optimization problem using Conditional Gradient:

    .. math::
        \text{srGW} = \min_{\mathbf{T}} \quad \sum_{i,j,k,l}
        L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l}) \mathbf{T}_{i,j} \mathbf{T}_{k,l}

        s.t. \ \mathbf{T} \mathbf{1} &= \mathbf{p}

             \mathbf{T} &\geq 0

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
    p : array-like, shape (ns,), optional
        Distribution in the source space.
        If let to its default value None, uniform distribution is taken.
    loss_fun : str
        loss function used for the solver either 'square_loss' or 'kl_loss'.
    symmetric : bool, optional
        Either C1 and C2 are to be assumed symmetric or not.
        If let to its default None value, a symmetry test will be conducted.
        Else if set to True (resp. False), C1 and C2 will be assumed symmetric (resp. asymmetric).
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

    .. [62] H. Van Assel, C. Vincent-Cuaz, T. Vayer, R. Flamary, N. Courty.
            "Interpolating between Clustering and Dimensionality Reduction with
            Gromov-Wasserstein". NeurIPS 2023 Workshop OTML.
    """
    # partial get_backend as the full one will be handled in gromov_wasserstein
    nx = get_backend(C1, C2)

    # init marginals if set as None
    if p is None:
        p = unif(C1.shape[0], type_as=C1)

    T, log_srgw = semirelaxed_gromov_wasserstein(
        C1, C2, p, loss_fun, symmetric, log=True, G0=G0,
        max_iter=max_iter, tol_rel=tol_rel, tol_abs=tol_abs, **kwargs)

    q = nx.sum(T, 0)
    log_srgw['T'] = T
    srgw = log_srgw['srgw_dist']

    if loss_fun == 'square_loss':
        gC1 = 2 * C1 * nx.outer(p, p) - 2 * nx.dot(T, nx.dot(C2, T.T))
        gC2 = 2 * C2 * nx.outer(q, q) - 2 * nx.dot(T.T, nx.dot(C1, T))

    elif loss_fun == 'kl_loss':
        gC1 = nx.log(C1 + 1e-15) * nx.outer(p, p) - nx.dot(T, nx.dot(nx.log(C2 + 1e-15), T.T))
        gC2 = - nx.dot(T.T, nx.dot(C1, T)) / (C2 + 1e-15) + nx.outer(q, q)

    srgw = nx.set_gradients(srgw, (C1, C2), (gC1, gC2))

    if log:
        return srgw, log_srgw
    else:
        return srgw


def semirelaxed_fused_gromov_wasserstein(
        M, C1, C2, p=None, loss_fun='square_loss', symmetric=None, alpha=0.5,
        G0=None, log=False, max_iter=1e4, tol_rel=1e-9, tol_abs=1e-9, **kwargs):
    r"""
    Computes the semi-relaxed Fused Gromov-Wasserstein transport between two graphs (see [48]).

    .. math::
        \mathbf{T}^* \in \mathop{\arg \min}_{\mathbf{T}} \quad (1 - \alpha) \langle \mathbf{T}, \mathbf{M} \rangle_F +
        \alpha \sum_{i,j,k,l} L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l}) T_{i,j} T_{k,l}

        s.t. \ \mathbf{T} \mathbf{1} &= \mathbf{p}

             \mathbf{T} &\geq 0

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
    p : array-like, shape (ns,), optional
        Distribution in the source space.
        If let to its default value None, uniform distribution is taken.
    loss_fun : str
        loss function used for the solver either 'square_loss' or 'kl_loss'.
    symmetric : bool, optional
        Either C1 and C2 are to be assumed symmetric or not.
        If let to its default None value, a symmetry test will be conducted.
        Else if set to True (resp. False), C1 and C2 will be assumed symmetric (resp. asymmetric).
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

    .. [62] H. Van Assel, C. Vincent-Cuaz, T. Vayer, R. Flamary, N. Courty.
            "Interpolating between Clustering and Dimensionality Reduction with
            Gromov-Wasserstein". NeurIPS 2023 Workshop OTML.
    """
    arr = [M, C1, C2]
    if p is not None:
        arr.append(list_to_array(p))
    else:
        p = unif(C1.shape[0], type_as=C1)

    if G0 is not None:
        arr.append(G0)

    nx = get_backend(*arr)

    if symmetric is None:
        symmetric = nx.allclose(C1, C1.T, atol=1e-10) and nx.allclose(C2, C2.T, atol=1e-10)

    if G0 is None:
        q = unif(C2.shape[0], type_as=p)
        G0 = nx.outer(p, q)
    else:
        q = nx.sum(G0, 0)
        # Check first marginal of G0
        assert nx.allclose(nx.sum(G0, 1), p, atol=1e-08)

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
            G, deltaG, cost_G, hC1, hC2, ones_p, M=(1 - alpha) * M, reg=alpha, fC2t=fC2t, nx=nx, **kwargs)

    if log:
        res, log = semirelaxed_cg(p, q, (1 - alpha) * M, alpha, f, df, G0, line_search, log=True, numItermax=max_iter, stopThr=tol_rel, stopThr2=tol_abs, **kwargs)
        log['srfgw_dist'] = log['loss'][-1]
        return res, log
    else:
        return semirelaxed_cg(p, q, (1 - alpha) * M, alpha, f, df, G0, line_search, log=False, numItermax=max_iter, stopThr=tol_rel, stopThr2=tol_abs, **kwargs)


def semirelaxed_fused_gromov_wasserstein2(M, C1, C2, p=None, loss_fun='square_loss', symmetric=None, alpha=0.5, G0=None, log=False,
                                          max_iter=1e4, tol_rel=1e-9, tol_abs=1e-9, **kwargs):
    r"""
    Computes the semi-relaxed FGW divergence between two graphs (see [48]).

    .. math::
        \mathbf{srFGW}_{\alpha} = \min_{\mathbf{T}} \quad (1 - \alpha) \langle \mathbf{T}, \mathbf{M} \rangle_F +
        \alpha \sum_{i,j,k,l} L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l}) T_{i,j} T_{k,l}

        s.t. \ \mathbf{T} \mathbf{1} &= \mathbf{p}

             \mathbf{T} &\geq 0

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
        If let to its default value None, uniform distribution is taken.
    loss_fun : str, optional
        loss function used for the solver either 'square_loss' or 'kl_loss'.
    symmetric : bool, optional
        Either C1 and C2 are to be assumed symmetric or not.
        If let to its default None value, a symmetry test will be conducted.
        Else if set to True (resp. False), C1 and C2 will be assumed symmetric (resp. asymmetric).
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
        Semi-relaxed Fused Gromov-Wasserstein divergence for the given parameters.
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

    .. [62] H. Van Assel, C. Vincent-Cuaz, T. Vayer, R. Flamary, N. Courty.
            "Interpolating between Clustering and Dimensionality Reduction with
            Gromov-Wasserstein". NeurIPS 2023 Workshop OTML.
    """
    # partial get_backend as the full one will be handled in gromov_wasserstein
    nx = get_backend(C1, C2)

    # init marginals if set as None
    if p is None:
        p = unif(C1.shape[0], type_as=C1)

    T, log_fgw = semirelaxed_fused_gromov_wasserstein(
        M, C1, C2, p, loss_fun, symmetric, alpha, G0, log=True,
        max_iter=max_iter, tol_rel=tol_rel, tol_abs=tol_abs, **kwargs)
    q = nx.sum(T, 0)
    srfgw_dist = log_fgw['srfgw_dist']
    log_fgw['T'] = T
    log_fgw['lin_loss'] = nx.sum(M * T) * (1 - alpha)
    log_fgw['quad_loss'] = srfgw_dist - log_fgw['lin_loss']

    if loss_fun == 'square_loss':
        gC1 = 2 * C1 * nx.outer(p, p) - 2 * nx.dot(T, nx.dot(C2, T.T))
        gC2 = 2 * C2 * nx.outer(q, q) - 2 * nx.dot(T.T, nx.dot(C1, T))

    elif loss_fun == 'kl_loss':
        gC1 = nx.log(C1 + 1e-15) * nx.outer(p, p) - nx.dot(T, nx.dot(nx.log(C2 + 1e-15), T.T))
        gC2 = - nx.dot(T.T, nx.dot(C1, T)) / (C2 + 1e-15) + nx.outer(q, q)

    if isinstance(alpha, int) or isinstance(alpha, float):
        srfgw_dist = nx.set_gradients(srfgw_dist, (C1, C2, M),
                                      (alpha * gC1, alpha * gC2, (1 - alpha) * T))
    else:
        lin_term = nx.sum(T * M)
        srgw_term = (srfgw_dist - (1 - alpha) * lin_term) / alpha
        srfgw_dist = nx.set_gradients(srfgw_dist, (C1, C2, M, alpha),
                                      (alpha * gC1, alpha * gC2, (1 - alpha) * T,
                                       srgw_term - lin_term))

    if log:
        return srfgw_dist, log_fgw
    else:
        return srfgw_dist


def solve_semirelaxed_gromov_linesearch(G, deltaG, cost_G, C1, C2, ones_p,
                                        M, reg, fC2t=None, alpha_min=None, alpha_max=None, nx=None, **kwargs):
    """
    Solve the linesearch in the Conditional Gradient iterations for the semi-relaxed Gromov-Wasserstein divergence.

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
        Note that for the 'square_loss' and 'kl_loss', we provide hC1 from ot.gromov.init_matrix_semirelaxed
    C2 : array-like (nt,nt), optional
        Transformed Structure matrix in the source domain.
        Note that for the 'square_loss' and 'kl_loss', we provide hC2 from ot.gromov.init_matrix_semirelaxed
    ones_p: array-like (ns,1)
        Array of ones of size ns
    M : array-like (ns,nt)
        Cost matrix between the features.
    reg : float
        Regularization parameter.
    fC2t: array-like (nt,nt), optional
        Transformed Structure matrix in the source domain.
        Note that for the 'square_loss' and 'kl_loss', we provide fC2t from ot.gromov.init_matrix_semirelaxed.
        If fC2t is not provided, it is by default fC2t corresponding to the 'square_loss'.
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

    .. [62] H. Van Assel, C. Vincent-Cuaz, T. Vayer, R. Flamary, N. Courty.
            "Interpolating between Clustering and Dimensionality Reduction with
            Gromov-Wasserstein". NeurIPS 2023 Workshop OTML.
    """
    if nx is None:
        if isinstance(M, int) or isinstance(M, float):
            nx = get_backend(G, deltaG, C1, C2)
        else:
            nx = get_backend(G, deltaG, C1, C2, M)

    qG, qdeltaG = nx.sum(G, 0), nx.sum(deltaG, 0)
    dot = nx.dot(nx.dot(C1, deltaG), C2.T)
    if fC2t is None:
        fC2t = C2.T ** 2
    dot_qG = nx.dot(nx.outer(ones_p, qG), fC2t)
    dot_qdeltaG = nx.dot(nx.outer(ones_p, qdeltaG), fC2t)

    a = reg * nx.sum((dot_qdeltaG - dot) * deltaG)
    b = nx.sum(M * deltaG) + reg * (nx.sum((dot_qdeltaG - dot) * G) + nx.sum((dot_qG - nx.dot(nx.dot(C1, G), C2.T)) * deltaG))

    alpha = solve_1d_linesearch_quad(a, b)
    if alpha_min is not None or alpha_max is not None:
        alpha = np.clip(alpha, alpha_min, alpha_max)

    # the new cost can be deduced from the line search quadratic function
    cost_G = cost_G + a * (alpha ** 2) + b * alpha

    return alpha, 1, cost_G


def entropic_semirelaxed_gromov_wasserstein(
        C1, C2, p=None, loss_fun='square_loss', epsilon=0.1, symmetric=None,
        G0=None, max_iter=1e4, tol=1e-9, log=False, verbose=False, **kwargs):
    r"""
    Returns the entropic-regularized semi-relaxed gromov-wasserstein divergence
    transport plan from :math:`(\mathbf{C_1}, \mathbf{p})` to :math:`\mathbf{C_2}`
    estimated using a Mirror Descent algorithm following the KL geometry.

    The function solves the following optimization problem:

    .. math::
        \mathbf{T}^* \in \mathop{\arg \min}_\mathbf{T} \quad \sum_{i,j,k,l}
        L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l}) \mathbf{T}_{i,j} \mathbf{T}_{k,l}

        s.t. \ \mathbf{T} \mathbf{1} &= \mathbf{p}

             \mathbf{T} &\geq 0
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
    p : array-like, shape (ns,), optional
        Distribution in the source space.
        If let to its default value None, uniform distribution is taken.
    loss_fun : str
        loss function used for the solver either 'square_loss' or 'kl_loss'.
    epsilon : float
        Regularization term >0
    symmetric : bool, optional
        Either C1 and C2 are to be assumed symmetric or not.
        If let to its default None value, a symmetry test will be conducted.
        Else if set to True (resp. False), C1 and C2 will be assumed symmetric (resp. asymetric).
    verbose : bool, optional
        Print information along iterations
    G0: array-like, shape (ns,nt), optional
        If None the initial transport plan of the solver is pq^T.
        Otherwise G0 must satisfy marginal constraints and will be used as initial transport of the solver.
    max_iter : int, optional
        Max number of iterations
    tol : float, optional
        Stop threshold on error computed on transport plans
    log : bool, optional
        record log if True
    verbose : bool, optional
        Print information along iterations
    Returns
    -------
    G : array-like, shape (`ns`, `nt`)
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
    arr = [C1, C2]
    if p is not None:
        arr.append(list_to_array(p))
    else:
        p = unif(C1.shape[0], type_as=C1)

    if G0 is not None:
        arr.append(G0)

    nx = get_backend(*arr)

    if symmetric is None:
        symmetric = nx.allclose(C1, C1.T, atol=1e-10) and nx.allclose(C2, C2.T, atol=1e-10)
    if G0 is None:
        q = unif(C2.shape[0], type_as=p)
        G0 = nx.outer(p, q)
    else:
        q = nx.sum(G0, 0)
        # Check first marginal of G0
        assert nx.allclose(nx.sum(G0, 1), p, atol=1e-08)

    constC, hC1, hC2, fC2t = init_matrix_semirelaxed(C1, C2, p, loss_fun, nx)

    ones_p = nx.ones(p.shape[0], type_as=p)

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

    cpt = 0
    err = 1e15
    G = G0

    if log:
        log = {'err': []}

    while (err > tol and cpt < max_iter):

        Gprev = G
        # compute the kernel
        K = G * nx.exp(- df(G) / epsilon)
        scaling = p / nx.sum(K, 1)
        G = nx.reshape(scaling, (-1, 1)) * K
        if cpt % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            err = nx.norm(G - Gprev)

            if log:
                log['err'].append(err)

            if verbose:
                if cpt % 200 == 0:
                    print('{:5s}|{:12s}'.format(
                        'It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))

        cpt += 1

    if log:
        qG = nx.sum(G, 0)
        marginal_product = nx.outer(ones_p, nx.dot(qG, fC2t))
        log['srgw_dist'] = gwloss(constC + marginal_product, hC1, hC2, G, nx)
        return G, log
    else:
        return G


def entropic_semirelaxed_gromov_wasserstein2(
        C1, C2, p=None, loss_fun='square_loss', epsilon=0.1, symmetric=None,
        G0=None, max_iter=1e4, tol=1e-9, log=False, verbose=False, **kwargs):
    r"""
    Returns the entropic-regularized semi-relaxed gromov-wasserstein divergence
    from :math:`(\mathbf{C_1}, \mathbf{p})` to :math:`\mathbf{C_2}`
    estimated using a Mirror Descent algorithm following the KL geometry.

    The function solves the following optimization problem:

    .. math::
        \mathbf{srGW} = \min_{\mathbf{T}} \quad \sum_{i,j,k,l}
        L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l}) \mathbf{T}_{i,j} \mathbf{T}_{k,l}

        s.t. \ \mathbf{T} \mathbf{1} &= \mathbf{p}

             \mathbf{T} &\geq 0
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
    p : array-like, shape (ns,), optional
        Distribution in the source space.
        If let to its default value None, uniform distribution is taken.
    loss_fun : str
        loss function used for the solver either 'square_loss' or 'kl_loss'.
    epsilon : float
        Regularization term >0
    symmetric : bool, optional
        Either C1 and C2 are to be assumed symmetric or not.
        If let to its default None value, a symmetry test will be conducted.
        Else if set to True (resp. False), C1 and C2 will be assumed symmetric (resp. asymetric).
    verbose : bool, optional
        Print information along iterations
    G0: array-like, shape (ns,nt), optional
        If None the initial transport plan of the solver is pq^T.
        Otherwise G0 must satisfy marginal constraints and will be used as initial transport of the solver.
    max_iter : int, optional
        Max number of iterations
    tol : float, optional
        Stop threshold on error computed on transport plans
    log : bool, optional
        record log if True
    verbose : bool, optional
        Print information along iterations
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
    T, log_srgw = entropic_semirelaxed_gromov_wasserstein(
        C1, C2, p, loss_fun, epsilon, symmetric, G0,
        max_iter, tol, log=True, verbose=verbose, **kwargs)

    log_srgw['T'] = T

    if log:
        return log_srgw['srgw_dist'], log_srgw
    else:
        return log_srgw['srgw_dist']


def entropic_semirelaxed_fused_gromov_wasserstein(
        M, C1, C2, p=None, loss_fun='square_loss', symmetric=None, epsilon=0.1,
        alpha=0.5, G0=None, max_iter=1e4, tol=1e-9, log=False, verbose=False, **kwargs):
    r"""
    Computes the entropic-regularized semi-relaxed FGW transport between two graphs (see :ref:`[48] <references-semirelaxed-fused-gromov-wasserstein>`)
    estimated using a Mirror Descent algorithm following the KL geometry.

    .. math::
        \mathbf{T}^* \in \mathop{\arg \min}_{\mathbf{T}} \quad (1 - \alpha) \langle \mathbf{T}, \mathbf{M} \rangle_F +
        \alpha \sum_{i,j,k,l} L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l}) \mathbf{T}_{i,j} \mathbf{T}_{k,l}

        s.t. \ \mathbf{T} \mathbf{1} &= \mathbf{p}

             \mathbf{T} &\geq 0

    where :

    - :math:`\mathbf{M}` is the (`ns`, `nt`) metric cost matrix between features
    - :math:`\mathbf{C_1}`: Metric cost matrix in the source space
    - :math:`\mathbf{C_2}`: Metric cost matrix in the target space
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
    p : array-like, shape (ns,), optional
        Distribution in the source space.
        If let to its default value None, uniform distribution is taken.
    loss_fun : str
        loss function used for the solver either 'square_loss' or 'kl_loss'.
    epsilon : float
        Regularization term >0
    symmetric : bool, optional
        Either C1 and C2 are to be assumed symmetric or not.
        If let to its default None value, a symmetry test will be conducted.
        Else if set to True (resp. False), C1 and C2 will be assumed symmetric (resp. asymetric).
    alpha : float, optional
        Trade-off parameter (0 < alpha < 1)
    G0: array-like, shape (ns,nt), optional
        If None the initial transport plan of the solver is pq^T.
        Otherwise G0 must satisfy marginal constraints and will be used as initial transport of the solver.
    max_iter : int, optional
        Max number of iterations
    tol : float, optional
        Stop threshold on error computed on transport plans
    log : bool, optional
        record log if True
    verbose : bool, optional
        Print information along iterations
    **kwargs : dict
        parameters can be directly passed to the ot.optim.cg solver

    Returns
    -------
    G : array-like, shape (`ns`, `nt`)
        Optimal transportation matrix for the given parameters.
    log : dict
        Log dictionary return only if log==True in parameters.


    .. _references-semirelaxed-fused-gromov-wasserstein:
    References
    ----------
    .. [48] Cédric Vincent-Cuaz, Rémi Flamary, Marco Corneli, Titouan Vayer, Nicolas Courty.
            "Semi-relaxed Gromov-Wasserstein divergence and applications on graphs"
            International Conference on Learning Representations (ICLR), 2022.
    """
    arr = [M, C1, C2]
    if p is not None:
        arr.append(list_to_array(p))
    else:
        p = unif(C1.shape[0], type_as=C1)

    if G0 is not None:
        arr.append(G0)

    nx = get_backend(*arr)

    if symmetric is None:
        symmetric = nx.allclose(C1, C1.T, atol=1e-10) and nx.allclose(C2, C2.T, atol=1e-10)
    if G0 is None:
        q = unif(C2.shape[0], type_as=p)
        G0 = nx.outer(p, q)
    else:
        q = nx.sum(G0, 0)
        # Check first marginal of G0
        assert nx.allclose(nx.sum(G0, 1), p, atol=1e-08)

    constC, hC1, hC2, fC2t = init_matrix_semirelaxed(C1, C2, p, loss_fun, nx)

    ones_p = nx.ones(p.shape[0], type_as=p)
    dM = (1 - alpha) * M
    if symmetric:
        def df(G):
            qG = nx.sum(G, 0)
            marginal_product = nx.outer(ones_p, nx.dot(qG, fC2t))
            return alpha * gwggrad(constC + marginal_product, hC1, hC2, G, nx) + dM
    else:
        constCt, hC1t, hC2t, fC2 = init_matrix_semirelaxed(C1.T, C2.T, p, loss_fun, nx)

        def df(G):
            qG = nx.sum(G, 0)
            marginal_product_1 = nx.outer(ones_p, nx.dot(qG, fC2t))
            marginal_product_2 = nx.outer(ones_p, nx.dot(qG, fC2))
            return 0.5 * alpha * (gwggrad(constC + marginal_product_1, hC1, hC2, G, nx) + gwggrad(constCt + marginal_product_2, hC1t, hC2t, G, nx)) + dM

    cpt = 0
    err = 1e15
    G = G0

    if log:
        log = {'err': []}

    while (err > tol and cpt < max_iter):

        Gprev = G
        # compute the kernel
        K = G * nx.exp(- df(G) / epsilon)
        scaling = p / nx.sum(K, 1)
        G = nx.reshape(scaling, (-1, 1)) * K
        if cpt % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            err = nx.norm(G - Gprev)

            if log:
                log['err'].append(err)

            if verbose:
                if cpt % 200 == 0:
                    print('{:5s}|{:12s}'.format(
                        'It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))

        cpt += 1

    if log:
        qG = nx.sum(G, 0)
        marginal_product = nx.outer(ones_p, nx.dot(qG, fC2t))
        log['lin_loss'] = nx.sum(M * G) * (1 - alpha)
        log['quad_loss'] = alpha * gwloss(constC + marginal_product, hC1, hC2, G, nx)
        log['srfgw_dist'] = log['lin_loss'] + log['quad_loss']
        return G, log
    else:
        return G


def entropic_semirelaxed_fused_gromov_wasserstein2(
        M, C1, C2, p=None, loss_fun='square_loss', symmetric=None, epsilon=0.1,
        alpha=0.5, G0=None, max_iter=1e4, tol=1e-9, log=False, verbose=False, **kwargs):
    r"""
    Computes the entropic-regularized semi-relaxed FGW divergence between two graphs (see :ref:`[48] <references-semirelaxed-fused-gromov-wasserstein>`)
    estimated using a Mirror Descent algorithm following the KL geometry.

    .. math::
        \mathbf{srFGW}_{\alpha} = \min_{\mathbf{T}} \quad (1 - \alpha) \langle \mathbf{T}, \mathbf{M} \rangle_F +
        \alpha \sum_{i,j,k,l} L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l}) \mathbf{T}_{i,j} \mathbf{T}_{k,l}

        s.t. \ \mathbf{T} \mathbf{1} &= \mathbf{p}

             \mathbf{T} &\geq 0

    where :

    - :math:`\mathbf{M}` is the (`ns`, `nt`) metric cost matrix between features
    - :math:`\mathbf{C_1}`: Metric cost matrix in the source space
    - :math:`\mathbf{C_2}`: Metric cost matrix in the target space
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
        Metric cost matrix representative of the structure in the source space.
    C2 : array-like, shape (nt, nt)
        Metric cost matrix representative of the structure in the target space.
    p :  array-like, shape (ns,), optional
        Distribution in the source space.
        If let to its default value None, uniform distribution is taken.
    loss_fun : str, optional
        loss function used for the solver either 'square_loss' or 'kl_loss'.
    epsilon : float
        Regularization term >0
    symmetric : bool, optional
        Either C1 and C2 are to be assumed symmetric or not.
        If let to its default None value, a symmetry test will be conducted.
        Else if set to True (resp. False), C1 and C2 will be assumed symmetric (resp. asymetric).
    alpha : float, optional
        Trade-off parameter (0 < alpha < 1)
    G0: array-like, shape (ns,nt), optional
        If None the initial transport plan of the solver is pq^T.
        Otherwise G0 must satisfy marginal constraints and will be used as initial transport of the solver.
    max_iter : int, optional
        Max number of iterations
    tol : float, optional
        Stop threshold on error computed on transport plans
    log : bool, optional
        record log if True
    verbose : bool, optional
        Print information along iterations
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
    .. [48] Cédric Vincent-Cuaz, Rémi Flamary, Marco Corneli, Titouan Vayer, Nicolas Courty.
            "Semi-relaxed Gromov-Wasserstein divergence and applications on graphs"
            International Conference on Learning Representations (ICLR), 2022.
    """
    T, log_srfgw = entropic_semirelaxed_fused_gromov_wasserstein(
        M, C1, C2, p, loss_fun, symmetric, epsilon, alpha, G0,
        max_iter, tol, log=True, verbose=verbose, **kwargs)

    log_srfgw['T'] = T

    if log:
        return log_srfgw['srfgw_dist'], log_srfgw
    else:
        return log_srfgw['srfgw_dist']
