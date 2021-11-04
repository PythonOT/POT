# -*- coding: utf-8 -*-
"""
Gromov-Wasserstein and Fused-Gromov-Wasserstein solvers
"""

# Author: Erwan Vautier <erwan.vautier@gmail.com>
#         Nicolas Courty <ncourty@irisa.fr>
#         Rémi Flamary <remi.flamary@unice.fr>
#         Titouan Vayer <titouan.vayer@irisa.fr>
#
# License: MIT License

import numpy as np


from .bregman import sinkhorn
from .utils import dist, UndefinedParameter, list_to_array
from .optim import cg
from .lp import emd_1d, emd
from .utils import check_random_state
from .backend import get_backend


def init_matrix(C1, C2, p, q, loss_fun='square_loss'):
    r"""Return loss matrices and tensors for Gromov-Wasserstein fast computation

    Returns the value of :math:`\mathcal{L}(\mathbf{C_1}, \mathbf{C_2}) \otimes \mathbf{T}` with the
    selected loss function as the loss function of Gromow-Wasserstein discrepancy.

    The matrices are computed as described in Proposition 1 in :ref:`[12] <references-init-matrix>`

    Where :

    - :math:`\mathbf{C_1}`: Metric cost matrix in the source space
    - :math:`\mathbf{C_2}`: Metric cost matrix in the target space
    - :math:`\mathbf{T}`: A coupling between those two spaces

    The square-loss function :math:`L(a, b) = |a - b|^2` is read as :

    .. math::

        L(a, b) = f_1(a) + f_2(b) - h_1(a) h_2(b)

        \mathrm{with} \ f_1(a) &= a^2

                        f_2(b) &= b^2

                        h_1(a) &= a

                        h_2(b) &= 2b

    The kl-loss function :math:`L(a, b) = a \log\left(\frac{a}{b}\right) - a + b` is read as :

    .. math::

        L(a, b) = f_1(a) + f_2(b) - h_1(a) h_2(b)

        \mathrm{with} \ f_1(a) &= a \log(a) - a

                        f_2(b) &= b

                        h_1(a) &= a

                        h_2(b) &= \log(b)

    Parameters
    ----------
    C1 : array-like, shape (ns, ns)
        Metric cost matrix in the source space
    C2 : array-like, shape (nt, nt)
        Metric cost matrix in the target space
    T :  array-like, shape (ns, nt)
        Coupling between source and target spaces
    p : array-like, shape (ns,)

    Returns
    -------
    constC : array-like, shape (ns, nt)
        Constant :math:`\mathbf{C}` matrix in Eq. (6)
    hC1 : array-like, shape (ns, ns)
        :math:`\mathbf{h1}(\mathbf{C1})` matrix in Eq. (6)
    hC2 : array-like, shape (nt, nt)
        :math:`\mathbf{h2}(\mathbf{C2})` matrix in Eq. (6)


    .. _references-init-matrix:
    References
    ----------
    .. [12] Gabriel Peyré, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.

    """
    C1, C2, p, q = list_to_array(C1, C2, p, q)
    nx = get_backend(C1, C2, p, q)

    if loss_fun == 'square_loss':
        def f1(a):
            return (a**2)

        def f2(b):
            return (b**2)

        def h1(a):
            return a

        def h2(b):
            return 2 * b
    elif loss_fun == 'kl_loss':
        def f1(a):
            return a * nx.log(a + 1e-15) - a

        def f2(b):
            return b

        def h1(a):
            return a

        def h2(b):
            return nx.log(b + 1e-15)

    constC1 = nx.dot(
        nx.dot(f1(C1), nx.reshape(p, (-1, 1))),
        nx.ones((1, len(q)), type_as=q)
    )
    constC2 = nx.dot(
        nx.ones((len(p), 1), type_as=p),
        nx.dot(nx.reshape(q, (1, -1)), f2(C2).T)
    )
    constC = constC1 + constC2
    hC1 = h1(C1)
    hC2 = h2(C2)

    return constC, hC1, hC2


def tensor_product(constC, hC1, hC2, T):
    r"""Return the tensor for Gromov-Wasserstein fast computation

    The tensor is computed as described in Proposition 1 Eq. (6) in :ref:`[12] <references-tensor-product>`

    Parameters
    ----------
    constC : array-like, shape (ns, nt)
        Constant :math:`\mathbf{C}` matrix in Eq. (6)
    hC1 : array-like, shape (ns, ns)
        :math:`\mathbf{h1}(\mathbf{C1})` matrix in Eq. (6)
    hC2 : array-like, shape (nt, nt)
        :math:`\mathbf{h2}(\mathbf{C2})` matrix in Eq. (6)

    Returns
    -------
    tens : array-like, shape (`ns`, `nt`)
        :math:`\mathcal{L}(\mathbf{C_1}, \mathbf{C_2}) \otimes \mathbf{T}` tensor-matrix multiplication result


    .. _references-tensor-product:
    References
    ----------
    .. [12] Gabriel Peyré, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.

    """
    constC, hC1, hC2, T = list_to_array(constC, hC1, hC2, T)
    nx = get_backend(constC, hC1, hC2, T)

    A = - nx.dot(
        nx.dot(hC1, T), hC2.T
    )
    tens = constC + A
    # tens -= tens.min()
    return tens


def gwloss(constC, hC1, hC2, T):
    r"""Return the Loss for Gromov-Wasserstein

    The loss is computed as described in Proposition 1 Eq. (6) in :ref:`[12] <references-gwloss>`

    Parameters
    ----------
    constC : array-like, shape (ns, nt)
        Constant :math:`\mathbf{C}` matrix in Eq. (6)
    hC1 : array-like, shape (ns, ns)
        :math:`\mathbf{h1}(\mathbf{C1})` matrix in Eq. (6)
    hC2 : array-like, shape (nt, nt)
        :math:`\mathbf{h2}(\mathbf{C2})` matrix in Eq. (6)
    T : array-like, shape (ns, nt)
        Current value of transport matrix :math:`\mathbf{T}`

    Returns
    -------
    loss : float
        Gromov Wasserstein loss


    .. _references-gwloss:
    References
    ----------
    .. [12] Gabriel Peyré, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.

    """

    tens = tensor_product(constC, hC1, hC2, T)

    tens, T = list_to_array(tens, T)
    nx = get_backend(tens, T)

    return nx.sum(tens * T)


def gwggrad(constC, hC1, hC2, T):
    r"""Return the gradient for Gromov-Wasserstein

    The gradient is computed as described in Proposition 2 in :ref:`[12] <references-gwggrad>`

    Parameters
    ----------
    constC : array-like, shape (ns, nt)
        Constant :math:`\mathbf{C}` matrix in Eq. (6)
    hC1 : array-like, shape (ns, ns)
        :math:`\mathbf{h1}(\mathbf{C1})` matrix in Eq. (6)
    hC2 : array-like, shape (nt, nt)
        :math:`\mathbf{h2}(\mathbf{C2})` matrix in Eq. (6)
    T : array-like, shape (ns, nt)
        Current value of transport matrix :math:`\mathbf{T}`

    Returns
    -------
    grad : array-like, shape (`ns`, `nt`)
           Gromov Wasserstein gradient


    .. _references-gwggrad:
    References
    ----------
    .. [12] Gabriel Peyré, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.

    """
    return 2 * tensor_product(constC, hC1, hC2,
                              T)  # [12] Prop. 2 misses a 2 factor


def update_square_loss(p, lambdas, T, Cs):
    r"""
    Updates :math:`\mathbf{C}` according to the L2 Loss kernel with the `S` :math:`\mathbf{T}_s`
    couplings calculated at each iteration

    Parameters
    ----------
    p : array-like, shape (N,)
        Masses in the targeted barycenter.
    lambdas : list of float
        List of the `S` spaces' weights.
    T : list of S array-like of shape (ns,N)
        The `S` :math:`\mathbf{T}_s` couplings calculated at each iteration.
    Cs : list of S array-like, shape(ns,ns)
        Metric cost matrices.

    Returns
    ----------
    C : array-like, shape (`nt`, `nt`)
        Updated :math:`\mathbf{C}` matrix.
    """
    T = list_to_array(*T)
    Cs = list_to_array(*Cs)
    p = list_to_array(p)
    nx = get_backend(p, *T, *Cs)

    tmpsum = sum([
        lambdas[s] * nx.dot(
            nx.dot(T[s].T, Cs[s]),
            T[s]
        ) for s in range(len(T))
    ])
    ppt = nx.outer(p, p)

    return tmpsum / ppt


def update_kl_loss(p, lambdas, T, Cs):
    r"""
    Updates :math:`\mathbf{C}` according to the KL Loss kernel with the `S` :math:`\mathbf{T}_s` couplings calculated at each iteration


    Parameters
    ----------
    p  : array-like, shape (N,)
        Weights in the targeted barycenter.
    lambdas : list of float
        List of the `S` spaces' weights
    T : list of S array-like of shape (ns,N)
        The `S` :math:`\mathbf{T}_s` couplings calculated at each iteration.
    Cs : list of S array-like, shape(ns,ns)
        Metric cost matrices.

    Returns
    ----------
    C : array-like, shape (`ns`, `ns`)
        updated :math:`\mathbf{C}` matrix
    """
    Cs = list_to_array(*Cs)
    T = list_to_array(*T)
    p = list_to_array(p)
    nx = get_backend(p, *T, *Cs)

    tmpsum = sum([
        lambdas[s] * nx.dot(
            nx.dot(T[s].T, Cs[s]),
            T[s]
        ) for s in range(len(T))
    ])
    ppt = nx.outer(p, p)

    return nx.exp(tmpsum / ppt)


def gromov_wasserstein(C1, C2, p, q, loss_fun='square_loss', log=False, armijo=False, **kwargs):
    r"""
    Returns the gromov-wasserstein transport between :math:`(\mathbf{C_1}, \mathbf{p})` and :math:`(\mathbf{C_2}, \mathbf{q})`

    The function solves the following optimization problem:

    .. math::
        \mathbf{GW} = \mathop{\arg \min}_\mathbf{T} \quad \sum_{i,j,k,l}
        L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l}) \mathbf{T}_{i,j} \mathbf{T}_{k,l}

    Where :

    - :math:`\mathbf{C_1}`: Metric cost matrix in the source space
    - :math:`\mathbf{C_2}`: Metric cost matrix in the target space
    - :math:`\mathbf{p}`: distribution in the source space
    - :math:`\mathbf{q}`: distribution in the target space
    - `L`: loss function to account for the misfit between the similarity matrices

    Parameters
    ----------
    C1 : array-like, shape (ns, ns)
        Metric cost matrix in the source space
    C2 : array-like, shape (nt, nt)
        Metric cost matrix in the target space
    p : array-like, shape (ns,)
        Distribution in the source space
    q : array-like, shape (nt,)
        Distribution in the target space
    loss_fun : str
        loss function used for the solver either 'square_loss' or 'kl_loss'
    max_iter : int, optional
        Max number of iterations
    tol : float, optional
        Stop threshold on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    armijo : bool, optional
        If True the step of the line-search is found via an armijo research. Else closed form is used.
        If there are convergence issues use False.
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
    .. [12] Gabriel Peyré, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.

    .. [13] Mémoli, Facundo. Gromov–Wasserstein distances and the
        metric approach to object matching. Foundations of computational
        mathematics 11.4 (2011): 417-487.

    """
    p, q = list_to_array(p, q)

    p0, q0, C10, C20 = p, q, C1, C2
    nx = get_backend(p0, q0, C10, C20)

    p = nx.to_numpy(p)
    q = nx.to_numpy(q)
    C1 = nx.to_numpy(C10)
    C2 = nx.to_numpy(C20)

    constC, hC1, hC2 = init_matrix(C1, C2, p, q, loss_fun)

    G0 = p[:, None] * q[None, :]

    def f(G):
        return gwloss(constC, hC1, hC2, G)

    def df(G):
        return gwggrad(constC, hC1, hC2, G)

    if log:
        res, log = cg(p, q, 0, 1, f, df, G0, log=True, armijo=armijo, C1=C1, C2=C2, constC=constC, **kwargs)
        log['gw_dist'] = nx.from_numpy(gwloss(constC, hC1, hC2, res), type_as=C10)
        log['u'] = nx.from_numpy(log['u'], type_as=C10)
        log['v'] = nx.from_numpy(log['v'], type_as=C10)
        return nx.from_numpy(res, type_as=C10), log
    else:
        return nx.from_numpy(cg(p, q, 0, 1, f, df, G0, armijo=armijo, C1=C1, C2=C2, constC=constC, log=False, **kwargs), type_as=C10)


def gromov_wasserstein2(C1, C2, p, q, loss_fun='square_loss', log=False, armijo=False, **kwargs):
    r"""
    Returns the gromov-wasserstein discrepancy between :math:`(\mathbf{C_1}, \mathbf{p})` and :math:`(\mathbf{C_2}, \mathbf{q})`

    The function solves the following optimization problem:

    .. math::
        GW = \min_\mathbf{T} \quad \sum_{i,j,k,l}
        L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l}) \mathbf{T}_{i,j} \mathbf{T}_{k,l}

    Where :

    - :math:`\mathbf{C_1}`: Metric cost matrix in the source space
    - :math:`\mathbf{C_2}`: Metric cost matrix in the target space
    - :math:`\mathbf{p}`: distribution in the source space
    - :math:`\mathbf{q}`: distribution in the target space
    - `L`: loss function to account for the misfit between the similarity
      matrices

    Note that when using backends, this loss function is differentiable wrt the
    marices and weights for quadratic loss using the gradients from [38]_.

    Parameters
    ----------
    C1 : array-like, shape (ns, ns)
        Metric cost matrix in the source space
    C2 : array-like, shape (nt, nt)
        Metric cost matrix in the target space
    p : array-like, shape (ns,)
        Distribution in the source space.
    q :  array-like, shape (nt,)
        Distribution in the target space.
    loss_fun :  str
        loss function used for the solver either 'square_loss' or 'kl_loss'
    max_iter : int, optional
        Max number of iterations
    tol : float, optional
        Stop threshold on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    armijo : bool, optional
        If True the step of the line-search is found via an armijo research. Else closed form is used.
        If there are convergence issues use False.

    Returns
    -------
    gw_dist : float
        Gromov-Wasserstein distance
    log : dict
        convergence information and Coupling marix

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

    """
    p, q = list_to_array(p, q)

    p0, q0, C10, C20 = p, q, C1, C2
    nx = get_backend(p0, q0, C10, C20)

    p = nx.to_numpy(p)
    q = nx.to_numpy(q)
    C1 = nx.to_numpy(C10)
    C2 = nx.to_numpy(C20)

    constC, hC1, hC2 = init_matrix(C1, C2, p, q, loss_fun)

    G0 = p[:, None] * q[None, :]

    def f(G):
        return gwloss(constC, hC1, hC2, G)

    def df(G):
        return gwggrad(constC, hC1, hC2, G)

    T, log_gw = cg(p, q, 0, 1, f, df, G0, log=True, armijo=armijo, C1=C1, C2=C2, constC=constC, **kwargs)

    T0 = nx.from_numpy(T, type_as=C10)

    log_gw['gw_dist'] = nx.from_numpy(gwloss(constC, hC1, hC2, T), type_as=C10)
    log_gw['u'] = nx.from_numpy(log_gw['u'], type_as=C10)
    log_gw['v'] = nx.from_numpy(log_gw['v'], type_as=C10)
    log_gw['T'] = T0

    gw = log_gw['gw_dist']

    if loss_fun == 'square_loss':
        gC1 = nx.from_numpy(2 * C1 * (p[:, None] * p[None, :]) - 2 * T.dot(C2).dot(T.T))
        gC2 = nx.from_numpy(2 * C2 * (q[:, None] * q[None, :]) - 2 * T.T.dot(C1).dot(T))
        gw = nx.set_gradients(gw, (p0, q0, C10, C20),
                              (log_gw['u'], log_gw['v'], gC1, gC2))

    if log:
        return gw, log_gw
    else:
        return gw


def fused_gromov_wasserstein(M, C1, C2, p, q, loss_fun='square_loss', alpha=0.5, armijo=False, log=False, **kwargs):
    r"""
    Computes the FGW transport between two graphs (see :ref:`[24] <references-fused-gromov-wasserstein>`)

    .. math::
        \gamma = \mathop{\arg \min}_\gamma \quad (1 - \alpha) \langle \gamma, \mathbf{M} \rangle_F +
        \alpha \sum_{i,j,k,l} L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l}) \mathbf{T}_{i,j} \mathbf{T}_{k,l}

        s.t. \ \mathbf{\gamma} \mathbf{1} &= \mathbf{p}

             \mathbf{\gamma}^T \mathbf{1} &= \mathbf{q}

             \mathbf{\gamma} &\geq 0

    where :

    - :math:`\mathbf{M}` is the (`ns`, `nt`) metric cost matrix
    - :math:`\mathbf{p}` and :math:`\mathbf{q}` are source and target weights (sum to 1)
    - `L` is a loss function to account for the misfit between the similarity matrices

    The algorithm used for solving the problem is conditional gradient as discussed in :ref:`[24] <references-fused-gromov-wasserstein>`

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
    q : array-like, shape (nt,)
        Distribution in the target space
    loss_fun : str, optional
        Loss function used for the solver
    alpha : float, optional
        Trade-off parameter (0 < alpha < 1)
    armijo : bool, optional
        If True the step of the line-search is found via an armijo research. Else closed form is used.
        If there are convergence issues use False.
    log : bool, optional
        record log if True
    **kwargs : dict
        parameters can be directly passed to the ot.optim.cg solver

    Returns
    -------
    gamma : array-like, shape (`ns`, `nt`)
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
    """
    p, q = list_to_array(p, q)

    p0, q0, C10, C20, M0 = p, q, C1, C2, M
    nx = get_backend(p0, q0, C10, C20, M0)

    p = nx.to_numpy(p)
    q = nx.to_numpy(q)
    C1 = nx.to_numpy(C10)
    C2 = nx.to_numpy(C20)
    M = nx.to_numpy(M0)

    constC, hC1, hC2 = init_matrix(C1, C2, p, q, loss_fun)

    G0 = p[:, None] * q[None, :]

    def f(G):
        return gwloss(constC, hC1, hC2, G)

    def df(G):
        return gwggrad(constC, hC1, hC2, G)

    if log:
        res, log = cg(p, q, (1 - alpha) * M, alpha, f, df, G0, armijo=armijo, C1=C1, C2=C2, constC=constC, log=True, **kwargs)

        fgw_dist = nx.from_numpy(log['loss'][-1], type_as=C10)

        log['fgw_dist'] = fgw_dist
        log['u'] = nx.from_numpy(log['u'], type_as=C10)
        log['v'] = nx.from_numpy(log['v'], type_as=C10)
        return nx.from_numpy(res, type_as=C10), log

    else:
        return nx.from_numpy(cg(p, q, (1 - alpha) * M, alpha, f, df, G0, armijo=armijo, C1=C1, C2=C2, constC=constC, **kwargs), type_as=C10)


def fused_gromov_wasserstein2(M, C1, C2, p, q, loss_fun='square_loss', alpha=0.5, armijo=False, log=False, **kwargs):
    r"""
    Computes the FGW distance between two graphs see (see :ref:`[24] <references-fused-gromov-wasserstein2>`)

    .. math::
        \min_\gamma \quad (1 - \alpha) \langle \gamma, \mathbf{M} \rangle_F + \alpha \sum_{i,j,k,l}
        L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l}) \mathbf{T}_{i,j} \mathbf{T}_{k,l}

        s.t. \ \mathbf{\gamma} \mathbf{1} &= \mathbf{p}

             \mathbf{\gamma}^T \mathbf{1} &= \mathbf{q}

             \mathbf{\gamma} &\geq 0

    where :

    - :math:`\mathbf{M}` is the (`ns`, `nt`) metric cost matrix
    - :math:`\mathbf{p}` and :math:`\mathbf{q}` are source and target weights (sum to 1)
    - `L` is a loss function to account for the misfit between the similarity matrices

    The algorithm used for solving the problem is conditional gradient as
    discussed in :ref:`[24] <references-fused-gromov-wasserstein2>`

    Note that when using backends, this loss function is differentiable wrt the
    marices and weights for quadratic loss using the gradients from [38]_.

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
    q :  array-like, shape (nt,)
        Distribution in the target space.
    loss_fun : str, optional
        Loss function used for the solver.
    alpha : float, optional
        Trade-off parameter (0 < alpha < 1)
    armijo : bool, optional
        If True the step of the line-search is found via an armijo research.
        Else closed form is used. If there are convergence issues use False.
    log : bool, optional
        Record log if True.
    **kwargs : dict
        Parameters can be directly passed to the ot.optim.cg solver.

    Returns
    -------
    fgw-distance : float
        Fused gromov wasserstein distance for the given parameters.
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
    """
    p, q = list_to_array(p, q)

    p0, q0, C10, C20, M0 = p, q, C1, C2, M
    nx = get_backend(p0, q0, C10, C20, M0)

    p = nx.to_numpy(p)
    q = nx.to_numpy(q)
    C1 = nx.to_numpy(C10)
    C2 = nx.to_numpy(C20)
    M = nx.to_numpy(M0)

    constC, hC1, hC2 = init_matrix(C1, C2, p, q, loss_fun)

    G0 = p[:, None] * q[None, :]

    def f(G):
        return gwloss(constC, hC1, hC2, G)

    def df(G):
        return gwggrad(constC, hC1, hC2, G)

    T, log_fgw = cg(p, q, (1 - alpha) * M, alpha, f, df, G0, armijo=armijo, C1=C1, C2=C2, constC=constC, log=True, **kwargs)

    fgw_dist = nx.from_numpy(log_fgw['loss'][-1], type_as=C10)

    T0 = nx.from_numpy(T, type_as=C10)

    log_fgw['fgw_dist'] = fgw_dist
    log_fgw['u'] = nx.from_numpy(log_fgw['u'], type_as=C10)
    log_fgw['v'] = nx.from_numpy(log_fgw['v'], type_as=C10)
    log_fgw['T'] = T0

    if loss_fun == 'square_loss':
        gC1 = nx.from_numpy(2 * C1 * (p[:, None] * p[None, :]) - 2 * T.dot(C2).dot(T.T))
        gC2 = nx.from_numpy(2 * C2 * (q[:, None] * q[None, :]) - 2 * T.T.dot(C1).dot(T))
        fgw_dist = nx.set_gradients(fgw_dist, (p0, q0, C10, C20, M0),
                                    (log_fgw['u'], log_fgw['v'], alpha * gC1, alpha * gC2, (1 - alpha) * T0))

    if log:
        return fgw_dist, log_fgw
    else:
        return fgw_dist


def GW_distance_estimation(C1, C2, p, q, loss_fun, T,
                           nb_samples_p=None, nb_samples_q=None, std=True, random_state=None):
    r"""
    Returns an approximation of the gromov-wasserstein cost between :math:`(\mathbf{C_1}, \mathbf{p})` and :math:`(\mathbf{C_2}, \mathbf{q})`
    with a fixed transport plan :math:`\mathbf{T}`.

    The function gives an unbiased approximation of the following equation:

    .. math::

        GW = \sum_{i,j,k,l} L(\mathbf{C_{1}}_{i,k}, \mathbf{C_{2}}_{j,l}) \mathbf{T}_{i,j} \mathbf{T}_{k,l}

    Where :

    - :math:`\mathbf{C_1}`: Metric cost matrix in the source space
    - :math:`\mathbf{C_2}`: Metric cost matrix in the target space
    - `L` : Loss function to account for the misfit between the similarity matrices
    - :math:`\mathbf{T}`: Matrix with marginal :math:`\mathbf{p}` and :math:`\mathbf{q}`

    Parameters
    ----------
    C1 : array-like, shape (ns, ns)
        Metric cost matrix in the source space
    C2 : array-like, shape (nt, nt)
        Metric cost matrix in the target space
    p :  array-like, shape (ns,)
        Distribution in the source space
    q :  array-like, shape (nt,)
        Distribution in the target space
    loss_fun :  function: :math:`\mathbb{R} \times \mathbb{R} \mapsto \mathbb{R}`
        Loss function used for the distance, the transport plan does not depend on the loss function
    T : csr or array-like, shape (ns, nt)
        Transport plan matrix, either a sparse csr or a dense matrix
    nb_samples_p : int, optional
        `nb_samples_p` is the number of samples (without replacement) along the first dimension of :math:`\mathbf{T}`
    nb_samples_q : int, optional
        `nb_samples_q` is the number of samples along the second dimension of :math:`\mathbf{T}`, for each sample along the first
    std : bool, optional
        Standard deviation associated with the prediction of the gromov-wasserstein cost
    random_state : int or RandomState instance, optional
        Fix the seed for reproducibility

    Returns
    -------
    : float
        Gromov-wasserstein cost

    References
    ----------
    .. [14] Kerdoncuff, Tanguy, Emonet, Rémi, Sebban, Marc
        "Sampled Gromov Wasserstein."
        Machine Learning Journal (MLJ). 2021.

    """
    C1, C2, p, q = list_to_array(C1, C2, p, q)
    nx = get_backend(C1, C2, p, q)

    generator = check_random_state(random_state)

    len_p = p.shape[0]
    len_q = q.shape[0]

    # It is always better to sample from the biggest distribution first.
    if len_p < len_q:
        p, q = q, p
        len_p, len_q = len_q, len_p
        C1, C2 = C2, C1
        T = T.T

    if nb_samples_p is None:
        if nx.issparse(T):
            # If T is sparse, it probably mean that PoGroW was used, thus the number of sample is reduced
            nb_samples_p = min(int(5 * (len_p * np.log(len_p)) ** 0.5), len_p)
        else:
            nb_samples_p = len_p
    else:
        # The number of sample along the first dimension is without replacement.
        nb_samples_p = min(nb_samples_p, len_p)
    if nb_samples_q is None:
        nb_samples_q = 1
    if std:
        nb_samples_q = max(2, nb_samples_q)

    index_k = np.zeros((nb_samples_p, nb_samples_q), dtype=int)
    index_l = np.zeros((nb_samples_p, nb_samples_q), dtype=int)

    index_i = generator.choice(len_p, size=nb_samples_p, p=p, replace=False)
    index_j = generator.choice(len_p, size=nb_samples_p, p=p, replace=False)

    for i in range(nb_samples_p):
        if nx.issparse(T):
            T_indexi = nx.reshape(nx.todense(T[index_i[i], :]), (-1,))
            T_indexj = nx.reshape(nx.todense(T[index_j[i], :]), (-1,))
        else:
            T_indexi = T[index_i[i], :]
            T_indexj = T[index_j[i], :]
        # For each of the row sampled, the column is sampled.
        index_k[i] = generator.choice(
            len_q,
            size=nb_samples_q,
            p=T_indexi / nx.sum(T_indexi),
            replace=True
        )
        index_l[i] = generator.choice(
            len_q,
            size=nb_samples_q,
            p=T_indexj / nx.sum(T_indexj),
            replace=True
        )

    list_value_sample = nx.stack([
        loss_fun(
            C1[np.ix_(index_i, index_j)],
            C2[np.ix_(index_k[:, n], index_l[:, n])]
        ) for n in range(nb_samples_q)
    ], axis=2)

    if std:
        std_value = nx.sum(nx.std(list_value_sample, axis=2) ** 2) ** 0.5
        return nx.mean(list_value_sample), std_value / (nb_samples_p * nb_samples_p)
    else:
        return nx.mean(list_value_sample)


def pointwise_gromov_wasserstein(C1, C2, p, q, loss_fun,
                                 alpha=1, max_iter=100, threshold_plan=0, log=False, verbose=False, random_state=None):
    r"""
    Returns the gromov-wasserstein transport between :math:`(\mathbf{C_1}, \mathbf{p})` and :math:`(\mathbf{C_2}, \mathbf{q})` using a stochastic Frank-Wolfe.
    This method has a :math:`\mathcal{O}(\mathrm{max\_iter} \times PN^2)` time complexity with `P` the number of Sinkhorn iterations.

    The function solves the following optimization problem:

    .. math::
        \mathbf{GW} = \mathop{\arg \min}_\mathbf{T} \quad \sum_{i,j,k,l}
        L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l}) \mathbf{T}_{i,j} \mathbf{T}_{k,l}

        s.t. \ \mathbf{T} \mathbf{1} &= \mathbf{p}

                \mathbf{T}^T \mathbf{1} &= \mathbf{q}

                \mathbf{T} &\geq 0

    Where :

    - :math:`\mathbf{C_1}`: Metric cost matrix in the source space
    - :math:`\mathbf{C_2}`: Metric cost matrix in the target space
    - :math:`\mathbf{p}`: distribution in the source space
    - :math:`\mathbf{q}`: distribution in the target space
    - `L`: loss function to account for the misfit between the similarity matrices

    Parameters
    ----------
    C1 : array-like, shape (ns, ns)
        Metric cost matrix in the source space
    C2 : array-like, shape (nt, nt)
        Metric cost matrix in the target space
    p :  array-like, shape (ns,)
        Distribution in the source space
    q :  array-like, shape (nt,)
        Distribution in the target space
    loss_fun :  function: :math:`\mathbb{R} \times \mathbb{R} \mapsto \mathbb{R}`
        Loss function used for the distance, the transport plan does not depend on the loss function
    alpha : float
        Step of the Frank-Wolfe algorithm, should be between 0 and 1
    max_iter : int, optional
        Max number of iterations
    threshold_plan : float, optional
        Deleting very small values in the transport plan. If above zero, it violates the marginal constraints.
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        Gives the distance estimated and the standard deviation
    random_state : int or RandomState instance, optional
        Fix the seed for reproducibility

    Returns
    -------
    T : array-like, shape (`ns`, `nt`)
        Optimal coupling between the two spaces

    References
    ----------
    .. [14] Kerdoncuff, Tanguy, Emonet, Rémi, Sebban, Marc
        "Sampled Gromov Wasserstein."
        Machine Learning Journal (MLJ). 2021.

    """
    C1, C2, p, q = list_to_array(C1, C2, p, q)
    nx = get_backend(C1, C2, p, q)

    len_p = p.shape[0]
    len_q = q.shape[0]

    generator = check_random_state(random_state)

    index = np.zeros(2, dtype=int)

    # Initialize with default marginal
    index[0] = generator.choice(len_p, size=1, p=p)
    index[1] = generator.choice(len_q, size=1, p=q)
    T = nx.tocsr(emd_1d(C1[index[0]], C2[index[1]], a=p, b=q, dense=False))

    best_gw_dist_estimated = np.inf
    for cpt in range(max_iter):
        index[0] = generator.choice(len_p, size=1, p=p)
        T_index0 = nx.reshape(nx.todense(T[index[0], :]), (-1,))
        index[1] = generator.choice(len_q, size=1, p=T_index0 / T_index0.sum())

        if alpha == 1:
            T = nx.tocsr(
                emd_1d(C1[index[0]], C2[index[1]], a=p, b=q, dense=False)
            )
        else:
            new_T = nx.tocsr(
                emd_1d(C1[index[0]], C2[index[1]], a=p, b=q, dense=False)
            )
            T = (1 - alpha) * T + alpha * new_T
            # To limit the number of non 0, the values below the threshold are set to 0.
            T = nx.eliminate_zeros(T, threshold=threshold_plan)

        if cpt % 10 == 0 or cpt == (max_iter - 1):
            gw_dist_estimated = GW_distance_estimation(
                C1=C1, C2=C2, loss_fun=loss_fun,
                p=p, q=q, T=T, std=False, random_state=generator
            )

            if gw_dist_estimated < best_gw_dist_estimated:
                best_gw_dist_estimated = gw_dist_estimated
                best_T = nx.copy(T)

            if verbose:
                if cpt % 200 == 0:
                    print('{:5s}|{:12s}'.format('It.', 'Best gw estimated') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, best_gw_dist_estimated))

    if log:
        log = {}
        log["gw_dist_estimated"], log["gw_dist_std"] = GW_distance_estimation(
            C1=C1, C2=C2, loss_fun=loss_fun,
            p=p, q=q, T=best_T, random_state=generator
        )
        return best_T, log
    return best_T


def sampled_gromov_wasserstein(C1, C2, p, q, loss_fun,
                               nb_samples_grad=100, epsilon=1, max_iter=500, log=False, verbose=False,
                               random_state=None):
    r"""
    Returns the gromov-wasserstein transport between :math:`(\mathbf{C_1}, \mathbf{p})` and :math:`(\mathbf{C_2}, \mathbf{q})` using a 1-stochastic Frank-Wolfe.
    This method has a :math:`\mathcal{O}(\mathrm{max\_iter} \times N \log(N))` time complexity by relying on the 1D Optimal Transport solver.

    The function solves the following optimization problem:

    .. math::
        \mathbf{GW} = \mathop{\arg \min}_\mathbf{T} \quad \sum_{i,j,k,l}
        L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l}) \mathbf{T}_{i,j} \mathbf{T}_{k,l}

        s.t. \ \mathbf{T} \mathbf{1} &= \mathbf{p}

                \mathbf{T}^T \mathbf{1} &= \mathbf{q}

                \mathbf{T} &\geq 0

    Where :

    - :math:`\mathbf{C_1}`: Metric cost matrix in the source space
    - :math:`\mathbf{C_2}`: Metric cost matrix in the target space
    - :math:`\mathbf{p}`: distribution in the source space
    - :math:`\mathbf{q}`: distribution in the target space
    - `L`: loss function to account for the misfit between the similarity matrices

    Parameters
    ----------
    C1 : array-like, shape (ns, ns)
        Metric cost matrix in the source space
    C2 : array-like, shape (nt, nt)
        Metric cost matrix in the target space
    p :  array-like, shape (ns,)
        Distribution in the source space
    q :  array-like, shape (nt,)
        Distribution in the target space
    loss_fun :  function: :math:`\mathbb{R} \times \mathbb{R} \mapsto \mathbb{R}`
        Loss function used for the distance, the transport plan does not depend on the loss function
    nb_samples_grad : int
        Number of samples to approximate the gradient
    epsilon : float
        Weight of the Kullback-Leibler regularization
    max_iter : int, optional
        Max number of iterations
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        Gives the distance estimated and the standard deviation
    random_state : int or RandomState instance, optional
        Fix the seed for reproducibility

    Returns
    -------
    T : array-like, shape (`ns`, `nt`)
        Optimal coupling between the two spaces

    References
    ----------
    .. [14] Kerdoncuff, Tanguy, Emonet, Rémi, Sebban, Marc
        "Sampled Gromov Wasserstein."
        Machine Learning Journal (MLJ). 2021.

    """
    C1, C2, p, q = list_to_array(C1, C2, p, q)
    nx = get_backend(C1, C2, p, q)

    len_p = p.shape[0]
    len_q = q.shape[0]

    generator = check_random_state(random_state)

    # The most natural way to define nb_sample is with a simple integer.
    if isinstance(nb_samples_grad, int):
        if nb_samples_grad > len_p:
            # As the sampling along the first dimension is done without replacement, the rest is reported to the second
            # dimension.
            nb_samples_grad_p, nb_samples_grad_q = len_p, nb_samples_grad // len_p
        else:
            nb_samples_grad_p, nb_samples_grad_q = nb_samples_grad, 1
    else:
        nb_samples_grad_p, nb_samples_grad_q = nb_samples_grad
    T = nx.outer(p, q)
    # continue_loop allows to stop the loop if there is several successive small modification of T.
    continue_loop = 0

    # The gradient of GW is more complex if the two matrices are not symmetric.
    C_are_symmetric = nx.allclose(C1, C1.T, rtol=1e-10, atol=1e-10) and nx.allclose(C2, C2.T, rtol=1e-10, atol=1e-10)

    for cpt in range(max_iter):
        index0 = generator.choice(len_p, size=nb_samples_grad_p, p=p, replace=False)
        Lik = 0
        for i, index0_i in enumerate(index0):
            index1 = generator.choice(len_q,
                                      size=nb_samples_grad_q,
                                      p=T[index0_i, :] / nx.sum(T[index0_i, :]),
                                      replace=False)
            # If the matrices C are not symmetric, the gradient has 2 terms, thus the term is chosen randomly.
            if (not C_are_symmetric) and generator.rand(1) > 0.5:
                Lik += nx.mean(loss_fun(
                    C1[:, [index0[i]] * nb_samples_grad_q][:, None, :],
                    C2[:, index1][None, :, :]
                ), axis=2)
            else:
                Lik += nx.mean(loss_fun(
                    C1[[index0[i]] * nb_samples_grad_q, :][:, :, None],
                    C2[index1, :][:, None, :]
                ), axis=0)

        max_Lik = nx.max(Lik)
        if max_Lik == 0:
            continue
        # This division by the max is here to facilitate the choice of epsilon.
        Lik /= max_Lik

        if epsilon > 0:
            # Set to infinity all the numbers below exp(-200) to avoid log of 0.
            log_T = nx.log(nx.clip(T, np.exp(-200), 1))
            log_T = nx.where(log_T == -200, -np.inf, log_T)
            Lik = Lik - epsilon * log_T

            try:
                new_T = sinkhorn(a=p, b=q, M=Lik, reg=epsilon)
            except (RuntimeWarning, UserWarning):
                print("Warning catched in Sinkhorn: Return last stable T")
                break
        else:
            new_T = emd(a=p, b=q, M=Lik)

        change_T = nx.mean((T - new_T) ** 2)
        if change_T <= 10e-20:
            continue_loop += 1
            if continue_loop > 100:  # Number max of low modifications of T
                T = nx.copy(new_T)
                break
        else:
            continue_loop = 0

        if verbose and cpt % 10 == 0:
            if cpt % 200 == 0:
                print('{:5s}|{:12s}'.format('It.', '||T_n - T_{n+1}||') + '\n' + '-' * 19)
            print('{:5d}|{:8e}|'.format(cpt, change_T))
        T = nx.copy(new_T)

    if log:
        log = {}
        log["gw_dist_estimated"], log["gw_dist_std"] = GW_distance_estimation(
            C1=C1, C2=C2, loss_fun=loss_fun,
            p=p, q=q, T=T, random_state=generator
        )
        return T, log
    return T


def entropic_gromov_wasserstein(C1, C2, p, q, loss_fun, epsilon,
                                max_iter=1000, tol=1e-9, verbose=False, log=False):
    r"""
    Returns the gromov-wasserstein transport between :math:`(\mathbf{C_1}, \mathbf{p})` and :math:`(\mathbf{C_2}, \mathbf{q})`

    The function solves the following optimization problem:

    .. math::
        \mathbf{GW} = \mathop{\arg\min}_\mathbf{T} \quad \sum_{i,j,k,l} L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l}) \mathbf{T}_{i,j} \mathbf{T}_{k,l} - \epsilon(H(\mathbf{T}))

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

    Parameters
    ----------
    C1 : array-like, shape (ns, ns)
        Metric cost matrix in the source space
    C2 : array-like, shape (nt, nt)
        Metric cost matrix in the target space
    p :  array-like, shape (ns,)
        Distribution in the source space
    q :  array-like, shape (nt,)
        Distribution in the target space
    loss_fun :  string
        Loss function used for the solver either 'square_loss' or 'kl_loss'
    epsilon : float
        Regularization term >0
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
    T : array-like, shape (`ns`, `nt`)
        Optimal coupling between the two spaces

    References
    ----------
    .. [12] Gabriel Peyré, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.

    """
    C1, C2, p, q = list_to_array(C1, C2, p, q)
    nx = get_backend(C1, C2, p, q)

    T = nx.outer(p, q)

    constC, hC1, hC2 = init_matrix(C1, C2, p, q, loss_fun)

    cpt = 0
    err = 1

    if log:
        log = {'err': []}

    while (err > tol and cpt < max_iter):

        Tprev = T

        # compute the gradient
        tens = gwggrad(constC, hC1, hC2, T)

        T = sinkhorn(p, q, tens, epsilon, method='sinkhorn')

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

    if log:
        log['gw_dist'] = gwloss(constC, hC1, hC2, T)
        return T, log
    else:
        return T


def entropic_gromov_wasserstein2(C1, C2, p, q, loss_fun, epsilon,
                                 max_iter=1000, tol=1e-9, verbose=False, log=False):
    r"""
    Returns the entropic gromov-wasserstein discrepancy between the two measured similarity matrices :math:`(\mathbf{C_1}, \mathbf{p})` and :math:`(\mathbf{C_2}, \mathbf{q})`

    The function solves the following optimization problem:

    .. math::
        GW = \min_\mathbf{T} \quad \sum_{i,j,k,l} L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l})
        \mathbf{T}_{i,j} \mathbf{T}_{k,l} - \epsilon(H(\mathbf{T}))

    Where :

    - :math:`\mathbf{C_1}`: Metric cost matrix in the source space
    - :math:`\mathbf{C_2}`: Metric cost matrix in the target space
    - :math:`\mathbf{p}`: distribution in the source space
    - :math:`\mathbf{q}`: distribution in the target space
    - `L`: loss function to account for the misfit between the similarity matrices
    - `H`: entropy

    Parameters
    ----------
    C1 : array-like, shape (ns, ns)
        Metric cost matrix in the source space
    C2 : array-like, shape (nt, nt)
        Metric cost matrix in the target space
    p :  array-like, shape (ns,)
        Distribution in the source space
    q :  array-like, shape (nt,)
        Distribution in the target space
    loss_fun : str
        Loss function used for the solver either 'square_loss' or 'kl_loss'
    epsilon : float
        Regularization term >0
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
    gw_dist : float
        Gromov-Wasserstein distance

    References
    ----------
    .. [12] Gabriel Peyré, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.

    """
    gw, logv = entropic_gromov_wasserstein(
        C1, C2, p, q, loss_fun, epsilon, max_iter, tol, verbose, log=True)

    logv['T'] = gw

    if log:
        return logv['gw_dist'], logv
    else:
        return logv['gw_dist']


def entropic_gromov_barycenters(N, Cs, ps, p, lambdas, loss_fun, epsilon,
                                max_iter=1000, tol=1e-9, verbose=False, log=False, init_C=None, random_state=None):
    r"""
    Returns the gromov-wasserstein barycenters of `S` measured similarity matrices :math:`(\mathbf{C}_s)_{1 \leq s \leq S}`

    The function solves the following optimization problem:

    .. math::

        \mathbf{C} = \mathop{\arg \min}_{\mathbf{C}\in \mathbb{R}^{N \times N}} \quad \sum_s \lambda_s \mathrm{GW}(\mathbf{C}, \mathbf{C}_s, \mathbf{p}, \mathbf{p}_s)

    Where :

    - :math:`\mathbf{C}_s`: metric cost matrix
    - :math:`\mathbf{p}_s`: distribution

    Parameters
    ----------
    N : int
        Size of the targeted barycenter
    Cs : list of S array-like of shape (ns,ns)
        Metric cost matrices
    ps : list of S array-like of shape (ns,)
        Sample weights in the `S` spaces
    p : array-like, shape(N,)
        Weights in the targeted barycenter
    lambdas : list of float
        List of the `S` spaces' weights.
    loss_fun : callable
        Tensor-matrix multiplication function based on specific loss function.
    update : callable
        function(:math:`\mathbf{p}`, lambdas, :math:`\mathbf{T}`, :math:`\mathbf{Cs}`) that updates
        :math:`\mathbf{C}` according to a specific Kernel with the `S` :math:`\mathbf{T}_s` couplings
        calculated at each iteration
    epsilon : float
        Regularization term >0
    max_iter : int, optional
        Max number of iterations
    tol : float, optional
        Stop threshold on error (>0)
    verbose : bool, optional
        Print information along iterations.
    log : bool, optional
        Record log if True.
    init_C : bool | array-like, shape (N, N)
        Random initial value for the :math:`\mathbf{C}` matrix provided by user.
    random_state : int or RandomState instance, optional
        Fix the seed for reproducibility

    Returns
    -------
    C : array-like, shape (`N`, `N`)
        Similarity matrix in the barycenter space (permutated arbitrarily)

    References
    ----------
    .. [12] Gabriel Peyré, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.
    """
    Cs = list_to_array(*Cs)
    ps = list_to_array(*ps)
    p = list_to_array(p)
    nx = get_backend(*Cs, *ps, p)

    S = len(Cs)

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

    while (err > tol) and (cpt < max_iter):
        Cprev = C

        T = [entropic_gromov_wasserstein(Cs[s], C, ps[s], p, loss_fun, epsilon,
                                         max_iter, 1e-4, verbose, log) for s in range(S)]
        if loss_fun == 'square_loss':
            C = update_square_loss(p, lambdas, T, Cs)

        elif loss_fun == 'kl_loss':
            C = update_kl_loss(p, lambdas, T, Cs)

        if cpt % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            err = nx.norm(C - Cprev)
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


def gromov_barycenters(N, Cs, ps, p, lambdas, loss_fun,
                       max_iter=1000, tol=1e-9, verbose=False, log=False, init_C=None, random_state=None):
    r"""
    Returns the gromov-wasserstein barycenters of `S` measured similarity matrices :math:`(\mathbf{C}_s)_{1 \leq s \leq S}`

    The function solves the following optimization problem with block coordinate descent:

    .. math::

        \mathbf{C} = \mathop{\arg \min}_{\mathbf{C}\in \mathbb{R}^{N \times N}} \quad \sum_s \lambda_s \mathrm{GW}(\mathbf{C}, \mathbf{C}_s, \mathbf{p}, \mathbf{p}_s)

    Where :

    - :math:`\mathbf{C}_s`: metric cost matrix
    - :math:`\mathbf{p}_s`: distribution

    Parameters
    ----------
    N : int
        Size of the targeted barycenter
    Cs : list of S array-like of shape (ns, ns)
        Metric cost matrices
    ps : list of S array-like of shape (ns,)
        Sample weights in the `S` spaces
    p : array-like, shape (N,)
        Weights in the targeted barycenter
    lambdas : list of float
        List of the `S` spaces' weights
    loss_fun : callable
        tensor-matrix multiplication function based on specific loss function
    update : callable
        function(:math:`\mathbf{p}`, lambdas, :math:`\mathbf{T}`, :math:`\mathbf{Cs}`) that updates
        :math:`\mathbf{C}` according to a specific Kernel with the `S` :math:`\mathbf{T}_s` couplings
        calculated at each iteration
    max_iter : int, optional
        Max number of iterations
    tol : float, optional
        Stop threshold on error (>0).
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

    References
    ----------
    .. [12] Gabriel Peyré, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.

    """
    Cs = list_to_array(*Cs)
    ps = list_to_array(*ps)
    p = list_to_array(p)
    nx = get_backend(*Cs, *ps, p)

    S = len(Cs)

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

    while(err > tol and cpt < max_iter):
        Cprev = C

        T = [gromov_wasserstein(Cs[s], C, ps[s], p, loss_fun,
                                numItermax=max_iter, stopThr=1e-5, verbose=verbose, log=log) for s in range(S)]
        if loss_fun == 'square_loss':
            C = update_square_loss(p, lambdas, T, Cs)

        elif loss_fun == 'kl_loss':
            C = update_kl_loss(p, lambdas, T, Cs)

        if cpt % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            err = nx.norm(C - Cprev)
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


def fgw_barycenters(N, Ys, Cs, ps, lambdas, alpha, fixed_structure=False, fixed_features=False,
                    p=None, loss_fun='square_loss', max_iter=100, tol=1e-9,
                    verbose=False, log=False, init_C=None, init_X=None, random_state=None):
    r"""Compute the fgw barycenter as presented eq (5) in :ref:`[24] <references-fgw-barycenters>`

    Parameters
    ----------
    N : int
        Desired number of samples of the target barycenter
    Ys: list of array-like, each element has shape (ns,d)
        Features of all samples
    Cs : list of array-like, each element has shape (ns,ns)
        Structure matrices of all samples
    ps : list of array-like, each element has shape (ns,)
        Masses of all samples.
    lambdas : list of float
        List of the `S` spaces' weights
    alpha : float
        Alpha parameter for the fgw distance
    fixed_structure : bool
        Whether to fix the structure of the barycenter during the updates
    fixed_features : bool
        Whether to fix the feature of the barycenter during the updates
    loss_fun : str
        Loss function used for the solver either 'square_loss' or 'kl_loss'
    max_iter : int, optional
        Max number of iterations
    tol : float, optional
        Stop threshold on error (>0).
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
        - :math:`(\mathbf{M}_s)_s`: all distance matrices between the feature of the barycenter and the other features :math:`(dist(\mathbf{X}, \mathbf{Y}_s))_s` shape (`N`, `ns`)


    .. _references-fgw-barycenters:
    References
    ----------
    .. [24] Vayer Titouan, Chapel Laetitia, Flamary Rémi, Tavenard Romain
        and Courty Nicolas
        "Optimal Transport for structured data with application on graphs"
        International Conference on Machine Learning (ICML). 2019.
    """
    Cs = list_to_array(*Cs)
    ps = list_to_array(*ps)
    Ys = list_to_array(*Ys)
    p = list_to_array(p)
    nx = get_backend(*Cs, *Ys, *ps)

    S = len(Cs)
    d = Ys[0].shape[1]  # dimension on the node features
    if p is None:
        p = nx.ones(N, type_as=Cs[0]) / N

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

    T = [nx.outer(p, q) for q in ps]

    Ms = [dist(X, Ys[s]) for s in range(len(Ys))]

    cpt = 0
    err_feature = 1
    err_structure = 1

    if log:
        log_ = {}
        log_['err_feature'] = []
        log_['err_structure'] = []
        log_['Ts_iter'] = []

    while((err_feature > tol or err_structure > tol) and cpt < max_iter):
        Cprev = C
        Xprev = X

        if not fixed_features:
            Ys_temp = [y.T for y in Ys]
            X = update_feature_matrix(lambdas, Ys_temp, T, p).T

        Ms = [dist(X, Ys[s]) for s in range(len(Ys))]

        if not fixed_structure:
            if loss_fun == 'square_loss':
                T_temp = [t.T for t in T]
                C = update_structure_matrix(p, lambdas, T_temp, Cs)

        T = [fused_gromov_wasserstein(Ms[s], C, Cs[s], p, ps[s], loss_fun, alpha,
                                      numItermax=max_iter, stopThr=1e-5, verbose=verbose) for s in range(S)]

        # T is N,ns
        err_feature = nx.norm(X - nx.reshape(Xprev, (N, d)))
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

    if log:
        log_['T'] = T  # from target to Ys
        log_['p'] = p
        log_['Ms'] = Ms

    if log:
        return X, C, log_
    else:
        return X, C


def update_structure_matrix(p, lambdas, T, Cs):
    r"""Updates :math:`\mathbf{C}` according to the L2 Loss kernel with the `S` :math:`\mathbf{T}_s` couplings.

    It is calculated at each iteration

    Parameters
    ----------
    p : array-like, shape (N,)
        Masses in the targeted barycenter.
    lambdas : list of float
        List of the `S` spaces' weights.
    T : list of S array-like of shape (ns, N)
        The `S` :math:`\mathbf{T}_s` couplings calculated at each iteration.
    Cs : list of S array-like, shape (ns, ns)
        Metric cost matrices.

    Returns
    -------
    C : array-like, shape (`nt`, `nt`)
        Updated :math:`\mathbf{C}` matrix.
    """
    p = list_to_array(p)
    T = list_to_array(*T)
    Cs = list_to_array(*Cs)
    nx = get_backend(*Cs, *T, p)

    tmpsum = sum([
        lambdas[s] * nx.dot(
            nx.dot(T[s].T, Cs[s]),
            T[s]
        ) for s in range(len(T))
    ])
    ppt = nx.outer(p, p)
    return tmpsum / ppt


def update_feature_matrix(lambdas, Ys, Ts, p):
    r"""Updates the feature with respect to the `S` :math:`\mathbf{T}_s` couplings.


    See "Solving the barycenter problem with Block Coordinate Descent (BCD)"
    in :ref:`[24] <references-update-feature-matrix>` calculated at each iteration

    Parameters
    ----------
    p : array-like, shape (N,)
        masses in the targeted barycenter
    lambdas : list of float
        List of the `S` spaces' weights
    Ts : list of S array-like, shape (ns,N)
        The `S` :math:`\mathbf{T}_s` couplings calculated at each iteration
    Ys : list of S array-like, shape (d,ns)
        The features.

    Returns
    -------
    X : array-like, shape (`d`, `N`)


    .. _references-update-feature-matrix:
    References
    ----------
    .. [24] Vayer Titouan, Chapel Laetitia, Flamary Rémi, Tavenard Romain and Courty Nicolas
        "Optimal Transport for structured data with application on graphs"
        International Conference on Machine Learning (ICML). 2019.
    """
    p = list_to_array(p)
    Ts = list_to_array(*Ts)
    Ys = list_to_array(*Ys)
    nx = get_backend(*Ys, *Ts, p)

    p = 1. / p
    tmpsum = sum([
        lambdas[s] * nx.dot(Ys[s], Ts[s].T) * p[None, :]
        for s in range(len(Ts))
    ])
    return tmpsum
