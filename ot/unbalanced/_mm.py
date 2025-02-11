# -*- coding: utf-8 -*-
"""
Regularized Unbalanced OT solvers
"""

# Author: Hicham Janati <hicham.janati@inria.fr>
#         Laetitia Chapel <laetitia.chapel@univ-ubs.fr>
#         Quang Huy Tran <quang-huy.tran@univ-ubs.fr>
#
# License: MIT License

from ..backend import get_backend
from ..utils import list_to_array, get_parameter_pair


def mm_unbalanced(
    a,
    b,
    M,
    reg_m,
    c=None,
    reg=0,
    div="kl",
    G0=None,
    numItermax=1000,
    stopThr=1e-15,
    verbose=False,
    log=False,
):
    r"""
    Solve the unbalanced optimal transport problem and return the OT plan.
    The function solves the following optimization problem:

    .. math::
        W = \arg \min_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F +
        \mathrm{reg_{m1}} \cdot \mathrm{div}(\gamma \mathbf{1}, \mathbf{a}) +
        \mathrm{reg_{m2}} \cdot \mathrm{div}(\gamma^T \mathbf{1}, \mathbf{b}) +
        \mathrm{reg} \cdot \mathrm{div}(\gamma, \mathbf{c})

        s.t.
             \gamma \geq 0

    where:

    - :math:`\mathbf{M}` is the (`dim_a`, `dim_b`) metric cost matrix
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are source and target
      unbalanced distributions
    - :math:`\mathbf{c}` is a reference distribution for the regularization
    - div is a divergence, either Kullback-Leibler or half-squared :math:`\ell_2` divergence

    The algorithm used for solving the problem is a maximization-
    minimization algorithm as proposed in :ref:`[41] <references-regpath>`

    Parameters
    ----------
    a : array-like (dim_a,)
        Unnormalized histogram of dimension `dim_a`
        If `a` is an empty list or array ([]),
        then `a` is set to uniform distribution.
    b : array-like (dim_b,)
        Unnormalized histogram of dimension `dim_b`
        If `b` is an empty list or array ([]),
        then `b` is set to uniform distribution.
    M : array-like (dim_a, dim_b)
        loss matrix
    reg_m: float or indexable object of length 1 or 2
        Marginal relaxation term: nonnegative but cannot be infinity.
        If :math:`\mathrm{reg_{m}}` is a scalar or an indexable object of length 1,
        then the same :math:`\mathrm{reg_{m}}` is applied to both marginal relaxations.
        If :math:`\mathrm{reg_{m}}` is an array,
        it must have the same backend as input arrays `(a, b, M)`.
    reg : float, optional (default = 0)
        Regularization term >= 0.
        By default, solve the unregularized problem
    c : array-like (dim_a, dim_b), optional (default = None)
        Reference measure for the regularization.
        If None, then use :math:`\mathbf{c} = \mathbf{a} \mathbf{b}^T`.
    div: string, optional
        Divergence to quantify the difference between the marginals.
        Can take two values: 'kl' (Kullback-Leibler) or 'l2' (half-squared)
    G0: array-like (dim_a, dim_b)
        Initialization of the transport matrix
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshold on error (> 0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    Returns
    -------
    gamma : (dim_a, dim_b) array-like
            Optimal transportation matrix for the given parameters
    log : dict
            log dictionary returned only if `log` is `True`

    Examples
    --------
    >>> import ot
    >>> import numpy as np
    >>> a=[.5, .5]
    >>> b=[.5, .5]
    >>> M=[[1., 36.],[9., 4.]]
    >>> np.round(ot.unbalanced.mm_unbalanced(a, b, M, 5, div='kl'), 2)
    array([[0.45, 0.  ],
           [0.  , 0.34]])
    >>> np.round(ot.unbalanced.mm_unbalanced(a, b, M, 5, div='l2'), 2)
    array([[0.4, 0. ],
           [0. , 0.1]])


    .. _references-regpath:
    References
    ----------
    .. [41] Chapel, L., Flamary, R., Wu, H., Févotte, C., and Gasso, G. (2021).
        Unbalanced optimal transport through non-negative penalized
        linear regression. NeurIPS.

    See Also
    --------
    ot.lp.emd : Unregularized OT
    ot.unbalanced.sinkhorn_unbalanced : Entropic regularized OT
    """

    M, a, b = list_to_array(M, a, b)
    nx = get_backend(M, a, b)

    dim_a, dim_b = M.shape

    if len(a) == 0:
        a = nx.ones(dim_a, type_as=M) / dim_a
    if len(b) == 0:
        b = nx.ones(dim_b, type_as=M) / dim_b

    G = a[:, None] * b[None, :] if G0 is None else G0
    if reg > 0:  # regularized case
        c = a[:, None] * b[None, :] if c is None else c
    else:  # unregularized case
        c = 0

    reg_m1, reg_m2 = get_parameter_pair(reg_m)

    if log:
        log = {"err": [], "G": []}

    div = div.lower()
    if div == "kl":
        sum_r = reg + reg_m1 + reg_m2
        r1, r2, r = reg_m1 / sum_r, reg_m2 / sum_r, reg / sum_r
        K = (a[:, None] ** r1) * (b[None, :] ** r2) * (c**r) * nx.exp(-M / sum_r)
    elif div == "l2":
        K = (reg_m1 * a[:, None]) + (reg_m2 * b[None, :]) + reg * c - M
        K = nx.maximum(K, nx.zeros((dim_a, dim_b), type_as=M))
    else:
        raise ValueError("Unknown div = {}. Must be either 'kl' or 'l2'".format(div))

    for i in range(numItermax):
        Gprev = G

        if div == "kl":
            Gd = (nx.sum(G, 1, keepdims=True) ** r1) * (
                nx.sum(G, 0, keepdims=True) ** r2
            ) + 1e-16
            G = K * G ** (r1 + r2) / Gd
        elif div == "l2":
            Gd = (
                reg_m1 * nx.sum(G, 1, keepdims=True)
                + reg_m2 * nx.sum(G, 0, keepdims=True)
                + reg * G
                + 1e-16
            )
            G = K * G / Gd

        err = nx.sqrt(nx.sum((G - Gprev) ** 2))
        if log:
            log["err"].append(err)
            log["G"].append(G)
        if verbose:
            print("{:5d}|{:8e}|".format(i, err))
        if err < stopThr:
            break

    if log:
        linear_cost = nx.sum(G * M)
        log["cost"] = linear_cost

        m1, m2 = nx.sum(G, 1), nx.sum(G, 0)
        if div == "kl":
            cost = (
                linear_cost
                + reg_m1 * nx.kl_div(m1, a, mass=True)
                + reg_m2 * nx.kl_div(m2, b, mass=True)
            )
            if reg > 0:
                cost = cost + reg * nx.kl_div(G, c, mass=True)
        else:
            cost = (
                linear_cost
                + reg_m1 * 0.5 * nx.sum((m1 - a) ** 2)
                + reg_m2 * 0.5 * nx.sum((m2 - b) ** 2)
            )
            if reg > 0:
                cost = cost + reg * 0.5 * nx.sum((G - c) ** 2)

        log["total_cost"] = cost

        return G, log
    else:
        return G


def mm_unbalanced2(
    a,
    b,
    M,
    reg_m,
    c=None,
    reg=0,
    div="kl",
    G0=None,
    returnCost="linear",
    numItermax=1000,
    stopThr=1e-15,
    verbose=False,
    log=False,
):
    r"""
    Solve the unbalanced optimal transport problem and return the OT cost.
    The function solves the following optimization problem:

    .. math::
        \min_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F +
        \mathrm{reg_{m1}} \cdot \mathrm{div}(\gamma \mathbf{1}, \mathbf{a}) +
        \mathrm{reg_{m2}} \cdot \mathrm{div}(\gamma^T \mathbf{1}, \mathbf{b}) +
        \mathrm{reg} \cdot \mathrm{div}(\gamma, \mathbf{c})

        s.t.
             \gamma \geq 0

    where:

    - :math:`\mathbf{M}` is the (`dim_a`, `dim_b`) metric cost matrix
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are source and target
      unbalanced distributions
    - :math:`\mathbf{c}` is a reference distribution for the regularization
    - :math:`\mathrm{div}` is a divergence, either Kullback-Leibler or half-squared :math:`\ell_2` divergence

    The algorithm used for solving the problem is a maximization-
    minimization algorithm as proposed in :ref:`[41] <references-regpath>`

    Parameters
    ----------
    a : array-like (dim_a,)
        Unnormalized histogram of dimension `dim_a`
        If `a` is an empty list or array ([]),
        then `a` is set to uniform distribution.
    b : array-like (dim_b,)
        Unnormalized histogram of dimension `dim_b`
        If `b` is an empty list or array ([]),
        then `b` is set to uniform distribution.
    M : array-like (dim_a, dim_b)
        loss matrix
    reg_m: float or indexable object of length 1 or 2
        Marginal relaxation term: nonnegative but cannot be infinity.
        If :math:`\mathrm{reg_{m}}` is a scalar or an indexable object of length 1,
        then the same :math:`\mathrm{reg_{m}}` is applied to both marginal relaxations.
        If :math:`\mathrm{reg_{m}}` is an array,
        it must have the same backend as input arrays `(a, b, M)`.
    reg : float, optional (default = 0)
        Entropy regularization term >= 0.
        By default, solve the unregularized problem
    c : array-like (dim_a, dim_b), optional (default = None)
        Reference measure for the regularization.
        If None, then use :math:`\mathbf{c} = mathbf{a} mathbf{b}^T`.
    div: string, optional
        Divergence to quantify the difference between the marginals.
        Can take two values: 'kl' (Kullback-Leibler) or 'l2' (half-squared)
    G0: array-like (dim_a, dim_b)
        Initialization of the transport matrix
    returnCost: string, optional (default = "linear")
        If `returnCost` = "linear", then return the linear part of the unbalanced OT loss.
        If `returnCost` = "total", then return the total unbalanced OT loss.
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshold on error (> 0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True

    Returns
    -------
    ot_cost : array-like
        the OT cost between :math:`\mathbf{a}` and :math:`\mathbf{b}`
    log : dict
        log dictionary returned only if `log` is `True`

    Examples
    --------
    >>> import ot
    >>> import numpy as np
    >>> a=[.5, .5]
    >>> b=[.5, .5]
    >>> M=[[1., 36.],[9., 4.]]
    >>> np.round(ot.unbalanced.mm_unbalanced2(a, b, M, 5, div='l2'), 2)
    0.8
    >>> np.round(ot.unbalanced.mm_unbalanced2(a, b, M, 5, div='kl'), 2)
    1.79

    References
    ----------
    .. [41] Chapel, L., Flamary, R., Wu, H., Févotte, C., and Gasso, G. (2021).
        Unbalanced optimal transport through non-negative penalized
        linear regression. NeurIPS.
    See Also
    --------
    ot.lp.emd2 : Unregularized OT loss
    ot.unbalanced.sinkhorn_unbalanced2 : Entropic regularized OT loss
    """

    _, log_mm = mm_unbalanced(
        a,
        b,
        M,
        reg_m,
        c=c,
        reg=reg,
        div=div,
        G0=G0,
        numItermax=numItermax,
        stopThr=stopThr,
        verbose=verbose,
        log=True,
    )

    if returnCost == "linear":
        cost = log_mm["cost"]
    elif returnCost == "total":
        cost = log_mm["total_cost"]
    else:
        raise ValueError("Unknown returnCost = {}".format(returnCost))

    if log:
        return cost, log_mm
    else:
        return cost
