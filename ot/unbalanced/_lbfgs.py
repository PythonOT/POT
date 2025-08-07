# -*- coding: utf-8 -*-
"""
Regularized Unbalanced OT solvers
"""

# Author: Hicham Janati <hicham.janati@inria.fr>
#         Laetitia Chapel <laetitia.chapel@univ-ubs.fr>
#         Quang Huy Tran <quang-huy.tran@univ-ubs.fr>
#
# License: MIT License

import numpy as np
from scipy.optimize import minimize, Bounds

from ..backend import get_backend
from ..utils import list_to_array, get_parameter_pair, fun_to_numpy


def _get_loss_unbalanced(a, b, c, M, reg, reg_m1, reg_m2, reg_div="kl", regm_div="kl"):
    r"""
    Return loss function for the L-BFGS-B solver

    .. note:: This function will be fed into scipy.optimize, so all input arrays must be Numpy arrays.

    Parameters
    ----------
    a : array-like (dim_a,)
        Unnormalized histogram of dimension `dim_a`
    b : array-like (dim_b,)
        Unnormalized histogram of dimension `dim_b`
    M : array-like (dim_a, dim_b)
        loss matrix
    reg: float
        regularization term >=0
    c : array-like (dim_a, dim_b), optional (default = None)
        Reference measure for the regularization.
        If None, then use :math:`\mathbf{c} = \mathbf{a} \mathbf{b}^T`.
    reg_m1: float
        Marginal relaxation term with respect to the first marginal:
        nonnegative (including 0) but cannot be infinity.
    reg_m2: float
        Marginal relaxation term with respect to the second marginal:
        nonnegative (including 0) but cannot be infinity.
    reg_div: string, optional
        Divergence used for regularization.
        Can take three values: 'entropy' (negative entropy), or
        'kl' (Kullback-Leibler) or 'l2' (half-squared) or a tuple
        of two callable functions returning the reg term and its derivative.
        Note that the callable functions should be able to handle Numpy arrays
        and not tensors from the backend
    regm_div: string, optional
        Divergence to quantify the difference between the marginals.
        Can take three values: 'kl' (Kullback-Leibler) or 'l2' (half-squared) or 'tv' (Total Variation)

    Returns
    -------
    Loss function (scipy.optimize compatible) for regularized unbalanced OT
    """

    m, n = M.shape
    nx_numpy = get_backend(M, a, b)

    def reg_l2(G):
        return np.sum((G - c) ** 2) / 2

    def grad_l2(G):
        return G - c

    def reg_kl(G):
        return nx_numpy.kl_div(G, c, mass=True)

    def grad_kl(G):
        return np.log(G / c + 1e-16)

    def reg_entropy(G):
        return np.sum(G * np.log(G + 1e-16)) - np.sum(G)

    def grad_entropy(G):
        return np.log(G + 1e-16)

    if reg_div == "kl":
        reg_fun = reg_kl
        grad_reg_fun = grad_kl
    elif reg_div == "entropy":
        reg_fun = reg_entropy
        grad_reg_fun = grad_entropy
    elif isinstance(reg_div, tuple):
        reg_fun = reg_div[0]
        grad_reg_fun = reg_div[1]
    else:
        reg_fun = reg_l2
        grad_reg_fun = grad_l2

    def marg_l2(G):
        return reg_m1 * 0.5 * np.sum((G.sum(1) - a) ** 2) + reg_m2 * 0.5 * np.sum(
            (G.sum(0) - b) ** 2
        )

    def grad_marg_l2(G):
        return reg_m1 * np.outer((G.sum(1) - a), np.ones(n)) + reg_m2 * np.outer(
            np.ones(m), (G.sum(0) - b)
        )

    def marg_kl(G):
        return reg_m1 * nx_numpy.kl_div(
            G.sum(1), a, mass=True
        ) + reg_m2 * nx_numpy.kl_div(G.sum(0), b, mass=True)

    def grad_marg_kl(G):
        return reg_m1 * np.outer(
            np.log(G.sum(1) / a + 1e-16), np.ones(n)
        ) + reg_m2 * np.outer(np.ones(m), np.log(G.sum(0) / b + 1e-16))

    def marg_tv(G):
        return reg_m1 * np.sum(np.abs(G.sum(1) - a)) + reg_m2 * np.sum(
            np.abs(G.sum(0) - b)
        )

    def grad_marg_tv(G):
        return reg_m1 * np.outer(np.sign(G.sum(1) - a), np.ones(n)) + reg_m2 * np.outer(
            np.ones(m), np.sign(G.sum(0) - b)
        )

    if regm_div == "kl":
        regm_fun = marg_kl
        grad_regm_fun = grad_marg_kl
    elif regm_div == "tv":
        regm_fun = marg_tv
        grad_regm_fun = grad_marg_tv
    else:
        regm_fun = marg_l2
        grad_regm_fun = grad_marg_l2

    def _func(G):
        G = G.reshape((m, n))

        # compute loss
        val = np.sum(G * M) + regm_fun(G)
        if reg > 0:
            val = val + reg * reg_fun(G)
        # compute gradient
        grad = M + grad_regm_fun(G)
        if reg > 0:
            grad = grad + reg * grad_reg_fun(G)

        return val, grad.ravel()

    return _func


def lbfgsb_unbalanced(
    a,
    b,
    M,
    reg,
    reg_m,
    c=None,
    reg_div="kl",
    regm_div="kl",
    G0=None,
    numItermax=1000,
    stopThr=1e-15,
    method="L-BFGS-B",
    verbose=False,
    log=False,
):
    r"""
    Solve the unbalanced optimal transport problem and return the OT plan using L-BFGS-B algorithm.
    The function solves the following optimization problem:

    .. math::
        W = \arg \min_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F +
        \mathrm{reg} \mathrm{div}(\gamma, \mathbf{c}) +
        \mathrm{reg_{m1}} \cdot \mathrm{div_m}(\gamma \mathbf{1}, \mathbf{a}) +
        \mathrm{reg_{m2}} \cdot \mathrm{div_m}(\gamma^T \mathbf{1}, \mathbf{b})

        s.t.
             \gamma \geq 0

    where:

    - :math:`\mathbf{M}` is the (`dim_a`, `dim_b`) metric cost matrix
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are source and target unbalanced distributions
    - :math:`\mathbf{c}` is a reference distribution for the regularization
    - :math:`\mathrm{div_m}` is a divergence, either Kullback-Leibler divergence,
    or half-squared :math:`\ell_2` divergence, or Total variation
    - :math:`\mathrm{div}` is a divergence, either Kullback-Leibler divergence,
    or half-squared :math:`\ell_2` divergence

    .. note:: This function is backend-compatible and will work on arrays
        from all compatible backends. First, it converts all arrays into Numpy arrays,
        then uses the L-BFGS-B algorithm from scipy.optimize to solve the optimization problem.

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
    reg: float
        regularization term >=0
    reg_m: float or indexable object of length 1 or 2
        Marginal relaxation term: nonnegative (including 0) but cannot be infinity.
        If :math:`\mathrm{reg_{m}}` is a scalar or an indexable object of length 1,
        then the same :math:`\mathrm{reg_{m}}` is applied to both marginal relaxations.
        If :math:`\mathrm{reg_{m}}` is an array, it must be a Numpy array.
    c : array-like (dim_a, dim_b), optional (default = None)
        Reference measure for the regularization.
        If None, then use :math:`\mathbf{c} = \mathbf{a} \mathbf{b}^T`.
    reg_div: string or pair of callable functions, optional (default = 'kl')
        Divergence used for regularization.
        Can take three values: 'entropy' (negative entropy), or
        'kl' (Kullback-Leibler) or 'l2' (half-squared) or a tuple
        of two callable functions returning the reg term and its derivative.
        Note that the callable functions should be able to handle Numpy arrays
        and not tensors from the backend, otherwise functions will be converted to Numpy
        leading to a computational overhead.
    regm_div: string, optional (default = 'kl')
        Divergence to quantify the difference between the marginals.
        Can take three values: 'kl' (Kullback-Leibler) or 'l2' (half-squared) or 'tv' (Total Variation)
    G0: array-like (dim_a, dim_b), optional (default = None)
        Initialization of the transport matrix. None corresponds to uniform product.
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
    >>> np.round(ot.unbalanced.lbfgsb_unbalanced(a, b, M, reg=0, reg_m=5, reg_div='kl', regm_div='kl'), 2)
    array([[0.45, 0.  ],
           [0.  , 0.34]])
    >>> np.round(ot.unbalanced.lbfgsb_unbalanced(a, b, M, reg=0, reg_m=5, reg_div='l2', regm_div='l2'), 2)
    array([[0.4, 0. ],
           [0. , 0.1]])

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

    # test settings
    regm_div = regm_div.lower()
    if regm_div not in ["kl", "l2", "tv"]:
        raise ValueError(
            "Unknown regm_div = {}. Must be either 'kl', 'l2' or 'tv'".format(regm_div)
        )

    if isinstance(reg_div, str):
        reg_div = reg_div.lower()
        if reg_div not in ["entropy", "kl", "l2"]:
            raise ValueError(
                "Unknown reg_div = {}. Must be either 'entropy', 'kl' or 'l2', or a tuple".format(
                    reg_div
                )
            )

    # convert all inputs to numpy arrays
    reg_m1, reg_m2 = get_parameter_pair(reg_m)

    M, a, b = list_to_array(M, a, b)
    nx = get_backend(M, a, b, G0)
    M0 = M

    dim_a, dim_b = M.shape

    if len(a) == 0:
        a = nx.ones(dim_a, type_as=M) / dim_a
    if len(b) == 0:
        b = nx.ones(dim_b, type_as=M) / dim_b

    # convert to numpy
    if nx.__name__ == "numpy":  # remaining parameters which can be arrays
        reg_m1, reg_m2, reg = nx.to_numpy(reg_m1, reg_m2, reg)
    else:
        a, b, M, reg_m1, reg_m2, reg = nx.to_numpy(a, b, M, reg_m1, reg_m2, reg)

    G0 = a[:, None] * b[None, :] if G0 is None else nx.to_numpy(G0)
    c = a[:, None] * b[None, :] if c is None else nx.to_numpy(c)

    # potentially convert the callable function to handle numpy arrays
    if isinstance(reg_div, tuple):
        f0, df0 = reg_div
        f = fun_to_numpy(f0, G0, nx, warn=True)
        df = fun_to_numpy(df0, G0, nx, warn=True)

        reg_div = (f, df)

    _func = _get_loss_unbalanced(a, b, c, M, reg, reg_m1, reg_m2, reg_div, regm_div)

    res = minimize(
        _func,
        G0.ravel(),
        method=method,
        jac=True,
        bounds=Bounds(0, np.inf),
        tol=stopThr,
        options=dict(maxiter=numItermax, disp=verbose),
    )

    G = nx.from_numpy(res.x.reshape(M.shape), type_as=M0)

    if log:
        log = {"cost": nx.sum(G * M), "res": res}
        log["total_cost"] = nx.from_numpy(res.fun, type_as=M0)
        return G, log
    else:
        return G


def lbfgsb_unbalanced2(
    a,
    b,
    M,
    reg,
    reg_m,
    c=None,
    reg_div="kl",
    regm_div="kl",
    G0=None,
    returnCost="linear",
    numItermax=1000,
    stopThr=1e-15,
    method="L-BFGS-B",
    verbose=False,
    log=False,
):
    r"""
    Solve the unbalanced optimal transport problem and return the OT cost using L-BFGS-B.
    The function solves the following optimization problem:

    .. math::
        \min_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F +
        \mathrm{reg} \mathrm{div}(\gamma, \mathbf{c}) +
        \mathrm{reg_{m1}} \cdot \mathrm{div_m}(\gamma \mathbf{1}, \mathbf{a}) +
        \mathrm{reg_{m2}} \cdot \mathrm{div_m}(\gamma^T \mathbf{1}, \mathbf{b})

        s.t.
             \gamma \geq 0

    where:

    - :math:`\mathbf{M}` is the (`dim_a`, `dim_b`) metric cost matrix
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are source and target unbalanced distributions
    - :math:`\mathbf{c}` is a reference distribution for the regularization
    - :math:`\mathrm{div_m}` is a divergence, either Kullback-Leibler divergence,
    or half-squared :math:`\ell_2` divergence, or Total variation
    - :math:`\mathrm{div}` is a divergence, either Kullback-Leibler divergence,
    or half-squared :math:`\ell_2` divergence

    .. note:: This function is backend-compatible and will work on arrays
        from all compatible backends. First, it converts all arrays into Numpy arrays,
        then uses the L-BFGS-B algorithm from scipy.optimize to solve the optimization problem.

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
    reg: float
        regularization term >=0
    reg_m: float or indexable object of length 1 or 2
        Marginal relaxation term: nonnegative (including 0) but cannot be infinity.
        If :math:`\mathrm{reg_{m}}` is a scalar or an indexable object of length 1,
        then the same :math:`\mathrm{reg_{m}}` is applied to both marginal relaxations.
        If :math:`\mathrm{reg_{m}}` is an array, it must be a Numpy array.
    c : array-like (dim_a, dim_b), optional (default = None)
        Reference measure for the regularization.
        If None, then use :math:`\mathbf{c} = \mathbf{a} \mathbf{b}^T`.
    reg_div: string or pair of callable functions, optional (default = 'kl')
        Divergence used for regularization.
        Can take three values: 'entropy' (negative entropy), or
        'kl' (Kullback-Leibler) or 'l2' (half-squared) or a tuple
        of two callable functions returning the reg term and its derivative.
        Note that the callable functions should be able to handle Numpy arrays
        and not tensors from the backend, otherwise functions will be converted to Numpy
        leading to a computational overhead.
    regm_div: string, optional (default = 'kl')
        Divergence to quantify the difference between the marginals.
        Can take three values: 'kl' (Kullback-Leibler) or 'l2' (half-squared) or 'tv' (Total Variation)
    G0: array-like (dim_a, dim_b), optional (default = None)
        Initialization of the transport matrix. None corresponds to uniform product.
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
    >>> np.round(ot.unbalanced.lbfgsb_unbalanced2(a, b, M, reg=0, reg_m=5, reg_div='kl', regm_div='kl'), 2)
    1.79
    >>> np.round(ot.unbalanced.lbfgsb_unbalanced2(a, b, M, reg=0, reg_m=5, reg_div='l2', regm_div='l2'), 2)
    0.8

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

    _, log_lbfgs = lbfgsb_unbalanced(
        a=a,
        b=b,
        M=M,
        reg=reg,
        reg_m=reg_m,
        c=c,
        reg_div=reg_div,
        regm_div=regm_div,
        G0=G0,
        numItermax=numItermax,
        stopThr=stopThr,
        method=method,
        verbose=verbose,
        log=True,
    )

    if returnCost == "linear":
        cost = log_lbfgs["cost"]
    elif returnCost == "total":
        cost = log_lbfgs["total_cost"]
    else:
        raise ValueError("Unknown returnCost = {}".format(returnCost))

    if log:
        return cost, log_lbfgs
    else:
        return cost
