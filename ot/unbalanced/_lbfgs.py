# -*- coding: utf-8 -*-
"""
Regularized Unbalanced OT solvers
"""

# Author: Hicham Janati <hicham.janati@inria.fr>
#         Laetitia Chapel <laetitia.chapel@univ-ubs.fr>
#         Quang Huy Tran <quang-huy.tran@univ-ubs.fr>
#
# License: MIT License

import warnings
import numpy as np
from scipy.optimize import minimize, Bounds

from ..backend import get_backend
from ..utils import list_to_array, get_parameter_pair


def _get_loss_unbalanced(a, b, c, M, reg, reg_m1, reg_m2, reg_div='kl', regm_div='kl'):
    """
    return the loss function (scipy.optimize compatible) for regularized
    unbalanced OT
    """

    m, n = M.shape

    def kl(p, q):
        return np.sum(p * np.log(p / q + 1e-16)) - np.sum(p) + np.sum(q)

    def reg_l2(G):
        return np.sum((G - c)**2) / 2

    def grad_l2(G):
        return G - c

    def reg_kl(G):
        return kl(G, c)

    def grad_kl(G):
        return np.log(G / c + 1e-16)

    if reg_div == 'kl':
        reg_fun = reg_kl
        grad_reg_fun = grad_kl
    elif isinstance(reg_div, tuple):
        reg_fun = reg_div[0]
        grad_reg_fun = reg_div[1]
    else:
        reg_fun = reg_l2
        grad_reg_fun = grad_l2

    def marg_l2(G):
        return reg_m1 * 0.5 * np.sum((G.sum(1) - a)**2) + \
            reg_m2 * 0.5 * np.sum((G.sum(0) - b)**2)

    def grad_marg_l2(G):
        return reg_m1 * np.outer((G.sum(1) - a), np.ones(n)) + \
            reg_m2 * np.outer(np.ones(m), (G.sum(0) - b))

    def marg_kl(G):
        return reg_m1 * kl(G.sum(1), a) + reg_m2 * kl(G.sum(0), b)

    def grad_marg_kl(G):
        return reg_m1 * np.outer(np.log(G.sum(1) / a + 1e-16), np.ones(n)) + \
            reg_m2 * np.outer(np.ones(m), np.log(G.sum(0) / b + 1e-16))

    def marg_tv(G):
        return reg_m1 * np.sum(np.abs(G.sum(1) - a)) + \
            reg_m2 * np.sum(np.abs(G.sum(0) - b))

    def grad_marg_tv(G):
        return reg_m1 * np.outer(np.sign(G.sum(1) - a), np.ones(n)) + \
            reg_m2 * np.outer(np.ones(m), np.sign(G.sum(0) - b))

    if regm_div == 'kl':
        regm_fun = marg_kl
        grad_regm_fun = grad_marg_kl
    elif regm_div == 'tv':
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


def lbfgsb_unbalanced(a, b, M, reg, reg_m, c=None, reg_div='kl', regm_div='kl', G0=None, numItermax=1000,
                      stopThr=1e-15, method='L-BFGS-B', verbose=False, log=False):
    r"""
    Solve the unbalanced optimal transport problem and return the OT plan using L-BFGS-B.
    The function solves the following optimization problem:

    .. math::
        W = \min_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F +
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

    The algorithm used for solving the problem is a L-BFGS-B from scipy.optimize

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
        If None, then use `\mathbf{c} = \mathbf{a} \mathbf{b}^T`.
    reg_m: float or indexable object of length 1 or 2
        Marginal relaxation term >= 0, but cannot be infinity.
        If reg_m is a scalar or an indexable object of length 1,
        then the same reg_m is applied to both marginal relaxations.
        If reg_m is an array, it must be a Numpy array.
    reg_div: string, optional
        Divergence used for regularization.
        Can take two values: 'kl' (Kullback-Leibler) or 'l2' (quadratic) or a tuple
        of two calable functions returning the reg term and its derivative.
        Note that the callable functions should be able to handle numpy arrays
        and not tesors from the backend
    regm_div: string, optional
        Divergence to quantify the difference between the marginals.
        Can take three values: 'kl' (Kullback-Leibler) or 'l2' (quadratic) or 'tv' (Total Variation)
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

    M, a, b = list_to_array(M, a, b)
    nx = get_backend(M, a, b)
    M0 = M

    # convert to numpy
    a, b, M = nx.to_numpy(a, b, M)
    G0 = np.zeros(M.shape) if G0 is None else nx.to_numpy(G0)
    if reg > 0:  # regularized case
        c = a[:, None] * b[None, :] if c is None else nx.to_numpy(c)

    # wrap the callable function to handle numpy arrays
    if isinstance(reg_div, tuple):
        f0, df0 = reg_div
        try:
            f0(G0)
            df0(G0)
        except BaseException:
            warnings.warn("The callable functions should be able to handle numpy arrays, wrapper ar added to handle this which comes with overhead")

            def f(x):
                return nx.to_numpy(f0(nx.from_numpy(x, type_as=M0)))

            def df(x):
                return nx.to_numpy(df0(nx.from_numpy(x, type_as=M0)))

            reg_div = (f, df)

    reg_m1, reg_m2 = get_parameter_pair(reg_m)
    _func = _get_loss_unbalanced(a, b, c, M, reg, reg_m1, reg_m2, reg_div, regm_div)

    res = minimize(_func, G0.ravel(), method=method, jac=True, bounds=Bounds(0, np.inf),
                   tol=stopThr, options=dict(maxiter=numItermax, disp=verbose))

    G = nx.from_numpy(res.x.reshape(M.shape), type_as=M0)

    if log:
        log = {'cost': nx.sum(G * M), 'res': res}
        log['total_cost'] = nx.from_numpy(res.fun, type_as=M0)
        return G, log
    else:
        return G


def lbfgsb_unbalanced2(a, b, M, reg, reg_m, c=None, reg_div='kl', regm_div='kl',
                       G0=None, returnCost="linear", numItermax=1000, stopThr=1e-15,
                       method='L-BFGS-B', verbose=False, log=False):
    r"""
    Solve the unbalanced optimal transport problem and return the OT plan using L-BFGS-B.
    The function solves the following optimization problem:

    .. math::
        W = \min_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F +
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

    The algorithm used for solving the problem is a L-BFGS-B from scipy.optimize

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
        If None, then use `\mathbf{c} = \mathbf{a} \mathbf{b}^T`.
    reg_m: float or indexable object of length 1 or 2
        Marginal relaxation term >= 0, but cannot be infinity.
        If reg_m is a scalar or an indexable object of length 1,
        then the same reg_m is applied to both marginal relaxations.
        If reg_m is an array, it must be a Numpy array.
    reg_div: string, optional
        Divergence used for regularization.
        Can take two values: 'kl' (Kullback-Leibler) or 'l2' (quadratic) or a tuple
        of two calable functions returning the reg term and its derivative.
        Note that the callable functions should be able to handle numpy arrays
        and not tesors from the backend
    regm_div: string, optional
        Divergence to quantify the difference between the marginals.
        Can take three values: 'kl' (Kullback-Leibler) or 'l2' (quadratic) or 'tv' (Total Variation)
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
    >>> np.round(ot.unbalanced.lbfgsb_unbalanced2(a, b, M, reg=0, reg_m=5, reg_div='kl', regm_div='kl'), 2)
    0.8
    >>> np.round(ot.unbalanced.lbfgsb_unbalanced2(a, b, M, reg=0, reg_m=5, reg_div='l2', regm_div='l2'), 2)
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

    _, log_lbfgs = lbfgsb_unbalanced(a=a, b=b, M=M, reg=reg, reg_m=reg_m, c=c,
                                     reg_div=reg_div, regm_div=regm_div, G0=G0,
                                     numItermax=numItermax, stopThr=stopThr,
                                     method=method, verbose=verbose, log=True)

    if returnCost == "linear":
        cost = log_lbfgs['cost']
    elif returnCost == "total":
        cost = log_lbfgs['total_cost']
    else:
        raise ValueError("Unknown returnCost = {}".format(returnCost))

    if log:
        return cost, log_lbfgs
    else:
        return cost
