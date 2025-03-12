# -*- coding: utf-8 -*-
"""
Generic solvers for regularized OT or its semi-relaxed version.
"""

# Author: Remi Flamary <remi.flamary@unice.fr>
#         Titouan Vayer <titouan.vayer@irisa.fr>
#         Cédric Vincent-Cuaz <cedvincentcuaz@gmail.com>
# License: MIT License

import numpy as np
import warnings
from .lp import emd
from .bregman import sinkhorn
from .backend import get_backend

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        from scipy.optimize._linesearch import scalar_search_armijo
    except ModuleNotFoundError:
        # scipy<1.8.0
        from scipy.optimize.linesearch import scalar_search_armijo

# The corresponding scipy function does not work for matrices


def line_search_armijo(
    f,
    xk,
    pk,
    gfk,
    old_fval,
    args=(),
    c1=1e-4,
    alpha0=0.99,
    alpha_min=0.0,
    alpha_max=None,
    nx=None,
    **kwargs,
):
    r"""
    Armijo linesearch function that works with matrices

    Find an approximate minimum of :math:`f(x_k + \alpha \cdot p_k)` that satisfies the
    armijo conditions.

    .. note:: If the loss function f returns a float (resp. a 1d array) then
        the returned alpha and fa are float (resp. 1d arrays).

    Parameters
    ----------
    f : callable
        loss function
    xk : array-like
        initial position
    pk : array-like
        descent direction
    gfk : array-like
        gradient of `f` at :math:`x_k`
    old_fval : float or 1d array
        loss value at :math:`x_k`
    args : tuple, optional
        arguments given to `f`
    c1 : float, optional
        :math:`c_1` const in armijo rule (>0)
    alpha0 : float, optional
        initial step (>0)
    alpha_min : float, default=0.
        minimum value for alpha
    alpha_max : float, optional
        maximum value for alpha
    nx : backend, optional
        If let to its default value None, a backend test will be conducted.
    Returns
    -------
    alpha : float or 1d array
        step that satisfy armijo conditions
    fc : int
        nb of function call
    fa : float or 1d array
        loss value at step alpha

    """
    if nx is None:
        xk0, pk0 = xk, pk
        nx = get_backend(xk0, pk0)
    else:
        xk0, pk0 = xk, pk

    if len(xk.shape) == 0:
        xk = nx.reshape(xk, (-1,))

    xk = nx.to_numpy(xk)
    pk = nx.to_numpy(pk)
    gfk = nx.to_numpy(gfk)

    fc = [0]

    def phi(alpha1):
        # it's necessary to check boundary condition here for the coefficient
        # as the callback could be evaluated for negative value of alpha by
        # `scalar_search_armijo` function here:
        #
        # https://github.com/scipy/scipy/blob/11509c4a98edded6c59423ac44ca1b7f28fba1fd/scipy/optimize/linesearch.py#L686
        #
        # see more details https://github.com/PythonOT/POT/issues/502
        alpha1 = np.clip(alpha1, alpha_min, alpha_max)
        # The callable function operates on nx backend
        fc[0] += 1
        alpha10 = nx.from_numpy(alpha1)
        fval = f(xk0 + alpha10 * pk0, *args)
        if isinstance(fval, float):
            # prevent bug from nx.to_numpy that can look for .cpu or .gpu
            return fval
        else:
            return nx.to_numpy(fval)

    if old_fval is None:
        phi0 = phi(0.0)
    elif isinstance(old_fval, float):
        # prevent bug from nx.to_numpy that can look for .cpu or .gpu
        phi0 = old_fval
    else:
        phi0 = nx.to_numpy(old_fval)

    derphi0 = np.sum(pk * gfk)  # Quickfix for matrices
    alpha, phi1 = scalar_search_armijo(
        phi, phi0, derphi0, c1=c1, alpha0=alpha0, amin=alpha_min
    )

    if alpha is None:
        return 0.0, fc[0], nx.from_numpy(phi0, type_as=xk0)
    else:
        alpha = np.clip(alpha, alpha_min, alpha_max)
        return (
            nx.from_numpy(alpha, type_as=xk0),
            fc[0],
            nx.from_numpy(phi1, type_as=xk0),
        )


def generic_conditional_gradient(
    a,
    b,
    M,
    f,
    df,
    reg1,
    reg2,
    lp_solver,
    line_search,
    G0=None,
    numItermax=200,
    stopThr=1e-9,
    stopThr2=1e-9,
    verbose=False,
    log=False,
    nx=None,
    **kwargs,
):
    r"""
    Solve the general regularized OT problem or its semi-relaxed version with
    conditional gradient or generalized conditional gradient depending on the
    provided linear program solver.

        The function solves the following optimization problem if set as a conditional gradient:

    .. math::
        \gamma = \mathop{\arg \min}_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F +
        \mathrm{reg_1} \cdot f(\gamma)

        s.t. \ \gamma \mathbf{1} &= \mathbf{a}

             \gamma^T \mathbf{1} &= \mathbf{b} (optional constraint)

             \gamma &\geq 0

    where :

    - :math:`\mathbf{M}` is the (`ns`, `nt`) metric cost matrix
    - :math:`f` is the regularization term (and `df` is its gradient)
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are source and target weights (sum to 1)

    The algorithm used for solving the problem is conditional gradient as discussed in :ref:`[1] <references-cg>`

        The function solves the following optimization problem if set a generalized conditional gradient:

    .. math::
        \gamma = \mathop{\arg \min}_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F +
        \mathrm{reg_1}\cdot f(\gamma) + \mathrm{reg_2}\cdot\Omega(\gamma)

        s.t. \ \gamma \mathbf{1} &= \mathbf{a}

             \gamma^T \mathbf{1} &= \mathbf{b}

             \gamma &\geq 0

    where :

    - :math:`\Omega` is the entropic regularization term :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`

    The algorithm used for solving the problem is the generalized conditional gradient as discussed in :ref:`[5, 7] <references-gcg>`

    Parameters
    ----------
    a : array-like, shape (ns,)
        samples weights in the source domain
    b : array-like, shape (nt,)
        samples weights in the target domain
    M : array-like, shape (ns, nt)
        loss matrix
    f : function
        Regularization function taking a transportation matrix as argument
    df: function
        Gradient of the regularization function taking a transportation matrix as argument
    reg1 : float
        Regularization term >0
    reg2 : float,
        Entropic Regularization term >0. Ignored if set to None.
    lp_solver: function,
        linear program solver for direction finding of the (generalized) conditional gradient.
        This function must take the form `lp_solver(a, b, Mi, **kwargs)` with p:
        `a` and `b` are sample weights in both domains; `Mi` is the gradient of
        the regularized objective; optimal arguments via kwargs.
        It must output an admissible transport plan.

        For instance, for the general regularized OT problem with conditional gradient :ref:`[1] <references-cg>`:

            def lp_solver(a, b, M, **kwargs):
                return ot.emd(a, b, M)

        or with the generalized conditional gradient instead :ref:`[5, 7] <references-gcg>`:

            def lp_solver(a, b, Mi, **kwargs):
                return ot.sinkhorn(a, b, Mi)

    line_search: function,
        Function to find the optimal step. This function must take the form
        `line_search(cost, G, deltaG, Mi, cost_G, df_G, **kwargs)` with: `cost`
        the cost function, `G` the transport plan, `deltaG` the conditional
        gradient direction given by lp_solver, `Mi` the gradient of regularized
        objective, `cost_G` the cost at G, `df_G` the gradient of the regularizer
        at G. Two types of outputs are supported:

        Instances such as `ot.optim.line_search_armijo` (generic solver),
        `ot.gromov.solve_gromov_linesearch` (FGW problems),
        `solve_semirelaxed_gromov_linesearch` (srFGW problems) and
        `gcg_linesearch` (generalized cg), output : the line-search step alpha,
        the number of iterations used in the solver if applicable and the loss
        value at step alpha. These can be called e.g as:

            def line_search(cost, G, deltaG, Mi, cost_G, df_G, **kwargs):
                return ot.optim.line_search_armijo(cost, G, deltaG, Mi, cost_G, **kwargs)

        Instances such as `ot.gromov.solve_partial_gromov_linesearch` for partial
        (F)GW problems add as finale output, the next step gradient reading as
        a convex combination of previously computed gradients, taking advantage of the regularizer
        quadratic form.

    G0 :  array-like, shape (ns,nt), optional
        initial guess (default is indep joint density)
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshold on the relative variation (>0)
    stopThr2 : float, optional
        Stop threshold on the absolute variation (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    nx : backend, optional
        If let to its default value None, the backend will be deduced from other inputs.
    **kwargs : dict
             Parameters for linesearch

    Returns
    -------
    gamma : (ns x nt) ndarray
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters


    .. _references-cg:
    .. _references_gcg:
    References
    ----------
    .. [1] Ferradans, S., Papadakis, N., Peyré, G., & Aujol, J. F. (2014).
        Regularized discrete optimal transport. SIAM Journal on Imaging Sciences, 7(3), 1853-1882.

    .. [5] N. Courty; R. Flamary; D. Tuia; A. Rakotomamonjy, "Optimal Transport for Domain Adaptation,"
        in IEEE Transactions on Pattern Analysis and Machine Intelligence , vol.PP, no.99, pp.1-1

    .. [7] Rakotomamonjy, A., Flamary, R., & Courty, N. (2015). Generalized conditional gradient:
        analysis of convergence and applications. arXiv preprint arXiv:1510.06567.

    See Also
    --------
    ot.lp.emd : Unregularized optimal transport
    ot.bregman.sinkhorn : Entropic regularized optimal transport
    """

    if nx is None:
        if isinstance(M, int) or isinstance(M, float):
            nx = get_backend(a, b)
        else:
            nx = get_backend(a, b, M)

    loop = 1

    if log:
        log = {"loss": []}

    if G0 is None:
        G = nx.outer(a, b)
    else:
        # to not change G0 in place.
        G = nx.copy(G0)

    if reg2 is None:

        def cost(G):
            return nx.sum(M * G) + reg1 * f(G)
    else:

        def cost(G):
            return nx.sum(M * G) + reg1 * f(G) + reg2 * nx.sum(G * nx.log(G))

    cost_G = cost(G)
    if log:
        log["loss"].append(cost_G)

    df_G = None
    it = 0

    if verbose:
        print(
            "{:5s}|{:12s}|{:8s}|{:8s}".format(
                "It.", "Loss", "Relative loss", "Absolute loss"
            )
            + "\n"
            + "-" * 48
        )
        print("{:5d}|{:8e}|{:8e}|{:8e}".format(it, cost_G, 0, 0))

    while loop:
        it += 1
        old_cost_G = cost_G
        # problem linearization
        if df_G is None:
            df_G = df(G)
        Mi = M + reg1 * df_G

        if reg2 is not None:
            Mi = Mi + reg2 * (1 + nx.log(G))

        # solve linear program
        Gc, innerlog_ = lp_solver(a, b, Mi, **kwargs)
        # line search
        deltaG = Gc - G

        res_line_search = line_search(cost, G, deltaG, Mi, cost_G, df_G, **kwargs)
        if len(res_line_search) == 3:
            # the line-search does not allow to update the gradient
            alpha, fc, cost_G = res_line_search
            df_G = None
        else:
            # the line-search allows to update the gradient directly
            # e.g. while using quadratic losses as the gromov-wasserstein loss
            alpha, fc, cost_G, df_G = res_line_search

        G = G + alpha * deltaG

        # test convergence
        if it >= numItermax:
            loop = 0

        abs_delta_cost_G = abs(cost_G - old_cost_G)
        relative_delta_cost_G = (
            abs_delta_cost_G / abs(cost_G) if cost_G != 0.0 else np.nan
        )
        if relative_delta_cost_G < stopThr or abs_delta_cost_G < stopThr2:
            loop = 0

        if log:
            log["loss"].append(cost_G)

        if verbose:
            if it % 20 == 0:
                print(
                    "{:5s}|{:12s}|{:8s}|{:8s}".format(
                        "It.", "Loss", "Relative loss", "Absolute loss"
                    )
                    + "\n"
                    + "-" * 48
                )
            print(
                "{:5d}|{:8e}|{:8e}|{:8e}".format(
                    it, cost_G, relative_delta_cost_G, abs_delta_cost_G
                )
            )

    if log:
        log.update(innerlog_)
        return G, log
    else:
        return G


def cg(
    a,
    b,
    M,
    reg,
    f,
    df,
    G0=None,
    line_search=None,
    numItermax=200,
    numItermaxEmd=100000,
    stopThr=1e-9,
    stopThr2=1e-9,
    verbose=False,
    log=False,
    nx=None,
    **kwargs,
):
    r"""
    Solve the general regularized OT problem with conditional gradient

        The function solves the following optimization problem:

    .. math::
        \gamma = \mathop{\arg \min}_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F +
        \mathrm{reg} \cdot f(\gamma)

        s.t. \ \gamma \mathbf{1} &= \mathbf{a}

             \gamma^T \mathbf{1} &= \mathbf{b}

             \gamma &\geq 0

    where :

    - :math:`\mathbf{M}` is the (`ns`, `nt`) metric cost matrix
    - :math:`f` is the regularization term (and `df` is its gradient)
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are source and target weights (sum to 1)

    The algorithm used for solving the problem is conditional gradient as discussed in :ref:`[1] <references-cg>`


    Parameters
    ----------
    a : array-like, shape (ns,)
        samples weights in the source domain
    b : array-like, shape (nt,)
        samples in the target domain
    M : array-like, shape (ns, nt)
        loss matrix
    reg : float
        Regularization term >0
    G0 :  array-like, shape (ns,nt), optional
        initial guess (default is indep joint density)
    line_search: function,
        Function to find the optimal step.
        Default is None and calls a wrapper to line_search_armijo.
    numItermax : int, optional
        Max number of iterations
    numItermaxEmd : int, optional
        Max number of iterations for emd
    stopThr : float, optional
        Stop threshold on the relative variation (>0)
    stopThr2 : float, optional
        Stop threshold on the absolute variation (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    nx : backend, optional
        If let to its default value None, the backend will be deduced from other inputs.
    **kwargs : dict
             Parameters for linesearch

    Returns
    -------
    gamma : (ns x nt) ndarray
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters


    .. _references-cg:
    References
    ----------
    .. [1] Ferradans, S., Papadakis, N., Peyré, G., & Aujol, J. F. (2014).
        Regularized discrete optimal transport. SIAM Journal on Imaging Sciences, 7(3), 1853-1882.

    See Also
    --------
    ot.lp.emd : Unregularized optimal transport
    ot.bregman.sinkhorn : Entropic regularized optimal transport

    """
    if nx is None:
        if isinstance(M, int) or isinstance(M, float):
            nx = get_backend(a, b)
        else:
            nx = get_backend(a, b, M)

    if line_search is None:

        def line_search(cost, G, deltaG, Mi, cost_G, df_G, **kwargs):
            return line_search_armijo(cost, G, deltaG, Mi, cost_G, nx=nx, **kwargs)

    def lp_solver(a, b, M, **kwargs):
        return emd(a, b, M, numItermaxEmd, log=True)

    return generic_conditional_gradient(
        a,
        b,
        M,
        f,
        df,
        reg,
        None,
        lp_solver,
        line_search,
        G0=G0,
        numItermax=numItermax,
        stopThr=stopThr,
        stopThr2=stopThr2,
        verbose=verbose,
        log=log,
        nx=nx,
        **kwargs,
    )


def semirelaxed_cg(
    a,
    b,
    M,
    reg,
    f,
    df,
    G0=None,
    line_search=None,
    numItermax=200,
    stopThr=1e-9,
    stopThr2=1e-9,
    verbose=False,
    log=False,
    nx=None,
    **kwargs,
):
    r"""
    Solve the general regularized and semi-relaxed OT problem with conditional gradient

        The function solves the following optimization problem:

    .. math::
        \gamma = \mathop{\arg \min}_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F +
        \mathrm{reg} \cdot f(\gamma)

        s.t. \ \gamma \mathbf{1} &= \mathbf{a}

             \gamma &\geq 0

    where :

    - :math:`\mathbf{M}` is the (`ns`, `nt`) metric cost matrix
    - :math:`f` is the regularization term (and `df` is its gradient)
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are source and target weights (sum to 1)

    The algorithm used for solving the problem is conditional gradient as discussed in :ref:`[1] <references-cg>`


    Parameters
    ----------
    a : array-like, shape (ns,)
        samples weights in the source domain
    b : array-like, shape (nt,)
        currently estimated samples weights in the target domain
    M : array-like, shape (ns, nt)
        loss matrix
    reg : float
        Regularization term >0
    G0 :  array-like, shape (ns,nt), optional
        initial guess (default is indep joint density)
    line_search: function,
        Function to find the optimal step.
        Default is None and calls a wrapper to line_search_armijo.
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshold on the relative variation (>0)
    stopThr2 : float, optional
        Stop threshold on the absolute variation (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    nx : backend, optional
        If let to its default value None, the backend will be deduced from other inputs.
    **kwargs : dict
             Parameters for linesearch

    Returns
    -------
    gamma : (ns x nt) ndarray
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters


    .. _references-cg:
    References
    ----------
    .. [48] Cédric Vincent-Cuaz, Rémi Flamary, Marco Corneli, Titouan Vayer, Nicolas Courty.
        "Semi-relaxed Gromov-Wasserstein divergence and applications on graphs"
        International Conference on Learning Representations (ICLR), 2021.

    """
    if nx is None:
        if isinstance(M, int) or isinstance(M, float):
            nx = get_backend(a, b)
        else:
            nx = get_backend(a, b, M)

    if line_search is None:

        def line_search(cost, G, deltaG, Mi, cost_G, df_G, **kwargs):
            return line_search_armijo(cost, G, deltaG, Mi, cost_G, nx=nx, **kwargs)

    def lp_solver(a, b, Mi, **kwargs):
        # get minimum by rows as binary mask
        min_ = nx.reshape(nx.min(Mi, axis=1), (-1, 1))
        # instead of exact elements equal to min_ we consider a small margin (1e-15)
        # for float precision issues. Then the mass is split uniformly
        # between these elements.
        Gc = nx.ones(1, type_as=a) * (Mi <= min_ + 1e-15)
        Gc *= nx.reshape((a / nx.sum(Gc, axis=1)), (-1, 1))
        # return by default an empty inner_log
        return Gc, {}

    return generic_conditional_gradient(
        a,
        b,
        M,
        f,
        df,
        reg,
        None,
        lp_solver,
        line_search,
        G0=G0,
        numItermax=numItermax,
        stopThr=stopThr,
        stopThr2=stopThr2,
        verbose=verbose,
        log=log,
        nx=nx,
        **kwargs,
    )


def partial_cg(
    a,
    b,
    a_extended,
    b_extended,
    M,
    reg,
    f,
    df,
    G0=None,
    line_search=line_search_armijo,
    numItermax=200,
    stopThr=1e-9,
    stopThr2=1e-9,
    warn=True,
    verbose=False,
    log=False,
    **kwargs,
):
    r"""
    Solve the general regularized partial OT problem with conditional gradient

        The function solves the following optimization problem:

    .. math::
        \gamma = \mathop{\arg \min}_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F +
        \mathrm{reg} \cdot f(\gamma)

        s.t. \ \gamma \mathbf{1} &= \mathbf{a}

             \gamma \mathbf{1} &= \mathbf{b}

             \mathbf{1}^T \gamma^T \mathbf{1} = m &\leq \min\{\|\mathbf{p}\|_1, \|\mathbf{q}\|_1\}

             \gamma &\geq 0

    where :

    - :math:`\mathbf{M}` is the (`ns`, `nt`) metric cost matrix
    - :math:`f` is the regularization term (and `df` is its gradient)
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are source and target weights
    - `m` is the amount of mass to be transported

    The algorithm used for solving the problem is conditional gradient as discussed in :ref:`[1] <references-cg>`

    Parameters
    ----------
    a : array-like, shape (ns,)
        samples weights in the source domain
    b : array-like, shape (nt,)
        currently estimated samples weights in the target domain
    a_extended : array-like, shape (ns + nb_dummies,)
        samples weights in the source domain with added dummy nodes
    b_extended : array-like, shape (nt + nb_dummies,)
        currently estimated samples weights in the target domain with added dummy nodes
    M : array-like, shape (ns, nt)
        loss matrix
    reg : float
        Regularization term >0
    G0 :  array-like, shape (ns,nt), optional
        initial guess (default is indep joint density)
    line_search: function,
        Function to find the optimal step.
        Default is the armijo line-search.
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshold on the relative variation (>0)
    stopThr2 : float, optional
        Stop threshold on the absolute variation (>0)
    warn: bool, optional.
        Whether to raise a warning when EMD did not converge.
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    **kwargs : dict
             Parameters for linesearch

    Returns
    -------
    gamma : (ns x nt) ndarray
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters


    .. _references-partial-cg:
    References
    ----------
    ..  [29] Chapel, L., Alaya, M., Gasso, G. (2020). "Partial Optimal
        Transport with Applications on Positive-Unlabeled Learning".
        NeurIPS.

    """
    n, m = a.shape[0], b.shape[0]
    n_extended, m_extended = a_extended.shape[0], b_extended.shape[0]
    nb_dummies = n_extended - n

    def lp_solver(a, b, Mi, **kwargs):
        # add dummy nodes to Mi
        Mi_extended = np.zeros((n_extended, m_extended), dtype=Mi.dtype)
        Mi_extended[:n, :m] = Mi
        Mi_extended[-nb_dummies:, -nb_dummies:] = np.max(M) * 1e2

        G_extended, log_ = emd(
            a_extended, b_extended, Mi_extended, numItermax, log=True
        )
        Gc = G_extended[:n, :m]

        if warn:
            if log_["warning"] is not None:
                raise ValueError(
                    "Error in the EMD resolution: try to increase the"
                    " number of dummy points"
                )

        return Gc, log_

    return generic_conditional_gradient(
        a,
        b,
        M,
        f,
        df,
        reg,
        None,
        lp_solver,
        line_search,
        G0=G0,
        numItermax=numItermax,
        stopThr=stopThr,
        stopThr2=stopThr2,
        verbose=verbose,
        log=log,
        **kwargs,
    )


def gcg(
    a,
    b,
    M,
    reg1,
    reg2,
    f,
    df,
    G0=None,
    numItermax=10,
    numInnerItermax=200,
    stopThr=1e-9,
    stopThr2=1e-9,
    verbose=False,
    log=False,
    **kwargs,
):
    r"""
    Solve the general regularized OT problem with the generalized conditional gradient

        The function solves the following optimization problem:

    .. math::
        \gamma = \mathop{\arg \min}_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F +
        \mathrm{reg_1}\cdot\Omega(\gamma) + \mathrm{reg_2}\cdot f(\gamma)

        s.t. \ \gamma \mathbf{1} &= \mathbf{a}

             \gamma^T \mathbf{1} &= \mathbf{b}

             \gamma &\geq 0

    where :

    - :math:`\mathbf{M}` is the (`ns`, `nt`) metric cost matrix
    - :math:`\Omega` is the entropic regularization term :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - :math:`f` is the regularization term (and `df` is its gradient)
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are source and target weights (sum to 1)

    The algorithm used for solving the problem is the generalized conditional gradient as discussed in :ref:`[5, 7] <references-gcg>`


    Parameters
    ----------
    a : array-like, shape (ns,)
        samples weights in the source domain
    b : array-like, (nt,)
        samples in the target domain
    M : array-like, shape (ns, nt)
        loss matrix
    reg1 : float
        Entropic Regularization term >0
    reg2 : float
        Second Regularization term >0
    G0 : array-like, shape (ns, nt), optional
        initial guess (default is indep joint density)
    numItermax : int, optional
        Max number of iterations
    numInnerItermax : int, optional
        Max number of iterations of Sinkhorn
    stopThr : float, optional
        Stop threshold on the relative variation (>0)
    stopThr2 : float, optional
        Stop threshold on the absolute variation (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True

    Returns
    -------
    gamma : ndarray, shape (ns, nt)
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters


    .. _references-gcg:
    References
    ----------
    .. [5] N. Courty; R. Flamary; D. Tuia; A. Rakotomamonjy, "Optimal Transport for Domain Adaptation,"
        in IEEE Transactions on Pattern Analysis and Machine Intelligence , vol.PP, no.99, pp.1-1

    .. [7] Rakotomamonjy, A., Flamary, R., & Courty, N. (2015). Generalized conditional gradient:
        analysis of convergence and applications. arXiv preprint arXiv:1510.06567.

    See Also
    --------
    ot.optim.cg : conditional gradient

    """

    def lp_solver(a, b, Mi, **kwargs):
        return sinkhorn(a, b, Mi, reg1, numItermax=numInnerItermax, log=True, **kwargs)

    def line_search(cost, G, deltaG, Mi, cost_G, df_G, **kwargs):
        return line_search_armijo(cost, G, deltaG, Mi, cost_G, **kwargs)

    return generic_conditional_gradient(
        a,
        b,
        M,
        f,
        df,
        reg2,
        reg1,
        lp_solver,
        line_search,
        G0=G0,
        numItermax=numItermax,
        stopThr=stopThr,
        stopThr2=stopThr2,
        verbose=verbose,
        log=log,
        **kwargs,
    )


def solve_1d_linesearch_quad(a, b):
    r"""
    For any convex or non-convex 1d quadratic function `f`, solve the following problem:

    .. math::

        \mathop{\arg \min}_{0 \leq x \leq 1} \quad f(x) = ax^{2} + bx + c

    Parameters
    ----------
    a,b : float or tensors (1,)
        The coefficients of the quadratic function

    Returns
    -------
    x : float
        The optimal value which leads to the minimal cost
    """
    if a > 0:  # convex
        minimum = min(1.0, max(0.0, -b / (2.0 * a)))
        return minimum
    else:  # non convex
        if a + b < 0:
            return 1.0
        else:
            return 0.0
