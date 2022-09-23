# -*- coding: utf-8 -*-
"""
Generic solvers for regularized OT or its semi-relaxed version.
"""

# Author: Remi Flamary <remi.flamary@unice.fr>
#         Titouan Vayer <titouan.vayer@irisa.fr>
#         Cédric Vincent-Cuaz <cedric.vincent-cuaz@inria.fr>
# License: MIT License

import numpy as np
import warnings
from .lp import emd
from .bregman import sinkhorn
from .utils import list_to_array
from .backend import get_backend

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        from scipy.optimize import scalar_search_armijo
    except ImportError:
        from scipy.optimize.linesearch import scalar_search_armijo

# The corresponding scipy function does not work for matrices


def line_search_armijo(
    f, xk, pk, gfk, old_fval, args=(), c1=1e-4,
    alpha0=0.99, alpha_min=None, alpha_max=None
):
    r"""
    Armijo linesearch function that works with matrices

    Find an approximate minimum of :math:`f(x_k + \alpha \cdot p_k)` that satisfies the
    armijo conditions.

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
    old_fval : float
        loss value at :math:`x_k`
    args : tuple, optional
        arguments given to `f`
    c1 : float, optional
        :math:`c_1` const in armijo rule (>0)
    alpha0 : float, optional
        initial step (>0)
    alpha_min : float, optional
        minimum value for alpha
    alpha_max : float, optional
        maximum value for alpha

    Returns
    -------
    alpha : float
        step that satisfy armijo conditions
    fc : int
        nb of function call
    fa : float
        loss value at step alpha

    """

    xk, pk, gfk = list_to_array(xk, pk, gfk)
    nx = get_backend(xk, pk)

    if len(xk.shape) == 0:
        xk = nx.reshape(xk, (-1,))

    fc = [0]

    def phi(alpha1):
        fc[0] += 1
        return f(xk + alpha1 * pk, *args)

    if old_fval is None:
        phi0 = phi(0.)
    else:
        phi0 = old_fval

    derphi0 = nx.sum(pk * gfk)  # Quickfix for matrices
    alpha, phi1 = scalar_search_armijo(
        phi, phi0, derphi0, c1=c1, alpha0=alpha0)

    if alpha is None:
        return 0., fc[0], phi0
    else:
        if alpha_min is not None or alpha_max is not None:
            alpha = np.clip(alpha, alpha_min, alpha_max)
        return float(alpha), fc[0], phi1


def solve_gromov_linesearch(G, deltaG, Mi, f_val, C1, C2t,
                            reg=None, Gc=None, M=None, alpha_min=None, alpha_max=None, **kwargs):
    """
    Solve the linesearch in the FW iterations

    Parameters
    ----------

    G : array-like, shape(ns,nt)
        The transport map at a given iteration of the FW
    deltaG : array-like (ns,nt)
        Difference between the optimal map found by linearization in the FW algorithm and the value at a given iteration
    Mi : array-like (ns,nt)
        Cost matrix of the linearized transport problem. Corresponds to the gradient of the cost
    f_val : float
        Value of the cost at `G`
    armijo : bool, optional
        If True the steps of the line-search is found via an armijo research. Else closed form is used.
        If there is convergence issues use False.
    C1 : array-like (ns,ns), optional
        Structure matrix in the source domain.
    C2t : array-like (nt,nt), optional
        Transposed Structure matrix in the target domain.
    reg : float, optional
        Regularization parameter.
    Gc : array-like (ns,nt), optional
        Optimal map found by linearization in the FW algorithm.
    M : array-like (ns,nt), optional
        Cost matrix between the features. Only used and necessary when armijo=False
    alpha_min : float, optional
        Minimum value for alpha
    alpha_max : float, optional
        Maximum value for alpha

    Returns
    -------
    alpha : float
        The optimal step size of the FW
    fc : int
        nb of function call. Useless here
    f_val : float
        The value of the cost for the next iteration


    .. _references-solve-linesearch:
    References
    ----------
    .. [24] Vayer Titouan, Chapel Laetitia, Flamary Rémi, Tavenard Romain and Courty Nicolas
        "Optimal Transport for structured data with application on graphs"
        International Conference on Machine Learning (ICML). 2019.
    """
    G, deltaG, C1, C2t, M = list_to_array(G, deltaG, C1, C2t, M)

    if isinstance(M, int) or isinstance(M, float):
        nx = get_backend(G, deltaG, C1, C2t)
    else:
        nx = get_backend(G, deltaG, C1, C2t, M)

    dot = nx.dot(nx.dot(C1, deltaG), C2t)
    a = -2 * reg * nx.sum(dot * deltaG)
    b = nx.sum(M * deltaG) - 2 * reg * (nx.sum(dot * G) + nx.sum(nx.dot(nx.dot(C1, G), C2t) * deltaG))

    alpha = new_solve_1d_linesearch_quad(a, b)
    if alpha_min is not None or alpha_max is not None:
        alpha = np.clip(alpha, alpha_min, alpha_max)

    fc = None
    # f_val = cost(G + alpha * deltaG) > removed to reduce numerical complexity
    # as the new cost can be deduced from the line search quadratic function
    f_val += a * (alpha ** 2) + b * alpha

    return alpha, fc, f_val


def solve_semirelaxed_gromov_linesearch(
    G, deltaG, Mi, f_val, C1, C2t, ones_p, qG, qdeltaG,
    reg=None, Gc=None, M=None, alpha_min=None, alpha_max=None, **kwargs
):
    """
    Solve the linesearch in the FW iterations

    Parameters
    ----------

    G : array-like, shape(ns,nt)
        The transport map at a given iteration of the FW
    deltaG : array-like (ns,nt)
        Difference between the optimal map found by linearization in the FW algorithm and the value at a given iteration
    Mi : array-like (ns,nt)
        Cost matrix of the linearized transport problem. Corresponds to the gradient of the cost
    f_val : float
        Value of the cost at `G`
    armijo : bool, optional
        If True the steps of the line-search is found via an armijo research. Else closed form is used.
        If there is convergence issues use False.
    C1 : array-like (ns,ns)
        Structure matrix in the source domain.
    C2t : array-like (nt,nt)
        Transposed Structure matrix in the target domain.
    ones_p: array-like (ns,1)
        Array of ones of size ns
    qG: array-like (nt,1)
        Second marginal of the current transportation matrix G
    qdeltaG: array-like (nt,1)
        Difference of the second marginals of G and Gc
    reg : float, optional
        Regularization parameter.
    Gc : array-like (ns,nt), optional
        Optimal map found by linearization in the FW algorithm.
    M : array-like (ns,nt), optional
        Cost matrix between the features. Only used and necessary when armijo=False
    alpha_min : float, optional
        Minimum value for alpha
    alpha_max : float, optional
        Maximum value for alpha

    Returns
    -------
    alpha : float
        The optimal step size of the FW
    fc : int
        nb of function call. Useless here
    f_val : float
        The value of the cost for the next iteration


    .. _references-solve-linesearch:
    References
    ----------
    .. [24] Vayer Titouan, Chapel Laetitia, Flamary Rémi, Tavenard Romain and Courty Nicolas
        "Optimal Transport for structured data with application on graphs"
        International Conference on Machine Learning (ICML). 2019.
    """
    G, deltaG, C1, C2t, M = list_to_array(G, deltaG, C1, C2t, M)

    if isinstance(M, int) or isinstance(M, float):
        nx = get_backend(G, deltaG, C1, C2t)
    else:
        nx = get_backend(G, deltaG, C1, C2t, M)

    dot = nx.dot(nx.dot(C1, deltaG), C2t)
    C2t_square = C2t ** 2
    dot_qG = nx.dot(ones_p[:, None] * qG[None, :], C2t_square)
    dot_qdeltaG = nx.dot(ones_p[:, None] * qdeltaG[None, :], C2t_square)
    a = reg * nx.sum((dot_qdeltaG - 2 * dot) * deltaG)
    b = nx.sum(M * deltaG) + reg * (nx.sum((dot_qdeltaG - 2 * dot) * G) + nx.sum((dot_qG - 2 * nx.dot(nx.dot(C1, G), C2t)) * deltaG))

    alpha = new_solve_1d_linesearch_quad(a, b)
    if alpha_min is not None or alpha_max is not None:
        alpha = np.clip(alpha, alpha_min, alpha_max)

    fc = None
    # f_val = cost(G + alpha * deltaG) > removed to reduce numerical complexity
    # as the new cost can be deduced from the line search quadratic function
    f_val += a * (alpha ** 2) + b * alpha

    return alpha, fc, f_val


def solve_linesearch(
    cost, G, deltaG, Mi, f_val, armijo=True, C1=None, C2=None,
    reg=None, Gc=None, constC=None, M=None, alpha_min=None, alpha_max=None
):
    """
    Solve the linesearch in the FW iterations

    Parameters
    ----------
    cost : method
        Cost in the FW for the linesearch
    G : array-like, shape(ns,nt)
        The transport map at a given iteration of the FW
    deltaG : array-like (ns,nt)
        Difference between the optimal map found by linearization in the FW algorithm and the value at a given iteration
    Mi : array-like (ns,nt)
        Cost matrix of the linearized transport problem. Corresponds to the gradient of the cost
    f_val : float
        Value of the cost at `G`
    armijo : bool, optional
        If True the steps of the line-search is found via an armijo research. Else closed form is used.
        If there is convergence issues use False.
    C1 : array-like (ns,ns), optional
        Structure matrix in the source domain. Only used and necessary when armijo=False
    C2 : array-like (nt,nt), optional
        Structure matrix in the target domain. Only used and necessary when armijo=False
    reg : float, optional
        Regularization parameter. Only used and necessary when armijo=False
    Gc : array-like (ns,nt)
        Optimal map found by linearization in the FW algorithm. Only used and necessary when armijo=False
    constC : array-like (ns,nt)
        Constant for the gromov cost. See :ref:`[24] <references-solve-linesearch>`. Only used and necessary when armijo=False
    M : array-like (ns,nt), optional
        Cost matrix between the features. Only used and necessary when armijo=False
    alpha_min : float, optional
        Minimum value for alpha
    alpha_max : float, optional
        Maximum value for alpha

    Returns
    -------
    alpha : float
        The optimal step size of the FW
    fc : int
        nb of function call. Useless here
    f_val : float
        The value of the cost for the next iteration


    .. _references-solve-linesearch:
    References
    ----------
    .. [24] Vayer Titouan, Chapel Laetitia, Flamary Rémi, Tavenard Romain and Courty Nicolas
        "Optimal Transport for structured data with application on graphs"
        International Conference on Machine Learning (ICML). 2019.
    """
    if armijo:
        alpha, fc, f_val = line_search_armijo(
            cost, G, deltaG, Mi, f_val, alpha_min=alpha_min, alpha_max=alpha_max
        )
    else:  # requires symetric matrices
        G, deltaG, C1, C2, constC, M = list_to_array(G, deltaG, C1, C2, constC, M)
        if isinstance(M, int) or isinstance(M, float):
            nx = get_backend(G, deltaG, C1, C2, constC)
        else:
            nx = get_backend(G, deltaG, C1, C2, constC, M)

        dot = nx.dot(nx.dot(C1, deltaG), C2)
        a = -2 * reg * nx.sum(dot * deltaG)
        b = nx.sum((M + reg * constC) * deltaG) - 2 * reg * (nx.sum(dot * G) + nx.sum(nx.dot(nx.dot(C1, G), C2) * deltaG))
        c = cost(G)

        alpha = solve_1d_linesearch_quad(a, b, c)
        if alpha_min is not None or alpha_max is not None:
            alpha = np.clip(alpha, alpha_min, alpha_max)
        fc = None
        f_val = cost(G + alpha * deltaG)

    return alpha, fc, f_val


def generic_cg(a, b, M, f, df, reg1, reg2, lp_solver, line_search_solver, G0=None,
               semirelaxed=False, numItermax=200, numInnerItermax=100000, stopThr=1e-9,
               stopThr2=1e-9, verbose=False, log=False, innerlog=True, **kwargs):
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
        If set to emd will solve the general regularized OT problem using cg.
        If set to lp_semi_relaxed_OT will solve the general regularized semi-relaxed OT problem using cg.
        If set to sinkhorn will solve the general regularized OT problem using generalized cg.
    line_search_solver: function,
        Function to find the optimal set. Currently used instances are:
        solve_linesearch for the (Fused) Gromov-Wasserstein problem.
        gcg_linesearch for the line search step of the Generalized cg.
    semirelaxed: bool
        either the program corresponds to a semi-relaxed OT problem or not
    G0 :  array-like, shape (ns,nt), optional
        initial guess (default is indep joint density)
    numItermax : int, optional
        Max number of iterations
    numInnerItermax: int, optional
        Max number of iterations for lp solver
    stopThr : float, optional
        Stop threshold on the relative variation (>0)
    stopThr2 : float, optional
        Stop threshold on the absolute variation (>0)
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


    .. _references-cg:
    .. _references_gcg:
    References
    ----------

    .. [1] Ferradans, S., Papadakis, N., Peyré, G., & Aujol, J. F. (2014). Regularized discrete optimal transport. SIAM Journal on Imaging Sciences, 7(3), 1853-1882.

    .. [5] N. Courty; R. Flamary; D. Tuia; A. Rakotomamonjy, "Optimal Transport for Domain Adaptation," in IEEE Transactions on Pattern Analysis and Machine Intelligence , vol.PP, no.99, pp.1-1

    .. [7] Rakotomamonjy, A., Flamary, R., & Courty, N. (2015). Generalized conditional gradient: analysis of convergence and applications. arXiv preprint arXiv:1510.06567.

    See Also
    --------
    ot.lp.emd : Unregularized optimal ransport
    ot.bregman.sinkhorn : Entropic regularized optimal transport
    """
    a, b, M, G0 = list_to_array(a, b, M, G0)
    if isinstance(M, int) or isinstance(M, float):
        nx = get_backend(a, b)
    else:
        nx = get_backend(a, b, M)

    loop = 1

    if log:
        log = {'loss': []}

    if G0 is None:
        G = nx.outer(a, b)
    else:
        G = G0
    bdeltaG = None

    if reg2 is None:
        def cost(G, qG=None):
            return nx.sum(M * G) + reg1 * f(G, qG)
    else:
        def cost(G, qG=None):
            return nx.sum(M * G) + reg1 * f(G, qG) + reg2 * nx.sum(G * nx.log(G))
    f_val = cost(G, b)
    if log:
        log['loss'].append(f_val)

    it = 0

    if verbose:
        print('{:5s}|{:12s}|{:8s}|{:8s}'.format(
            'It.', 'Loss', 'Relative loss', 'Absolute loss') + '\n' + '-' * 48)
        print('{:5d}|{:8e}|{:8e}|{:8e}'.format(it, f_val, 0, 0))

    while loop:

        it += 1
        old_fval = f_val

        # problem linearization
        Mi = M + reg1 * df(G, b)
        if not (reg2 is None):
            Mi += reg2 * (1 + nx.log(G))
        # set M positive
        Mi += nx.min(Mi)

        # solve linear program
        if innerlog:
            Gc, innerlog_ = lp_solver(a, b, Mi, numItermax=numInnerItermax, log=innerlog, reg=reg1, **kwargs)
        else:
            Gc = lp_solver(a, b, Mi, reg=reg1, numItermax=numInnerItermax, log=innerlog, **kwargs)

        if semirelaxed:
            bdeltaG = b - Gc.sum(0)
        # line search
        deltaG = Gc - G

        alpha, fc, f_val = line_search_solver(cost=cost, G=G, deltaG=deltaG, Mi=Mi, f_val=f_val,
                                              reg=reg1, M=M, Gc=Gc, alpha_min=0., alpha_max=1., qG=b, qdeltaG=bdeltaG, **kwargs)

        G = G + alpha * deltaG
        if semirelaxed:
            b = b + alpha * bdeltaG
        # test convergence
        if it >= numItermax:
            loop = 0

        abs_delta_fval = abs(f_val - old_fval)
        relative_delta_fval = abs_delta_fval / abs(f_val)
        if relative_delta_fval < stopThr or abs_delta_fval < stopThr2:
            loop = 0

        if log:
            log['loss'].append(f_val)

        if verbose:
            if it % 20 == 0:
                print('{:5s}|{:12s}|{:8s}|{:8s}'.format(
                    'It.', 'Loss', 'Relative loss', 'Absolute loss') + '\n' + '-' * 48)
            print('{:5d}|{:8e}|{:8e}|{:8e}'.format(it, f_val, relative_delta_fval, abs_delta_fval))

    if log:
        if innerlog:
            log.update(innerlog_)
        return G, log
    else:
        return G


def cg(a, b, M, reg, f, df, G0=None, numItermax=200, numItermaxEmd=100000,
       stopThr=1e-9, stopThr2=1e-9, verbose=False, log=False, **kwargs):
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

    .. [1] Ferradans, S., Papadakis, N., Peyré, G., & Aujol, J. F. (2014). Regularized discrete optimal transport. SIAM Journal on Imaging Sciences, 7(3), 1853-1882.

    See Also
    --------
    ot.lp.emd : Unregularized optimal ransport
    ot.bregman.sinkhorn : Entropic regularized optimal transport

    """
    a, b, M, G0 = list_to_array(a, b, M, G0)
    if isinstance(M, int) or isinstance(M, float):
        nx = get_backend(a, b)
    else:
        nx = get_backend(a, b, M)

    loop = 1

    if log:
        log = {'loss': []}

    if G0 is None:
        G = nx.outer(a, b)
    else:
        G = G0

    def cost(G):
        return nx.sum(M * G) + reg * f(G)

    f_val = cost(G)
    if log:
        log['loss'].append(f_val)

    it = 0

    if verbose:
        print('{:5s}|{:12s}|{:8s}|{:8s}'.format(
            'It.', 'Loss', 'Relative loss', 'Absolute loss') + '\n' + '-' * 48)
        print('{:5d}|{:8e}|{:8e}|{:8e}'.format(it, f_val, 0, 0))

    while loop:

        it += 1
        old_fval = f_val

        # problem linearization
        Mi = M + reg * df(G)
        # set M positive
        Mi += nx.min(Mi)

        # solve linear program
        Gc, logemd = emd(a, b, Mi, numItermax=numItermaxEmd, log=True)

        deltaG = Gc - G

        # line search
        alpha, fc, f_val = solve_linesearch(
            cost, G, deltaG, Mi, f_val, reg=reg, M=M, Gc=Gc,
            alpha_min=0., alpha_max=1., **kwargs
        )

        G = G + alpha * deltaG

        # test convergence
        if it >= numItermax:
            loop = 0

        abs_delta_fval = abs(f_val - old_fval)
        relative_delta_fval = abs_delta_fval / abs(f_val)
        if relative_delta_fval < stopThr or abs_delta_fval < stopThr2:
            loop = 0

        if log:
            log['loss'].append(f_val)

        if verbose:
            if it % 20 == 0:
                print('{:5s}|{:12s}|{:8s}|{:8s}'.format(
                    'It.', 'Loss', 'Relative loss', 'Absolute loss') + '\n' + '-' * 48)
            print('{:5d}|{:8e}|{:8e}|{:8e}'.format(it, f_val, relative_delta_fval, abs_delta_fval))

    if log:
        log.update(logemd)
        return G, log
    else:
        return G


def new_gcg(a, b, M, reg1, reg2, f, df, G0=None, numItermax=10,
            numInnerItermax=200, stopThr=1e-9, stopThr2=1e-9, verbose=False, log=False, **kwargs):
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

    .. [5] N. Courty; R. Flamary; D. Tuia; A. Rakotomamonjy, "Optimal Transport for Domain Adaptation," in IEEE Transactions on Pattern Analysis and Machine Intelligence , vol.PP, no.99, pp.1-1

    .. [7] Rakotomamonjy, A., Flamary, R., & Courty, N. (2015). Generalized conditional gradient: analysis of convergence and applications. arXiv preprint arXiv:1510.06567.

    See Also
    --------
    ot.optim.cg : conditional gradient

    """

    def lp_solver(a, b, Mi, numItermax, log, **kwargs):
        return sinkhorn(a, b, Mi, reg1, numItermax=numItermax)

    def line_search_solver(cost, G, deltaG, Mi, f_val, reg, M, Gc, alpha_min, alpha_max, qG, qdeltaG, **kwargs):
        return line_search_armijo(cost, G, deltaG, Mi, f_val, alpha_min=alpha_min, alpha_max=alpha_max)

    if log:
        G, log = generic_cg(a, b, M, f, df, reg2, reg1, lp_solver, line_search_solver, G0=None,
                            semirelaxed=False, numItermax=numItermax, numInnerItermax=numInnerItermax,
                            stopThr=stopThr, stopThr2=stopThr2, verbose=verbose, log=log, innerlog=log, **kwargs)
        return G, log
    else:
        G = generic_cg(a, b, M, f, df, reg2, reg1, lp_solver, line_search_solver, G0=None,
                       semirelaxed=False, numItermax=numItermax, numInnerItermax=numInnerItermax,
                       stopThr=stopThr, stopThr2=stopThr2, verbose=verbose, log=log, innerlog=log, **kwargs)
        return G


def gcg(a, b, M, reg1, reg2, f, df, G0=None, numItermax=10,
        numInnerItermax=200, stopThr=1e-9, stopThr2=1e-9, verbose=False, log=False):
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

    .. [5] N. Courty; R. Flamary; D. Tuia; A. Rakotomamonjy, "Optimal Transport for Domain Adaptation," in IEEE Transactions on Pattern Analysis and Machine Intelligence , vol.PP, no.99, pp.1-1

    .. [7] Rakotomamonjy, A., Flamary, R., & Courty, N. (2015). Generalized conditional gradient: analysis of convergence and applications. arXiv preprint arXiv:1510.06567.

    See Also
    --------
    ot.optim.cg : conditional gradient

    """
    a, b, M, G0 = list_to_array(a, b, M, G0)
    nx = get_backend(a, b, M)

    loop = 1

    if log:
        log = {'loss': []}

    if G0 is None:
        G = nx.outer(a, b)
    else:
        G = G0

    def cost(G):
        return nx.sum(M * G) + reg1 * nx.sum(G * nx.log(G)) + reg2 * f(G)

    f_val = cost(G)
    if log:
        log['loss'].append(f_val)

    it = 0

    if verbose:
        print('{:5s}|{:12s}|{:8s}|{:8s}'.format(
            'It.', 'Loss', 'Relative loss', 'Absolute loss') + '\n' + '-' * 48)
        print('{:5d}|{:8e}|{:8e}|{:8e}'.format(it, f_val, 0, 0))

    while loop:

        it += 1
        old_fval = f_val

        # problem linearization
        Mi = M + reg2 * df(G)

        # solve linear program with Sinkhorn
        # Gc = sinkhorn_stabilized(a,b, Mi, reg1, numItermax = numInnerItermax)
        Gc = sinkhorn(a, b, Mi, reg1, numItermax=numInnerItermax)

        deltaG = Gc - G

        # line search
        dcost = Mi + reg1 * (1 + nx.log(G))  # ??
        alpha, fc, f_val = line_search_armijo(
            cost, G, deltaG, dcost, f_val, alpha_min=0., alpha_max=1.
        )

        G = G + alpha * deltaG

        # test convergence
        if it >= numItermax:
            loop = 0

        abs_delta_fval = abs(f_val - old_fval)
        relative_delta_fval = abs_delta_fval / abs(f_val)

        if relative_delta_fval < stopThr or abs_delta_fval < stopThr2:
            loop = 0

        if log:
            log['loss'].append(f_val)

        if verbose:
            if it % 20 == 0:
                print('{:5s}|{:12s}|{:8s}|{:8s}'.format(
                    'It.', 'Loss', 'Relative loss', 'Absolute loss') + '\n' + '-' * 48)
            print('{:5d}|{:8e}|{:8e}|{:8e}'.format(it, f_val, relative_delta_fval, abs_delta_fval))

    if log:
        return G, log
    else:
        return G


def solve_1d_linesearch_quad(a, b, c):
    r"""
    For any convex or non-convex 1d quadratic function `f`, solve the following problem:
    .. math::
        \mathop{\arg \min}_{0 \leq x \leq 1} \quad f(x) = ax^{2} + bx + c
    Parameters
    ----------
    a,b,c : float
        The coefficients of the quadratic function
    Returns
    -------
    x : float
        The optimal value which leads to the minimal cost
    """
    f0 = c
    df0 = b
    f1 = a + f0 + df0

    if a > 0:  # convex
        minimum = min(1, max(0, np.divide(-b, 2.0 * a)))
        return minimum
    else:  # non convex
        if f0 > f1:
            return 1
        else:
            return 0


def new_solve_1d_linesearch_quad(a, b):
    r"""
    For any convex or non-convex 1d quadratic function `f`, solve the following problem:

    .. math::

        \mathop{\arg \min}_{0 \leq x \leq 1} \quad f(x) = ax^{2} + bx + c

    Parameters
    ----------
    a,b,c : float
        The coefficients of the quadratic function

    Returns
    -------
    x : float
        The optimal value which leads to the minimal cost
    """

    if a > 0:  # convex
        minimum = min(1., max(0., np.divide(-b, 2.0 * a)))
        return minimum
    else:  # non convex
        if a + b < 0:
            return 1.
        else:
            return 0.
