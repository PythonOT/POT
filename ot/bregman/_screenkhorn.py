# -*- coding: utf-8 -*-
"""
Screening Sinkhorn Algorithms for Regularized Optimal Transport
"""

# Author: Remi Flamary <remi.flamary@unice.fr>
#         Mokhtar Z. Alaya <mokhtarzahdi.alaya@gmail.com>
#
# License: MIT License

import warnings

import numpy as np
from scipy.optimize import fmin_l_bfgs_b

from ..utils import list_to_array
from ..backend import get_backend


def screenkhorn(a, b, M, reg, ns_budget=None, nt_budget=None, uniform=False,
                restricted=True, maxiter=10000, maxfun=10000, pgtol=1e-09,
                verbose=False, log=False):
    r"""
    Screening Sinkhorn Algorithm for Regularized Optimal Transport

    The function solves an approximate dual of Sinkhorn divergence :ref:`[2]
    <references-screenkhorn>` which is written as the following optimization problem:

    .. math::

        (\mathbf{u}, \mathbf{v}) = \mathop{\arg \min}_{\mathbf{u}, \mathbf{v}} \quad
        \mathbf{1}_{ns}^T \mathbf{B}(\mathbf{u}, \mathbf{v}) \mathbf{1}_{nt} -
        \langle \kappa \mathbf{u}, \mathbf{a} \rangle -
        \langle \frac{1}{\kappa} \mathbf{v}, \mathbf{b} \rangle

    where:

    .. math::

        \mathbf{B}(\mathbf{u}, \mathbf{v}) = \mathrm{diag}(e^\mathbf{u}) \mathbf{K} \mathrm{diag}(e^\mathbf{v}) \text{, with } \mathbf{K} = e^{-\mathbf{M} / \mathrm{reg}} \text{ and}

    .. math::

        s.t. \ e^{u_i} &\geq \epsilon / \kappa, \forall i \in \{1, \ldots, ns\}

             e^{v_j} &\geq \epsilon \kappa, \forall j \in \{1, \ldots, nt\}

    The parameters `kappa` and `epsilon` are determined w.r.t the couple number
    budget of points (`ns_budget`, `nt_budget`), see Equation (5)
    in :ref:`[26] <references-screenkhorn>`


    Parameters
    ----------
    a: array-like, shape=(ns,)
        samples weights in the source domain
    b: array-like, shape=(nt,)
        samples weights in the target domain
    M: array-like, shape=(ns, nt)
        Cost matrix
    reg: `float`
        Level of the entropy regularisation
    ns_budget: `int`, default=None
        Number budget of points to be kept in the source domain.
        If it is None then 50% of the source sample points will be kept
    nt_budget: `int`, default=None
        Number budget of points to be kept in the target domain.
        If it is None then 50% of the target sample points will be kept
    uniform: `bool`, default=False
        If `True`, the source and target distribution are supposed to be uniform,
        i.e., :math:`a_i = 1 / ns` and :math:`b_j = 1 / nt`
    restricted : `bool`, default=True
         If `True`, a warm-start initialization for the  L-BFGS-B solver
         using a restricted Sinkhorn algorithm with at most 5 iterations
    maxiter: `int`, default=10000
      Maximum number of iterations in LBFGS solver
    maxfun: `int`, default=10000
      Maximum number of function evaluations in LBFGS solver
    pgtol: `float`, default=1e-09
      Final objective function accuracy in LBFGS solver
    verbose: `bool`, default=False
        If `True`, display informations about the cardinals of the active sets
        and the parameters kappa and epsilon


    .. admonition:: Dependency

        To gain more efficiency, :py:func:`ot.bregman.screenkhorn` needs to call the "Bottleneck"
        package (https://pypi.org/project/Bottleneck/) in the screening pre-processing step.

        If Bottleneck isn't installed, the following error message appears:

        "Bottleneck module doesn't exist. Install it from https://pypi.org/project/Bottleneck/"


    Returns
    -------
    gamma : array-like, shape=(ns, nt)
        Screened optimal transportation matrix for the given parameters

    log : `dict`, default=False
      Log dictionary return only if log==True in parameters


    .. _references-screenkhorn:
    References
    -----------

    .. [2] M. Cuturi, Sinkhorn Distances : Lightspeed Computation of Optimal Transport,
        Advances in Neural Information Processing Systems (NIPS) 26, 2013

    .. [26] Alaya M. Z., Bérar M., Gasso G., Rakotomamonjy A. (2019).
        Screening Sinkhorn Algorithm for Regularized Optimal Transport (NIPS) 33, 2019

    """
    # check if bottleneck module exists
    try:
        import bottleneck
    except ImportError:
        warnings.warn(
            "Bottleneck module is not installed. Install it from"
            " https://pypi.org/project/Bottleneck/ for better performance.")
        bottleneck = np

    a, b, M = list_to_array(a, b, M)

    nx = get_backend(M, a, b)
    if nx.__name__ in ("jax", "tf"):
        raise TypeError("JAX or TF arrays have been received but screenkhorn is not "
                        "compatible with neither JAX nor TF.")

    ns, nt = M.shape

    # by default, we keep only 50% of the sample data points
    if ns_budget is None:
        ns_budget = int(np.floor(0.5 * ns))
    if nt_budget is None:
        nt_budget = int(np.floor(0.5 * nt))

    # calculate the Gibbs kernel
    K = nx.exp(-M / reg)

    def projection(u, epsilon):
        u = nx.maximum(u, epsilon)
        return u

    # ----------------------------------------------------------------------------------------------------------------#
    #                                          Step 1: Screening pre-processing                                       #
    # ----------------------------------------------------------------------------------------------------------------#

    if ns_budget == ns and nt_budget == nt:
        # full number of budget points (ns, nt) = (ns_budget, nt_budget)
        Isel = nx.from_numpy(np.ones(ns, dtype=bool))
        Jsel = nx.from_numpy(np.ones(nt, dtype=bool))
        epsilon = 0.0
        kappa = 1.0

        cst_u = 0.
        cst_v = 0.

        bounds_u = [(0.0, np.inf)] * ns
        bounds_v = [(0.0, np.inf)] * nt

        a_I = a
        b_J = b
        K_IJ = K
        K_IJc = []
        K_IcJ = []

        vec_eps_IJc = nx.zeros((nt,), type_as=M)
        vec_eps_IcJ = nx.zeros((ns,), type_as=M)

    else:
        # sum of rows and columns of K
        K_sum_cols = nx.sum(K, axis=1)
        K_sum_rows = nx.sum(K, axis=0)

        if uniform:
            if ns / ns_budget < 4:
                aK_sort = nx.sort(K_sum_cols)
                epsilon_u_square = a[0] / aK_sort[ns_budget - 1]
            else:
                aK_sort = nx.from_numpy(
                    bottleneck.partition(nx.to_numpy(
                        K_sum_cols), ns_budget - 1)[ns_budget - 1],
                    type_as=M
                )
                epsilon_u_square = a[0] / aK_sort

            if nt / nt_budget < 4:
                bK_sort = nx.sort(K_sum_rows)
                epsilon_v_square = b[0] / bK_sort[nt_budget - 1]
            else:
                bK_sort = nx.from_numpy(
                    bottleneck.partition(nx.to_numpy(
                        K_sum_rows), nt_budget - 1)[nt_budget - 1],
                    type_as=M
                )
                epsilon_v_square = b[0] / bK_sort
        else:
            aK = a / K_sum_cols
            bK = b / K_sum_rows

            aK_sort = nx.flip(nx.sort(aK), axis=0)
            epsilon_u_square = aK_sort[ns_budget - 1]

            bK_sort = nx.flip(nx.sort(bK), axis=0)
            epsilon_v_square = bK_sort[nt_budget - 1]

        # active sets I and J (see Lemma 1 in [26])
        Isel = a >= epsilon_u_square * K_sum_cols
        Jsel = b >= epsilon_v_square * K_sum_rows

        if nx.sum(Isel) != ns_budget:
            if uniform:
                aK = a / K_sum_cols
                aK_sort = nx.flip(nx.sort(aK), axis=0)
            epsilon_u_square = nx.mean(aK_sort[ns_budget - 1:ns_budget + 1])
            Isel = a >= epsilon_u_square * K_sum_cols
            ns_budget = nx.sum(Isel)

        if nx.sum(Jsel) != nt_budget:
            if uniform:
                bK = b / K_sum_rows
                bK_sort = nx.flip(nx.sort(bK), axis=0)
            epsilon_v_square = nx.mean(bK_sort[nt_budget - 1:nt_budget + 1])
            Jsel = b >= epsilon_v_square * K_sum_rows
            nt_budget = nx.sum(Jsel)

        epsilon = (epsilon_u_square * epsilon_v_square) ** (1 / 4)
        kappa = (epsilon_v_square / epsilon_u_square) ** (1 / 2)

        if verbose:
            print("epsilon = %s\n" % epsilon)
            print("kappa = %s\n" % kappa)
            print('Cardinality of selected points: |Isel| = %s \t |Jsel| = %s \n'
                  % (sum(Isel), sum(Jsel)))

        # Ic, Jc: complementary of the active sets I and J
        Ic = ~Isel
        Jc = ~Jsel

        K_IJ = K[np.ix_(Isel, Jsel)]
        K_IcJ = K[np.ix_(Ic, Jsel)]
        K_IJc = K[np.ix_(Isel, Jc)]

        K_min = nx.min(K_IJ)
        if K_min == 0:
            K_min = float(np.finfo(float).tiny)

        # a_I, b_J, a_Ic, b_Jc
        a_I = a[Isel]
        b_J = b[Jsel]
        if not uniform:
            a_I_min = nx.min(a_I)
            a_I_max = nx.max(a_I)
            b_J_max = nx.max(b_J)
            b_J_min = nx.min(b_J)
        else:
            a_I_min = a_I[0]
            a_I_max = a_I[0]
            b_J_max = b_J[0]
            b_J_min = b_J[0]

        # box constraints in L-BFGS-B (see Proposition 1 in [26])
        bounds_u = [(max(a_I_min / ((nt - nt_budget) * epsilon + nt_budget * (b_J_max / (
                    ns * epsilon * kappa * K_min))), epsilon / kappa), a_I_max / (nt * epsilon * K_min))] * ns_budget

        bounds_v = [(
            max(b_J_min / ((ns - ns_budget) * epsilon + ns_budget * (kappa * a_I_max / (nt * epsilon * K_min))),
                epsilon * kappa), b_J_max / (ns * epsilon * K_min))] * nt_budget

        # pre-calculated constants for the objective
        vec_eps_IJc = epsilon * kappa * nx.sum(
            K_IJc * nx.ones((nt - nt_budget,), type_as=M)[None, :],
            axis=1
        )
        vec_eps_IcJ = (epsilon / kappa) * nx.sum(
            nx.ones((ns - ns_budget,), type_as=M)[:, None] * K_IcJ,
            axis=0
        )

    # initialisation
    u0 = nx.full((ns_budget,), 1. / ns_budget + epsilon / kappa, type_as=M)
    v0 = nx.full((nt_budget,), 1. / nt_budget + epsilon * kappa, type_as=M)

    # pre-calculed constants for Restricted Sinkhorn (see Algorithm 1 in supplementary of [26])
    if restricted:
        if ns_budget != ns or nt_budget != nt:
            cst_u = kappa * epsilon * nx.sum(K_IJc, axis=1)
            cst_v = epsilon * nx.sum(K_IcJ, axis=0) / kappa

        for _ in range(5):  # 5 iterations
            K_IJ_v = nx.dot(K_IJ.T, u0) + cst_v
            v0 = b_J / (kappa * K_IJ_v)
            KIJ_u = nx.dot(K_IJ, v0) + cst_u
            u0 = (kappa * a_I) / KIJ_u

        u0 = projection(u0, epsilon / kappa)
        v0 = projection(v0, epsilon * kappa)

    else:
        u0 = u0
        v0 = v0

    def restricted_sinkhorn(usc, vsc, max_iter=5):
        """
        Restricted Sinkhorn Algorithm as a warm-start initialized pointfor L-BFGS-B)
        """
        for _ in range(max_iter):
            K_IJ_v = nx.dot(K_IJ.T, usc) + cst_v
            vsc = b_J / (kappa * K_IJ_v)
            KIJ_u = nx.dot(K_IJ, vsc) + cst_u
            usc = (kappa * a_I) / KIJ_u

        usc = projection(usc, epsilon / kappa)
        vsc = projection(vsc, epsilon * kappa)

        return usc, vsc

    def screened_obj(usc, vsc):
        part_IJ = (
            nx.dot(nx.dot(usc, K_IJ), vsc)
            - kappa * nx.dot(a_I, nx.log(usc))
            - (1. / kappa) * nx.dot(b_J, nx.log(vsc))
        )
        part_IJc = nx.dot(usc, vec_eps_IJc)
        part_IcJ = nx.dot(vec_eps_IcJ, vsc)
        psi_epsilon = part_IJ + part_IJc + part_IcJ
        return psi_epsilon

    def screened_grad(usc, vsc):
        # gradients of Psi_(kappa,epsilon) w.r.t u and v
        grad_u = nx.dot(K_IJ, vsc) + vec_eps_IJc - kappa * a_I / usc
        grad_v = nx.dot(K_IJ.T, usc) + vec_eps_IcJ - (1. / kappa) * b_J / vsc
        return grad_u, grad_v

    def bfgspost(theta):
        u = theta[:ns_budget]
        v = theta[ns_budget:]
        # objective
        f = screened_obj(u, v)
        # gradient
        g_u, g_v = screened_grad(u, v)
        g = nx.concatenate([g_u, g_v], axis=0)
        return nx.to_numpy(f), nx.to_numpy(g)

    # ----------------------------------------------------------------------------------------------------------------#
    #                                           Step 2: L-BFGS-B solver                                              #
    # ----------------------------------------------------------------------------------------------------------------#

    u0, v0 = restricted_sinkhorn(u0, v0)
    theta0 = nx.concatenate([u0, v0], axis=0)

    bounds = bounds_u + bounds_v  # constraint bounds

    def obj(theta):
        return bfgspost(nx.from_numpy(theta, type_as=M))

    theta, _, _ = fmin_l_bfgs_b(func=obj,
                                x0=theta0,
                                bounds=bounds,
                                maxfun=maxfun,
                                pgtol=pgtol,
                                maxiter=maxiter)
    theta = nx.from_numpy(theta, type_as=M)

    usc = theta[:ns_budget]
    vsc = theta[ns_budget:]

    usc_full = nx.full((ns,), epsilon / kappa, type_as=M)
    vsc_full = nx.full((nt,), epsilon * kappa, type_as=M)
    usc_full[Isel] = usc
    vsc_full[Jsel] = vsc

    if log:
        log = {}
        log['u'] = usc_full
        log['v'] = vsc_full
        log['Isel'] = Isel
        log['Jsel'] = Jsel

    gamma = usc_full[:, None] * K * vsc_full[None, :]
    gamma = gamma / nx.sum(gamma)

    if log:
        return gamma, log
    else:
        return gamma
