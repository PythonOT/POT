# -*- coding: utf-8 -*-
"""
General OT solvers with unified API
"""

# Author: Remi Flamary <remi.flamary@polytechnique.edu>
#
# License: MIT License

from .utils import OTResult
from .lp import emd2
from .backend import get_backend
from .unbalanced import mm_unbalanced, sinkhorn_knopp_unbalanced, lbfgsb_unbalanced
from .bregman import sinkhorn_log
from .partial import partial_wasserstein_lagrange
from .smooth import smooth_ot_dual
from .gromov import (gromov_wasserstein2, fused_gromov_wasserstein2,
                     entropic_gromov_wasserstein2, entropic_fused_gromov_wasserstein2,
                     semirelaxed_gromov_wasserstein2, semirelaxed_fused_gromov_wasserstein2,
                     entropic_semirelaxed_fused_gromov_wasserstein2,
                     entropic_semirelaxed_gromov_wasserstein2)
from .partial import partial_gromov_wasserstein2, entropic_partial_gromov_wasserstein2

#, entropic_gromov_wasserstein2, entropic_fused_gromov_wasserstein2


def solve(M, a=None, b=None, reg=None, reg_type="KL", unbalanced=None,
          unbalanced_type='KL', n_threads=1, max_iter=None, plan_init=None,
          potentials_init=None, tol=None, verbose=False):
    r"""Solve the discrete optimal transport problem and return :any:`OTResult` object

    The function solves the following general optimal transport problem

    .. math::
        \min_{\mathbf{T}\geq 0} \quad \sum_{i,j} T_{i,j}M_{i,j} + \lambda_r R(\mathbf{T}) +
        \lambda_u U(\mathbf{T}\mathbf{1},\mathbf{a}) +
        \lambda_u U(\mathbf{T}^T\mathbf{1},\mathbf{b})

    The regularization is selected with `reg` (:math:`\lambda_r`) and `reg_type`. By
    default ``reg=None`` and there is no regularization. The unbalanced marginal
    penalization can be selected with `unbalanced` (:math:`\lambda_u`) and
    `unbalanced_type`. By default ``unbalanced=None`` and the function
    solves the exact optimal transport problem (respecting the marginals).

    Parameters
    ----------
    M : array_like, shape (dim_a, dim_b)
        Loss matrix
    a : array-like, shape (dim_a,), optional
        Samples weights in the source domain (default is uniform)
    b : array-like, shape (dim_b,), optional
        Samples weights in the source domain (default is uniform)
    reg : float, optional
        Regularization weight :math:`\lambda_r`, by default None (no reg., exact
        OT)
    reg_type : str, optional
        Type of regularization :math:`R`  either "KL", "L2", "entropy", by default "KL"
    unbalanced : float, optional
        Unbalanced penalization weight :math:`\lambda_u`, by default None
        (balanced OT)
    unbalanced_type : str, optional
        Type of unbalanced penalization function :math:`U`  either "KL", "L2", "TV", by default "KL"
    n_threads : int, optional
        Number of OMP threads for exact OT solver, by default 1
    max_iter : int, optional
        Maximum number of iterations, by default None (default values in each solvers)
    plan_init : array_like, shape (dim_a, dim_b), optional
        Initialization of the OT plan for iterative methods, by default None
    potentials_init : (array_like(dim_a,),array_like(dim_b,)), optional
        Initialization of the OT dual potentials for iterative methods, by default None
    tol : _type_, optional
        Tolerance for solution precision, by default None (default values in each solvers)
    verbose : bool, optional
        Print information in the solver, by default False

    Returns
    -------
    res : OTResult()
        Result of the optimization problem. The information can be obtained as follows:

        - res.plan : OT plan :math:`\mathbf{T}`
        - res.potentials : OT dual potentials
        - res.value : Optimal value of the optimization problem
        - res.value_linear : Linear OT loss with the optimal OT plan

        See :any:`OTResult` for more information.

    Notes
    -----

    The following methods are available for solving the OT problems:

    - **Classical exact OT problem** (default parameters):

    .. math::
        \min_\mathbf{T} \quad \langle \mathbf{T}, \mathbf{M} \rangle_F

        s.t. \ \mathbf{T} \mathbf{1} = \mathbf{a}

             \mathbf{T}^T \mathbf{1} = \mathbf{b}

             \mathbf{T} \geq 0

    can be solved with the following code:

    .. code-block:: python

        res = ot.solve(M, a, b)

    - **Entropic regularized OT** (when ``reg!=None``):

    .. math::
        \min_\mathbf{T} \quad \langle \mathbf{T}, \mathbf{M} \rangle_F + \lambda R(\mathbf{T})

        s.t. \ \mathbf{T} \mathbf{1} = \mathbf{a}

             \mathbf{T}^T \mathbf{1} = \mathbf{b}

             \mathbf{T} \geq 0

    can be solved with the following code:

    .. code-block:: python

        # default is ``"KL"`` regularization (``reg_type="KL"``)
        res = ot.solve(M, a, b, reg=1.0)
        # or for original Sinkhorn paper formulation [2]
        res = ot.solve(M, a, b, reg=1.0, reg_type='entropy')

    - **Quadratic regularized OT** (when ``reg!=None`` and ``reg_type="L2"``):

    .. math::
        \min_\mathbf{T} \quad \langle \mathbf{T}, \mathbf{M} \rangle_F + \lambda R(\mathbf{T})

        s.t. \ \mathbf{T} \mathbf{1} = \mathbf{a}

             \mathbf{T}^T \mathbf{1} = \mathbf{b}

             \mathbf{T} \geq 0

    can be solved with the following code:

    .. code-block:: python

        res = ot.solve(M,a,b,reg=1.0,reg_type='L2')

    - **Unbalanced OT** (when ``unbalanced!=None``):

    .. math::
        \min_{\mathbf{T}\geq 0} \quad \sum_{i,j} T_{i,j}M_{i,j} + \lambda_u U(\mathbf{T}\mathbf{1},\mathbf{a}) + \lambda_u U(\mathbf{T}^T\mathbf{1},\mathbf{b})

    can be solved with the following code:

    .. code-block:: python

        # default is ``"KL"``
        res = ot.solve(M,a,b,reg=1.0,unbalanced=1.0)
        # quadratic unbalanced OT
        res = ot.solve(M,a,b,reg=1.0,unbalanced=1.0,unbalanced_type='L2')
        # TV = partial OT
        res = ot.solve(M,a,b,reg=1.0,unbalanced=1.0,unbalanced_type='TV')


    - **Regularized unbalanced regularized OT** (when ``unbalanced!=None`` and ``reg!=None``):

    .. math::
        \min_{\mathbf{T}\geq 0} \quad \sum_{i,j} T_{i,j}M_{i,j} + \lambda_r R(\mathbf{T}) + \lambda_u U(\mathbf{T}\mathbf{1},\mathbf{a}) + \lambda_u U(\mathbf{T}^T\mathbf{1},\mathbf{b})

    can be solved with the following code:

    .. code-block:: python

        # default is ``"KL"`` for both
        res = ot.solve(M,a,b,reg=1.0,unbalanced=1.0)
        # quadratic unbalanced OT with KL regularization
        res = ot.solve(M,a,b,reg=1.0,unbalanced=1.0,unbalanced_type='L2')
        # both quadratic
        res = ot.solve(M,a,b,reg=1.0, reg_type='L2',unbalanced=1.0,unbalanced_type='L2')


    .. _references-solve:
    References
    ----------

    .. [2] M. Cuturi, Sinkhorn Distances : Lightspeed Computation
        of Optimal Transport, Advances in Neural Information Processing
        Systems (NIPS) 26, 2013

    .. [10] Chizat, L., Peyré, G., Schmitzer, B., & Vialard, F. X. (2016).
        Scaling algorithms for unbalanced transport problems.
        arXiv preprint arXiv:1607.05816.

    .. [17] Blondel, M., Seguy, V., & Rolet, A. (2018). Smooth and Sparse
        Optimal Transport. Proceedings of the Twenty-First International
        Conference on Artificial Intelligence and Statistics (AISTATS).

    .. [34] Feydy, J., Séjourné, T., Vialard, F. X., Amari, S. I., Trouvé,
        A., & Peyré, G. (2019, April). Interpolating between optimal transport
        and MMD using Sinkhorn divergences. In The 22nd International Conference
        on Artificial Intelligence and Statistics (pp. 2681-2690). PMLR.

    """

    # detect backend
    arr = [M]
    if a is not None:
        arr.append(a)
    if b is not None:
        arr.append(b)
    nx = get_backend(*arr)

    # create uniform weights if not given
    if a is None:
        a = nx.ones(M.shape[0], type_as=M) / M.shape[0]
    if b is None:
        b = nx.ones(M.shape[1], type_as=M) / M.shape[1]

    # default values for solutions
    potentials = None
    value = None
    value_linear = None
    plan = None
    status = None

    if reg is None or reg == 0:  # exact OT

        if unbalanced is None:  # Exact balanced OT

            # default values for EMD solver
            if max_iter is None:
                max_iter = 1000000

            value_linear, log = emd2(a, b, M, numItermax=max_iter, log=True, return_matrix=True, numThreads=n_threads)

            value = value_linear
            potentials = (log['u'], log['v'])
            plan = log['G']
            status = log["warning"] if log["warning"] is not None else 'Converged'

        elif unbalanced_type.lower() in ['kl', 'l2']:  # unbalanced exact OT

            # default values for exact unbalanced OT
            if max_iter is None:
                max_iter = 1000
            if tol is None:
                tol = 1e-12

            plan, log = mm_unbalanced(a, b, M, reg_m=unbalanced,
                                      div=unbalanced_type.lower(), numItermax=max_iter,
                                      stopThr=tol, log=True,
                                      verbose=verbose, G0=plan_init)

            value_linear = log['cost']

            if unbalanced_type.lower() == 'kl':
                value = value_linear + unbalanced * (nx.kl_div(nx.sum(plan, 1), a) + nx.kl_div(nx.sum(plan, 0), b))
            else:
                err_a = nx.sum(plan, 1) - a
                err_b = nx.sum(plan, 0) - b
                value = value_linear + unbalanced * nx.sum(err_a**2) + unbalanced * nx.sum(err_b**2)

        elif unbalanced_type.lower() == 'tv':

            if max_iter is None:
                max_iter = 1000000

            plan, log = partial_wasserstein_lagrange(a, b, M, reg_m=unbalanced**2, log=True, numItermax=max_iter)

            value_linear = nx.sum(M * plan)
            err_a = nx.sum(plan, 1) - a
            err_b = nx.sum(plan, 0) - b
            value = value_linear + nx.sqrt(unbalanced**2 / 2.0 * (nx.sum(nx.abs(err_a)) +
                                                                  nx.sum(nx.abs(err_b))))

        else:
            raise (NotImplementedError('Unknown unbalanced_type="{}"'.format(unbalanced_type)))

    else:  # regularized OT

        if unbalanced is None:  # Balanced regularized OT

            if reg_type.lower() in ['entropy', 'kl']:

                # default values for sinkhorn
                if max_iter is None:
                    max_iter = 1000
                if tol is None:
                    tol = 1e-9

                plan, log = sinkhorn_log(a, b, M, reg=reg, numItermax=max_iter,
                                         stopThr=tol, log=True,
                                         verbose=verbose)

                value_linear = nx.sum(M * plan)

                if reg_type.lower() == 'entropy':
                    value = value_linear + reg * nx.sum(plan * nx.log(plan + 1e-16))
                else:
                    value = value_linear + reg * nx.kl_div(plan, a[:, None] * b[None, :])

                potentials = (log['log_u'], log['log_v'])

            elif reg_type.lower() == 'l2':

                if max_iter is None:
                    max_iter = 1000
                if tol is None:
                    tol = 1e-9

                plan, log = smooth_ot_dual(a, b, M, reg=reg, numItermax=max_iter, stopThr=tol, log=True, verbose=verbose)

                value_linear = nx.sum(M * plan)
                value = value_linear + reg * nx.sum(plan**2)
                potentials = (log['alpha'], log['beta'])

            else:
                raise (NotImplementedError('Not implemented reg_type="{}"'.format(reg_type)))

        else:  # unbalanced AND regularized OT

            if reg_type.lower() in ['kl'] and unbalanced_type.lower() == 'kl':

                if max_iter is None:
                    max_iter = 1000
                if tol is None:
                    tol = 1e-9

                plan, log = sinkhorn_knopp_unbalanced(a, b, M, reg=reg, reg_m=unbalanced, numItermax=max_iter, stopThr=tol, verbose=verbose, log=True)

                value_linear = nx.sum(M * plan)

                value = value_linear + reg * nx.kl_div(plan, a[:, None] * b[None, :]) + unbalanced * (nx.kl_div(nx.sum(plan, 1), a) + nx.kl_div(nx.sum(plan, 0), b))

                potentials = (log['logu'], log['logv'])

            elif reg_type.lower() in ['kl', 'l2', 'entropy'] and unbalanced_type.lower() in ['kl', 'l2']:

                if max_iter is None:
                    max_iter = 1000
                if tol is None:
                    tol = 1e-12

                plan, log = lbfgsb_unbalanced(a, b, M, reg=reg, reg_m=unbalanced, reg_div=reg_type.lower(), regm_div=unbalanced_type.lower(), numItermax=max_iter, stopThr=tol, verbose=verbose, log=True)

                value_linear = nx.sum(M * plan)

                value = log['loss']

            else:
                raise (NotImplementedError('Not implemented reg_type="{}" and unbalanced_type="{}"'.format(reg_type, unbalanced_type)))

    res = OTResult(potentials=potentials, value=value,
                   value_linear=value_linear, plan=plan, status=status, backend=nx)

    return res


def solve_gromov(Ca, Cb, M=None, a=None, b=None, loss='L2', symmetric=None,
                 alpha=0.5, reg=None,
                 reg_type="entropy", unbalanced=None, unbalanced_type='KL',
                 n_threads=1, method=None, max_iter=None, plan_init=None, tol=None,
                 verbose=False):
    r""" Solve the discrete (Fused) Gromov-Wasserstein and return :any:`OTResult` object

    The function solves the following optimization problem:

    .. math::
        \min_{\mathbf{T}\geq 0} \quad (1 - \alpha) \langle \mathbf{T}, \mathbf{M} \rangle_F +
        \alpha \sum_{i,j,k,l} L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l}) \mathbf{T}_{i,j} + \lambda_r R(\mathbf{T}) + \lambda_u U(\mathbf{T}\mathbf{1},\mathbf{a}) + \lambda_u U(\mathbf{T}^T\mathbf{1},\mathbf{b})

    The regularization is selected with `reg` (:math:`\lambda_r`) and
    `reg_type`. By default ``reg=None`` and there is no regularization. The
    unbalanced marginal penalization can be selected with `unbalanced`
    (:math:`\lambda_u`) and `unbalanced_type`. By default ``unbalanced=None``
    and the function solves the exact optimal transport problem (respecting the
    marginals).

    Parameters
    ----------
    Ca : array_like, shape (dim_a, dim_a)
        Cost matrix in the source domain
    Cb : array_like, shape (dim_b, dim_b)
        Cost matrix in the target domain
    M : array_like, shape (dim_a, dim_b), optional
        Linear cost matrix for Fused Gromov-Wasserstein (default is None).
    a : array-like, shape (dim_a,), optional
        Samples weights in the source domain (default is uniform)
    b : array-like, shape (dim_b,), optional
        Samples weights in the source domain (default is uniform)
    loss : str, optional
        Type of loss function, either ``"L2"`` or ``"KL"``, by default ``"L2"``
    symmetric : bool, optional
        Use symmetric version of the Gromov-Wasserstein problem, by default None
        tests whether the matrices are symmetric or True/False to avoid the test.
    reg : float, optional
        Regularization weight :math:`\lambda_r`, by default None (no reg., exact
        OT)
    reg_type : str, optional
        Type of regularization :math:`R`, by default "entropy" (only used when
        ``reg!=None``)
    alpha : float, optional
        Weight the quadratic term (alpha*Gromov) and the linear term
        ((1-alpha)*Wass) in the Fused Gromov-Wasserstein problem. Not used for
        Gromov problem (when M is not provided). By default ``alpha=None``
        corresponds to ``alpha=1`` for Gromov problem (``M==None``) and
        ``alpha=0.5`` for Fused Gromov-Wasserstein problem (``M!=None``)
    unbalanced : float, optional
        Unbalanced penalization weight :math:`\lambda_u`, by default None
        (balanced OT), Not implemented yet
    unbalanced_type : str, optional
        Type of unbalanced penalization function :math:`U` either "KL", "semirelaxed",
        "partial", by default "KL" but note that it is not implemented yet.
    n_threads : int, optional
        Number of OMP threads for exact OT solver, by default 1
    method : str, optional
        Method for solving the problem, for entropic problems "PGD" is projected
        gradient descent and "PPA" for proximal point, default None for
        automatic selection ("PGD").
    max_iter : int, optional
        Maximum number of iterations, by default None (default values in each
        solvers)
    plan_init : array_like, shape (dim_a, dim_b), optional
        Initialization of the OT plan for iterative methods, by default None
    tol : float, optional
        Tolerance for solution precision, by default None (default values in
        each solvers)
    verbose : bool, optional
        Print information in the solver, by default False

    Returns
    -------
    res : OTResult()
        Result of the optimization problem. The information can be obtained as follows:

        - res.plan : OT plan :math:`\mathbf{T}`
        - res.potentials : OT dual potentials
        - res.value : Optimal value of the optimization problem
        - res.value_linear : Linear OT loss with the optimal OT plan
        - res.value_quad : Quadratic (GW) part of the OT loss with the optimal OT plan

        See :any:`OTResult` for more information.

    Notes
    -----
    The following methods are available for solving the Gromov-Wasserstein
    problem:

    - **Classical Gromov-Wasserstein (GW) problem [3]** (default parameters):

    .. math::
        \min_{\mathbf{T}\geq 0} \sum_{i,j,k,l} L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l}) \mathbf{T}_{i,j}\mathbf{T}_{k,l}

        s.t. \ \mathbf{T} \mathbf{1} = \mathbf{a}

             \mathbf{T}^T \mathbf{1} = \mathbf{b}

             \mathbf{T} \geq 0

    can be solved with the following code:

    .. code-block:: python

        res = ot.solve_gromov(Ca, Cb) # uniform weights
        res = ot.solve_gromov(Ca, Cb, a=a, b=b) # given weights
        res = ot.solve_gromov(Ca, Cb, loss='KL') # KL loss

        plan = res.plan # GW plan
        value = res.value # GW value

    - **Fused Gromov-Wasserstein (FGW) problem [24]** (when ``M!=None``):

    .. math::
        \min_{\mathbf{T}\geq 0} \quad (1 - \alpha) \langle \mathbf{T}, \mathbf{M} \rangle_F +
        \alpha \sum_{i,j,k,l} L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l}) \mathbf{T}_{i,j}\mathbf{T}_{k,l}

        s.t. \ \mathbf{T} \mathbf{1} = \mathbf{a}

             \mathbf{T}^T \mathbf{1} = \mathbf{b}

             \mathbf{T} \geq 0

    can be solved with the following code:

    .. code-block:: python

        res = ot.solve_gromov(Ca, Cb, M) # uniform weights, alpha=0.5 (default)
        res = ot.solve_gromov(Ca, Cb, M, a=a, b=b, alpha=0.1) # given weights and alpha

        plan = res.plan # FGW plan
        loss_linear_term = res.value_linear # Wasserstein part of the loss
        loss_quad_term = res.value_quad # Gromov part of the loss
        loss = res.value # FGW value

    - **Regularized (Fused) Gromov-Wasserstein (GW) problem [12]** (when  ``reg!=None``):

    .. math::
        \min_{\mathbf{T}\geq 0} \quad (1 - \alpha) \langle \mathbf{T}, \mathbf{M} \rangle_F +
        \alpha \sum_{i,j,k,l} L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l}) \mathbf{T}_{i,j}\mathbf{T}_{k,l} + \lambda_r R(\mathbf{T})

        s.t. \ \mathbf{T} \mathbf{1} = \mathbf{a}

             \mathbf{T}^T \mathbf{1} = \mathbf{b}

             \mathbf{T} \geq 0

    can be solved with the following code:

    .. code-block:: python

        res = ot.solve_gromov(Ca, Cb, reg=1.0) # GW entropy regularization (default)
        res = ot.solve_gromov(Ca, Cb, M, a=a, b=b, reg=10, alpha=0.1) # FGW with entropy

        plan = res.plan # FGW plan
        loss_linear_term = res.value_linear # Wasserstein part of the loss
        loss_quad_term = res.value_quad # Gromov part of the loss
        loss = res.value # FGW value (including regularization)

    - **Semi-relaxed (Fused) Gromov-Wasserstein (GW) [48]** (when  ``unbalanced='semirelaxed'``):

    .. math::
        \min_{\mathbf{T}\geq 0} \quad (1 - \alpha) \langle \mathbf{T}, \mathbf{M} \rangle_F +
        \alpha \sum_{i,j,k,l} L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l}) \mathbf{T}_{i,j}\mathbf{T}_{k,l}

        s.t. \ \mathbf{T} \mathbf{1} = \mathbf{a}

             \mathbf{T} \geq 0

    can be solved with the following code:

    .. code-block:: python

        res = ot.solve_gromov(Ca, Cb, unbalanced='semirelaxed') # semirelaxed GW
        res = ot.solve_gromov(Ca, Cb, unbalanced='semirelaxed', reg=1) # entropic semirelaxed GW
        res = ot.solve_gromov(Ca, Cb, M, unbalanced='semirelaxed', alpha=0.1) # semirelaxed FGW

        plan = res.plan # FGW plan
        right_marginal = res.marginal_b # right marginal of the plan

    - **Partial (Fused) Gromov-Wasserstein (GW) problem [29]** (when  ``unbalanced='partial'``):

    .. math::
        \min_{\mathbf{T}\geq 0} \quad (1 - \alpha) \langle \mathbf{T}, \mathbf{M} \rangle_F +
        \alpha \sum_{i,j,k,l} L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l}) \mathbf{T}_{i,j}\mathbf{T}_{k,l}

        s.t. \ \mathbf{T} \mathbf{1} \leq \mathbf{a}

                \mathbf{T}^T \mathbf{1} \leq \mathbf{b}

                \mathbf{T} \geq 0

                \mathbf{1}^T\mathbf{T}\mathbf{1} = m

    can be solved with the following code:

    .. code-block:: python

        res = ot.solve_gromov(Ca, Cb, unbalanced_type='partial', unbalanced=0.8) # partial GW with m=0.8


    .. _references-solve-gromov:
    References
    ----------

    .. [3] Mémoli, F. (2011). Gromov–Wasserstein distances and the metric
        approach to object matching. Foundations of computational mathematics,
        11(4), 417-487.

    .. [12] Gabriel Peyré, Marco Cuturi, and Justin Solomon (2016),
        Gromov-Wasserstein averaging of kernel and distance matrices
        International Conference on Machine Learning (ICML).

    .. [24] Vayer, T., Chapel, L., Flamary, R., Tavenard, R. and Courty, N.
        (2019). Optimal Transport for structured data with application on graphs
        Proceedings of the 36th International Conference on Machine Learning
        (ICML).

    .. [48] Cédric Vincent-Cuaz, Rémi Flamary, Marco Corneli, Titouan Vayer,
        Nicolas Courty (2022). Semi-relaxed Gromov-Wasserstein divergence and
        applications on graphs. International Conference on Learning
        Representations (ICLR), 2022.

    .. [29] Chapel, L., Alaya, M., Gasso, G. (2020). Partial Optimal Transport
        with Applications on Positive-Unlabeled Learning, Advances in Neural
        Information Processing Systems (NeurIPS), 2020.

    """

    # detect backend
    nx = get_backend(Ca, Cb, M, a, b)

    # create uniform weights if not given
    if a is None:
        a = nx.ones(Ca.shape[0], type_as=Ca) / Ca.shape[0]
    if b is None:
        b = nx.ones(Cb.shape[1], type_as=Cb) / Cb.shape[1]

    # default values for solutions
    potentials = None
    value = None
    value_linear = None
    value_quad = None
    plan = None
    status = None

    loss_dict = {'l2': 'square_loss', 'kl': 'kl_loss'}

    if loss.lower() not in loss_dict.keys():
        raise (NotImplementedError('Not implemented GW loss="{}"'.format(loss)))
    loss_fun = loss_dict[loss.lower()]

    if reg is None or reg == 0:  # exact OT

        if unbalanced is None and unbalanced_type.lower() not in ['semirelaxed']:  # Exact balanced OT

            if M is None or alpha == 1:  # Gromov-Wasserstein problem

                # default values for solver
                if max_iter is None:
                    max_iter = 10000
                if tol is None:
                    tol = 1e-9

                value, log = gromov_wasserstein2(Ca, Cb, a, b, loss_fun=loss_fun, log=True, symmetric=symmetric, max_iter=max_iter, G0=plan_init, tol_rel=tol, tol_abs=tol, verbose=verbose)

                value_quad = value
                if alpha == 1:  # set to 0 for FGW with alpha=1
                    value_linear = 0
                plan = log['T']
                potentials = (log['u'], log['v'])

            elif alpha == 0:  # Wasserstein problem

                # default values for EMD solver
                if max_iter is None:
                    max_iter = 1000000

                value_linear, log = emd2(a, b, M, numItermax=max_iter, log=True, return_matrix=True, numThreads=n_threads)

                value = value_linear
                potentials = (log['u'], log['v'])
                plan = log['G']
                status = log["warning"] if log["warning"] is not None else 'Converged'
                value_quad = 0

            else:  # Fused Gromov-Wasserstein problem

                # default values for solver
                if max_iter is None:
                    max_iter = 10000
                if tol is None:
                    tol = 1e-9

                value, log = fused_gromov_wasserstein2(M, Ca, Cb, a, b, loss_fun=loss_fun, alpha=alpha, log=True, symmetric=symmetric, max_iter=max_iter, G0=plan_init, tol_rel=tol, tol_abs=tol, verbose=verbose)

                value_linear = log['lin_loss']
                value_quad = log['quad_loss']
                plan = log['T']
                potentials = (log['u'], log['v'])

        elif unbalanced_type.lower() in ['semirelaxed']:  # Semi-relaxed  OT

            if M is None or alpha == 1:  # Semi relaxed Gromov-Wasserstein problem

                # default values for solver
                if max_iter is None:
                    max_iter = 10000
                if tol is None:
                    tol = 1e-9

                value, log = semirelaxed_gromov_wasserstein2(Ca, Cb, a, loss_fun=loss_fun, log=True, symmetric=symmetric, max_iter=max_iter, G0=plan_init, tol_rel=tol, tol_abs=tol, verbose=verbose)

                value_quad = value
                if alpha == 1:  # set to 0 for FGW with alpha=1
                    value_linear = 0
                plan = log['T']
                # potentials = (log['u'], log['v']) TODO

            else:  # Semi relaxed Fused Gromov-Wasserstein problem

                # default values for solver
                if max_iter is None:
                    max_iter = 10000
                if tol is None:
                    tol = 1e-9

                value, log = semirelaxed_fused_gromov_wasserstein2(M, Ca, Cb, a, loss_fun=loss_fun, alpha=alpha, log=True, symmetric=symmetric, max_iter=max_iter, G0=plan_init, tol_rel=tol, tol_abs=tol, verbose=verbose)

                value_linear = log['lin_loss']
                value_quad = log['quad_loss']
                plan = log['T']
                # potentials = (log['u'], log['v']) TODO

        elif unbalanced_type.lower() in ['partial']:  # Partial OT

            if M is None:  # Partial Gromov-Wasserstein problem

                if unbalanced > nx.sum(a) or unbalanced > nx.sum(b):
                    raise (ValueError('Partial GW mass given in reg is too large'))
                if loss.lower() != 'l2':
                    raise (NotImplementedError('Partial GW only implemented with L2 loss'))
                if symmetric is not None:
                    raise (NotImplementedError('Partial GW only implemented with symmetric=True'))

                # default values for solver
                if max_iter is None:
                    max_iter = 1000
                if tol is None:
                    tol = 1e-7

                value, log = partial_gromov_wasserstein2(Ca, Cb, a, b, m=unbalanced, log=True, numItermax=max_iter, G0=plan_init, tol=tol, verbose=verbose)

                value_quad = value
                plan = log['T']
                # potentials = (log['u'], log['v']) TODO

            else:  # partial FGW

                raise (NotImplementedError('Partial FGW not implemented yet'))

        elif unbalanced_type.lower() in ['kl', 'l2']:  # unbalanced exact OT

            raise (NotImplementedError('Unbalanced_type="{}"'.format(unbalanced_type)))

        else:
            raise (NotImplementedError('Unknown unbalanced_type="{}"'.format(unbalanced_type)))

    else:  # regularized OT

        if unbalanced is None and unbalanced_type.lower() not in ['semirelaxed']:  # Balanced regularized OT

            if reg_type.lower() in ['entropy'] and (M is None or alpha == 1):  # Entropic Gromov-Wasserstein problem

                # default values for solver
                if max_iter is None:
                    max_iter = 1000
                if tol is None:
                    tol = 1e-9
                if method is None:
                    method = 'PGD'

                value_quad, log = entropic_gromov_wasserstein2(Ca, Cb, a, b, epsilon=reg, loss_fun=loss_fun, log=True, symmetric=symmetric, solver=method, max_iter=max_iter, G0=plan_init, tol_rel=tol, tol_abs=tol, verbose=verbose)

                plan = log['T']
                value_linear = 0
                value = value_quad + reg * nx.sum(plan * nx.log(plan + 1e-16))
                # potentials = (log['log_u'], log['log_v'])  #TODO

            elif reg_type.lower() in ['entropy'] and M is not None and alpha == 0:  # Entropic Wasserstein problem

                # default values for solver
                if max_iter is None:
                    max_iter = 1000
                if tol is None:
                    tol = 1e-9

                plan, log = sinkhorn_log(a, b, M, reg=reg, numItermax=max_iter,
                                         stopThr=tol, log=True,
                                         verbose=verbose)

                value_linear = nx.sum(M * plan)
                value = value_linear + reg * nx.sum(plan * nx.log(plan + 1e-16))
                potentials = (log['log_u'], log['log_v'])

            elif reg_type.lower() in ['entropy'] and M is not None:  # Entropic Fused Gromov-Wasserstein problem

                # default values for solver
                if max_iter is None:
                    max_iter = 1000
                if tol is None:
                    tol = 1e-9
                if method is None:
                    method = 'PGD'

                value_noreg, log = entropic_fused_gromov_wasserstein2(M, Ca, Cb, a, b, loss_fun=loss_fun, alpha=alpha, log=True, symmetric=symmetric, solver=method, max_iter=max_iter, G0=plan_init, tol_rel=tol, tol_abs=tol, verbose=verbose)

                value_linear = log['lin_loss']
                value_quad = log['quad_loss']
                plan = log['T']
                # potentials = (log['u'], log['v'])
                value = value_noreg + reg * nx.sum(plan * nx.log(plan + 1e-16))

            else:
                raise (NotImplementedError('Not implemented reg_type="{}"'.format(reg_type)))

        elif unbalanced_type.lower() in ['semirelaxed']:  # Semi-relaxed  OT

            if reg_type.lower() in ['entropy'] and (M is None or alpha == 1):  # Entropic Semi-relaxed Gromov-Wasserstein problem

                # default values for solver
                if max_iter is None:
                    max_iter = 1000
                if tol is None:
                    tol = 1e-9

                value_quad, log = entropic_semirelaxed_gromov_wasserstein2(Ca, Cb, a, epsilon=reg, loss_fun=loss_fun, log=True, symmetric=symmetric, max_iter=max_iter, G0=plan_init, tol_rel=tol, tol_abs=tol, verbose=verbose)

                plan = log['T']
                value_linear = 0
                value = value_quad + reg * nx.sum(plan * nx.log(plan + 1e-16))

            else:  # Entropic Semi-relaxed FGW problem

                # default values for solver
                if max_iter is None:
                    max_iter = 1000
                if tol is None:
                    tol = 1e-9

                value_noreg, log = entropic_semirelaxed_fused_gromov_wasserstein2(M, Ca, Cb, a, loss_fun=loss_fun, alpha=alpha, log=True, symmetric=symmetric, max_iter=max_iter, G0=plan_init, tol_rel=tol, tol_abs=tol, verbose=verbose)

                value_linear = log['lin_loss']
                value_quad = log['quad_loss']
                plan = log['T']
                value = value_noreg + reg * nx.sum(plan * nx.log(plan + 1e-16))

        elif unbalanced_type.lower() in ['partial']:  # Partial OT

            if M is None:  # Partial Gromov-Wasserstein problem

                if unbalanced > nx.sum(a) or unbalanced > nx.sum(b):
                    raise (ValueError('Partial GW mass given in reg is too large'))
                if loss.lower() != 'l2':
                    raise (NotImplementedError('Partial GW only implemented with L2 loss'))
                if symmetric is not None:
                    raise (NotImplementedError('Partial GW only implemented with symmetric=True'))

                # default values for solver
                if max_iter is None:
                    max_iter = 1000
                if tol is None:
                    tol = 1e-7

                value_quad, log = entropic_partial_gromov_wasserstein2(Ca, Cb, a, b, reg=reg, m=unbalanced, log=True, numItermax=max_iter, G0=plan_init, tol=tol, verbose=verbose)

                value_quad = value
                plan = log['T']
                # potentials = (log['u'], log['v']) TODO

            else:  # partial FGW

                raise (NotImplementedError('Partial entropic FGW not implemented yet'))

        else:  # unbalanced AND regularized OT

            raise (NotImplementedError('Not implemented reg_type="{}" and unbalanced_type="{}"'.format(reg_type, unbalanced_type)))

    res = OTResult(potentials=potentials, value=value,
                   value_linear=value_linear, value_quad=value_quad, plan=plan, status=status, backend=nx)

    return res






