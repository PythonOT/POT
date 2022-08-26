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


def KL(nx, p, q):
    return nx.sum(p * nx.log(p / q + 1e-16))


def solve(M, a=None, b=None, reg=None, reg_type="KL", unbalanced=None,
          unbalanced_type='KL', n_threads=1, max_iter=None, plan_init=None,
          potentials_init=None,
          tol=None, verbose=False):
    r"""Solve the discrete optimal transport problem and return solution

    The function returns an object of type :any:`OTResult`

    The function solves the following general optimal transport problem

    .. math::
        \min_{\mathbf{T}\geq 0} \quad \sum_{i,j} T_{i,j}M_{i,j} + \lambda_r R(\mathbf{T}) + \lambda_u U(\mathbf{T}\mathbf{1},\mathbf{a}) + \lambda_u U(\mathbf{T}^T\mathbf{1},\mathbf{b})

    The regularization is selected with :any:`reg` (:math:`\lambda_r`) and :any:`reg_type`. By
    default ``reg=None`` and there is no regularization. The unbalanced marginal
    penalization can be selected with :any:`unbalanced` (:math:`\lambda_u`) and
    :any:`unbalanced_type`. By default ``reg=None``

    When ``reg>0`` you can choose the type of regularization from :

    - ``reg="entropy"``: entropic regularization :math:`R(\mathbf{T})=\sum_{i,j} T_{i,j}\log(T_{i,j})` from seminal paper :ref:`[2] <references-solve>`.
    - ``reg="KL"`` (**default**): Kullback–Leibler regularization :math:`R(\mathbf{T})=\sum_{i,j} T_{i,j}\log(\frac{T_{i,j}}{a_i,b_j})` from paper :ref:`[34] <references-solve>`.
    - ``reg="L2"``: Quadratic regularization :math:`R(\mathbf{T})=\frac{1}{2}\sum_{i,j}
      T_{i,j}^2` from  paper :ref:`[17] <references-solve>`.



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
        Type of regularization :math:`R`  either "KL", "L2", 'entropy', by default "KL"
    unbalanced : float, optional
        Unbalanced penalization weight :math:`\lambda_u`, by default None
        (balanced OT)
    unbalanced_type : str, optional
        Type of unbalanced penalization unction :math:`U`  either "KL", "L2", 'TV', by default 'KL'
    n_threads : int, optional
        Number of OMP threads for exact OT solver, by default 1
    max_iter : int, optional
        Maximum number of iteration, by default None (default values in each solvers)
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

        - res.T : OT plan
        - res.potentials : OT dual potentials
        - res.value : Optimal value of the optimization problem
        - res.value_linear : Linear OT loss with the optimal OT plan
        

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
                value = value_linear + unbalanced * (KL(nx, nx.sum(plan, 1), a) + KL(nx, nx.sum(plan, 0), b))
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
            raise(NotImplementedError('Unknown unbalanced_type="{}"'.format(unbalanced_type)))

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
                    value = value_linear + reg * KL(nx, plan, a[:, None] * b[None, :])

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
                raise(NotImplementedError('Not implemented reg_type="{}"'.format(reg_type)))

        else:  # unbalanced AND regularized OT

            if reg_type.lower() in ['kl'] and unbalanced_type.lower() == 'kl':

                if max_iter is None:
                    max_iter = 1000
                if tol is None:
                    tol = 1e-9

                plan, log = sinkhorn_knopp_unbalanced(a, b, M, reg=reg, reg_m=unbalanced, numItermax=max_iter, stopThr=tol, verbose=verbose, log=True)

                value_linear = nx.sum(M * plan)

                value = value_linear + reg * KL(nx, plan, a[:, None] * b[None, :]) + unbalanced * (KL(nx, nx.sum(plan, 1), a) + KL(nx, nx.sum(plan, 0), b))

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
                raise(NotImplementedError('Not implemented reg_type="{}" and unbalanced_type="{}"'.format(reg_type, unbalanced_type)))

    res = OTResult(potentials=potentials, value=value,
                   value_linear=value_linear, plan=plan, status=status, backend=nx)

    return res
