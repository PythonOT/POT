# -*- coding: utf-8 -*-
"""
LP solvers for optimal transport using cvxopt
"""

# Author: Remi Flamary <remi.flamary@unice.fr>
#
# License: MIT License

import numpy as np
import scipy as sp
import scipy.sparse as sps

try:
    import cvxopt
    from cvxopt import solvers, matrix, spmatrix
except ImportError:
    cvxopt = False


def scipy_sparse_to_spmatrix(A):
    """Efficient conversion from scipy sparse matrix to cvxopt sparse matrix"""
    coo = A.tocoo()
    SP = spmatrix(coo.data.tolist(), coo.row.tolist(), coo.col.tolist(), size=A.shape)
    return SP


def barycenter(A, M, weights=None, verbose=False, log=False, solver='highs-ipm'):
    r"""Compute the Wasserstein barycenter of distributions A

     The function solves the following optimization problem [16]:

    .. math::
       \mathbf{a} = arg\min_\mathbf{a} \sum_i W_{1}(\mathbf{a},\mathbf{a}_i)

    where :

    - :math:`W_1(\cdot,\cdot)` is the Wasserstein distance (see ot.emd.sinkhorn)
    - :math:`\mathbf{a}_i` are training distributions in the columns of matrix :math:`\mathbf{A}`

    The linear program is solved using the interior point solver from scipy.optimize.
    If cvxopt solver if installed it can use cvxopt

    Note that this problem do not scale well (both in memory and computational time).

    Parameters
    ----------
    A : np.ndarray (d,n)
        n training distributions a_i of size d
    M : np.ndarray (d,d)
        loss matrix   for OT
    reg : float
        Regularization term >0
    weights : np.ndarray (n,)
        Weights of each histogram a_i on the simplex (barycentric coordinates)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    solver : string, optional
        the solver used, default 'interior-point' use the lp solver from
        scipy.optimize. None, or 'glpk' or 'mosek' use the solver from cvxopt.

    Returns
    -------
    a : (d,) ndarray
        Wasserstein barycenter
    log : dict
        log dictionary return only if log==True in parameters


    References
    ----------

    .. [16] Agueh, M., & Carlier, G. (2011). Barycenters in the Wasserstein space. SIAM Journal on Mathematical Analysis, 43(2), 904-924.


    """

    if weights is None:
        weights = np.ones(A.shape[1]) / A.shape[1]
    else:
        assert len(weights) == A.shape[1]

    n_distributions = A.shape[1]
    n = A.shape[0]

    n2 = n * n
    c = np.zeros((0))
    b_eq1 = np.zeros((0))
    for i in range(n_distributions):
        c = np.concatenate((c, M.ravel() * weights[i]))
        b_eq1 = np.concatenate((b_eq1, A[:, i]))
    c = np.concatenate((c, np.zeros(n)))

    lst_idiag1 = [sps.kron(sps.eye(n), np.ones((1, n))) for i in range(n_distributions)]
    #  row constraints
    A_eq1 = sps.hstack((sps.block_diag(lst_idiag1), sps.coo_matrix((n_distributions * n, n))))

    # columns constraints
    lst_idiag2 = []
    lst_eye = []
    for i in range(n_distributions):
        if i == 0:
            lst_idiag2.append(sps.kron(np.ones((1, n)), sps.eye(n)))
            lst_eye.append(-sps.eye(n))
        else:
            lst_idiag2.append(sps.kron(np.ones((1, n)), sps.eye(n - 1, n)))
            lst_eye.append(-sps.eye(n - 1, n))

    A_eq2 = sps.hstack((sps.block_diag(lst_idiag2), sps.vstack(lst_eye)))
    b_eq2 = np.zeros((A_eq2.shape[0]))

    # full problem
    A_eq = sps.vstack((A_eq1, A_eq2))
    b_eq = np.concatenate((b_eq1, b_eq2))

    if not cvxopt or solver in ['interior-point', 'highs', 'highs-ipm', 'highs-ds']:
        # cvxopt not installed or interior point

        if solver is None:
            solver = 'interior-point'

        options = {'disp': verbose}
        sol = sp.optimize.linprog(c, A_eq=A_eq, b_eq=b_eq, method=solver,
                                  options=options)
        x = sol.x
        b = x[-n:]

    else:

        h = np.zeros((n_distributions * n2 + n))
        G = -sps.eye(n_distributions * n2 + n)

        sol = solvers.lp(matrix(c), scipy_sparse_to_spmatrix(G), matrix(h),
                         A=scipy_sparse_to_spmatrix(A_eq), b=matrix(b_eq),
                         solver=solver)

        x = np.array(sol['x'])
        b = x[-n:].ravel()

    if log:
        return b, sol
    else:
        return b
