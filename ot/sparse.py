"""
Sparsity-constrained optimal transport solvers.

Implementation of :
Sparsity-Constrained Optimal Transport.
Liu, T., Puigcerver, J., & Blondel, M. (2023).
Sparsity-constrained optimal transport.
Proceedings of the Eleventh International Conference on
Learning Representations (ICLR).
https://arxiv.org/abs/2209.15466

[50] Liu, T., Puigcerver, J., & Blondel, M. (2023).
Sparsity-constrained optimal transport.
Proceedings of the Eleventh International Conference on
Learning Representations (ICLR).
"""

# Author: Tianlin Liu <t.liu@unibas.ch>
#
# License: MIT License


import numpy as np
import ot
from .backend import get_backend


def projection_sparse_simplex(V, max_nz, z=1, axis=None):
    r"""Projection of :math:`\mathbf{V}` onto the simplex with cardinality constraint (maximum number of non-zero elements) and then scaled by `z`.

    .. math::
        P\left(\mathbf{V}, max_nz, z\right) = \mathop{\arg \min}_{\substack{\mathbf{y} >= 0 \\ \sum_i \mathbf{y}_i = z} \\ ||p||_0 \le \text{max_nz}} \quad \|\mathbf{y} - \mathbf{V}\|^2

    Parameters
    ----------
    V: ndarray, rank 2
    z: float or array
        If array, len(z) must be compatible with :math:`\mathbf{V}`
    axis: None or int
        - axis=None: project :math:`\mathbf{V}` by :math:`P(\mathbf{V}.\mathrm{ravel}(), max_nz, z)`
        - axis=1: project each :math:`\mathbf{V}_i` by :math:`P(\mathbf{V}_i, max_nz, z_i)`
        - axis=0: project each :math:`\mathbf{V}_{:, j}` by :math:`P(\mathbf{V}_{:, j}, max_nz, z_j)`

    Returns
    -------
    projection: ndarray, shape :math:`\mathbf{V}`.shape

    References:
        Sparse projections onto the simplex
        Anastasios Kyrillidis, Stephen Becker, Volkan Cevher and, Christoph Koch
        ICML 2013
        https://arxiv.org/abs/1206.1529
    """
    if axis == 1:
        max_nz_indices = np.argpartition(
            V,
            kth=-max_nz,
            axis=1)[:, -max_nz:]
        # Record nonzero column indices in a descending order
        max_nz_indices = max_nz_indices[:, ::-1]

        row_indices = np.arange(V.shape[0])[:, np.newaxis]

        # Extract the top max_nz values for each row
        # and then project to simplex.
        U = V[row_indices, max_nz_indices]
        z = np.ones(len(U)) * z
        cssv = np.cumsum(U, axis=1) - z[:, np.newaxis]
        ind = np.arange(max_nz) + 1
        cond = U - cssv / ind > 0
        rho = np.count_nonzero(cond, axis=1)
        theta = cssv[np.arange(len(U)), rho - 1] / rho
        nz_projection = np.maximum(U - theta[:, np.newaxis], 0)

        # Put the projection of max_nz_values to their original column indices
        # while keeping other values zero.
        sparse_projection = np.zeros_like(V)
        sparse_projection[row_indices, max_nz_indices] = nz_projection
        return sparse_projection

    elif axis == 0:
        return projection_sparse_simplex(V.T, max_nz, z, axis=1).T

    else:
        V = V.ravel().reshape(1, -1)
        return projection_sparse_simplex(V, max_nz, z, axis=1).ravel()


class SparsityConstrained(ot.smooth.Regularization):
    """ Squared L2 regularization with sparsity constraints """

    def __init__(self, max_nz, gamma=1.0):
        self.max_nz = max_nz
        self.gamma = gamma

    def delta_Omega(self, X):
        # For each column of X, find entries that are not among the top max_nz.
        non_top_indices = np.argpartition(
            -X, self.max_nz, axis=0)[self.max_nz:]
        # Set these entries to -inf.
        X[non_top_indices, np.arange(X.shape[1])] = -np.inf
        max_X = np.maximum(X, 0)
        val = np.sum(max_X ** 2, axis=0) / (2 * self.gamma)
        G = max_X / self.gamma
        return val, G

    def max_Omega(self, X, b):
        # Project the scaled X onto the simplex with sparsity constraint.
        G = projection_sparse_simplex(
            X / (b * self.gamma), self.max_nz, axis=0)
        val = np.sum(X * G, axis=0)
        val -= 0.5 * self.gamma * b * np.sum(G * G, axis=0)
        return val, G

    def Omega(self, T):
        return 0.5 * self.gamma * np.sum(T ** 2)


def sparsity_constrained_ot_dual(
        a, b, M, reg, max_nz,
        method="L-BFGS-B", stopThr=1e-9,
        numItermax=500, verbose=False, log=False):
    r"""
    Solve the sparsity-constrained OT problem in the dual and return the OT matrix.

    The function solves the sparsity-contrained OT in dual formulation in
    :ref:`[50] <references-sparsity-constrained-ot-dual>`.


    Parameters
    ----------
    a : np.ndarray (ns,)
        samples weights in the source domain
    b : np.ndarray (nt,) or np.ndarray (nt,nbb)
        samples in the target domain, compute sinkhorn with multiple targets
        and fixed :math:`\mathbf{M}` if :math:`\mathbf{b}` is a matrix
        (return OT loss + dual variables in log)
    M : np.ndarray (ns,nt)
        loss matrix
    reg : float
        Regularization term >0
    max_nz: int
        Maximum number of non-zero entries permitted in each column of the
        optimal transport matrix.
    method : str
        Solver to use for scipy.optimize.minimize
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshold on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True


    Returns
    -------
    gamma : (ns, nt) ndarray
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters


    .. _references-sparsity-constrained-ot-dual:
    References
    ----------
    .. [50] Liu, T., Puigcerver, J., & Blondel, M. (2023). Sparsity-constrained optimal transport. Proceedings of the Eleventh International Conference on Learning Representations (ICLR).

    See Also
    --------
    ot.lp.emd : Unregularized OT
    ot.sinhorn : Entropic regularized OT
    ot.smooth : Entropic regularized and squared l2 regularized OT
    ot.optim.cg : General regularized OT

    """

    nx = get_backend(a, b, M)
    max_nz = min(max_nz, M.shape[0])
    regul = SparsityConstrained(gamma=reg, max_nz=max_nz)

    a0, b0, M0 = a, b, M

    # convert to humpy
    a, b, M = nx.to_numpy(a, b, M)

    # solve dual
    alpha, beta, res = ot.smooth.solve_dual(
        a, b, M, regul,
        max_iter=numItermax,
        tol=stopThr, verbose=verbose)

    # reconstruct transport matrix
    G = nx.from_numpy(ot.smooth.get_plan_from_dual(alpha, beta, M, regul),
                      type_as=M0)

    if log:
        log = {'alpha': nx.from_numpy(alpha, type_as=a0),
               'beta': nx.from_numpy(beta, type_as=b0), 'res': res}
        return G, log
    else:
        return G


def sparsity_constrained_ot_semi_dual(
        a, b, M, reg, max_nz,
        method="L-BFGS-B", stopThr=1e-9,
        numItermax=500, verbose=False, log=False):
    r"""
    Solve the regularized OT problem in the semi-dual and return the OT matrix

    The function solves the sparsity-contrained OT in semi-dual formulation in
    :ref:`[50] <references-sparsity-constrained-ot-semi-dual>`.


    Parameters
    ----------
    a : np.ndarray (ns,)
        samples weights in the source domain
    b : np.ndarray (nt,) or np.ndarray (nt,nbb)
        samples in the target domain, compute sinkhorn with multiple targets
        and fixed:math:`\mathbf{M}` if :math:`\mathbf{b}` is a matrix
        (return OT loss + dual variables in log)
    M : np.ndarray (ns,nt)
        loss matrix
    reg : float
        Regularization term >0
    max_nz: int
        Maximum number of non-zero entries permitted in each column of the optimal transport matrix.
    method : str
        Solver to use for scipy.optimize.minimize
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshold on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True


    Returns
    -------
    gamma : (ns, nt) ndarray
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters


    .. _references-sparsity-constrained-ot-semi-dual:
    References
    ----------
    .. [50] Liu, T., Puigcerver, J., & Blondel, M. (2023). Sparsity-constrained optimal transport. Proceedings of the Eleventh International Conference on Learning Representations (ICLR).

    See Also
    --------
    ot.lp.emd : Unregularized OT
    ot.sinhorn : Entropic regularized OT
    ot.smooth : Entropic regularized and squared l2 regularized OT
    ot.optim.cg : General regularized OT

    """

    max_nz = min(max_nz, M.shape[0])
    regul = SparsityConstrained(gamma=reg, max_nz=max_nz)
    # solve dual
    alpha, res = ot.smooth.solve_semi_dual(
        a, b, M, regul, max_iter=numItermax,
        tol=stopThr, verbose=verbose)

    # reconstruct transport matrix
    G = ot.smooth.get_plan_from_semi_dual(alpha, b, M, regul)

    if log:
        log = {'alpha': alpha, 'res': res}
        return G, log
    else:
        return G
