"""
Sparsity-constrained optimal transport solvers.

Implementation of :
Sparsity-Constrained Optimal Transport.
Tianlin Liu, Joan Puigcerver, Mathieu Blondel.
In Proc. of AISTATS 2018.
https://arxiv.org/abs/1710.06276

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
        # For each column of X, find top max_nz values and
        # their corresponding indices.
        max_nz_indices = np.argpartition(
            X,
            kth=-self.max_nz,
            axis=0)[-self.max_nz:]
        max_nz_values = X[max_nz_indices, np.arange(X.shape[1])]

        # Project the top max_nz values onto the simplex.
        G_nz_values = ot.smooth.projection_simplex(
            max_nz_values / (b * self.gamma), axis=0)

        # Put the projection of max_nz_values to their original indices
        # and set all other values zero.
        G = np.zeros_like(X)
        G[max_nz_indices, np.arange(X.shape[1])] = G_nz_values
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
