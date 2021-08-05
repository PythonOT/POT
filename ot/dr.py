# -*- coding: utf-8 -*-
"""
Dimension reduction with OT


.. warning::
    Note that by default the module is not imported in :mod:`ot`. In order to
    use it you need to explicitely import :mod:`ot.dr`

"""

# Author: Remi Flamary <remi.flamary@unice.fr>
#         Minhui Huang <mhhuang@ucdavis.edu>
#
# License: MIT License

from scipy import linalg
import autograd.numpy as np
from pymanopt.manifolds import Stiefel
from pymanopt import Problem
from pymanopt.solvers import SteepestDescent, TrustRegions


def dist(x1, x2):
    """ Compute squared euclidean distance between samples (autograd)
    """
    x1p2 = np.sum(np.square(x1), 1)
    x2p2 = np.sum(np.square(x2), 1)
    return x1p2.reshape((-1, 1)) + x2p2.reshape((1, -1)) - 2 * np.dot(x1, x2.T)


def sinkhorn(w1, w2, M, reg, k):
    """Sinkhorn algorithm with fixed number of iteration (autograd)
    """
    K = np.exp(-M / reg)
    ui = np.ones((M.shape[0],))
    vi = np.ones((M.shape[1],))
    for i in range(k):
        vi = w2 / (np.dot(K.T, ui))
        ui = w1 / (np.dot(K, vi))
    G = ui.reshape((M.shape[0], 1)) * K * vi.reshape((1, M.shape[1]))
    return G


def split_classes(X, y):
    """split samples in X by classes in y
    """
    lstsclass = np.unique(y)
    return [X[y == i, :].astype(np.float32) for i in lstsclass]


def fda(X, y, p=2, reg=1e-16):
    """Fisher Discriminant Analysis

    Parameters
    ----------
    X : ndarray, shape (n, d)
        Training samples.
    y : ndarray, shape (n,)
        Labels for training samples.
    p : int, optional
        Size of dimensionnality reduction.
    reg : float, optional
        Regularization term >0 (ridge regularization)

    Returns
    -------
    P : ndarray, shape (d, p)
        Optimal transportation matrix for the given parameters
    proj : callable
        projection function including mean centering
    """

    mx = np.mean(X)
    X -= mx.reshape((1, -1))

    # data split between classes
    d = X.shape[1]
    xc = split_classes(X, y)
    nc = len(xc)

    p = min(nc - 1, p)

    Cw = 0
    for x in xc:
        Cw += np.cov(x, rowvar=False)
    Cw /= nc

    mxc = np.zeros((d, nc))

    for i in range(nc):
        mxc[:, i] = np.mean(xc[i])

    mx0 = np.mean(mxc, 1)
    Cb = 0
    for i in range(nc):
        Cb += (mxc[:, i] - mx0).reshape((-1, 1)) * \
            (mxc[:, i] - mx0).reshape((1, -1))

    w, V = linalg.eig(Cb, Cw + reg * np.eye(d))

    idx = np.argsort(w.real)

    Popt = V[:, idx[-p:]]

    def proj(X):
        return (X - mx.reshape((1, -1))).dot(Popt)

    return Popt, proj


def wda(X, y, p=2, reg=1, k=10, solver=None, maxiter=100, verbose=0, P0=None):
    r"""
    Wasserstein Discriminant Analysis [11]_

    The function solves the following optimization problem:

    .. math::
        P = \\text{arg}\min_P \\frac{\\sum_i W(PX^i,PX^i)}{\\sum_{i,j\\neq i} W(PX^i,PX^j)}

    where :

    - :math:`P` is a linear projection operator in the Stiefel(p,d) manifold
    - :math:`W` is entropic regularized Wasserstein distances
    - :math:`X^i` are samples in the dataset corresponding to class i

    Parameters
    ----------
    X : ndarray, shape (n, d)
        Training samples.
    y : ndarray, shape (n,)
        Labels for training samples.
    p : int, optional
        Size of dimensionnality reduction.
    reg : float, optional
        Regularization term >0 (entropic regularization)
    solver : None | str, optional
        None for steepest descent or 'TrustRegions' for trust regions algorithm
        else should be a pymanopt.solvers
    P0 : ndarray, shape (d, p)
        Initial starting point for projection.
    verbose : int, optional
        Print information along iterations.

    Returns
    -------
    P : ndarray, shape (d, p)
        Optimal transportation matrix for the given parameters
    proj : callable
        Projection function including mean centering.

    References
    ----------
    .. [11] Flamary, R., Cuturi, M., Courty, N., & Rakotomamonjy, A. (2016).
            Wasserstein Discriminant Analysis. arXiv preprint arXiv:1608.08063.
    """  # noqa

    mx = np.mean(X)
    X -= mx.reshape((1, -1))

    # data split between classes
    d = X.shape[1]
    xc = split_classes(X, y)
    # compute uniform weighs
    wc = [np.ones((x.shape[0]), dtype=np.float32) / x.shape[0] for x in xc]

    def cost(P):
        # wda loss
        loss_b = 0
        loss_w = 0

        for i, xi in enumerate(xc):
            xi = np.dot(xi, P)
            for j, xj in enumerate(xc[i:]):
                xj = np.dot(xj, P)
                M = dist(xi, xj)
                G = sinkhorn(wc[i], wc[j + i], M, reg, k)
                if j == 0:
                    loss_w += np.sum(G * M)
                else:
                    loss_b += np.sum(G * M)

        # loss inversed because minimization
        return loss_w / loss_b

    # declare manifold and problem
    manifold = Stiefel(d, p)
    problem = Problem(manifold=manifold, cost=cost)

    # declare solver and solve
    if solver is None:
        solver = SteepestDescent(maxiter=maxiter, logverbosity=verbose)
    elif solver in ['tr', 'TrustRegions']:
        solver = TrustRegions(maxiter=maxiter, logverbosity=verbose)

    Popt = solver.solve(problem, x=P0)

    def proj(X):
        return (X - mx.reshape((1, -1))).dot(Popt)

    return Popt, proj


def prw(X, Y, a, b, tau, U0 = None,  reg=0.1, k=2, stopThr=1e-3, maxiter=100, verbose=0):
    r"""
    Projection Robust Wasserstein Distance _[12],[13]

    The function solves the following optimization problem:

    .. math::
        max_{U \in St(d, k)} min_{\pi \in \Pi(\mu,\nu)} \sum_{i,j} \pi_{i,j}*||U^T(x_i - y_j)||^2 - reg * H(\pi)
    
    - :math:`U` is a linear projection operator in the Stiefel(d, k) manifold
    - :math:`H(\pi)` is entropy regularizer
    - :math:`x_i`, `y_j` are samples of measures \mu and \nu respectively
    
    Parameters
    ----------
    X : ndarray, shape (n, d)
        Samples from measure \mu
    Y : ndarray, shape (n, d)
        Samples from measure \nu
    a : ndarray, shape (n, 1)
        weights for measure \mu
    b : ndarray, shape (n, 1)
        weights for measure \nu
    tau : float
        stepsize for Riemannian Gradient Descent
    U0 : ndarray, shape (d, p)
        Initial starting point for projection.
    reg : float, optional
        Regularization term >0 (entropic regularization)
    k : int
        Subspace dimension
    stopThr : float, optional
        Accuracy
    verbose : int, optional
        Print information along iterations.

    Returns
    -------
    pi : ndarray, shape (n, n)
        Optimal transportation matrix for the given parameters
    U : ndarray, shape (d, k)
        Projection operator.

    References
    ----------
    .. [12] Paty, F. & Cuturi, M. (2019).
            Subspace Robust Wasserstein Distance, ICML
    .. [13] Huang, M. , Ma S. & Lai L. (2021).
            A Riemannian Block Coordinate Descent Method for Computing 
            the Projection Robust Wasserstein Distance, ICML.
    """  # noqa

    # initialization
    n, d = X.shape
    m, d = Y.shape
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    u = np.ones(n) / n
    v = np.ones(m) / m
    ones = np.ones((n, m))

    assert d > k

    if U0 is None:
        U = np.random.randn(d, k)
        U, _ = np.linalg.qr(U)
    else:
        U = U0

    def Vpi(X, Y, a, b, pi):
        # Return the second order matrix of the displacements: sum_ij { (pi)_ij (X_i-Y_j)(X_i-Y_j)^T }.
        A = X.T.dot(pi).dot(Y)
        return X.T.dot(np.diag(a)).dot(X) + Y.T.dot(np.diag(np.sum(pi, 0))).dot(Y) - A - A.T

    err = 1
    iter = 0

    while err > stopThr and iter < maxiter:

        # Projected cost matrix
        UUT = U.dot(U.T)
        M = np.diag(np.diag(X.dot(UUT.dot(X.T)))).dot(ones) + ones.dot(
            np.diag(np.diag(Y.dot(UUT.dot(Y.T))))) - 2 * X.dot(UUT.dot(Y.T))

        A = np.empty(M.shape, dtype=M.dtype)
        np.divide(M, -reg, out=A)
        np.exp(A, out=A)

        # Sinkhorn update
        Ap = (1 / a).reshape(-1, 1) * A
        AtransposeU = np.dot(A.T, u)
        v = np.divide(b, AtransposeU)
        u = 1. / np.dot(Ap, v)
        pi = u.reshape((-1, 1)) * A * v.reshape((1, -1))

        V = Vpi(X, Y, a, b, pi)

        # Riemannian gradient descent
        G = 2 / reg * V.dot(U)
        GTU = G.T.dot(U)
        xi = G - U.dot(GTU + GTU.T) / 2  # Riemannian gradient
        U, _ = np.linalg.qr(U + tau * xi) # Retraction by QR decomposition

        grad_norm = np.linalg.norm(xi)
        err = max(reg*grad_norm, np.linalg.norm(np.sum(pi, 0) - b, 1))

        f_val = np.trace(U.T.dot(V.dot(U)))
        if verbose:
            print('RBCD Iteration: ', iter, ' error', err, '\t fval: ', f_val)

        iter = iter + 1


    return pi, U
