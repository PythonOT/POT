#Copyright (c) 2018, Mathieu Blondel
#All rights reserved.
#
#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:
#
#1. Redistributions of source code must retain the above copyright notice, this
#list of conditions and the following disclaimer.
#
#2. Redistributions in binary form must reproduce the above copyright notice,
#this list of conditions and the following disclaimer in the documentation and/or
#other materials provided with the distribution.
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
#IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
#INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
#NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
#OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
#LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
#OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
#THE POSSIBILITY OF SUCH DAMAGE.

# Author: Mathieu Blondel
#         Remi Flamary <remi.flamary@unice.fr>
#         Tianlin Liu <t.liu@unibas.ch>

"""
Smooth and Sparse (KL an L2 reg.) and sparsity-constrained OT solvers.

Implementation of :
Smooth and Sparse Optimal Transport.
Mathieu Blondel, Vivien Seguy, Antoine Rolet.
In Proc. of AISTATS 2018.
https://arxiv.org/abs/1710.06276

(Original code from https://github.com/mblondel/smooth-ot/)

Sparsity-Constrained Optimal Transport.
Liu, T., Puigcerver, J., & Blondel, M. (2023).
Sparsity-constrained optimal transport.
Proceedings of the Eleventh International Conference on
Learning Representations (ICLR).
https://arxiv.org/abs/2209.15466


[17] Blondel, M., Seguy, V., & Rolet, A. (2018). Smooth and Sparse Optimal
Transport. Proceedings of the Twenty-First International Conference on
Artificial Intelligence and Statistics (AISTATS).

[50] Liu, T., Puigcerver, J., & Blondel, M. (2023).
Sparsity-constrained optimal transport.
Proceedings of the Eleventh International Conference on
Learning Representations (ICLR).

"""

import numpy as np
from scipy.optimize import minimize
from .backend import get_backend
import ot


def projection_simplex(V, z=1, axis=None):
    r""" Projection of :math:`\mathbf{V}` onto the simplex, scaled by `z`

    .. math::
        P\left(\mathbf{V}, z\right) = \mathop{\arg \min}_{\substack{\mathbf{y} >= 0 \\ \sum_i \mathbf{y}_i = z}} \quad \|\mathbf{y} - \mathbf{V}\|^2

    Parameters
    ----------
    V: ndarray, rank 2
    z: float or array
        If array, len(z) must be compatible with :math:`\mathbf{V}`
    axis: None or int
        - axis=None: project :math:`\mathbf{V}` by :math:`P(\mathbf{V}.\mathrm{ravel}(), z)`
        - axis=1: project each :math:`\mathbf{V}_i` by :math:`P(\mathbf{V}_i, z_i)`
        - axis=0: project each :math:`\mathbf{V}_{:, j}` by :math:`P(\mathbf{V}_{:, j}, z_j)`

    Returns
    -------
    projection: ndarray, shape :math:`\mathbf{V}`.shape
    """
    if axis == 1:
        n_features = V.shape[1]
        U = np.sort(V, axis=1)[:, ::-1]
        z = np.ones(len(V)) * z
        cssv = np.cumsum(U, axis=1) - z[:, np.newaxis]
        ind = np.arange(n_features) + 1
        cond = U - cssv / ind > 0
        rho = np.count_nonzero(cond, axis=1)
        theta = cssv[np.arange(len(V)), rho - 1] / rho
        return np.maximum(V - theta[:, np.newaxis], 0)

    elif axis == 0:
        return projection_simplex(V.T, z, axis=1).T

    else:
        V = V.ravel().reshape(1, -1)
        return projection_simplex(V, z, axis=1).ravel()


class Regularization(object):
    r"""Base class for Regularization objects

        Notes
        -----
        This class is not intended for direct use but as apparent for true
        regularization implementation.
    """

    def __init__(self, gamma=1.0):
        """

        Parameters
        ----------
        gamma: float
            Regularization parameter.
            We recover unregularized OT when gamma -> 0.

        """
        self.gamma = gamma

    def delta_Omega(X):
        r"""
        Compute :math:`\delta_\Omega(\mathbf{X}_{:, j})` for each :math:`\mathbf{X}_{:, j}`.

        .. math::
            \delta_\Omega(\mathbf{x}) = \sup_{\mathbf{y} >= 0} \
            \mathbf{y}^T \mathbf{x} - \Omega(\mathbf{y})

        Parameters
        ----------
        X: array, shape = (len(a), len(b))
            Input array.

        Returns
        -------
        v: array, (len(b), )
            Values: :math:`\mathbf{v}_j = \delta_\Omega(\mathbf{X}_{:, j})`
        G: array, (len(a), len(b))
            Gradients: :math:`\mathbf{G}_{:, j} = \nabla \delta_\Omega(\mathbf{X}_{:, j})`
        """
        raise NotImplementedError

    def max_Omega(X, b):
        r"""
        Compute :math:`\mathrm{max}_{\Omega, j}(\mathbf{X}_{:, j})` for each :math:`\mathbf{X}_{:, j}`.

        .. math::
            \mathrm{max}_{\Omega, j}(\mathbf{x}) =
            \sup_{\substack{\mathbf{y} >= 0 \ \sum_i \mathbf{y}_i = 1}}
            \mathbf{y}^T \mathbf{x} - \frac{1}{\mathbf{b}_j} \Omega(\mathbf{b}_j \mathbf{y})

        Parameters
        ----------
        X: array, shape = (len(a), len(b))
            Input array.
        b: array, shape = (len(b), )

        Returns
        -------
        v: array, (len(b), )
            Values: :math:`\mathbf{v}_j = \mathrm{max}_{\Omega, j}(\mathbf{X}_{:, j})`
        G: array, (len(a), len(b))
            Gradients: :math:`\mathbf{G}_{:, j} = \nabla \mathrm{max}_{\Omega, j}(\mathbf{X}_{:, j})`
        """
        raise NotImplementedError

    def Omega(T):
        """
        Compute regularization term.

        Parameters
        ----------
        T: array, shape = len(a) x len(b)
            Input array.

        Returns
        -------
        value: float
            Regularization term.
        """
        raise NotImplementedError


class NegEntropy(Regularization):
    """ NegEntropy regularization """

    def delta_Omega(self, X):
        G = np.exp(X / self.gamma - 1)
        val = self.gamma * np.sum(G, axis=0)
        return val, G

    def max_Omega(self, X, b):
        max_X = np.max(X, axis=0) / self.gamma
        exp_X = np.exp(X / self.gamma - max_X)
        val = self.gamma * (np.log(np.sum(exp_X, axis=0)) + max_X)
        val -= self.gamma * np.log(b)
        G = exp_X / np.sum(exp_X, axis=0)
        return val, G

    def Omega(self, T):
        return self.gamma * np.sum(T * np.log(T))


class SquaredL2(Regularization):
    """ Squared L2 regularization """

    def delta_Omega(self, X):
        max_X = np.maximum(X, 0)
        val = np.sum(max_X ** 2, axis=0) / (2 * self.gamma)
        G = max_X / self.gamma
        return val, G

    def max_Omega(self, X, b):
        G = projection_simplex(X / (b * self.gamma), axis=0)
        val = np.sum(X * G, axis=0)
        val -= 0.5 * self.gamma * b * np.sum(G * G, axis=0)
        return val, G

    def Omega(self, T):
        return 0.5 * self.gamma * np.sum(T ** 2)


class SparsityConstrained(Regularization):
    """ Squared L2 regularization with sparsity constraints """

    def __init__(self, max_nz, gamma=1.0):
        self.max_nz = max_nz
        self.gamma = gamma

    def delta_Omega(self, X):
        # For each column of X, find entries that are not among the top max_nz.
        non_top_indices = np.argpartition(
            -X, self.max_nz, axis=0)[self.max_nz:]
        # Set these entries to -inf.
        if X.ndim == 1:
            X[non_top_indices] = 0.0
        else:
            X[non_top_indices, np.arange(X.shape[1])] = 0.0
        max_X = np.maximum(X, 0)
        val = np.sum(max_X ** 2, axis=0) / (2 * self.gamma)
        G = max_X / self.gamma
        return val, G

    def max_Omega(self, X, b):
        # Project the scaled X onto the simplex with sparsity constraint.
        G = ot.utils.projection_sparse_simplex(
            X / (b * self.gamma), self.max_nz, axis=0)
        val = np.sum(X * G, axis=0)
        val -= 0.5 * self.gamma * b * np.sum(G * G, axis=0)
        return val, G

    def Omega(self, T):
        return 0.5 * self.gamma * np.sum(T ** 2)


def dual_obj_grad(alpha, beta, a, b, C, regul):
    r"""
    Compute objective value and gradients of dual objective.

    Parameters
    ----------
    alpha: array, shape = len(a)
    beta: array, shape = len(b)
        Current iterate of dual potentials.
    a: array, shape = len(a)
    b: array, shape = len(b)
        Input histograms (should be non-negative and sum to 1).
    C: array, shape = (len(a), len(b))
        Ground cost matrix.
    regul: Regularization object
        Should implement a `delta_Omega(X)` method.

    Returns
    -------
    obj: float
        Objective value (higher is better).
    grad_alpha: array, shape = len(a)
        Gradient w.r.t. `alpha`.
    grad_beta: array, shape = len(b)
        Gradient w.r.t. `beta`.
    """
    obj = np.dot(alpha, a) + np.dot(beta, b)
    grad_alpha = a.copy()
    grad_beta = b.copy()

    # X[:, j] = alpha + beta[j] - C[:, j]
    X = alpha[:, np.newaxis] + beta - C

    # val.shape = len(b)
    # G.shape = len(a) x len(b)
    val, G = regul.delta_Omega(X)

    obj -= np.sum(val)
    grad_alpha -= G.sum(axis=1)
    grad_beta -= G.sum(axis=0)

    return obj, grad_alpha, grad_beta


def solve_dual(a, b, C, regul, method="L-BFGS-B", tol=1e-3, max_iter=500,
               verbose=False):
    """
    Solve the "smoothed" dual objective.

    Parameters
    ----------
    a: array, shape = (len(a), )
    b: array, shape = (len(b), )
        Input histograms (should be non-negative and sum to 1).
    C: array, shape = (len(a), len(b))
        Ground cost matrix.
    regul: Regularization object
        Should implement a `delta_Omega(X)` method.
    method: str
        Solver to be used (passed to `scipy.optimize.minimize`).
    tol: float
        Tolerance parameter.
    max_iter: int
        Maximum number of iterations.

    Returns
    -------
    alpha: array, shape = (len(a), )
    beta: array, shape = (len(b), )
        Dual potentials.
    """

    def _func(params):
        # Unpack alpha and beta.
        alpha = params[:len(a)]
        beta = params[len(a):]

        obj, grad_alpha, grad_beta = dual_obj_grad(alpha, beta, a, b, C, regul)

        # Pack grad_alpha and grad_beta.
        grad = np.concatenate((grad_alpha, grad_beta))

        # We need to maximize the dual.
        return -obj, -grad

    # Unfortunately, `minimize` only supports functions whose argument is a
    # vector. So, we need to concatenate alpha and beta.
    alpha_init = np.zeros(len(a))
    beta_init = np.zeros(len(b))
    params_init = np.concatenate((alpha_init, beta_init))

    res = minimize(_func, params_init, method=method, jac=True,
                   tol=tol, options=dict(maxiter=max_iter, disp=verbose))

    alpha = res.x[:len(a)]
    beta = res.x[len(a):]

    return alpha, beta, res


def semi_dual_obj_grad(alpha, a, b, C, regul):
    """
    Compute objective value and gradient of semi-dual objective.

    Parameters
    ----------
    alpha: array, shape = len(a)
        Current iterate of semi-dual potentials.
    a: array, shape = len(a)
    b: array, shape = len(b)
        Input histograms (should be non-negative and sum to 1).
    C: array, shape = (len(a), len(b))
        Ground cost matrix.
    regul: Regularization object
        Should implement a `max_Omega(X)` method.

    Returns
    -------
    obj: float
        Objective value (higher is better).
    grad: array, shape = len(a)
        Gradient w.r.t. alpha.
    """
    obj = np.dot(alpha, a)
    grad = a.copy()

    # X[:, j] = alpha - C[:, j]
    X = alpha[:, np.newaxis] - C

    # val.shape = len(b)
    # G.shape = len(a) x len(b)
    val, G = regul.max_Omega(X, b)

    obj -= np.dot(b, val)
    grad -= np.dot(G, b)

    return obj, grad


def solve_semi_dual(a, b, C, regul, method="L-BFGS-B", tol=1e-3, max_iter=500,
                    verbose=False):
    """
    Solve the "smoothed" semi-dual objective.

    Parameters
    ----------
    a: array, shape = (len(a), )
    b: array, shape = (len(b), )
        Input histograms (should be non-negative and sum to 1).
    C: array, shape = (len(a), len(b))
        Ground cost matrix.
    regul: Regularization object
        Should implement a `max_Omega(X)` method.
    method: str
        Solver to be used (passed to `scipy.optimize.minimize`).
    tol: float
        Tolerance parameter.
    max_iter: int
        Maximum number of iterations.

    Returns
    -------
    alpha: array, shape = (len(a), )
        Semi-dual potentials.
    """

    def _func(alpha):
        obj, grad = semi_dual_obj_grad(alpha, a, b, C, regul)
        # We need to maximize the semi-dual.
        return -obj, -grad

    alpha_init = np.zeros(len(a))

    res = minimize(_func, alpha_init, method=method, jac=True,
                   tol=tol, options=dict(maxiter=max_iter, disp=verbose))

    return res.x, res


def get_plan_from_dual(alpha, beta, C, regul):
    r"""
    Retrieve optimal transportation plan from optimal dual potentials.

    Parameters
    ----------
    alpha: array, shape = len(a)
    beta: array, shape = len(b)
        Optimal dual potentials.
    C: array, shape = (len(a), len(b))
        Ground cost matrix.
    regul: Regularization object
        Should implement a `delta_Omega(X)` method.

    Returns
    -------
    T: array, shape = (len(a), len(b))
        Optimal transportation plan.
    """
    X = alpha[:, np.newaxis] + beta - C
    return regul.delta_Omega(X)[1]


def get_plan_from_semi_dual(alpha, b, C, regul):
    r"""
    Retrieve optimal transportation plan from optimal semi-dual potentials.

    Parameters
    ----------
    alpha: array, shape = len(a)
        Optimal semi-dual potentials.
    b: array, shape = len(b)
        Second input histogram (should be non-negative and sum to 1).
    C: array, shape = (len(a), len(b))
        Ground cost matrix.
    regul: Regularization object
        Should implement a `delta_Omega(X)` method.

    Returns
    -------
    T: array, shape = (len(a), len(b))
        Optimal transportation plan.
    """
    X = alpha[:, np.newaxis] - C
    return regul.max_Omega(X, b)[1] * b


def smooth_ot_dual(a, b, M, reg, reg_type='l2',
                   method="L-BFGS-B", stopThr=1e-9,
                   numItermax=500, verbose=False, log=False, max_nz=None):
    r"""
    Solve the regularized OT problem in the dual and return the OT matrix

    The function solves the smooth relaxed dual formulation (7) in
    :ref:`[17] <references-smooth-ot-dual>`:

    .. math::
        \max_{\alpha,\beta}\quad \mathbf{a}^T\alpha + \mathbf{b}^T\beta -
        \sum_j \delta_\Omega \left(\alpha+\beta_j-\mathbf{m}_j \right)

    where :

    - :math:`\mathbf{m}_j` is the j-th column of the cost matrix
    - :math:`\delta_\Omega` is the convex conjugate of the regularization term :math:`\Omega`
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are source and target weights (sum to 1)

    The OT matrix can is reconstructed from the gradient of :math:`\delta_\Omega`
    (See :ref:`[17] <references-smooth-ot-dual>` Proposition 1).
    The optimization algorithm is using gradient decent (L-BFGS by default).


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
    reg_type : str
        Regularization type, can be the following (default ='l2'):

            - 'kl' : Kullback Leibler (~ Neg-entropy used in sinkhorn
              :ref:`[2] <references-smooth-ot-dual>`)

            - 'l2' : Squared Euclidean regularization
            - 'sparsity_constrained' : Sparsity-constrained regularization [50]
    max_nz : int or None, optional. Used only in the case of reg_type = 'sparsity_constrained' to specify the maximum number of nonzeros per column of the optimal plan;
        not used for other regularization types.
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


    .. _references-smooth-ot-dual:
    References
    ----------
    .. [2] M. Cuturi, Sinkhorn Distances : Lightspeed Computation of Optimal Transport, Advances in Neural Information Processing Systems (NIPS) 26, 2013

    .. [17] Blondel, M., Seguy, V., & Rolet, A. (2018). Smooth and Sparse Optimal Transport. Proceedings of the Twenty-First International Conference on Artificial Intelligence and Statistics (AISTATS).

    .. [50] Liu, T., Puigcerver, J., & Blondel, M. (2023). Sparsity-constrained optimal transport. Proceedings of the Eleventh International Conference on Learning Representations (ICLR).

    See Also
    --------
    ot.lp.emd : Unregularized OT
    ot.sinhorn : Entropic regularized OT
    ot.optim.cg : General regularized OT

    """

    nx = get_backend(a, b, M)

    if reg_type.lower() in ['l2', 'squaredl2']:
        regul = SquaredL2(gamma=reg)
    elif reg_type.lower() in ['entropic', 'negentropy', 'kl']:
        regul = NegEntropy(gamma=reg)
    elif reg_type.lower() in ['sparsity_constrained', 'sparsity-constrained']:
        if not isinstance(max_nz, int):
            raise ValueError(
                f'max_nz {max_nz} must be an integer')
        regul = SparsityConstrained(gamma=reg, max_nz=max_nz)
    else:
        raise NotImplementedError('Unknown regularization')

    a0, b0, M0 = a, b, M
    # convert to humpy
    a, b, M = nx.to_numpy(a, b, M)

    # solve dual
    alpha, beta, res = solve_dual(a, b, M, regul, max_iter=numItermax,
                                  tol=stopThr, verbose=verbose)

    # reconstruct transport matrix
    G = nx.from_numpy(get_plan_from_dual(alpha, beta, M, regul), type_as=M0)

    if log:
        log = {'alpha': nx.from_numpy(alpha, type_as=a0), 'beta': nx.from_numpy(beta, type_as=b0), 'res': res}
        return G, log
    else:
        return G


def smooth_ot_semi_dual(a, b, M, reg, reg_type='l2', max_nz=None,
                        method="L-BFGS-B", stopThr=1e-9,
                        numItermax=500, verbose=False, log=False):
    r"""
    Solve the regularized OT problem in the semi-dual and return the OT matrix

    The function solves the smooth relaxed dual formulation (10) in
    :ref:`[17] <references-smooth-ot-semi-dual>`:

    .. math::
        \max_{\alpha}\quad \mathbf{a}^T\alpha- \mathrm{OT}_\Omega^*(\alpha, \mathbf{b})

    where :

    .. math::
        \mathrm{OT}_\Omega^*(\alpha,b)=\sum_j \mathbf{b}_j

    - :math:`\mathbf{m}_j` is the j-th column of the cost matrix
    - :math:`\mathrm{OT}_\Omega^*(\alpha,b)` is defined in Eq. (9) in
      :ref:`[17] <references-smooth-ot-semi-dual>`
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are source and target weights (sum to 1)

    The OT matrix can is reconstructed using :ref:`[17] <references-smooth-ot-semi-dual>` Proposition 2.
    The optimization algorithm is using gradient decent (L-BFGS by default).


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
    reg_type : str
        Regularization type, can be the following (default ='l2'):

            - 'kl' : Kullback Leibler (~ Neg-entropy used in sinkhorn
              :ref:`[2] <references-smooth-ot-semi-dual>`)

            - 'l2' : Squared Euclidean regularization
            - 'sparsity_constrained' : Sparsity-constrained regularization [50]
    max_nz : int or None, optional. Used only in the case of reg_type = 'sparsity_constrained' to specify the maximum number of nonzeros per column of the optimal plan;
        not used for other regularization types.
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


    .. _references-smooth-ot-semi-dual:
    References
    ----------
    .. [2] M. Cuturi, Sinkhorn Distances : Lightspeed Computation of Optimal Transport, Advances in Neural Information Processing Systems (NIPS) 26, 2013

    .. [17] Blondel, M., Seguy, V., & Rolet, A. (2018). Smooth and Sparse Optimal Transport. Proceedings of the Twenty-First International Conference on Artificial Intelligence and Statistics (AISTATS).

    .. [50] Liu, T., Puigcerver, J., & Blondel, M. (2023). Sparsity-constrained optimal transport. Proceedings of the Eleventh International Conference on Learning Representations (ICLR).

    See Also
    --------
    ot.lp.emd : Unregularized OT
    ot.sinhorn : Entropic regularized OT
    ot.optim.cg : General regularized OT

    """
    if reg_type.lower() in ['l2', 'squaredl2']:
        regul = SquaredL2(gamma=reg)
    elif reg_type.lower() in ['entropic', 'negentropy', 'kl']:
        regul = NegEntropy(gamma=reg)
    elif reg_type.lower() in ['sparsity_constrained', 'sparsity-constrained']:
        if not isinstance(max_nz, int):
            raise ValueError(
                f'max_nz {max_nz} must be an integer')
        regul = SparsityConstrained(gamma=reg, max_nz=max_nz)
    else:
        raise NotImplementedError('Unknown regularization')

    # solve dual
    alpha, res = solve_semi_dual(a, b, M, regul, max_iter=numItermax,
                                 tol=stopThr, verbose=verbose)

    # reconstruct transport matrix
    G = get_plan_from_semi_dual(alpha, b, M, regul)

    if log:
        log = {'alpha': alpha, 'res': res}
        return G, log
    else:
        return G
