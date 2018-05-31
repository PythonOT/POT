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

"""
Implementation of
Smooth and Sparse Optimal Transport.
Mathieu Blondel, Vivien Seguy, Antoine Rolet.
In Proc. of AISTATS 2018.
https://arxiv.org/abs/1710.06276
"""

import numpy as np
from scipy.optimize import minimize


def projection_simplex(V, z=1, axis=None):
    """ Projection of x onto the simplex, scaled by z
    
        P(x; z) = argmin_{y >= 0, sum(y) = z} ||y - x||^2
    z: float or array
        If array, len(z) must be compatible with V
    axis: None or int
        - axis=None: project V by P(V.ravel(); z)
        - axis=1: project each V[i] by P(V[i]; z[i])
        - axis=0: project each V[:, j] by P(V[:, j]; z[j])
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
        """
        Compute delta_Omega(X[:, j]) for each X[:, j].
        delta_Omega(x) = sup_{y >= 0} y^T x - Omega(y).
        
        Parameters
        ----------
        X: array, shape = len(a) x len(b)
            Input array.
            
        Returns
        -------
        v: array, len(b)
            Values: v[j] = delta_Omega(X[:, j])
        G: array, len(a) x len(b)
            Gradients: G[:, j] = nabla delta_Omega(X[:, j])
        """
        raise NotImplementedError

    def max_Omega(X, b):
        """
        Compute max_Omega_j(X[:, j]) for each X[:, j].
        max_Omega_j(x) = sup_{y >= 0, sum(y) = 1} y^T x - Omega(b[j] y) / b[j].
        
        Parameters
        ----------
        X: array, shape = len(a) x len(b)
            Input array.
            
        Returns
        -------
        v: array, len(b)
            Values: v[j] = max_Omega_j(X[:, j])
        G: array, len(a) x len(b)
            Gradients: G[:, j] = nabla max_Omega_j(X[:, j])
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


def dual_obj_grad(alpha, beta, a, b, C, regul):
    """
    Compute objective value and gradients of dual objective.
    
    Parameters
    ----------
    alpha: array, shape = len(a)
    beta: array, shape = len(b)
        Current iterate of dual potentials.
    a: array, shape = len(a)
    b: array, shape = len(b)
        Input histograms (should be non-negative and sum to 1).
    C: array, shape = len(a) x len(b)
        Ground cost matrix.
    regul: Regularization object
        Should implement a delta_Omega(X) method.
        
    Returns
    -------
    obj: float
        Objective value (higher is better).
    grad_alpha: array, shape = len(a)
        Gradient w.r.t. alpha.
    grad_beta: array, shape = len(b)
        Gradient w.r.t. beta.
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


def solve_dual(a, b, C, regul, method="L-BFGS-B", tol=1e-3, max_iter=500):
    """
    Solve the "smoothed" dual objective.
    
    Parameters
    ----------
    a: array, shape = len(a)
    b: array, shape = len(b)
        Input histograms (should be non-negative and sum to 1).
    C: array, shape = len(a) x len(b)
        Ground cost matrix.
    regul: Regularization object
        Should implement a delta_Omega(X) method.
    method: str
        Solver to be used (passed to `scipy.optimize.minimize`).
    tol: float
        Tolerance parameter.
    max_iter: int
        Maximum number of iterations.
        
    Returns
    -------
    alpha: array, shape = len(a)
    beta: array, shape = len(b)
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
                   tol=tol, options=dict(maxiter=max_iter, disp=False))

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
    C: array, shape = len(a) x len(b)
        Ground cost matrix.
    regul: Regularization object
        Should implement a max_Omega(X) method.
        
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


def solve_semi_dual(a, b, C, regul, method="L-BFGS-B", tol=1e-3, max_iter=500):
    """
    Solve the "smoothed" semi-dual objective.
    
    Parameters
    ----------
    a: array, shape = len(a)
    b: array, shape = len(b)
        Input histograms (should be non-negative and sum to 1).
    C: array, shape = len(a) x len(b)
        Ground cost matrix.
    regul: Regularization object
        Should implement a max_Omega(X) method.
    method: str
        Solver to be used (passed to `scipy.optimize.minimize`).
    tol: float
        Tolerance parameter.
    max_iter: int
        Maximum number of iterations.
        
    Returns
    -------
    alpha: array, shape = len(a)
        Semi-dual potentials.
    """

    def _func(alpha):
        obj, grad = semi_dual_obj_grad(alpha, a, b, C, regul)
        # We need to maximize the semi-dual.
        return -obj, -grad

    alpha_init = np.zeros(len(a))

    res = minimize(_func, alpha_init, method=method, jac=True,
                   tol=tol, options=dict(maxiter=max_iter, disp=False))

    return res.x, res


def get_plan_from_dual(alpha, beta, C, regul):
    """
    Retrieve optimal transportation plan from optimal dual potentials.
    
    Parameters
    ----------
    alpha: array, shape = len(a)
    beta: array, shape = len(b)
        Optimal dual potentials.
    C: array, shape = len(a) x len(b)
        Ground cost matrix.
    regul: Regularization object
        Should implement a delta_Omega(X) method.
        
    Returns
    -------
    T: array, shape = len(a) x len(b)
        Optimal transportation plan.
    """
    X = alpha[:, np.newaxis] + beta - C
    return regul.delta_Omega(X)[1]


def get_plan_from_semi_dual(alpha, b, C, regul):
    """
    Retrieve optimal transportation plan from optimal semi-dual potentials.
    
    Parameters
    ----------
    alpha: array, shape = len(a)
        Optimal semi-dual potentials.
    b: array, shape = len(b)
        Second input histogram (should be non-negative and sum to 1).
    C: array, shape = len(a) x len(b)
        Ground cost matrix.
    regul: Regularization object
        Should implement a delta_Omega(X) method.
        
    Returns
    -------
    T: array, shape = len(a) x len(b)
        Optimal transportation plan.
    """
    X = alpha[:, np.newaxis] - C
    return regul.max_Omega(X, b)[1] * b


def smooth_ot_dual(a, b, M, reg, reg_type='l2', method="L-BFGS-B", stopThr=1e-9,
                   numItermax=500, log=False):
    
    if reg_type.lower()=='l2':
        regul = SquaredL2(gamma=reg)
    elif reg_type.lower() in ['entropic','negentropy','kl']:
        regul = NegEntropy(gamma=reg)    
    
    alpha, beta, res = solve_dual(a, b, M, regul, max_iter=numItermax,tol=stopThr)
    G = get_plan_from_dual(alpha, beta, M, regul)
    
    if log:
        log={'alpha':alpha,beta:'beta','res':res}
        return G, log
    else:
        return G
    
    
    
