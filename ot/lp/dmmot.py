# -*- coding: utf-8 -*-
"""
DEMD solvers for optimal transport
"""

# Author: Ronak Mehta <ronakrm@cs.wisc.edu>
#         Xizheng Yu <xyu354@wisc.edu>
#
# License: MIT License

import numpy as np
from ..backend import get_backend

# M -> obj

def greedy_primal_dual(A, verbose=False):
    r"""
    The greedy algorithm that solves both primal and dual generalized Earth
    mover’s programs.

    The algorithm accepts :math:`d` distributions (i.e., histograms) :math:`p_{1},
    \ldots, p_{d} \in \mathbb{R}_{+}^{n}` with :math:`e^{\prime} p_{j}=1`
    for all :math:`j \in[d]`. Although the algorithm states that all
    histograms have the same number of bins, the algorithm can be easily
    adapted to accept as inputs :math:`p_{i} \in \mathbb{R}_{+}^{n_{i}}`
    with :math:`n_{i} \neq n_{j}`.

    Parameters
    ----------
    A : list of numpy arrays -> nd array
        The input arrays are list of distributions
    

    Returns
    -------
    dict : dic
        A dictionary containing the solution of the primal-dual problem:
        - 'x': a dictionary that maps tuples of indices to the corresponding
          primal variables. The tuples are the indices of the entries that are
          set to their minimum value during the algorithm.
        - 'primal objective': a float, the value of the objective function
          evaluated at the solution.
        - 'dual': a list of numpy arrays, the dual variables corresponding to
          the input arrays. The i-th element of the list is the dual variable
          corresponding to the i-th dimension of the input arrays.
        - 'dual objective': a float, the value of the dual objective function
          evaluated at the solution.

    Examples
    --------
    >>> import numpy as np
    >>> A = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]
    >>> result = greedy_primal_dual(A)
    >>> result['primal objective']
    -12
    """

    pass


def discrete_mmot(A, verbose=False, log=False):
    r"""
    Solver of our proposed method: d−Dimensional Earch Mover’s Distance (DEMD).
    
    multi marginal optimal transport

    Parameters
    ----------
    A : numpy array, shape (d * n, )
        The input vector containing coordinates of n points in d dimensions.
    d : int
        The number of dimensions of the points.
    n : int
        The number of points.
    verbose : bool, optional
        If True, print debugging information during the execution of the
        algorithm. Default is False.
    return_dual_vars : bool, optional
        If True, also return the dual variables and the dual objective value of
        the DEMD problem. Default is False.

    Returns
    -------
    primal_obj : float
        the value of the primal objective function evaluated at the solution.
    dual_vars : numpy array, shape (d, n-1), optional
        the values of the dual variables corresponding to the input points.
        The i-th column of the array corresponds to the i-th point.
    dual_obj : float, optional
        the value of the dual objective function evaluated at the solution.

    References
    ----------
    .. [50] Ronak Mehta, Jeffery Kline, Vishnu Suresh Lokhande, Glenn Fung, &
        Vikas Singh (2023). Efficient Discrete Multi Marginal Optimal
        Transport Regularization. In The Eleventh International
        Conference on Learning Representations.
    .. [51] Jeffery Kline. Properties of the d-dimensional earth mover’s
        problem. Discrete Applied Mathematics, 265: 128–141, 2019.
    """
    
    def OBJ(i):
        return max(i) - min(i)

    # print(f"A type is: {type(A)}")
    nx = get_backend(A)

    AA = [nx.copy(_) for _ in A]

    dims = tuple([len(_) for _ in AA])
    xx = {}
    dual = [nx.zeros(d) for d in dims]

    idx = [0, ] * len(AA)
    obj = 0
    if verbose:
        print('i minval oldidx\t\tobj\t\tvals')
    while all([i < _ for _, i in zip(dims, idx)]):
        vals = [v[i] for v, i in zip(AA, idx)]
        minval = min(vals)
        i = vals.index(minval)
        xx[tuple(idx)] = minval
        obj += (OBJ(idx)) * minval
        for v, j in zip(AA, idx):
            v[j] -= minval
        oldidx = nx.copy(idx)
        idx[i] += 1
        if idx[i] < dims[i]:
            dual[i][idx[i]] += OBJ(idx) - OBJ(oldidx) + dual[i][idx[i]-1]
        if verbose:
            print(i, minval, oldidx, obj, '\t', vals)

    # the above terminates when any entry in idx equals the corresponding
    # value in dims this leaves other dimensions incomplete; the remaining
    # terms of the dual solution must be filled-in
    for _, i in enumerate(idx):
        try:
            dual[_][i:] = dual[_][i]
        except Exception:
            pass

    dualobj = nx.sum([nx.dot(arr, dual_arr) for arr, dual_arr in zip(A, dual)])

    log_dict = {'A': xx, 
           'primal objective': obj,
           'dual': dual, 
           'dual objective': dualobj}
    
    if log:
        return obj, log_dict
    else:
        return obj

    # if return_dual_vars:
    #     dual = log['dual']
    #     return_dual = np.array(dual)
    #     dualobj = log['dual objective']
    #     return log['primal objective'], return_dual, log['dual objective']
    # else:
    #     return log['primal objective'], log


def discrete_mmot_converge(A, niters=100, lr=0.1, print_rate=100, log=False):
    r"""
    Minimize a DEMD function using gradient descent.

    Parameters
    ----------
    f : callable
        The objective function to minimize. This function must take as input
        a matrix x of shape (d, n) and return a scalar value representing
        the objective function evaluated at x. It may also return a matrix of
        shape (d, n) representing the gradient of the objective function
        with respect to x, and/or any other dual variables needed for the
        optimization algorithm. The signature of this function should be:
        `f(x, d, n, return_dual_vars=False) -> float`
        or
        `f(x, d, n, return_dual_vars=True) -> (float, ndarray, ...)`
    A : ndarray, shape (d, n)
        The initial point for the optimization algorithm.
    niters : int, optional (default=100)
        The maximum number of iterations for the optimization algorithm.
    lr : float, optional (default=0.1)
        The learning rate (step size) for the optimization algorithm.
    print_rate : int, optional (default=100)
        The rate at which to print the objective value and gradient norm
        during the optimization algorithm.

    Returns
    -------
    list of ndarrays, each of shape (n,)
        The optimal solution as a list of n vectors, each of length vecsize.
    """

    # function body here
    nx = get_backend(A)
    d, n = A.shape

    def dualIter(A, lr):
        funcval, log_dict = discrete_mmot(A, log=True)
        grad = np.array(log_dict['dual'])
        A_new = nx.reshape(A, (d, n)) - grad * lr
        # A_new = A - grad * lr
        return funcval, A_new, grad, log_dict

    def renormalize(A):
        A = nx.reshape(A, (d, n))
        for i in range(A.shape[0]):
            if min(A[i, :]) < 0:
                A[i, :] -= min(A[i, :])
            A[i, :] /= nx.sum(A[i, :])
        return A

    def listify(A):
        return [A[i, :] for i in range(A.shape[0])]

    funcval, _, grad, log_dict = dualIter(A, lr)
    gn = nx.norm(grad)

    print(f'Inital:\t\tObj:\t{funcval:.4f}\tGradNorm:\t{gn:.4f}')

    for i in range(niters):

        A = renormalize(A)
        funcval, A, grad, log_dict = dualIter(A, lr)
        gn = nx.norm(grad)

        if i % print_rate == 0:
            print(f'Iter {i:2.0f}:\tObj:\t{funcval:.4f}\tGradNorm:\t{gn:.4f}')

    A = renormalize(A)
    a = listify(nx.reshape(A, (d, n)))
    
    if log:
        return a, log_dict
    else:
        return a
    