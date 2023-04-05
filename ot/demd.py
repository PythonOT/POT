# -*- coding: utf-8 -*-
"""
DEMD solvers for optimal transport
"""

# Author: Ronak Mehta <ronakrm@cs.wisc.edu> 
#         Xizheng Yu <xyu354@wisc.edu>
#
# License: MIT License

import numpy as np
from .backend import get_backend


def greedy_primal_dual(aa, verbose=False):
    r"""
    The greedy algorithm that solves both primal and dual generalized Earth
    mover’s programs.

    The algorithm accepts $d$ distributions (i.e., histograms) :math:`p_{1},
    \ldots, p_{d} \in \mathbb{R}_{+}^{n}` with :math:`e^{\prime} p_{j}=1`
    for all :math:`j \in[d]`. Although the algorithm states that all
    histograms have the same number of bins, the algorithm can be easily
    adapted to accept as inputs :math:`p_{i} \in \mathbb{R}_{+}^{n_{i}}`
    with :math:`n_{i} \neq n_{j}`.

    Parameters
    ----------
    aa : list of numpy arrays
        The input arrays defining the optimization problem. They must have the
        same shape.
    verbose : bool, optional
        If True, print debugging information during the execution of the
        algorithm. Default is False.

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

    References
    ----------
    .. [51] Jeffery Kline. Properties of the d-dimensional earth mover’s
        problem. Discrete Applied Mathematics, 265: 128–141, 2019.

    Examples
    --------
    >>> import numpy as np
    >>> aa = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]
    >>> result = greedy_primal_dual(aa)
    >>> result['primal objective']
    -12
    """
    # function body here

    def OBJ(i):
        return max(i) - min(i)
    
    # print(f"aa type is: {type(aa)}")
    nx = get_backend(aa)

    AA = [nx.copy(_) for _ in aa]

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

    dualobj = sum([_.dot(_d) for _, _d in zip(aa, dual)])

    return {'x': xx, 'primal objective': obj,
            'dual': dual, 'dual objective': dualobj}


def demd(x, d, n, return_dual_vars=False):
    r"""
    Solver of our proposed method: d−Dimensional Earch Mover’s Distance (DEMD).

    Parameters
    ----------
    x : numpy array, shape (d * n, )
        The input vector containing coordinates of n points in d dimensions.
    d : int
        The number of dimensions of the points.
    n : int
        The number of points.
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

    """
    
    # function body here
    nx = get_backend(x)
    log = greedy_primal_dual(x)

    if return_dual_vars:
        dual = log['dual']
        return_dual = np.array(dual)
        dualobj = log['dual objective']
        return log['primal objective'], return_dual, dualobj
    else:
        return log['primal objective']


def demd_minimize(f, x, d, n, vecsize, niters=100, lr=0.1, print_rate=100):
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
    x : ndarray, shape (d, n)
        The initial point for the optimization algorithm.
    d : int
        The number of rows in the matrix x.
    n : int
        The number of columns in the matrix x.
    vecsize : int
        The size of the vectors that make up the columns of x.
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
    nx = get_backend(x)
    
    def dualIter(f, x, d, n, vecsize, lr):
        funcval, grad, _ = f(x, d, n, return_dual_vars=True)
        xnew = nx.reshape(x, (d, n)) - grad * lr
        return funcval, xnew, grad

    def renormalize(x, d, n, vecsize):
        x = nx.reshape(x, (d, n))
        for i in range(x.shape[0]):
            if min(x[i, :]) < 0:
                x[i, :] -= min(x[i, :])
            x[i, :] /= nx.sum(x[i, :])
        return x

    def listify(x):
        return [x[i, :] for i in range(x.shape[0])]

    # print(f"x type is {type(x)}")
    funcval, _, grad = dualIter(f, x, d, n, vecsize, lr)
    gn = nx.norm(grad)

    print(f'Inital:\t\tObj:\t{funcval:.4f}\tGradNorm:\t{gn:.4f}')

    for i in range(niters):

        x = renormalize(x, d, n, vecsize)
        funcval, x, grad = dualIter(f, x, d, n, vecsize, lr)
        gn = nx.norm(grad)

        if i % print_rate == 0:
            print(f'Iter {i:2.0f}:\tObj:\t{funcval:.4f}\tGradNorm:\t{gn:.4f}')

    x = renormalize(x, d, n, vecsize)
    return listify(nx.reshape(x, (d, n)))