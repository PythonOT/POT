# -*- coding: utf-8 -*-
"""
d-MMOT solvers for optimal transport
"""

# Author: Ronak Mehta <ronakrm@cs.wisc.edu>
#         Xizheng Yu <xyu354@wisc.edu>
#
# License: MIT License

import numpy as np
from ..backend import get_backend


def dist_monge_max_min(i):
    r"""
    A tensor :math:c is Monge if for all valid :math:i_1, \ldots i_d and
    :math:j_1, \ldots, j_d,

    .. math::
        c(s_1, \ldots, s_d) + c(t_1, \ldots t_d) \leq c(i_1, \ldots i_d) +
        c(j_1, \ldots, j_d)

    where :math:s_k = \min(i_k, j_k) and :math:t_k = \max(i_k, j_k).

    Our focus is on a specific cost, which is known to be Monge:

    .. math::
        c(i_1,i_2,\ldots,i_d) = \max{i_k:k\in[d]} - \min{i_k:k\in[d]}.

    When :math:d=2, this cost reduces to :math:c(i_1,i_2)=|i_1-i_2|,
    which agrees with the classical EMD cost. This choice of :math:c is called
    the generalized EMD cost.

    Parameters
    ----------
    i : list
        The list of integer indexes.

    Returns
    -------
    cost : numeric value
        The ground cost (generalized EMD cost) of the tensor.

    References
    ----------
    .. [56] Jeffery Kline. Properties of the d-dimensional earth mover's
        problem. Discrete Applied Mathematics, 265: 128-141, 2019.
    .. [57] Wolfgang W. Bein, Peter Brucker, James K. Park, and Pramod K.
        Pathak. A monge property for the d-dimensional transportation problem.
        Discrete Applied Mathematics, 58(2):97-109, 1995. ISSN 0166-218X. doi:
        https://doi.org/10.1016/0166-218X(93)E0121-E. URL
        https://www.sciencedirect.com/ science/article/pii/0166218X93E0121E.
        Workshop on Discrete Algoritms.
    """

    return max(i) - min(i)


def dmmot_monge_1dgrid_loss(A, verbose=False, log=False):
    r"""
    Compute the discrete multi-marginal optimal transport of distributions A.

    This function operates on distributions whose supports are real numbers on
    the real line.

    The algorithm solves both primal and dual d-MMOT programs concurrently to
    produce the optimal transport plan as well as the total (minimal) cost.
    The cost is a ground cost, and the solution is independent of
    which Monge cost is desired.

    The algorithm accepts :math:`d` distributions (i.e., histograms)
    :math:`a_{1}, \ldots, a_{d} \in \mathbb{R}_{+}^{n}` with :math:`e^{\prime}
    a_{j}=1` for all :math:`j \in[d]`. Although the algorithm states that all
    histograms have the same number of bins, the algorithm can be easily
    adapted to accept as inputs :math:`a_{i} \in \mathbb{R}_{+}^{n_{i}}`
    with :math:`n_{i} \neq n_{j}` [50].

    The function solves the following optimization problem[51]:

    .. math::
        \begin{align}\begin{aligned}
            \underset{\gamma\in\mathbb{R}^{n^{d}}_{+}} {\textrm{min}}
            \sum_{i_1,\ldots,i_d} c(i_1,\ldots, i_d)\, \gamma(i_1,\ldots,i_d)
            \quad \textrm{s.t.}
            \sum_{i_2,\ldots,i_d} \gamma(i_1,\ldots,i_d) &= a_1(i_i),
            (\forall i_1\in[n])\\
            \qquad\vdots\\
            \sum_{i_1,\ldots,i_{d-1}} \gamma(i_1,\ldots,i_d) &= a_{d}(i_{d}),
            (\forall i_d\in[n]).
            \end{aligned}
        \end{align}


    Parameters
    ----------
    A : nx.ndarray, shape (dim, n_hists)
        The input ndarray containing distributions of n bins in d dimensions.
    verbose : bool, optional
        If True, print debugging information during execution. Default=False.
    log : bool, optional
        If True, record log. Default is False.

    Returns
    -------
    obj : float
        the value of the primal objective function evaluated at the solution.
    log : dict
        A dictionary containing the log of the discrete mmot problem:
        - 'A': a dictionary that maps tuples of indices to the corresponding
        primal variables. The tuples are the indices of the entries that are
        set to their minimum value during the algorithm.
        - 'primal objective': a float, the value of the objective function
        evaluated at the solution.
        - 'dual': a list of arrays, the dual variables corresponding to
        the input arrays. The i-th element of the list is the dual variable
        corresponding to the i-th dimension of the input arrays.
        - 'dual objective': a float, the value of the dual objective function
        evaluated at the solution.


    References
    ----------
    .. [55] Ronak Mehta, Jeffery Kline, Vishnu Suresh Lokhande, Glenn Fung, &
        Vikas Singh (2023). Efficient Discrete Multi Marginal Optimal
        Transport Regularization. In The Eleventh International
        Conference on Learning Representations.
    .. [56] Jeffery Kline. Properties of the d-dimensional earth mover's
        problem. Discrete Applied Mathematics, 265: 128-141, 2019.
    .. [58] Leonid V Kantorovich. On the translocation of masses. Dokl. Akad.
        Nauk SSSR, 37:227-229, 1942.

    See Also
    --------
    ot.lp.dmmot_monge_1dgrid_optimize : Optimize the d-Dimensional Earth
    Mover's Distance (d-MMOT)
    """

    nx = get_backend(A)
    A_copy = A
    A = nx.to_numpy(A)

    AA = [np.copy(A[:, j]) for j in range(A.shape[1])]

    dims = tuple([len(_) for _ in AA])
    xx = {}
    dual = [np.zeros(d) for d in dims]

    idx = [0, ] * len(AA)
    obj = 0

    if verbose:
        print('i minval oldidx\t\tobj\t\tvals')

    while all([i < _ for _, i in zip(dims, idx)]):
        vals = [v[i] for v, i in zip(AA, idx)]
        minval = min(vals)
        i = vals.index(minval)
        xx[tuple(idx)] = minval
        obj += (dist_monge_max_min(idx)) * minval
        for v, j in zip(AA, idx):
            v[j] -= minval
        # oldidx = nx.copy(idx)
        oldidx = idx.copy()
        idx[i] += 1
        if idx[i] < dims[i]:
            temp = (dist_monge_max_min(idx) -
                    dist_monge_max_min(oldidx) +
                    dual[i][idx[i] - 1])
            dual[i][idx[i]] += temp
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

    dualobj = sum([np.dot(A[:, i], arr) for i, arr in enumerate(dual)])
    obj = nx.from_numpy(obj)

    log_dict = {'A': xx,
                'primal objective': obj,
                'dual': dual,
                'dual objective': dualobj}

    # define forward/backward relations for pytorch
    obj = nx.set_gradients(obj, (A_copy), (dual))

    if log:
        return obj, log_dict
    else:
        return obj


def dmmot_monge_1dgrid_optimize(
        A,
        niters=100,
        lr_init=1e-5,
        lr_decay=0.995,
        print_rate=100,
        verbose=False,
        log=False):
    r"""Minimize the d-dimensional EMD using gradient descent.

    Discrete Multi-Marginal Optimal Transport (d-MMOT): Let :math:`a_1, \ldots,
    a_d\in\mathbb{R}^n_{+}` be discrete probability distributions. Here,
    the d-MMOT is the LP,

    .. math::
        \begin{align}\begin{aligned}
            \underset{x\in\mathbb{R}^{n^{d}}_{+}} {\textrm{min}}
            \sum_{i_1,\ldots,i_d} c(i_1,\ldots, i_d)\, x(i_1,\ldots,i_d) \quad
            \textrm{s.t.}
            \sum_{i_2,\ldots,i_d} x(i_1,\ldots,i_d) &= a_1(i_i),
            (\forall i_1\in[n])\\
            \qquad\vdots\\
            \sum_{i_1,\ldots,i_{d-1}} x(i_1,\ldots,i_d) &= a_{d}(i_{d}),
            (\forall i_d\in[n]).
            \end{aligned}
        \end{align}

    The dual linear program of the d-MMOT problem is:

    .. math::
        \underset{z_j\in\mathbb{R}^n, j\in[d]}{\textrm{maximize}}\qquad\sum_{j}
        a_j'z_j\qquad \textrm{subject to}\qquad z_{1}(i_1)+\cdots+z_{d}(i_{d})
        \leq c(i_1,\ldots,i_{d}),


    where the indices in the constraints include all :math:`i_j\in[n]`, :math:
    `j\in[d]`. Denote by :math:`\phi(a_1,\ldots,a_d)`, the optimal objective
    value of the LP in d-MMOT problem. Let :math:`z^*` be an optimal solution
    to the dual program. Then,

    .. math::
        \begin{align}
            \nabla \phi(a_1,\ldots,a_{d}) &= z^*,
            ~~\text{and for any $t\in \mathbb{R}$,}~~
            \phi(a_1,a_2,\ldots,a_{d}) = \sum_{j}a_j'
            (z_j^* + t\, \eta), \nonumber \\
            \text{where } \eta &:= (z_1^{*}(n)\,e, z^*_1(n)\,e, \cdots,
            z^*_{d}(n)\,e)
        \end{align}

    Using these dual variables naturally provided by the algorithm in
    ot.lp.dmmot_monge_1dgrid_loss, gradient steps move each input distribution
    to minimize their d-mmot distance.

    Parameters
    ----------
    A : nx.ndarray, shape (dim, n_hists)
        The input ndarray containing distributions of n bins in d dimensions.
    niters : int, optional (default=100)
        The maximum number of iterations for the optimization algorithm.
    lr_init : float, optional (default=1e-5)
        The initial learning rate (step size) for the optimization algorithm.
    lr_decay : float, optional (default=0.995)
        The learning rate decay rate in each iteration.
    print_rate : int, optional (default=100)
        The rate at which to print the objective value and gradient norm
        during the optimization algorithm.
    verbose : bool, optional
        If True, print debugging information during execution. Default=False.
    log : bool, optional
        If True, record log. Default is False.

    Returns
    -------
    a : list of ndarrays, each of shape (n,)
        The optimal solution as a list of n approximate barycenters, each of
        length vecsize.
    log : dict
        log dictionary return only if log==True in parameters

    References
    ----------
    .. [55] Ronak Mehta, Jeffery Kline, Vishnu Suresh Lokhande, Glenn Fung, &
        Vikas Singh (2023). Efficient Discrete Multi Marginal Optimal
        Transport Regularization. In The Eleventh International
        Conference on Learning Representations.
    .. [60] Olvi L Mangasarian and RR Meyer. Nonlinear perturbation of linear
        programs. SIAM Journal on Control and Optimization, 17(6):745-752, 1979
    .. [59] Michael C Ferris and Olvi L Mangasarian. Finite perturbation of
        convex programs. Applied Mathematics and Optimization, 23(1):263-273,
        1991.

    See Also
    --------
    ot.lp.dmmot_monge_1dgrid_loss: d-Dimensional Earth Mover's Solver
    """

    # function body here
    nx = get_backend(A)
    A = nx.to_numpy(A)
    n, d = A.shape  # n is dim, d is n_hists

    def dualIter(A, lr):
        funcval, log_dict = dmmot_monge_1dgrid_loss(
            A, verbose=verbose, log=True)
        grad = np.column_stack(log_dict['dual'])
        A_new = np.reshape(A, (n, d)) - grad * lr
        return funcval, A_new, grad, log_dict

    def renormalize(A):
        A = np.reshape(A, (n, d))
        for i in range(A.shape[1]):
            if min(A[:, i]) < 0:
                A[:, i] -= min(A[:, i])
            A[:, i] /= np.sum(A[:, i])
        return A

    def listify(A):
        return [A[:, i] for i in range(A.shape[1])]

    lr = lr_init

    funcval, _, grad, log_dict = dualIter(A, lr)
    gn = np.linalg.norm(grad)

    print(f'Inital:\t\tObj:\t{funcval:.4f}\tGradNorm:\t{gn:.4f}')

    for i in range(niters):

        A = renormalize(A)
        funcval, A, grad, log_dict = dualIter(A, lr)
        gn = np.linalg.norm(grad)

        if i % print_rate == 0:
            print(f'Iter {i:2.0f}:\tObj:\t{funcval:.4f}\tGradNorm:\t{gn:.4f}')

        lr *= lr_decay

    A = renormalize(A)
    a = listify(A)

    if log:
        return a, log_dict
    else:
        return a
