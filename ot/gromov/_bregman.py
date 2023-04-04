# -*- coding: utf-8 -*-
"""
Bregman projections solvers for entropic Gromov-Wasserstein
"""

# Author: Erwan Vautier <erwan.vautier@gmail.com>
#         Nicolas Courty <ncourty@irisa.fr>
#         Rémi Flamary <remi.flamary@unice.fr>
#         Titouan Vayer <titouan.vayer@irisa.fr>
#         Cédric Vincent-Cuaz <cedvincentcuaz@gmail.com>
#
# License: MIT License

from ..bregman import sinkhorn
from ..utils import dist, list_to_array, check_random_state
from ..backend import get_backend

from ._utils import init_matrix, gwloss, gwggrad
from ._utils import update_square_loss, update_kl_loss


def entropic_gromov_wasserstein(C1, C2, p, q, loss_fun, epsilon, symmetric=None, G0=None,
                                max_iter=1000, tol=1e-9, verbose=False, log=False):
    r"""
    Returns the gromov-wasserstein transport between :math:`(\mathbf{C_1}, \mathbf{p})` and :math:`(\mathbf{C_2}, \mathbf{q})`

    The function solves the following optimization problem:

    .. math::
        \mathbf{GW} = \mathop{\arg\min}_\mathbf{T} \quad \sum_{i,j,k,l} L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l}) \mathbf{T}_{i,j} \mathbf{T}_{k,l} - \epsilon(H(\mathbf{T}))

        s.t. \ \mathbf{T} \mathbf{1} &= \mathbf{p}

             \mathbf{T}^T \mathbf{1} &= \mathbf{q}

             \mathbf{T} &\geq 0

    Where :

    - :math:`\mathbf{C_1}`: Metric cost matrix in the source space
    - :math:`\mathbf{C_2}`: Metric cost matrix in the target space
    - :math:`\mathbf{p}`: distribution in the source space
    - :math:`\mathbf{q}`: distribution in the target space
    - `L`: loss function to account for the misfit between the similarity matrices
    - `H`: entropy

    .. note:: If the inner solver `ot.sinkhorn` did not convergence, the
        optimal coupling :math:`\mathbf{T}` returned by this function does not
        necessarily satisfy the marginal constraints
        :math:`\mathbf{T}\mathbf{1}=\mathbf{p}` and
        :math:`\mathbf{T}^T\mathbf{1}=\mathbf{q}`. So the returned
        Gromov-Wasserstein loss does not necessarily satisfy distance
        properties and may be negative.

    Parameters
    ----------
    C1 : array-like, shape (ns, ns)
        Metric cost matrix in the source space
    C2 : array-like, shape (nt, nt)
        Metric cost matrix in the target space
    p :  array-like, shape (ns,)
        Distribution in the source space
    q :  array-like, shape (nt,)
        Distribution in the target space
    loss_fun :  string
        Loss function used for the solver either 'square_loss' or 'kl_loss'
    epsilon : float
        Regularization term >0
    symmetric : bool, optional
        Either C1 and C2 are to be assumed symmetric or not.
        If let to its default None value, a symmetry test will be conducted.
        Else if set to True (resp. False), C1 and C2 will be assumed symmetric (resp. asymetric).
    G0: array-like, shape (ns,nt), optional
        If None the initial transport plan of the solver is pq^T.
        Otherwise G0 must satisfy marginal constraints and will be used as initial transport of the solver.
    max_iter : int, optional
        Max number of iterations
    tol : float, optional
        Stop threshold on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        Record log if True.

    Returns
    -------
    T : array-like, shape (`ns`, `nt`)
        Optimal coupling between the two spaces

    References
    ----------
    .. [12] Gabriel Peyré, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.

    .. [47] Chowdhury, S., & Mémoli, F. (2019). The gromov–wasserstein
        distance between networks and stable network invariants.
        Information and Inference: A Journal of the IMA, 8(4), 757-787.
    """
    C1, C2, p, q = list_to_array(C1, C2, p, q)
    if G0 is None:
        nx = get_backend(p, q, C1, C2)
        G0 = nx.outer(p, q)
    else:
        nx = get_backend(p, q, C1, C2, G0)
    T = G0
    constC, hC1, hC2 = init_matrix(C1, C2, p, q, loss_fun, nx)
    if symmetric is None:
        symmetric = nx.allclose(C1, C1.T, atol=1e-10) and nx.allclose(C2, C2.T, atol=1e-10)
    if not symmetric:
        constCt, hC1t, hC2t = init_matrix(C1.T, C2.T, p, q, loss_fun, nx)
    cpt = 0
    err = 1

    if log:
        log = {'err': []}

    while (err > tol and cpt < max_iter):

        Tprev = T

        # compute the gradient
        if symmetric:
            tens = gwggrad(constC, hC1, hC2, T, nx)
        else:
            tens = 0.5 * (gwggrad(constC, hC1, hC2, T, nx) + gwggrad(constCt, hC1t, hC2t, T, nx))
        T = sinkhorn(p, q, tens, epsilon, method='sinkhorn')

        if cpt % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            err = nx.norm(T - Tprev)

            if log:
                log['err'].append(err)

            if verbose:
                if cpt % 200 == 0:
                    print('{:5s}|{:12s}'.format(
                        'It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))

        cpt += 1

    if log:
        log['gw_dist'] = gwloss(constC, hC1, hC2, T, nx)
        return T, log
    else:
        return T


def entropic_gromov_wasserstein2(C1, C2, p, q, loss_fun, epsilon, symmetric=None, G0=None,
                                 max_iter=1000, tol=1e-9, verbose=False, log=False):
    r"""
    Returns the entropic gromov-wasserstein discrepancy between the two measured similarity matrices :math:`(\mathbf{C_1}, \mathbf{p})` and :math:`(\mathbf{C_2}, \mathbf{q})`

    The function solves the following optimization problem:

    .. math::
        GW = \min_\mathbf{T} \quad \sum_{i,j,k,l} L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l})
        \mathbf{T}_{i,j} \mathbf{T}_{k,l} - \epsilon(H(\mathbf{T}))

    Where :

    - :math:`\mathbf{C_1}`: Metric cost matrix in the source space
    - :math:`\mathbf{C_2}`: Metric cost matrix in the target space
    - :math:`\mathbf{p}`: distribution in the source space
    - :math:`\mathbf{q}`: distribution in the target space
    - `L`: loss function to account for the misfit between the similarity matrices
    - `H`: entropy

    .. note:: If the inner solver `ot.sinkhorn` did not convergence, the
        optimal coupling :math:`\mathbf{T}` returned by this function does not
        necessarily satisfy the marginal constraints
        :math:`\mathbf{T}\mathbf{1}=\mathbf{p}` and
        :math:`\mathbf{T}^T\mathbf{1}=\mathbf{q}`. So the returned
        Gromov-Wasserstein loss does not necessarily satisfy distance
        properties and may be negative.

    Parameters
    ----------
    C1 : array-like, shape (ns, ns)
        Metric cost matrix in the source space
    C2 : array-like, shape (nt, nt)
        Metric cost matrix in the target space
    p :  array-like, shape (ns,)
        Distribution in the source space
    q :  array-like, shape (nt,)
        Distribution in the target space
    loss_fun : str
        Loss function used for the solver either 'square_loss' or 'kl_loss'
    epsilon : float
        Regularization term >0
    symmetric : bool, optional
        Either C1 and C2 are to be assumed symmetric or not.
        If let to its default None value, a symmetry test will be conducted.
        Else if set to True (resp. False), C1 and C2 will be assumed symmetric (resp. asymetric).
    G0: array-like, shape (ns,nt), optional
        If None the initial transport plan of the solver is pq^T.
        Otherwise G0 must satisfy marginal constraints and will be used as initial transport of the solver.
    max_iter : int, optional
        Max number of iterations
    tol : float, optional
        Stop threshold on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        Record log if True.

    Returns
    -------
    gw_dist : float
        Gromov-Wasserstein distance

    References
    ----------
    .. [12] Gabriel Peyré, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.

    """
    gw, logv = entropic_gromov_wasserstein(
        C1, C2, p, q, loss_fun, epsilon, symmetric, G0, max_iter, tol, verbose, log=True)

    logv['T'] = gw

    if log:
        return logv['gw_dist'], logv
    else:
        return logv['gw_dist']


def entropic_gromov_barycenters(N, Cs, ps, p, lambdas, loss_fun, epsilon, symmetric=True,
                                max_iter=1000, tol=1e-9, verbose=False, log=False, init_C=None, random_state=None):
    r"""
    Returns the gromov-wasserstein barycenters of `S` measured similarity matrices :math:`(\mathbf{C}_s)_{1 \leq s \leq S}`

    The function solves the following optimization problem:

    .. math::

        \mathbf{C} = \mathop{\arg \min}_{\mathbf{C}\in \mathbb{R}^{N \times N}} \quad \sum_s \lambda_s \mathrm{GW}(\mathbf{C}, \mathbf{C}_s, \mathbf{p}, \mathbf{p}_s)

    Where :

    - :math:`\mathbf{C}_s`: metric cost matrix
    - :math:`\mathbf{p}_s`: distribution

    Parameters
    ----------
    N : int
        Size of the targeted barycenter
    Cs : list of S array-like of shape (ns,ns)
        Metric cost matrices
    ps : list of S array-like of shape (ns,)
        Sample weights in the `S` spaces
    p : array-like, shape(N,)
        Weights in the targeted barycenter
    lambdas : list of float
        List of the `S` spaces' weights.
    loss_fun : callable
        Tensor-matrix multiplication function based on specific loss function.
    update : callable
        function(:math:`\mathbf{p}`, lambdas, :math:`\mathbf{T}`, :math:`\mathbf{Cs}`) that updates
        :math:`\mathbf{C}` according to a specific Kernel with the `S` :math:`\mathbf{T}_s` couplings
        calculated at each iteration
    epsilon : float
        Regularization term >0
    symmetric : bool, optional.
        Either structures are to be assumed symmetric or not. Default value is True.
        Else if set to True (resp. False), C1 and C2 will be assumed symmetric (resp. asymmetric).
    max_iter : int, optional
        Max number of iterations
    tol : float, optional
        Stop threshold on error (>0)
    verbose : bool, optional
        Print information along iterations.
    log : bool, optional
        Record log if True.
    init_C : bool | array-like, shape (N, N)
        Random initial value for the :math:`\mathbf{C}` matrix provided by user.
    random_state : int or RandomState instance, optional
        Fix the seed for reproducibility

    Returns
    -------
    C : array-like, shape (`N`, `N`)
        Similarity matrix in the barycenter space (permutated arbitrarily)
    log : dict
        Log dictionary of error during iterations. Return only if `log=True` in parameters.

    References
    ----------
    .. [12] Gabriel Peyré, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.
    """
    Cs = list_to_array(*Cs)
    ps = list_to_array(*ps)
    p = list_to_array(p)
    nx = get_backend(*Cs, *ps, p)

    S = len(Cs)

    # Initialization of C : random SPD matrix (if not provided by user)
    if init_C is None:
        generator = check_random_state(random_state)
        xalea = generator.randn(N, 2)
        C = dist(xalea, xalea)
        C /= C.max()
        C = nx.from_numpy(C, type_as=p)
    else:
        C = init_C

    cpt = 0
    err = 1

    error = []

    while (err > tol) and (cpt < max_iter):
        Cprev = C

        T = [entropic_gromov_wasserstein(Cs[s], C, ps[s], p, loss_fun, epsilon, symmetric, None,
                                         max_iter, 1e-4, verbose, log=False) for s in range(S)]
        if loss_fun == 'square_loss':
            C = update_square_loss(p, lambdas, T, Cs)

        elif loss_fun == 'kl_loss':
            C = update_kl_loss(p, lambdas, T, Cs)

        if cpt % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            err = nx.norm(C - Cprev)
            error.append(err)

            if verbose:
                if cpt % 200 == 0:
                    print('{:5s}|{:12s}'.format(
                        'It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))

        cpt += 1

    if log:
        return C, {"err": error}
    else:
        return C
