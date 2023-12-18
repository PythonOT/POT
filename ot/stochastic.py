"""
Stochastic solvers for regularized OT.


"""

# Authors: Kilian Fatras <kilian.fatras@gmail.com>
#          Rémi Flamary <remi.flamary@polytechnique.edu>
#
# License: MIT License

import numpy as np
from .utils import dist, check_random_state
from .backend import get_backend

##############################################################################
# Optimization toolbox for SEMI - DUAL problems
##############################################################################


def coordinate_grad_semi_dual(b, M, reg, beta, i):
    r'''
    Compute the coordinate gradient update for regularized discrete distributions for :math:`(i, :)`

    The function computes the gradient of the semi dual problem:

    .. math::
        \max_\mathbf{v} \ \sum_i \mathbf{a}_i \left[ \sum_j \mathbf{v}_j \mathbf{b}_j - \mathrm{reg}
        \cdot \log \left( \sum_j \mathbf{b}_j
        \exp \left( \frac{\mathbf{v}_j - \mathbf{M}_{i,j}}{\mathrm{reg}}
        \right) \right) \right]

    Where :

    - :math:`\mathbf{M}` is the (`ns`, `nt`) metric cost matrix
    - :math:`\mathbf{v}` is a dual variable in :math:`\mathbb{R}^{nt}`
    - reg is the regularization term
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are source and target weights (sum to 1)

    The algorithm used for solving the problem is the ASGD & SAG algorithms
    as proposed in :ref:`[18] <references-coordinate-grad-semi-dual>` [alg.1 & alg.2]


    Parameters
    ----------
    b : ndarray, shape (nt,)
        Target measure.
    M : ndarray, shape (ns, nt)
        Cost matrix.
    reg : float
        Regularization term > 0.
    v : ndarray, shape (nt,)
        Dual variable.
    i : int
        Picked number `i`.

    Returns
    -------
    coordinate gradient : ndarray, shape (nt,)

    .. _references-coordinate-grad-semi-dual:
    References
    ----------
    .. [18] Genevay, A., Cuturi, M., Peyré, G. & Bach, F. (2016) Stochastic Optimization for Large-scale Optimal Transport. Advances in Neural Information Processing Systems (2016).
    '''
    r = M[i, :] - beta
    exp_beta = np.exp(-r / reg) * b
    khi = exp_beta / (np.sum(exp_beta))
    return b - khi


def sag_entropic_transport(a, b, M, reg, numItermax=10000, lr=None, random_state=None):
    r"""
    Compute the SAG algorithm to solve the regularized discrete measures optimal transport max problem

    The function solves the following optimization problem:

    .. math::
        \gamma = \mathop{\arg \min}_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F +
        \mathrm{reg} \cdot\Omega(\gamma)

        s.t. \ \gamma \mathbf{1} = \mathbf{a}

             \gamma^T \mathbf{1} = \mathbf{b}

             \gamma \geq 0

    Where :

    - :math:`\mathbf{M}` is the (`ns`, `nt`) metric cost matrix
    - :math:`\Omega` is the entropic regularization term with :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are source and target weights (sum to 1)

    The algorithm used for solving the problem is the SAG algorithm
    as proposed in :ref:`[18] <references-sag-entropic-transport>` [alg.1]


    Parameters
    ----------

    a : ndarray, shape (ns,),
        Source measure.
    b : ndarray, shape (nt,),
        Target measure.
    M : ndarray, shape (ns, nt),
        Cost matrix.
    reg : float
        Regularization term > 0
    numItermax : int
        Number of iteration.
    lr : float
        Learning rate.
    random_state : int, RandomState instance or None, default=None
        Determines random number generation. Pass an int for reproducible
        output across multiple function calls.

    Returns
    -------
    v : ndarray, shape (`nt`,)
        Dual variable.

    .. _references-sag-entropic-transport:
    References
    ----------
    .. [18] Genevay, A., Cuturi, M., Peyré, G. & Bach, F. (2016) Stochastic Optimization for Large-scale Optimal Transport. Advances in Neural Information Processing Systems (2016).
    """

    if lr is None:
        lr = 1. / max(a / reg)
    n_source = np.shape(M)[0]
    n_target = np.shape(M)[1]
    cur_beta = np.zeros(n_target)
    stored_gradient = np.zeros((n_source, n_target))
    sum_stored_gradient = np.zeros(n_target)
    rng = check_random_state(random_state)
    for _ in range(numItermax):
        i = rng.randint(n_source)
        cur_coord_grad = a[i] * coordinate_grad_semi_dual(b, M, reg,
                                                          cur_beta, i)
        sum_stored_gradient += (cur_coord_grad - stored_gradient[i])
        stored_gradient[i] = cur_coord_grad
        cur_beta += lr * (1. / n_source) * sum_stored_gradient
    return cur_beta


def averaged_sgd_entropic_transport(a, b, M, reg, numItermax=300000, lr=None, random_state=None):
    r'''
    Compute the ASGD algorithm to solve the regularized semi continous measures optimal transport max problem

    The function solves the following optimization problem:

    .. math::
        \gamma = \mathop{\arg \min}_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F +
        \mathrm{reg}\cdot\Omega(\gamma)

        s.t. \gamma \mathbf{1} = \mathbf{a}

             \gamma^T \mathbf{1} = \mathbf{b}

             \gamma \geq 0

    Where :

    - :math:`\mathbf{M}` is the (`ns`, `nt`) metric cost matrix
    - :math:`\Omega` is the entropic regularization term with :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are source and target weights (sum to 1)

    The algorithm used for solving the problem is the ASGD algorithm
    as proposed in :ref:`[18] <references-averaged-sgd-entropic-transport>` [alg.2]


    Parameters
    ----------
    b : ndarray, shape (nt,)
        target measure
    M : ndarray, shape (ns, nt)
        cost matrix
    reg : float
        Regularization term > 0
    numItermax : int
        Number of iteration.
    lr : float
        Learning rate.
    random_state : int, RandomState instance or None, default=None
        Determines random number generation. Pass an int for reproducible
        output across multiple function calls.

    Returns
    -------
    ave_v : ndarray, shape (`nt`,)
        dual variable

    .. _references-averaged-sgd-entropic-transport:
    References
    ----------
    .. [18] Genevay, A., Cuturi, M., Peyré, G. & Bach, F. (2016) Stochastic Optimization for Large-scale Optimal Transport. Advances in Neural Information Processing Systems (2016).
    '''

    if lr is None:
        lr = 1. / max(a / reg)
    n_source = np.shape(M)[0]
    n_target = np.shape(M)[1]
    cur_beta = np.zeros(n_target)
    ave_beta = np.zeros(n_target)
    rng = check_random_state(random_state)
    for cur_iter in range(numItermax):
        k = cur_iter + 1
        i = rng.randint(n_source)
        cur_coord_grad = coordinate_grad_semi_dual(b, M, reg, cur_beta, i)
        cur_beta += (lr / np.sqrt(k)) * cur_coord_grad
        ave_beta = (1. / k) * cur_beta + (1 - 1. / k) * ave_beta
    return ave_beta


def c_transform_entropic(b, M, reg, beta):
    r'''
    The goal is to recover u from the c-transform.

    The function computes the c-transform of a dual variable from the other
    dual variable:

    .. math::
        \mathbf{u} = \mathbf{v}^{c,reg} = - \mathrm{reg} \sum_j \mathbf{b}_j
        \exp\left( \frac{\mathbf{v} - \mathbf{M}}{\mathrm{reg}} \right)

    Where :

    - :math:`\mathbf{M}` is the (`ns`, `nt`) metric cost matrix
    - :math:`\mathbf{u}`, :math:`\mathbf{v}` are dual variables in :math:`\mathbb{R}^{ns} \times \mathbb{R}^{nt}`
    - reg is the regularization term

    It is used to recover an optimal u from optimal v solving the semi dual
    problem, see Proposition 2.1 of :ref:`[18] <references-c-transform-entropic>`


    Parameters
    ----------
    b : ndarray, shape (nt,)
        Target measure
    M : ndarray, shape (ns, nt)
        Cost matrix
    reg : float
        Regularization term > 0
    v : ndarray, shape (nt,)
        Dual variable.

    Returns
    -------
    u : ndarray, shape (`ns`,)
        Dual variable.

    .. _references-c-transform-entropic:
    References
    ----------
    .. [18] Genevay, A., Cuturi, M., Peyré, G. & Bach, F. (2016) Stochastic Optimization for Large-scale Optimal Transport. Advances in Neural Information Processing Systems (2016).
    '''

    n_source = np.shape(M)[0]
    alpha = np.zeros(n_source)
    for i in range(n_source):
        r = M[i, :] - beta
        min_r = np.min(r)
        exp_beta = np.exp(-(r - min_r) / reg) * b
        alpha[i] = min_r - reg * np.log(np.sum(exp_beta))
    return alpha


def solve_semi_dual_entropic(a, b, M, reg, method, numItermax=10000, lr=None,
                             log=False):
    r'''
    Compute the transportation matrix to solve the regularized discrete measures optimal transport max problem

    The function solves the following optimization problem:

    .. math::
        \gamma = \mathop{\arg \min}_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F +
        \mathrm{reg} \cdot\Omega(\gamma)

        s.t. \ \gamma \mathbf{1} = \mathbf{a}

             \gamma^T \mathbf{1} = \mathbf{b}

             \gamma \geq 0

    Where :

    - :math:`\mathbf{M}` is the (`ns`, `nt`) metric cost matrix
    - :math:`\Omega` is the entropic regularization term with :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are source and target weights (sum to 1)

    The algorithm used for solving the problem is the SAG or ASGD algorithms
    as proposed in :ref:`[18] <references-solve-semi-dual-entropic>`


    Parameters
    ----------

    a : ndarray, shape (ns,)
        source measure
    b : ndarray, shape (nt,)
        target measure
    M : ndarray, shape (ns, nt)
        cost matrix
    reg : float
        Regularization term > 0
    methode : str
        used method (SAG or ASGD)
    numItermax : int
        number of iteration
    lr : float
        learning rate
    n_source : int
        size of the source measure
    n_target : int
        size of the target measure
    log : bool, optional
        record log if True

    Returns
    -------
    pi : ndarray, shape (ns, nt)
        transportation matrix
    log : dict
        log dictionary return only if log==True in parameters

    .. _references-solve-semi-dual-entropic:
    References
    ----------
    .. [18] Genevay, A., Cuturi, M., Peyré, G. & Bach, F. (2016) Stochastic Optimization for Large-scale Optimal Transport. Advances in Neural Information Processing Systems (2016).
    '''

    if method.lower() == "sag":
        opt_beta = sag_entropic_transport(a, b, M, reg, numItermax, lr)
    elif method.lower() == "asgd":
        opt_beta = averaged_sgd_entropic_transport(a, b, M, reg, numItermax, lr)
    else:
        print("Please, select your method between SAG and ASGD")
        return None

    opt_alpha = c_transform_entropic(b, M, reg, opt_beta)
    pi = (np.exp((opt_alpha[:, None] + opt_beta[None, :] - M[:, :]) / reg) *
          a[:, None] * b[None, :])

    if log:
        log = {}
        log['alpha'] = opt_alpha
        log['beta'] = opt_beta
        return pi, log
    else:
        return pi


##############################################################################
# Optimization toolbox for DUAL problems
##############################################################################


def batch_grad_dual(a, b, M, reg, alpha, beta, batch_size, batch_alpha,
                    batch_beta):
    r'''
    Computes the partial gradient of the dual optimal transport problem.

    For each :math:`(i,j)` in a batch of coordinates, the partial gradients are :

    .. math::
        \partial_{\mathbf{u}_i} F = \frac{b_s}{l_v} \mathbf{u}_i -
        \sum_{j \in B_v} \mathbf{a}_i \mathbf{b}_j
        \exp\left( \frac{\mathbf{u}_i + \mathbf{v}_j - \mathbf{M}_{i,j}}{\mathrm{reg}} \right)

        \partial_{\mathbf{v}_j} F = \frac{b_s}{l_u} \mathbf{v}_j -
        \sum_{i \in B_u} \mathbf{a}_i \mathbf{b}_j
        \exp\left( \frac{\mathbf{u}_i + \mathbf{v}_j - \mathbf{M}_{i,j}}{\mathrm{reg}} \right)

    Where :

    - :math:`\mathbf{M}` is the (`ns`, `nt`) metric cost matrix
    - :math:`\mathbf{u}`, :math:`\mathbf{v}` are dual variables in :math:`\mathbb{R}^{ns} \times \mathbb{R}^{nt}`
    - reg is the regularization term
    - :math:`B_u` and :math:`B_v` are lists of index
    - :math:`b_s` is the size of the batches :math:`B_u` and :math:`B_v`
    - :math:`l_u` and :math:`l_v` are the lengths of :math:`B_u` and :math:`B_v`
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are source and target weights (sum to 1)


    The algorithm used for solving the dual problem is the SGD algorithm
    as proposed in :ref:`[19] <references-batch-grad-dual>` [alg.1]


    Parameters
    ----------
    a : ndarray, shape (ns,)
        source measure
    b : ndarray, shape (nt,)
        target measure
    M : ndarray, shape (ns, nt)
        cost matrix
    reg : float
        Regularization term > 0
    alpha : ndarray, shape (ns,)
        dual variable
    beta : ndarray, shape (nt,)
        dual variable
    batch_size : int
        size of the batch
    batch_alpha : ndarray, shape (bs,)
        batch of index of alpha
    batch_beta : ndarray, shape (bs,)
        batch of index of beta

    Returns
    -------
    grad : ndarray, shape (`ns`,)
        partial grad F

    .. _references-batch-grad-dual:
    References
    ----------
    .. [19] Seguy, V., Bhushan Damodaran, B., Flamary, R., Courty, N., Rolet, A.& Blondel, M. Large-scale Optimal Transport and Mapping Estimation. International Conference on Learning Representation (2018)
    '''
    G = - (np.exp((alpha[batch_alpha, None] + beta[None, batch_beta] -
                   M[batch_alpha, :][:, batch_beta]) / reg) *
           a[batch_alpha, None] * b[None, batch_beta])
    grad_beta = np.zeros(np.shape(M)[1])
    grad_alpha = np.zeros(np.shape(M)[0])
    grad_beta[batch_beta] = (b[batch_beta] * len(batch_alpha) / np.shape(M)[0]
                             + G.sum(0))
    grad_alpha[batch_alpha] = (a[batch_alpha] * len(batch_beta)
                               / np.shape(M)[1] + G.sum(1))

    return grad_alpha, grad_beta


def sgd_entropic_regularization(a, b, M, reg, batch_size, numItermax, lr, random_state=None):
    r'''
    Compute the sgd algorithm to solve the regularized discrete measures optimal transport dual problem

    The function solves the following optimization problem:

    .. math::
        \gamma = \mathop{\arg \min}_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F +
        \mathrm{reg} \cdot\Omega(\gamma)

        s.t. \ \gamma \mathbf{1} = \mathbf{a}

             \gamma^T \mathbf{1} = \mathbf{b}

             \gamma \geq 0

    Where :

    - :math:`\mathbf{M}` is the (`ns`, `nt`) metric cost matrix
    - :math:`\Omega` is the entropic regularization term with :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are source and target weights (sum to 1)

    Parameters
    ----------
    a : ndarray, shape (ns,)
        source measure
    b : ndarray, shape (nt,)
        target measure
    M : ndarray, shape (ns, nt)
        cost matrix
    reg : float
        Regularization term > 0
    batch_size : int
        size of the batch
    numItermax : int
        number of iteration
    lr : float
        learning rate
    random_state : int, RandomState instance or None, default=None
        Determines random number generation. Pass an int for reproducible
        output across multiple function calls.

    Returns
    -------
    alpha : ndarray, shape (ns,)
        dual variable
    beta : ndarray, shape (nt,)
        dual variable

    References
    ----------
    .. [19] Seguy, V., Bhushan Damodaran, B., Flamary, R., Courty, N., Rolet, A.& Blondel, M. Large-scale Optimal Transport and Mapping Estimation. International Conference on Learning Representation (2018)
    '''

    n_source = np.shape(M)[0]
    n_target = np.shape(M)[1]
    cur_alpha = np.zeros(n_source)
    cur_beta = np.zeros(n_target)
    rng = check_random_state(random_state)
    for cur_iter in range(numItermax):
        k = np.sqrt(cur_iter + 1)
        batch_alpha = rng.choice(n_source, batch_size, replace=False)
        batch_beta = rng.choice(n_target, batch_size, replace=False)
        update_alpha, update_beta = batch_grad_dual(a, b, M, reg, cur_alpha,
                                                    cur_beta, batch_size,
                                                    batch_alpha, batch_beta)
        cur_alpha[batch_alpha] += (lr / k) * update_alpha[batch_alpha]
        cur_beta[batch_beta] += (lr / k) * update_beta[batch_beta]

    return cur_alpha, cur_beta


def solve_dual_entropic(a, b, M, reg, batch_size, numItermax=10000, lr=1,
                        log=False):
    r'''
    Compute the transportation matrix to solve the regularized discrete measures optimal transport dual problem

    The function solves the following optimization problem:

    .. math::
        \gamma = \mathop{\arg \min}_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F +
        \mathrm{reg} \cdot\Omega(\gamma)

        s.t. \ \gamma \mathbf{1} = \mathbf{a}

             \gamma^T \mathbf{1} = \mathbf{b}

             \gamma \geq 0

    Where :

    - :math:`\mathbf{M}` is the (`ns`, `nt`) metric cost matrix
    - :math:`\Omega` is the entropic regularization term with :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are source and target weights (sum to 1)

    Parameters
    ----------
    a : ndarray, shape (ns,)
        source measure
    b : ndarray, shape (nt,)
        target measure
    M : ndarray, shape (ns, nt)
        cost matrix
    reg : float
        Regularization term > 0
    batch_size : int
        size of the batch
    numItermax : int
        number of iteration
    lr : float
        learning rate
    log : bool, optional
        record log if True

    Returns
    -------
    pi : ndarray, shape (ns, nt)
        transportation matrix
    log : dict
        log dictionary return only if log==True in parameters

    References
    ----------
    .. [19] Seguy, V., Bhushan Damodaran, B., Flamary, R., Courty, N., Rolet, A.& Blondel, M. Large-scale Optimal Transport and Mapping Estimation. International Conference on Learning Representation (2018)
    '''

    opt_alpha, opt_beta = sgd_entropic_regularization(a, b, M, reg, batch_size,
                                                      numItermax, lr)
    pi = (np.exp((opt_alpha[:, None] + opt_beta[None, :] - M[:, :]) / reg) *
          a[:, None] * b[None, :])
    if log:
        log = {}
        log['alpha'] = opt_alpha
        log['beta'] = opt_beta
        return pi, log
    else:
        return pi


################################################################################
# Losses for stochastic optimization
################################################################################

def loss_dual_entropic(u, v, xs, xt, reg=1, ws=None, wt=None, metric='sqeuclidean'):
    r"""
    Compute the dual loss of the entropic OT as in equation (6)-(7) of [19]

    This loss is backend compatible and can be used for stochastic optimization
    of the dual potentials. It can be used on the full dataset (beware of
    memory) or on minibatches.


    Parameters
    ----------
    u : array-like, shape (ns,)
        Source dual potential
    v : array-like, shape (nt,)
        Target dual potential
    xs : array-like, shape (ns,d)
        Source samples
    xt : array-like, shape (ns,d)
        Target samples
    reg : float
        Regularization term > 0 (default=1)
    ws : array-like, shape (ns,), optional
        Source sample weights (default unif)
    wt : array-like, shape (ns,), optional
        Target sample weights (default unif)
    metric : string, callable
        Ground metric for OT (default quadratic). Can be given as a callable
        function taking (xs,xt) as parameters.

    Returns
    -------
    dual_loss : array-like
        Dual loss (to maximize)


    References
    ----------
    .. [19] Seguy, V., Bhushan Damodaran, B., Flamary, R., Courty, N., Rolet, A.& Blondel, M. Large-scale Optimal Transport and Mapping Estimation. International Conference on Learning Representation (2018)
    """

    nx = get_backend(u, v, xs, xt)

    if ws is None:
        ws = nx.ones(xs.shape[0], type_as=xs) / xs.shape[0]

    if wt is None:
        wt = nx.ones(xt.shape[0], type_as=xt) / xt.shape[0]

    if callable(metric):
        M = metric(xs, xt)
    else:
        M = dist(xs, xt, metric=metric)

    F = -reg * nx.exp((u[:, None] + v[None, :] - M) / reg)

    return nx.sum(u * ws) + nx.sum(v * wt) + nx.sum(ws[:, None] * F * wt[None, :])


def plan_dual_entropic(u, v, xs, xt, reg=1, ws=None, wt=None, metric='sqeuclidean'):
    r"""
    Compute the primal OT plan the entropic OT as in equation (8) of [19]

    This loss is backend compatible and can be used for stochastic optimization
    of the dual potentials. It can be used on the full dataset (beware of
    memory) or on minibatches.


    Parameters
    ----------
    u : array-like, shape (ns,)
        Source dual potential
    v : array-like, shape (nt,)
        Target dual potential
    xs : array-like, shape (ns,d)
        Source samples
    xt : array-like, shape (ns,d)
        Target samples
    reg : float
        Regularization term > 0 (default=1)
    ws : array-like, shape (ns,), optional
        Source sample weights (default unif)
    wt : array-like, shape (ns,), optional
        Target sample weights (default unif)
    metric : string, callable
        Ground metric for OT (default quadratic). Can be given as a callable
        function taking (xs,xt) as parameters.

    Returns
    -------
    G : array-like
        Primal OT plan


    References
    ----------
    .. [19] Seguy, V., Bhushan Damodaran, B., Flamary, R., Courty, N., Rolet, A.& Blondel, M. Large-scale Optimal Transport and Mapping Estimation. International Conference on Learning Representation (2018)
    """

    nx = get_backend(u, v, xs, xt)

    if ws is None:
        ws = nx.ones(xs.shape[0], type_as=xs) / xs.shape[0]

    if wt is None:
        wt = nx.ones(xt.shape[0], type_as=xt) / xt.shape[0]

    if callable(metric):
        M = metric(xs, xt)
    else:
        M = dist(xs, xt, metric=metric)

    H = nx.exp((u[:, None] + v[None, :] - M) / reg)

    return ws[:, None] * H * wt[None, :]


def loss_dual_quadratic(u, v, xs, xt, reg=1, ws=None, wt=None, metric='sqeuclidean'):
    r"""
    Compute the dual loss of the quadratic regularized OT as in equation (6)-(7) of [19]

    This loss is backend compatible and can be used for stochastic optimization
    of the dual potentials. It can be used on the full dataset (beware of
    memory) or on minibatches.


    Parameters
    ----------
    u : array-like, shape (ns,)
        Source dual potential
    v : array-like, shape (nt,)
        Target dual potential
    xs : array-like, shape (ns,d)
        Source samples
    xt : array-like, shape (ns,d)
        Target samples
    reg : float
        Regularization term > 0 (default=1)
    ws : array-like, shape (ns,), optional
        Source sample weights (default unif)
    wt : array-like, shape (ns,), optional
        Target sample weights (default unif)
    metric : string, callable
        Ground metric for OT (default quadratic). Can be given as a callable
        function taking (xs,xt) as parameters.

    Returns
    -------
    dual_loss : array-like
        Dual loss (to maximize)


    References
    ----------
    .. [19] Seguy, V., Bhushan Damodaran, B., Flamary, R., Courty, N., Rolet, A.& Blondel, M. Large-scale Optimal Transport and Mapping Estimation. International Conference on Learning Representation (2018)
    """

    nx = get_backend(u, v, xs, xt)

    if ws is None:
        ws = nx.ones(xs.shape[0], type_as=xs) / xs.shape[0]

    if wt is None:
        wt = nx.ones(xt.shape[0], type_as=xt) / xt.shape[0]

    if callable(metric):
        M = metric(xs, xt)
    else:
        M = dist(xs, xt, metric=metric)

    F = -1.0 / (4 * reg) * nx.maximum(u[:, None] + v[None, :] - M, 0.0)**2

    return nx.sum(u * ws) + nx.sum(v * wt) + nx.sum(ws[:, None] * F * wt[None, :])


def plan_dual_quadratic(u, v, xs, xt, reg=1, ws=None, wt=None, metric='sqeuclidean'):
    r"""
    Compute the primal OT plan the quadratic regularized OT as in equation (8) of [19]

    This loss is backend compatible and can be used for stochastic optimization
    of the dual potentials. It can be used on the full dataset (beware of
    memory) or on minibatches.


    Parameters
    ----------
    u : array-like, shape (ns,)
        Source dual potential
    v : array-like, shape (nt,)
        Target dual potential
    xs : array-like, shape (ns,d)
        Source samples
    xt : array-like, shape (ns,d)
        Target samples
    reg : float
        Regularization term > 0 (default=1)
    ws : array-like, shape (ns,), optional
        Source sample weights (default unif)
    wt : array-like, shape (ns,), optional
        Target sample weights (default unif)
    metric : string, callable
        Ground metric for OT (default quadratic). Can be given as a callable
        function taking (xs,xt) as parameters.

    Returns
    -------
    G : array-like
        Primal OT plan


    References
    ----------
    .. [19] Seguy, V., Bhushan Damodaran, B., Flamary, R., Courty, N., Rolet, A.& Blondel, M. Large-scale Optimal Transport and Mapping Estimation. International Conference on Learning Representation (2018)
    """

    nx = get_backend(u, v, xs, xt)

    if ws is None:
        ws = nx.ones(xs.shape[0], type_as=xs) / xs.shape[0]

    if wt is None:
        wt = nx.ones(xt.shape[0], type_as=xt) / xt.shape[0]

    if callable(metric):
        M = metric(xs, xt)
    else:
        M = dist(xs, xt, metric=metric)

    H = 1.0 / (2 * reg) * nx.maximum(u[:, None] + v[None, :] - M, 0.0)

    return ws[:, None] * H * wt[None, :]
