# Author: Kilian Fatras <kilian.fatras@gmail.com>
#
# License: MIT License

import numpy as np


##############################################################################
# Optimization toolbox for SEMI - DUAL problems
##############################################################################


def coordinate_grad_semi_dual(b, M, reg, beta, i):
    '''
    Compute the coordinate gradient update for regularized discrete distributions for (i, :)

    The function computes the gradient of the semi dual problem:

    .. math::
        \max_v \sum_i (\sum_j v_j * b_j - reg * log(\sum_j exp((v_j - M_{i,j})/reg) * b_j)) * a_i

    Where :

    - M is the (ns,nt) metric cost matrix
    - v is a dual variable in R^J
    - reg is the regularization term
    - a and b are source and target weights (sum to 1)

    The algorithm used for solving the problem is the ASGD & SAG algorithms
    as proposed in [18]_ [alg.1 & alg.2]


    Parameters
    ----------

    b : np.ndarray(nt,)
        target measure
    M : np.ndarray(ns, nt)
        cost matrix
    reg : float nu
        Regularization term > 0
    v : np.ndarray(nt,)
        dual variable
    i : number int
        picked number i

    Returns
    -------

    coordinate gradient : np.ndarray(nt,)

    Examples
    --------

    >>> n_source = 7
    >>> n_target = 4
    >>> reg = 1
    >>> numItermax = 300000
    >>> a = ot.utils.unif(n_source)
    >>> b = ot.utils.unif(n_target)
    >>> rng = np.random.RandomState(0)
    >>> X_source = rng.randn(n_source, 2)
    >>> Y_target = rng.randn(n_target, 2)
    >>> M = ot.dist(X_source, Y_target)
    >>> method = "ASGD"
    >>> asgd_pi = stochastic.solve_semi_dual_entropic(a, b, M, reg,
                                                      method, numItermax)
    >>> print(asgd_pi)

    References
    ----------

    [Genevay et al., 2016] :
                    Stochastic Optimization for Large-scale Optimal Transport,
                     Advances in Neural Information Processing Systems (2016),
                      arXiv preprint arxiv:1605.08527.

    '''

    r = M[i, :] - beta
    exp_beta = np.exp(-r / reg) * b
    khi = exp_beta / (np.sum(exp_beta))
    return b - khi


def sag_entropic_transport(a, b, M, reg, numItermax=10000, lr=None):
    '''
    Compute the SAG algorithm to solve the regularized discrete measures
        optimal transport max problem

    The function solves the following optimization problem:

    .. math::
        \gamma = arg\min_\gamma <\gamma,M>_F + reg\cdot\Omega(\gamma)

        s.t. \gamma 1 = a

             \gamma^T 1 = b

             \gamma \geq 0

    Where :

    - M is the (ns,nt) metric cost matrix
    - :math:`\Omega` is the entropic regularization term with :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - a and b are source and target weights (sum to 1)

    The algorithm used for solving the problem is the SAG algorithm
    as proposed in [18]_ [alg.1]


    Parameters
    ----------

    a : np.ndarray(ns,),
        source measure
    b : np.ndarray(nt,),
        target measure
    M : np.ndarray(ns, nt),
        cost matrix
    reg : float number,
        Regularization term > 0
    numItermax : int number
        number of iteration
    lr : float number
        learning rate

    Returns
    -------

    v : np.ndarray(nt,)
        dual variable

    Examples
    --------

    >>> n_source = 7
    >>> n_target = 4
    >>> reg = 1
    >>> numItermax = 300000
    >>> a = ot.utils.unif(n_source)
    >>> b = ot.utils.unif(n_target)
    >>> rng = np.random.RandomState(0)
    >>> X_source = rng.randn(n_source, 2)
    >>> Y_target = rng.randn(n_target, 2)
    >>> M = ot.dist(X_source, Y_target)
    >>> method = "ASGD"
    >>> asgd_pi = stochastic.solve_semi_dual_entropic(a, b, M, reg,
                                                      method, numItermax)
    >>> print(asgd_pi)

    References
    ----------

    [Genevay et al., 2016] :
                    Stochastic Optimization for Large-scale Optimal Transport,
                     Advances in Neural Information Processing Systems (2016),
                      arXiv preprint arxiv:1605.08527.
    '''

    if lr is None:
        lr = 1. / max(a / reg)
    n_source = np.shape(M)[0]
    n_target = np.shape(M)[1]
    cur_beta = np.zeros(n_target)
    stored_gradient = np.zeros((n_source, n_target))
    sum_stored_gradient = np.zeros(n_target)
    for _ in range(numItermax):
        i = np.random.randint(n_source)
        cur_coord_grad = a[i] * coordinate_grad_semi_dual(b, M, reg,
                                                          cur_beta, i)
        sum_stored_gradient += (cur_coord_grad - stored_gradient[i])
        stored_gradient[i] = cur_coord_grad
        cur_beta += lr * (1. / n_source) * sum_stored_gradient
    return cur_beta


def averaged_sgd_entropic_transport(a, b, M, reg, numItermax=300000, lr=None):
    '''
    Compute the ASGD algorithm to solve the regularized semi continous measures optimal transport max problem

    The function solves the following optimization problem:

    .. math::
        \gamma = arg\min_\gamma <\gamma,M>_F + reg\cdot\Omega(\gamma)

        s.t. \gamma 1 = a

             \gamma^T 1= b

             \gamma \geq 0

    Where :

    - M is the (ns,nt) metric cost matrix
    - :math:`\Omega` is the entropic regularization term with :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - a and b are source and target weights (sum to 1)

    The algorithm used for solving the problem is the ASGD algorithm
    as proposed in [18]_ [alg.2]


    Parameters
    ----------

    b : np.ndarray(nt,)
        target measure
    M : np.ndarray(ns, nt)
        cost matrix
    reg : float number
        Regularization term > 0
    numItermax : int number
        number of iteration
    lr : float number
        learning rate


    Returns
    -------

    ave_v : np.ndarray(nt,)
        dual variable

    Examples
    --------

    >>> n_source = 7
    >>> n_target = 4
    >>> reg = 1
    >>> numItermax = 300000
    >>> a = ot.utils.unif(n_source)
    >>> b = ot.utils.unif(n_target)
    >>> rng = np.random.RandomState(0)
    >>> X_source = rng.randn(n_source, 2)
    >>> Y_target = rng.randn(n_target, 2)
    >>> M = ot.dist(X_source, Y_target)
    >>> method = "ASGD"
    >>> asgd_pi = stochastic.solve_semi_dual_entropic(a, b, M, reg,
                                                      method, numItermax)
    >>> print(asgd_pi)

    References
    ----------

    [Genevay et al., 2016] :
                    Stochastic Optimization for Large-scale Optimal Transport,
                     Advances in Neural Information Processing Systems (2016),
                      arXiv preprint arxiv:1605.08527.
    '''

    if lr is None:
        lr = 1. / max(a / reg)
    n_source = np.shape(M)[0]
    n_target = np.shape(M)[1]
    cur_beta = np.zeros(n_target)
    ave_beta = np.zeros(n_target)
    for cur_iter in range(numItermax):
        k = cur_iter + 1
        i = np.random.randint(n_source)
        cur_coord_grad = coordinate_grad_semi_dual(b, M, reg, cur_beta, i)
        cur_beta += (lr / np.sqrt(k)) * cur_coord_grad
        ave_beta = (1. / k) * cur_beta + (1 - 1. / k) * ave_beta
    return ave_beta


def c_transform_entropic(b, M, reg, beta):
    '''
    The goal is to recover u from the c-transform.

    The function computes the c_transform of a dual variable from the other
    dual variable:

    .. math::
        u = v^{c,reg} = -reg \sum_j exp((v - M)/reg) b_j

    Where :

    - M is the (ns,nt) metric cost matrix
    - u, v are dual variables in R^IxR^J
    - reg is the regularization term

    It is used to recover an optimal u from optimal v solving the semi dual
    problem, see Proposition 2.1 of [18]_


    Parameters
    ----------

    b : np.ndarray(nt,)
        target measure
    M : np.ndarray(ns, nt)
        cost matrix
    reg : float
        regularization term > 0
    v : np.ndarray(nt,)
        dual variable

    Returns
    -------

    u : np.ndarray(ns,)
        dual variable

    Examples
    --------

    >>> n_source = 7
    >>> n_target = 4
    >>> reg = 1
    >>> numItermax = 300000
    >>> a = ot.utils.unif(n_source)
    >>> b = ot.utils.unif(n_target)
    >>> rng = np.random.RandomState(0)
    >>> X_source = rng.randn(n_source, 2)
    >>> Y_target = rng.randn(n_target, 2)
    >>> M = ot.dist(X_source, Y_target)
    >>> method = "ASGD"
    >>> asgd_pi = stochastic.solve_semi_dual_entropic(a, b, M, reg,
                                                      method, numItermax)
    >>> print(asgd_pi)

    References
    ----------

    [Genevay et al., 2016] :
                    Stochastic Optimization for Large-scale Optimal Transport,
                     Advances in Neural Information Processing Systems (2016),
                      arXiv preprint arxiv:1605.08527.
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
    '''
    Compute the transportation matrix to solve the regularized discrete
        measures optimal transport max problem

    The function solves the following optimization problem:

    .. math::
        \gamma = arg\min_\gamma <\gamma,M>_F + reg\cdot\Omega(\gamma)
        s.t. \gamma 1 = a
             \gamma^T 1= b
             \gamma \geq 0

    Where :

    - M is the (ns,nt) metric cost matrix
    - :math:`\Omega` is the entropic regularization term with :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - a and b are source and target weights (sum to 1)
    The algorithm used for solving the problem is the SAG or ASGD algorithms
    as proposed in [18]_


    Parameters
    ----------

    a : np.ndarray(ns,)
        source measure
    b : np.ndarray(nt,)
        target measure
    M : np.ndarray(ns, nt)
        cost matrix
    reg : float number
        Regularization term > 0
    methode : str
        used method (SAG or ASGD)
    numItermax : int number
        number of iteration
    lr : float number
        learning rate
    n_source : int number
        size of the source measure
    n_target : int number
        size of the target measure
    log : bool, optional
        record log if True

    Returns
    -------

    pi : np.ndarray(ns, nt)
        transportation matrix
    log : dict
        log dictionary return only if log==True in parameters

    Examples
    --------

    >>> n_source = 7
    >>> n_target = 4
    >>> reg = 1
    >>> numItermax = 300000
    >>> a = ot.utils.unif(n_source)
    >>> b = ot.utils.unif(n_target)
    >>> rng = np.random.RandomState(0)
    >>> X_source = rng.randn(n_source, 2)
    >>> Y_target = rng.randn(n_target, 2)
    >>> M = ot.dist(X_source, Y_target)
    >>> method = "ASGD"
    >>> asgd_pi = stochastic.solve_semi_dual_entropic(a, b, M, reg,
                                                      method, numItermax)
    >>> print(asgd_pi)

    References
    ----------

    [Genevay et al., 2016] :
                    Stochastic Optimization for Large-scale Optimal Transport,
                     Advances in Neural Information Processing Systems (2016),
                      arXiv preprint arxiv:1605.08527.
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
    '''
    Computes the partial gradient of the dual optimal transport problem.

    For each (i,j) in a batch of coordinates, the partial gradients are :

    .. math::
        \partial_{u_i} F = u_i * b_s/l_{v} - \sum_{j \in B_v} exp((u_i + v_j - M_{i,j})/reg) * a_i * b_j

        \partial_{v_j} F = v_j * b_s/l_{u} - \sum_{i \in B_u} exp((u_i + v_j - M_{i,j})/reg) * a_i * b_j

    Where :

    - M is the (ns,nt) metric cost matrix
    - u, v are dual variables in R^ixR^J
    - reg is the regularization term
    - :math:`B_u` and :math:`B_v` are lists of index
    - :math:`b_s` is the size of the batchs :math:`B_u` and :math:`B_v`
    - :math:`l_u` and :math:`l_v` are the lenghts of :math:`B_u` and :math:`B_v`
    - a and b are source and target weights (sum to 1)


    The algorithm used for solving the dual problem is the SGD algorithm
    as proposed in [19]_ [alg.1]


    Parameters
    ----------

    a : np.ndarray(ns,)
        source measure
    b : np.ndarray(nt,)
        target measure
    M : np.ndarray(ns, nt)
        cost matrix
    reg : float number
        Regularization term > 0
    alpha : np.ndarray(ns,)
        dual variable
    beta : np.ndarray(nt,)
        dual variable
    batch_size : int number
        size of the batch
    batch_alpha : np.ndarray(bs,)
        batch of index of alpha
    batch_beta : np.ndarray(bs,)
        batch of index of beta

    Returns
    -------

    grad : np.ndarray(ns,)
        partial grad F

    Examples
    --------

    >>> n_source = 7
    >>> n_target = 4
    >>> reg = 1
    >>> numItermax = 20000
    >>> lr = 0.1
    >>> batch_size = 3
    >>> log = True
    >>> a = ot.utils.unif(n_source)
    >>> b = ot.utils.unif(n_target)
    >>> rng = np.random.RandomState(0)
    >>> X_source = rng.randn(n_source, 2)
    >>> Y_target = rng.randn(n_target, 2)
    >>> M = ot.dist(X_source, Y_target)
    >>> sgd_dual_pi, log = stochastic.solve_dual_entropic(a, b, M, reg,
                                                            batch_size,
                                                            numItermax, lr, log)
    >>> print(log['alpha'], log['beta'])
    >>> print(sgd_dual_pi)

    References
    ----------

    [Seguy et al., 2018] :
                    International Conference on Learning Representation (2018),
                      arXiv preprint arxiv:1711.02283.
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


def sgd_entropic_regularization(a, b, M, reg, batch_size, numItermax, lr):
    '''
    Compute the sgd algorithm to solve the regularized discrete measures
        optimal transport dual problem

    The function solves the following optimization problem:

    .. math::
        \gamma = arg\min_\gamma <\gamma,M>_F + reg\cdot\Omega(\gamma)

        s.t. \gamma 1 = a

             \gamma^T 1= b

             \gamma \geq 0

    Where :

    - M is the (ns,nt) metric cost matrix
    - :math:`\Omega` is the entropic regularization term with :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - a and b are source and target weights (sum to 1)

    Parameters
    ----------

    a : np.ndarray(ns,)
        source measure
    b : np.ndarray(nt,)
        target measure
    M : np.ndarray(ns, nt)
        cost matrix
    reg : float number
        Regularization term > 0
    batch_size : int number
        size of the batch
    numItermax : int number
        number of iteration
    lr : float number
        learning rate

    Returns
    -------

    alpha : np.ndarray(ns,)
        dual variable
    beta : np.ndarray(nt,)
        dual variable

    Examples
    --------

    >>> n_source = 7
    >>> n_target = 4
    >>> reg = 1
    >>> numItermax = 20000
    >>> lr = 0.1
    >>> batch_size = 3
    >>> log = True
    >>> a = ot.utils.unif(n_source)
    >>> b = ot.utils.unif(n_target)
    >>> rng = np.random.RandomState(0)
    >>> X_source = rng.randn(n_source, 2)
    >>> Y_target = rng.randn(n_target, 2)
    >>> M = ot.dist(X_source, Y_target)
    >>> sgd_dual_pi, log = stochastic.solve_dual_entropic(a, b, M, reg,
                                                          batch_size,
                                                          numItermax, lr, log)
    >>> print(log['alpha'], log['beta'])
    >>> print(sgd_dual_pi)

    References
    ----------

    [Seguy et al., 2018] :
                    International Conference on Learning Representation (2018),
                      arXiv preprint arxiv:1711.02283.
    '''

    n_source = np.shape(M)[0]
    n_target = np.shape(M)[1]
    cur_alpha = np.zeros(n_source)
    cur_beta = np.zeros(n_target)
    for cur_iter in range(numItermax):
        k = np.sqrt(cur_iter + 1)
        batch_alpha = np.random.choice(n_source, batch_size, replace=False)
        batch_beta = np.random.choice(n_target, batch_size, replace=False)
        update_alpha, update_beta = batch_grad_dual(a, b, M, reg, cur_alpha,
                                                    cur_beta, batch_size,
                                                    batch_alpha, batch_beta)
        cur_alpha[batch_alpha] += (lr / k) * update_alpha[batch_alpha]
        cur_beta[batch_beta] += (lr / k) * update_beta[batch_beta]

    return cur_alpha, cur_beta


def solve_dual_entropic(a, b, M, reg, batch_size, numItermax=10000, lr=1,
                        log=False):
    '''
    Compute the transportation matrix to solve the regularized discrete measures
        optimal transport dual problem

    The function solves the following optimization problem:

    .. math::
        \gamma = arg\min_\gamma <\gamma,M>_F + reg\cdot\Omega(\gamma)

        s.t. \gamma 1 = a

             \gamma^T 1= b

             \gamma \geq 0

    Where :

    - M is the (ns,nt) metric cost matrix
    - :math:`\Omega` is the entropic regularization term :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - a and b are source and target weights (sum to 1)

    Parameters
    ----------

    a : np.ndarray(ns,)
        source measure
    b : np.ndarray(nt,)
        target measure
    M : np.ndarray(ns, nt)
        cost matrix
    reg : float number
        Regularization term > 0
    batch_size : int number
        size of the batch
    numItermax : int number
        number of iteration
    lr : float number
        learning rate
    log : bool, optional
        record log if True

    Returns
    -------

    pi : np.ndarray(ns, nt)
        transportation matrix
    log : dict
        log dictionary return only if log==True in parameters

    Examples
    --------

    >>> n_source = 7
    >>> n_target = 4
    >>> reg = 1
    >>> numItermax = 20000
    >>> lr = 0.1
    >>> batch_size = 3
    >>> log = True
    >>> a = ot.utils.unif(n_source)
    >>> b = ot.utils.unif(n_target)
    >>> rng = np.random.RandomState(0)
    >>> X_source = rng.randn(n_source, 2)
    >>> Y_target = rng.randn(n_target, 2)
    >>> M = ot.dist(X_source, Y_target)
    >>> sgd_dual_pi, log = stochastic.solve_dual_entropic(a, b, M, reg,
                                                            batch_size,
                                                            numItermax, lr, log)
    >>> print(log['alpha'], log['beta'])
    >>> print(sgd_dual_pi)

    References
    ----------

    [Seguy et al., 2018] :
                    International Conference on Learning Representation (2018),
                      arXiv preprint arxiv:1711.02283.
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
