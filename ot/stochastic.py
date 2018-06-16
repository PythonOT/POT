# Author: Kilian Fatras <kilian.fatras@gmail.com>
#
# License: MIT License

import numpy as np


def coordinate_gradient(b, M, reg, v, i):
    '''
    Compute the coordinate gradient update for regularized discrete
        distributions for (i, :)

    The function computes the gradient of the semi dual problem:
    .. math::
        \W_varepsilon(a, b) = \max_\v \sum_i (\sum_j v_j * b_j
            - \reg log(\sum_j exp((v_j - M_{i,j})/reg) * b_j)) * a_i

    where :
    - M is the (ns,nt) metric cost matrix
    - v is a dual variable in R^J
    - reg is the regularization term
    - a and b are source and target weights (sum to 1)

    The algorithm used for solving the problem is the ASGD & SAG algorithms
    as proposed in [18]_ [alg.1 & alg.2]


    Parameters
    ----------

    b : np.ndarray(nt,),
        target measure
    M : np.ndarray(ns, nt),
        cost matrix
    reg : float nu,
        Regularization term > 0
    v : np.ndarray(nt,),
        optimization vector
    i : number int,
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
    >>> lr = 1
    >>> a = ot.utils.unif(n_source)
    >>> b = ot.utils.unif(n_target)
    >>> rng = np.random.RandomState(0)
    >>> X_source = rng.randn(n_source, 2)
    >>> Y_target = rng.randn(n_target, 2)
    >>> M = ot.dist(X_source, Y_target)
    >>> method = "ASGD"
    >>> asgd_pi = stochastic.transportation_matrix_entropic(a, b, M, reg,
                                                            method, numItermax,
                                                            lr)
    >>> print(asgd_pi)

    References
    ----------

    [Genevay et al., 2016] :
                    Stochastic Optimization for Large-scale Optimal Transport,
                     Advances in Neural Information Processing Systems (2016),
                      arXiv preprint arxiv:1605.08527.

    '''

    r = M[i, :] - v
    exp_v = np.exp(-r / reg) * b
    khi = exp_v / (np.sum(exp_v))
    return b - khi


def sag_entropic_transport(a, b, M, reg, numItermax=10000, lr=0.1):
    '''
    Compute the SAG algorithm to solve the regularized discrete measures
        optimal transport max problem

    The function solves the following optimization problem:
    .. math::
        \gamma = arg\min_\gamma <\gamma,M>_F + reg\cdot\Omega(\gamma)
        s.t. \gamma 1 = a
             \gamma^T 1= b
             \gamma \geq 0
    where :
    - M is the (ns,nt) metric cost matrix
    - :math:`\Omega` is the entropic regularization term
        :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
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
    >>> lr = 1
    >>> a = ot.utils.unif(n_source)
    >>> b = ot.utils.unif(n_target)
    >>> rng = np.random.RandomState(0)
    >>> X_source = rng.randn(n_source, 2)
    >>> Y_target = rng.randn(n_target, 2)
    >>> M = ot.dist(X_source, Y_target)
    >>> method = "SAG"
    >>> sag_pi = stochastic.transportation_matrix_entropic(a, b, M, reg,
                                                            method, numItermax,
                                                            lr)
    >>> print(asgd_pi)

    References
    ----------

    [Genevay et al., 2016] :
                    Stochastic Optimization for Large-scale Optimal Transport,
                     Advances in Neural Information Processing Systems (2016),
                      arXiv preprint arxiv:1605.08527.
    '''

    n_source = np.shape(M)[0]
    n_target = np.shape(M)[1]
    v = np.zeros(n_target)
    stored_gradient = np.zeros((n_source, n_target))
    sum_stored_gradient = np.zeros(n_target)
    for _ in range(numItermax):
        i = np.random.randint(n_source)
        cur_coord_grad = a[i] * coordinate_gradient(b, M, reg, v, i)
        sum_stored_gradient += (cur_coord_grad - stored_gradient[i])
        stored_gradient[i] = cur_coord_grad
        v += lr * (1. / n_source) * sum_stored_gradient
    return v


def averaged_sgd_entropic_transport(b, M, reg, numItermax=300000, lr=1):
    '''
    Compute the ASGD algorithm to solve the regularized semi contibous measures
        optimal transport max problem

        The function solves the following optimization problem:
        .. math::
            \gamma = arg\min_\gamma <\gamma,M>_F + reg\cdot\Omega(\gamma)
            s.t. \gamma 1 = a
                 \gamma^T 1= b
                 \gamma \geq 0
        where :
        - M is the (ns,nt) metric cost matrix
        - :math:`\Omega` is the entropic regularization term
            :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
        - a and b are source and target weights (sum to 1)
        The algorithm used for solving the problem is the ASGD algorithm
        as proposed in [18]_ [alg.2]


    Parameters
    ----------

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

    ave_v : np.ndarray(nt,)
        optimization vector

    Examples
    --------

    >>> n_source = 7
    >>> n_target = 4
    >>> reg = 1
    >>> numItermax = 300000
    >>> lr = 1
    >>> a = ot.utils.unif(n_source)
    >>> b = ot.utils.unif(n_target)
    >>> rng = np.random.RandomState(0)
    >>> X_source = rng.randn(n_source, 2)
    >>> Y_target = rng.randn(n_target, 2)
    >>> M = ot.dist(X_source, Y_target)
    >>> method = "ASGD"
    >>> asgd_pi = stochastic.transportation_matrix_entropic(a, b, M, reg,
                                                            method, numItermax,
                                                            lr)
    >>> print(asgd_pi)

    References
    ----------

    [Genevay et al., 2016] :
                    Stochastic Optimization for Large-scale Optimal Transport,
                     Advances in Neural Information Processing Systems (2016),
                      arXiv preprint arxiv:1605.08527.
    '''

    n_source = np.shape(M)[0]
    n_target = np.shape(M)[1]
    cur_v = np.zeros(n_target)
    ave_v = np.zeros(n_target)
    for cur_iter in range(numItermax):
        k = cur_iter + 1
        i = np.random.randint(n_source)
        cur_coord_grad = coordinate_gradient(b, M, reg, cur_v, i)
        cur_v += (lr / np.sqrt(k)) * cur_coord_grad
        ave_v = (1. / k) * cur_v + (1 - 1. / k) * ave_v
    return ave_v


def c_transform_entropic(b, M, reg, v):
    '''
    The goal is to recover u from the c-transform.

    The function computes the c_transform of a dual variable from the other
    dual variable:
    .. math::
        u = v^{c,reg} = -reg \sum_j exp((v - M)/reg) b_j

    where :
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

    Examples
    --------

    >>> n_source = 7
    >>> n_target = 4
    >>> reg = 1
    >>> numItermax = 300000
    >>> lr = 1
    >>> a = ot.utils.unif(n_source)
    >>> b = ot.utils.unif(n_target)
    >>> rng = np.random.RandomState(0)
    >>> X_source = rng.randn(n_source, 2)
    >>> Y_target = rng.randn(n_target, 2)
    >>> M = ot.dist(X_source, Y_target)
    >>> method = "ASGD"
    >>> asgd_pi = stochastic.transportation_matrix_entropic(a, b, M, reg,
                                                            method, numItermax,
                                                            lr)
    >>> print(asgd_pi)

    References
    ----------

    [Genevay et al., 2016] :
                    Stochastic Optimization for Large-scale Optimal Transport,
                     Advances in Neural Information Processing Systems (2016),
                      arXiv preprint arxiv:1605.08527.
    '''

    n_source = np.shape(M)[0]
    n_target = np.shape(M)[1]
    u = np.zeros(n_source)
    for i in range(n_source):
        r = M[i, :] - v
        exp_v = np.exp(-r / reg) * b
        u[i] = - reg * np.log(np.sum(exp_v))
    return u


def transportation_matrix_entropic(a, b, M, reg, method, numItermax=10000,
                                   lr=0.1):
    '''
    Compute the transportation matrix to solve the regularized discrete
        measures optimal transport max problem

    The function solves the following optimization problem:
    .. math::
        \gamma = arg\min_\gamma <\gamma,M>_F + reg\cdot\Omega(\gamma)
        s.t. \gamma 1 = a
             \gamma^T 1= b
             \gamma \geq 0
    where :
    - M is the (ns,nt) metric cost matrix
    - :math:`\Omega` is the entropic regularization term
        :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - a and b are source and target weights (sum to 1)
    The algorithm used for solving the problem is the SAG or ASGD algorithms
    as proposed in [18]_


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
    methode : str,
        used method (SAG or ASGD)
    numItermax : int number
        number of iteration
    lr : float number
        learning rate
    n_source : int number
        size of the source measure
    n_target : int number
        size of the target measure

    Returns
    -------

    pi : np.ndarray(ns, nt)
        transportation matrix

    Examples
    --------

    >>> n_source = 7
    >>> n_target = 4
    >>> reg = 1
    >>> numItermax = 300000
    >>> lr = 1
    >>> a = ot.utils.unif(n_source)
    >>> b = ot.utils.unif(n_target)
    >>> rng = np.random.RandomState(0)
    >>> X_source = rng.randn(n_source, 2)
    >>> Y_target = rng.randn(n_target, 2)
    >>> M = ot.dist(X_source, Y_target)
    >>> method = "ASGD"
    >>> asgd_pi = stochastic.transportation_matrix_entropic(a, b, M, reg,
                                                            method, numItermax,
                                                            lr)
    >>> print(asgd_pi)

    References
    ----------

    [Genevay et al., 2016] :
                    Stochastic Optimization for Large-scale Optimal Transport,
                     Advances in Neural Information Processing Systems (2016),
                      arXiv preprint arxiv:1605.08527.
    '''
    n_source = 7
    n_target = 4
    if method.lower() == "sag":
        opt_v = sag_entropic_transport(a, b, M, reg, numItermax, lr)
    elif method.lower() == "asgd":
        opt_v = averaged_sgd_entropic_transport(b, M, reg, numItermax, lr)
    else:
        print("Please, select your method between SAG and ASGD")
        return None

    opt_u = c_transform_entropic(b, M, reg, opt_v)
    pi = (np.exp((opt_u[:, None] + opt_v[None, :] - M[:, :]) / reg)
          * a[:, None] * b[None, :])
    return pi
