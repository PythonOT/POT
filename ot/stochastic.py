"""
Stochastic solvers for regularized OT.


"""

# Author: Kilian Fatras <kilian.fatras@gmail.com>
#
# License: MIT License

import numpy as np


##############################################################################
# Optimization toolbox for SEMI - DUAL problems
##############################################################################


def coordinate_grad_semi_dual(b, M, reg, beta, i):
    r'''
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
    b : ndarray, shape (nt,)
        Target measure.
    M : ndarray, shape (ns, nt)
        Cost matrix.
    reg : float
        Regularization term > 0.
    v : ndarray, shape (nt,)
        Dual variable.
    i : int
        Picked number i.

    Returns
    -------
    coordinate gradient : ndarray, shape (nt,)

    Examples
    --------
    >>> import ot
    >>> np.random.seed(0)
    >>> n_source = 7
    >>> n_target = 4
    >>> a = ot.utils.unif(n_source)
    >>> b = ot.utils.unif(n_target)
    >>> X_source = np.random.randn(n_source, 2)
    >>> Y_target = np.random.randn(n_target, 2)
    >>> M = ot.dist(X_source, Y_target)
    >>> ot.stochastic.solve_semi_dual_entropic(a, b, M, reg=1, method="ASGD", numItermax=300000)
    array([[2.53942342e-02, 9.98640673e-02, 1.75945647e-02, 4.27664307e-06],
           [1.21556999e-01, 1.26350515e-02, 1.30491795e-03, 7.36017394e-03],
           [3.54070702e-03, 7.63581358e-02, 6.29581672e-02, 1.32812798e-07],
           [2.60578198e-02, 3.35916645e-02, 8.28023223e-02, 4.05336238e-04],
           [9.86808864e-03, 7.59774324e-04, 1.08702729e-02, 1.21359007e-01],
           [2.17218856e-02, 9.12931802e-04, 1.87962526e-03, 1.18342700e-01],
           [4.14237512e-02, 2.67487857e-02, 7.23016955e-02, 2.38291052e-03]])


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
    r'''
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

    Returns
    -------
    v : ndarray, shape (nt,)
        Dual variable.

    Examples
    --------
    >>> import ot
    >>> np.random.seed(0)
    >>> n_source = 7
    >>> n_target = 4
    >>> a = ot.utils.unif(n_source)
    >>> b = ot.utils.unif(n_target)
    >>> X_source = np.random.randn(n_source, 2)
    >>> Y_target = np.random.randn(n_target, 2)
    >>> M = ot.dist(X_source, Y_target)
    >>> ot.stochastic.solve_semi_dual_entropic(a, b, M, reg=1, method="ASGD", numItermax=300000)
    array([[2.53942342e-02, 9.98640673e-02, 1.75945647e-02, 4.27664307e-06],
           [1.21556999e-01, 1.26350515e-02, 1.30491795e-03, 7.36017394e-03],
           [3.54070702e-03, 7.63581358e-02, 6.29581672e-02, 1.32812798e-07],
           [2.60578198e-02, 3.35916645e-02, 8.28023223e-02, 4.05336238e-04],
           [9.86808864e-03, 7.59774324e-04, 1.08702729e-02, 1.21359007e-01],
           [2.17218856e-02, 9.12931802e-04, 1.87962526e-03, 1.18342700e-01],
           [4.14237512e-02, 2.67487857e-02, 7.23016955e-02, 2.38291052e-03]])

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
    r'''
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

    Returns
    -------
    ave_v : ndarray, shape (nt,)
        dual variable

    Examples
    --------
    >>> import ot
    >>> np.random.seed(0)
    >>> n_source = 7
    >>> n_target = 4
    >>> a = ot.utils.unif(n_source)
    >>> b = ot.utils.unif(n_target)
    >>> X_source = np.random.randn(n_source, 2)
    >>> Y_target = np.random.randn(n_target, 2)
    >>> M = ot.dist(X_source, Y_target)
    >>> ot.stochastic.solve_semi_dual_entropic(a, b, M, reg=1, method="ASGD", numItermax=300000)
    array([[2.53942342e-02, 9.98640673e-02, 1.75945647e-02, 4.27664307e-06],
           [1.21556999e-01, 1.26350515e-02, 1.30491795e-03, 7.36017394e-03],
           [3.54070702e-03, 7.63581358e-02, 6.29581672e-02, 1.32812798e-07],
           [2.60578198e-02, 3.35916645e-02, 8.28023223e-02, 4.05336238e-04],
           [9.86808864e-03, 7.59774324e-04, 1.08702729e-02, 1.21359007e-01],
           [2.17218856e-02, 9.12931802e-04, 1.87962526e-03, 1.18342700e-01],
           [4.14237512e-02, 2.67487857e-02, 7.23016955e-02, 2.38291052e-03]])

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
    r'''
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
    u : ndarray, shape (ns,)
        Dual variable.

    Examples
    --------
    >>> import ot
    >>> np.random.seed(0)
    >>> n_source = 7
    >>> n_target = 4
    >>> a = ot.utils.unif(n_source)
    >>> b = ot.utils.unif(n_target)
    >>> X_source = np.random.randn(n_source, 2)
    >>> Y_target = np.random.randn(n_target, 2)
    >>> M = ot.dist(X_source, Y_target)
    >>> ot.stochastic.solve_semi_dual_entropic(a, b, M, reg=1, method="ASGD", numItermax=300000)
    array([[2.53942342e-02, 9.98640673e-02, 1.75945647e-02, 4.27664307e-06],
           [1.21556999e-01, 1.26350515e-02, 1.30491795e-03, 7.36017394e-03],
           [3.54070702e-03, 7.63581358e-02, 6.29581672e-02, 1.32812798e-07],
           [2.60578198e-02, 3.35916645e-02, 8.28023223e-02, 4.05336238e-04],
           [9.86808864e-03, 7.59774324e-04, 1.08702729e-02, 1.21359007e-01],
           [2.17218856e-02, 9.12931802e-04, 1.87962526e-03, 1.18342700e-01],
           [4.14237512e-02, 2.67487857e-02, 7.23016955e-02, 2.38291052e-03]])

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
    r'''
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

    Examples
    --------
    >>> import ot
    >>> np.random.seed(0)
    >>> n_source = 7
    >>> n_target = 4
    >>> a = ot.utils.unif(n_source)
    >>> b = ot.utils.unif(n_target)
    >>> X_source = np.random.randn(n_source, 2)
    >>> Y_target = np.random.randn(n_target, 2)
    >>> M = ot.dist(X_source, Y_target)
    >>> ot.stochastic.solve_semi_dual_entropic(a, b, M, reg=1, method="ASGD", numItermax=300000)
    array([[2.53942342e-02, 9.98640673e-02, 1.75945647e-02, 4.27664307e-06],
           [1.21556999e-01, 1.26350515e-02, 1.30491795e-03, 7.36017394e-03],
           [3.54070702e-03, 7.63581358e-02, 6.29581672e-02, 1.32812798e-07],
           [2.60578198e-02, 3.35916645e-02, 8.28023223e-02, 4.05336238e-04],
           [9.86808864e-03, 7.59774324e-04, 1.08702729e-02, 1.21359007e-01],
           [2.17218856e-02, 9.12931802e-04, 1.87962526e-03, 1.18342700e-01],
           [4.14237512e-02, 2.67487857e-02, 7.23016955e-02, 2.38291052e-03]])

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
    r'''
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
    grad : ndarray, shape (ns,)
        partial grad F

    Examples
    --------
    >>> import ot
    >>> np.random.seed(0)
    >>> n_source = 7
    >>> n_target = 4
    >>> a = ot.utils.unif(n_source)
    >>> b = ot.utils.unif(n_target)
    >>> X_source = np.random.randn(n_source, 2)
    >>> Y_target = np.random.randn(n_target, 2)
    >>> M = ot.dist(X_source, Y_target)
    >>> sgd_dual_pi, log = ot.stochastic.solve_dual_entropic(a, b, M, reg=1, batch_size=3, numItermax=30000, lr=0.1, log=True)
    >>> log['alpha']
    array([0.71759102, 1.57057384, 0.85576566, 0.1208211 , 0.59190466,
           1.197148  , 0.17805133])
    >>> log['beta']
    array([0.49741367, 0.57478564, 1.40075528, 2.75890102])
    >>> sgd_dual_pi
    array([[2.09730063e-02, 8.38169324e-02, 7.50365455e-03, 8.72731415e-09],
           [5.58432437e-03, 5.89881299e-04, 3.09558411e-05, 8.35469849e-07],
           [3.26489515e-03, 7.15536035e-02, 2.99778211e-02, 3.02601593e-10],
           [4.05390622e-02, 5.31085068e-02, 6.65191787e-02, 1.55812785e-06],
           [7.82299812e-02, 6.12099102e-03, 4.44989098e-02, 2.37719187e-03],
           [5.06266486e-02, 2.16230494e-03, 2.26215141e-03, 6.81514609e-04],
           [6.06713990e-02, 3.98139808e-02, 5.46829338e-02, 8.62371424e-06]])

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
    r'''
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

    Returns
    -------
    alpha : ndarray, shape (ns,)
        dual variable
    beta : ndarray, shape (nt,)
        dual variable

    Examples
    --------
    >>> import ot
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
    >>> sgd_dual_pi, log = ot.stochastic.solve_dual_entropic(a, b, M, reg, batch_size, numItermax, lr, log)
    >>> log['alpha']
    array([0.64171798, 1.27932201, 0.78132257, 0.15638935, 0.54888354,
           1.03663469, 0.20595781])
    >>> log['beta']
    array([0.51207194, 0.58033189, 1.28922676, 2.26859736])
    >>> sgd_dual_pi
    array([[1.97276541e-02, 7.81248547e-02, 6.22136048e-03, 4.95442423e-09],
           [4.23494310e-03, 4.43286263e-04, 2.06927079e-05, 3.82389139e-07],
           [3.07542414e-03, 6.67897769e-02, 2.48904999e-02, 1.72030247e-10],
           [4.26271990e-02, 5.53375455e-02, 6.16535024e-02, 9.88812650e-07],
           [7.60423265e-02, 5.89585256e-03, 3.81267087e-02, 1.39458256e-03],
           [4.37557504e-02, 1.85189176e-03, 1.72335760e-03, 3.55491279e-04],
           [6.33096109e-02, 4.11683954e-02, 5.02962051e-02, 5.43097516e-06]])

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
    r'''
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

    Examples
    --------
    >>> import ot
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
    >>> sgd_dual_pi, log = ot.stochastic.solve_dual_entropic(a, b, M, reg, batch_size, numItermax, lr, log)
    >>> log['alpha']
    array([0.64057733, 1.2683513 , 0.75610161, 0.16024284, 0.54926534,
           1.0514201 , 0.19958936])
    >>> log['beta']
    array([0.51372571, 0.58843489, 1.27993921, 2.24344807])
    >>> sgd_dual_pi
    array([[1.97377795e-02, 7.86706853e-02, 6.15682001e-03, 4.82586997e-09],
           [4.19566963e-03, 4.42016865e-04, 2.02777272e-05, 3.68823708e-07],
           [3.00379244e-03, 6.56562018e-02, 2.40462171e-02, 1.63579656e-10],
           [4.28626062e-02, 5.60031599e-02, 6.13193826e-02, 9.67977735e-07],
           [7.61972739e-02, 5.94609051e-03, 3.77886693e-02, 1.36046648e-03],
           [4.44810042e-02, 1.89476742e-03, 1.73285847e-03, 3.51826036e-04],
           [6.30118293e-02, 4.12398660e-02, 4.95148998e-02, 5.26247246e-06]])

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
