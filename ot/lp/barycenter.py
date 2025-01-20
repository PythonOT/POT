
def free_support_barycenter(
    measures_locations,
    measures_weights,
    X_init,
    b=None,
    weights=None,
    numItermax=100,
    stopThr=1e-7,
    verbose=False,
    log=None,
    numThreads=1,
):
    r"""
    Solves the free support (locations of the barycenters are optimized, not the weights) Wasserstein barycenter problem (i.e. the weighted Frechet mean for the 2-Wasserstein distance), formally:

    .. math::
        \min_\mathbf{X} \quad \sum_{i=1}^N w_i W_2^2(\mathbf{b}, \mathbf{X}, \mathbf{a}_i, \mathbf{X}_i)

    where :

    - :math:`w \in \mathbb{(0, 1)}^{N}`'s are the barycenter weights and sum to one
    - `measure_weights` denotes the :math:`\mathbf{a}_i \in \mathbb{R}^{k_i}`: empirical measures weights (on simplex)
    - `measures_locations` denotes the :math:`\mathbf{X}_i \in \mathbb{R}^{k_i, d}`: empirical measures atoms locations
    - :math:`\mathbf{b} \in \mathbb{R}^{k}` is the desired weights vector of the barycenter

    This problem is considered in :ref:`[20] <references-free-support-barycenter>` (Algorithm 2).
    There are two differences with the following codes:

    - we do not optimize over the weights
    - we do not do line search for the locations updates, we use i.e. :math:`\theta = 1` in
      :ref:`[20] <references-free-support-barycenter>` (Algorithm 2). This can be seen as a discrete
      implementation of the fixed-point algorithm of
      :ref:`[43] <references-free-support-barycenter>` proposed in the continuous setting.

    Parameters
    ----------
    measures_locations : list of N (k_i,d) array-like
        The discrete support of a measure supported on :math:`k_i` locations of a `d`-dimensional space
        (:math:`k_i` can be different for each element of the list)
    measures_weights : list of N (k_i,) array-like
        Numpy arrays where each numpy array has :math:`k_i` non-negatives values summing to one
        representing the weights of each discrete input measure

    X_init : (k,d) array-like
        Initialization of the support locations (on `k` atoms) of the barycenter
    b : (k,) array-like
        Initialization of the weights of the barycenter (non-negatives, sum to 1)
    weights : (N,) array-like
        Initialization of the coefficients of the barycenter (non-negatives, sum to 1)

    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshold on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    numThreads: int or "max", optional (default=1, i.e. OpenMP is not used)
        If compiled with OpenMP, chooses the number of threads to parallelize.
        "max" selects the highest number possible.


    Returns
    -------
    X : (k,d) array-like
        Support locations (on k atoms) of the barycenter


    .. _references-free-support-barycenter:

    References
    ----------
    .. [20] Cuturi, Marco, and Arnaud Doucet. "Fast computation of Wasserstein barycenters." International Conference on Machine Learning. 2014.

    .. [43] Álvarez-Esteban, Pedro C., et al. "A fixed-point approach to barycenters in Wasserstein space." Journal of Mathematical Analysis and Applications 441.2 (2016): 744-762.

    """

    nx = get_backend(*measures_locations, *measures_weights, X_init)

    iter_count = 0

    N = len(measures_locations)
    k = X_init.shape[0]
    d = X_init.shape[1]
    if b is None:
        b = nx.ones((k,), type_as=X_init) / k
    if weights is None:
        weights = nx.ones((N,), type_as=X_init) / N

    X = X_init

    log_dict = {}
    displacement_square_norms = []

    displacement_square_norm = stopThr + 1.0

    while displacement_square_norm > stopThr and iter_count < numItermax:
        T_sum = nx.zeros((k, d), type_as=X_init)

        for measure_locations_i, measure_weights_i, weight_i in zip(
            measures_locations, measures_weights, weights
        ):
            M_i = dist(X, measure_locations_i)
            T_i = emd(b, measure_weights_i, M_i, numThreads=numThreads)
            T_sum = T_sum + weight_i * 1.0 / b[:, None] * nx.dot(
                T_i, measure_locations_i
            )

        displacement_square_norm = nx.sum((T_sum - X) ** 2)
        if log:
            displacement_square_norms.append(displacement_square_norm)

        X = T_sum

        if verbose:
            print(
                "iteration %d, displacement_square_norm=%f\n",
                iter_count,
                displacement_square_norm,
            )

        iter_count += 1

    if log:
        log_dict["displacement_square_norms"] = displacement_square_norms
        return X, log_dict
    else:
        return X


def generalized_free_support_barycenter(
    X_list,
    a_list,
    P_list,
    n_samples_bary,
    Y_init=None,
    b=None,
    weights=None,
    numItermax=100,
    stopThr=1e-7,
    verbose=False,
    log=None,
    numThreads=1,
    eps=0,
):
    r"""
    Solves the free support generalized Wasserstein barycenter problem: finding a barycenter (a discrete measure with
    a fixed amount of points of uniform weights) whose respective projections fit the input measures.
    More formally:

    .. math::
        \min_\gamma \quad \sum_{i=1}^p w_i W_2^2(\nu_i, \mathbf{P}_i\#\gamma)

    where :

    - :math:`\gamma = \sum_{l=1}^n b_l\delta_{y_l}` is the desired barycenter with each :math:`y_l \in \mathbb{R}^d`
    - :math:`\mathbf{b} \in \mathbb{R}^{n}` is the desired weights vector of the barycenter
    - The input measures are :math:`\nu_i = \sum_{j=1}^{k_i}a_{i,j}\delta_{x_{i,j}}`
    - The :math:`\mathbf{a}_i \in \mathbb{R}^{k_i}` are the respective empirical measures weights (on the simplex)
    - The :math:`\mathbf{X}_i \in \mathbb{R}^{k_i, d_i}` are the respective empirical measures atoms locations
    - :math:`w = (w_1, \cdots w_p)` are the barycenter coefficients (on the simplex)
    - Each :math:`\mathbf{P}_i \in \mathbb{R}^{d, d_i}`, and :math:`P_i\#\nu_i = \sum_{j=1}^{k_i}a_{i,j}\delta_{P_ix_{i,j}}`

    As show by :ref:`[42] <references-generalized-free-support-barycenter>`,
    this problem can be re-written as a Wasserstein Barycenter problem,
    which we solve using the free support method :ref:`[20] <references-generalized-free-support-barycenter>`
    (Algorithm 2).

    Parameters
    ----------
    X_list : list of p (k_i,d_i) array-like
        Discrete supports of the input measures: each consists of :math:`k_i` locations of a `d_i`-dimensional space
        (:math:`k_i` can be different for each element of the list)
    a_list : list of p (k_i,) array-like
        Measure weights: each element is a vector (k_i) on the simplex
    P_list : list of p (d_i,d) array-like
        Each :math:`P_i` is a linear map :math:`\mathbb{R}^{d} \rightarrow \mathbb{R}^{d_i}`
    n_samples_bary : int
        Number of barycenter points
    Y_init : (n_samples_bary,d) array-like
        Initialization of the support locations (on `k` atoms) of the barycenter
    b : (n_samples_bary,) array-like
        Initialization of the weights of the barycenter measure (on the simplex)
    weights : (p,) array-like
        Initialization of the coefficients of the barycenter (on the simplex)
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshold on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    numThreads: int or "max", optional (default=1, i.e. OpenMP is not used)
        If compiled with OpenMP, chooses the number of threads to parallelize.
        "max" selects the highest number possible.
    eps: Stability coefficient for the change of variable matrix inversion
        If the :math:`\mathbf{P}_i^T` matrices don't span :math:`\mathbb{R}^d`, the problem is ill-defined and a matrix
        inversion will fail. In this case one may set eps=1e-8 and get a solution anyway (which may make little sense)


    Returns
    -------
    Y : (n_samples_bary,d) array-like
        Support locations (on n_samples_bary atoms) of the barycenter


    .. _references-generalized-free-support-barycenter:
    References
    ----------
    .. [20] Cuturi, M. and Doucet, A.. "Fast computation of Wasserstein barycenters." International Conference on Machine Learning. 2014.

    .. [42] Delon, J., Gozlan, N., and Saint-Dizier, A.. Generalized Wasserstein barycenters between probability measures living on different subspaces. arXiv preprint arXiv:2105.09755, 2021.

    """
    nx = get_backend(*X_list, *a_list, *P_list)
    d = P_list[0].shape[1]
    p = len(X_list)

    if weights is None:
        weights = nx.ones(p, type_as=X_list[0]) / p

    # variable change matrix to reduce the problem to a Wasserstein Barycenter (WB)
    A = eps * nx.eye(
        d, type_as=X_list[0]
    )  # if eps nonzero: will force the invertibility of A
    for P_i, lambda_i in zip(P_list, weights):
        A = A + lambda_i * P_i.T @ P_i
    B = nx.inv(nx.sqrtm(A))

    Z_list = [
        x @ Pi @ B.T for (x, Pi) in zip(X_list, P_list)
    ]  # change of variables -> (WB) problem on Z

    if Y_init is None:
        Y_init = nx.randn(n_samples_bary, d, type_as=X_list[0])

    if b is None:
        b = nx.ones(n_samples_bary, type_as=X_list[0]) / n_samples_bary  # not optimized

    out = free_support_barycenter(
        Z_list,
        a_list,
        Y_init,
        b,
        numItermax=numItermax,
        stopThr=stopThr,
        verbose=verbose,
        log=log,
        numThreads=numThreads,
    )

    if log:  # unpack
        Y, log_dict = out
    else:
        Y = out
        log_dict = None
    Y = Y @ B.T  # return to the Generalized WB formulation

    if log:
        return Y, log_dict
    else:
        return Y
