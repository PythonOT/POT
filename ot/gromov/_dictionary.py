# -*- coding: utf-8 -*-
"""
(Fused) Gromov-Wasserstein dictionary learning.
"""

# Author: Rémi Flamary <remi.flamary@unice.fr>
#         Cédric Vincent-Cuaz <cedvincentcuaz@gmail.com>
#
# License: MIT License

import numpy as np


from ..utils import unif, check_random_state
from ..backend import get_backend
from ._gw import gromov_wasserstein, fused_gromov_wasserstein


def gromov_wasserstein_dictionary_learning(
    Cs,
    D,
    nt,
    reg=0.0,
    ps=None,
    q=None,
    epochs=20,
    batch_size=32,
    learning_rate=1.0,
    Cdict_init=None,
    projection="nonnegative_symmetric",
    use_log=True,
    tol_outer=10 ** (-5),
    tol_inner=10 ** (-5),
    max_iter_outer=20,
    max_iter_inner=200,
    use_adam_optimizer=True,
    verbose=False,
    random_state=None,
    **kwargs,
):
    r"""
    Infer Gromov-Wasserstein linear dictionary :math:`\{ (\mathbf{C_{dict}[d]}, q) \}_{d \in [D]}`  from the list of structures :math:`\{ (\mathbf{C_s},\mathbf{p_s}) \}_s`

    .. math::
        \min_{\mathbf{C_{dict}}, \{\mathbf{w_s} \}_{s \leq S}} \sum_{s=1}^S  GW_2(\mathbf{C_s}, \sum_{d=1}^D w_{s,d}\mathbf{C_{dict}[d]}, \mathbf{p_s}, \mathbf{q}) - reg\| \mathbf{w_s}  \|_2^2

    such that, :math:`\forall s \leq S` :

        - :math:`\mathbf{w_s}^\top \mathbf{1}_D = 1`
        - :math:`\mathbf{w_s} \geq \mathbf{0}_D`

    Where :

    - :math:`\forall s \leq S, \mathbf{C_s}` is a (ns,ns) pairwise similarity matrix of variable size ns.
    - :math:`\mathbf{C_{dict}}` is a (D, nt, nt) tensor of D pairwise similarity matrix of fixed size nt.
    - :math:`\forall s \leq S, \mathbf{p_s}` is the source distribution corresponding to :math:`\mathbf{C_s}`
    - :math:`\mathbf{q}` is the target distribution assigned to every structures in the embedding space.
    - reg is the regularization coefficient.

    The stochastic algorithm used for estimating the graph dictionary atoms as proposed in [38]_

    Parameters
    ----------
    Cs : list of S symmetric array-like, shape (ns, ns)
        List of Metric/Graph cost matrices of variable size (ns, ns).
    D: int
        Number of dictionary atoms to learn
    nt: int
        Number of samples within each dictionary atoms
    reg : float, optional
        Coefficient of the negative quadratic regularization used to promote sparsity of w. The default is 0.
    ps : list of S array-like, shape (ns,), optional
        Distribution in each source space C of Cs. Default is None and corresponds to uniform distributions.
    q : array-like, shape (nt,), optional
        Distribution in the embedding space whose structure will be learned. Default is None and corresponds to uniform distributions.
    epochs: int, optional
        Number of epochs used to learn the dictionary. Default is 32.
    batch_size: int, optional
        Batch size for each stochastic gradient update of the dictionary. Set to the dataset size if the provided batch_size is higher than the dataset size. Default is 32.
    learning_rate: float, optional
        Learning rate used for the stochastic gradient descent. Default is 1.
    Cdict_init: list of D array-like with shape (nt, nt), optional
        Used to initialize the dictionary.
        If set to None (Default), the dictionary will be initialized randomly.
        Else Cdict must have shape (D, nt, nt) i.e match provided shape features.
    projection: str , optional
        If 'nonnegative' and/or 'symmetric' is in projection, the corresponding projection will be performed at each stochastic update of the dictionary
        Else the set of atoms is :math:`R^{nt * nt}`. Default is 'nonnegative_symmetric'
    log: bool, optional
        If set to True, losses evolution by batches and epochs are tracked. Default is False.
    use_adam_optimizer: bool, optional
        If set to True, adam optimizer with default settings is used as adaptative learning rate strategy.
        Else perform SGD with fixed learning rate. Default is True.
    tol_outer : float, optional
        Solver precision for the BCD algorithm, measured by absolute relative error on consecutive losses. Default is :math:`10^{-5}`.
    tol_inner : float, optional
        Solver precision for the Conjugate Gradient algorithm used to get optimal w at a fixed transport, measured by absolute relative error on consecutive losses. Default is :math:`10^{-5}`.
    max_iter_outer : int, optional
        Maximum number of iterations for the BCD. Default is 20.
    max_iter_inner : int, optional
        Maximum number of iterations for the Conjugate Gradient. Default is 200.
    verbose : bool, optional
        Print the reconstruction loss every epoch. Default is False.
    random_state : int, RandomState instance or None, default=None
        Determines random number generation. Pass an int for reproducible
        output across multiple function calls.

    Returns
    -------

    Cdict_best_state : D array-like, shape (D,nt,nt)
        Metric/Graph cost matrices composing the dictionary.
        The dictionary leading to the best loss over an epoch is saved and returned.
    log: dict
        If use_log is True, contains loss evolutions by batches and epochs.

    References
    -------
    .. [38] C. Vincent-Cuaz, T. Vayer, R. Flamary, M. Corneli, N. Courty, Online
        Graph Dictionary Learning, International Conference on Machine Learning
        (ICML), 2021.
    """
    # Handle backend of non-optional arguments
    Cs0 = Cs
    nx = get_backend(*Cs0)
    Cs = [nx.to_numpy(C) for C in Cs0]
    dataset_size = len(Cs)
    # Handle backend of optional arguments
    if ps is None:
        ps = [unif(C.shape[0]) for C in Cs]
    else:
        ps = [nx.to_numpy(p) for p in ps]
    if q is None:
        q = unif(nt)
    else:
        q = nx.to_numpy(q)
    rng = check_random_state(random_state)
    if Cdict_init is None:
        # Initialize randomly structures of dictionary atoms based on samples
        dataset_means = [C.mean() for C in Cs]
        Cdict = rng.normal(
            loc=np.mean(dataset_means), scale=np.std(dataset_means), size=(D, nt, nt)
        )
    else:
        Cdict = nx.to_numpy(Cdict_init).copy()
        assert Cdict.shape == (D, nt, nt)

    if "symmetric" in projection:
        Cdict = 0.5 * (Cdict + Cdict.transpose((0, 2, 1)))
        symmetric = True
    else:
        symmetric = False
    if "nonnegative" in projection:
        Cdict[Cdict < 0.0] = 0
    if use_adam_optimizer:
        adam_moments = _initialize_adam_optimizer(Cdict)

    log = {"loss_batches": [], "loss_epochs": []}
    const_q = q[:, None] * q[None, :]
    Cdict_best_state = Cdict.copy()
    loss_best_state = np.inf
    if batch_size > dataset_size:
        batch_size = dataset_size
    iter_by_epoch = dataset_size // batch_size + int((dataset_size % batch_size) > 0)

    for epoch in range(epochs):
        cumulated_loss_over_epoch = 0.0

        for _ in range(iter_by_epoch):
            # batch sampling
            batch = rng.choice(range(dataset_size), size=batch_size, replace=False)
            cumulated_loss_over_batch = 0.0
            unmixings = np.zeros((batch_size, D))
            Cs_embedded = np.zeros((batch_size, nt, nt))
            Ts = [None] * batch_size

            for batch_idx, C_idx in enumerate(batch):
                # BCD solver for Gromov-Wasserstein linear unmixing used independently on each structure of the sampled batch
                (
                    unmixings[batch_idx],
                    Cs_embedded[batch_idx],
                    Ts[batch_idx],
                    current_loss,
                ) = gromov_wasserstein_linear_unmixing(
                    Cs[C_idx],
                    Cdict,
                    reg=reg,
                    p=ps[C_idx],
                    q=q,
                    tol_outer=tol_outer,
                    tol_inner=tol_inner,
                    max_iter_outer=max_iter_outer,
                    max_iter_inner=max_iter_inner,
                    symmetric=symmetric,
                    **kwargs,
                )
                cumulated_loss_over_batch += current_loss
            cumulated_loss_over_epoch += cumulated_loss_over_batch

            if use_log:
                log["loss_batches"].append(cumulated_loss_over_batch)

            # Stochastic projected gradient step over dictionary atoms
            grad_Cdict = np.zeros_like(Cdict)
            for batch_idx, C_idx in enumerate(batch):
                shared_term_structures = Cs_embedded[batch_idx] * const_q - (
                    Cs[C_idx].dot(Ts[batch_idx])
                ).T.dot(Ts[batch_idx])
                grad_Cdict += (
                    unmixings[batch_idx][:, None, None]
                    * shared_term_structures[None, :, :]
                )
            grad_Cdict *= 2 / batch_size
            if use_adam_optimizer:
                Cdict, adam_moments = _adam_stochastic_updates(
                    Cdict, grad_Cdict, learning_rate, adam_moments
                )
            else:
                Cdict -= learning_rate * grad_Cdict
            if "symmetric" in projection:
                Cdict = 0.5 * (Cdict + Cdict.transpose((0, 2, 1)))
            if "nonnegative" in projection:
                Cdict[Cdict < 0.0] = 0.0

        if use_log:
            log["loss_epochs"].append(cumulated_loss_over_epoch)
        if loss_best_state > cumulated_loss_over_epoch:
            loss_best_state = cumulated_loss_over_epoch
            Cdict_best_state = Cdict.copy()
        if verbose:
            print(
                "--- epoch =",
                epoch,
                " cumulated reconstruction error: ",
                cumulated_loss_over_epoch,
            )

    return nx.from_numpy(Cdict_best_state), log


def _initialize_adam_optimizer(variable):
    # Initialization for our numpy implementation of adam optimizer
    atoms_adam_m = np.zeros_like(variable)  # Initialize first  moment tensor
    atoms_adam_v = np.zeros_like(variable)  # Initialize second moment tensor
    atoms_adam_count = 1

    return {"mean": atoms_adam_m, "var": atoms_adam_v, "count": atoms_adam_count}


def _adam_stochastic_updates(
    variable, grad, learning_rate, adam_moments, beta_1=0.9, beta_2=0.99, eps=1e-09
):
    adam_moments["mean"] = beta_1 * adam_moments["mean"] + (1 - beta_1) * grad
    adam_moments["var"] = beta_2 * adam_moments["var"] + (1 - beta_2) * (grad**2)
    unbiased_m = adam_moments["mean"] / (1 - beta_1 ** adam_moments["count"])
    unbiased_v = adam_moments["var"] / (1 - beta_2 ** adam_moments["count"])
    variable -= learning_rate * unbiased_m / (np.sqrt(unbiased_v) + eps)
    adam_moments["count"] += 1

    return variable, adam_moments


def gromov_wasserstein_linear_unmixing(
    C,
    Cdict,
    reg=0.0,
    p=None,
    q=None,
    tol_outer=10 ** (-5),
    tol_inner=10 ** (-5),
    max_iter_outer=20,
    max_iter_inner=200,
    symmetric=None,
    **kwargs,
):
    r"""
    Returns the Gromov-Wasserstein linear unmixing of :math:`(\mathbf{C},\mathbf{p})` onto the dictionary :math:`\{ (\mathbf{C_{dict}[d]}, \mathbf{q}) \}_{d \in [D]}`.

    .. math::
        \min_{ \mathbf{w}}  GW_2(\mathbf{C}, \sum_{d=1}^D w_d\mathbf{C_{dict}[d]}, \mathbf{p}, \mathbf{q}) - reg \| \mathbf{w}  \|_2^2

    such that:

        - :math:`\mathbf{w}^\top \mathbf{1}_D = 1`
        - :math:`\mathbf{w} \geq \mathbf{0}_D`

    Where :

    - :math:`\mathbf{C}` is the (ns,ns) pairwise similarity matrix.
    - :math:`\mathbf{C_{dict}}` is a (D, nt, nt) tensor of D pairwise similarity matrices of size nt.
    - :math:`\mathbf{p}` and :math:`\mathbf{q}` are source and target weights.
    - reg is the regularization coefficient.

    The algorithm used for solving the problem is a Block Coordinate Descent as discussed in [38]_ , algorithm 1.

    Parameters
    ----------
    C : array-like, shape (ns, ns)
        Metric/Graph cost matrix.
    Cdict : D array-like, shape (D,nt,nt)
        Metric/Graph cost matrices composing the dictionary on which to embed C.
    reg : float, optional.
        Coefficient of the negative quadratic regularization used to promote sparsity of w. Default is 0.
    p : array-like, shape (ns,), optional
        Distribution in the source space C. Default is None and corresponds to uniform distribution.
    q : array-like, shape (nt,), optional
        Distribution in the space depicted by the dictionary. Default is None and corresponds to uniform distribution.
    tol_outer : float, optional
        Solver precision for the BCD algorithm.
    tol_inner : float, optional
        Solver precision for the Conjugate Gradient algorithm used to get optimal w at a fixed transport. Default is :math:`10^{-5}`.
    max_iter_outer : int, optional
        Maximum number of iterations for the BCD. Default is 20.
    max_iter_inner : int, optional
        Maximum number of iterations for the Conjugate Gradient. Default is 200.

    Returns
    -------
    w: array-like, shape (D,)
        Gromov-Wasserstein linear unmixing of :math:`(\mathbf{C},\mathbf{p})` onto the span of the dictionary.
    Cembedded: array-like, shape (nt,nt)
        embedded structure of :math:`(\mathbf{C},\mathbf{p})` onto the dictionary, :math:`\sum_d w_d\mathbf{C_{dict}[d]}`.
    T: array-like (ns, nt)
        Gromov-Wasserstein transport plan between :math:`(\mathbf{C},\mathbf{p})` and :math:`(\sum_d w_d\mathbf{C_{dict}[d]}, \mathbf{q})`
    current_loss: float
        reconstruction error
    References
    -------
    .. [38] C. Vincent-Cuaz, T. Vayer, R. Flamary, M. Corneli, N. Courty, Online
        Graph Dictionary Learning, International Conference on Machine Learning
        (ICML), 2021.
    """
    C0, Cdict0 = C, Cdict
    nx = get_backend(C0, Cdict0)
    C = nx.to_numpy(C0)
    Cdict = nx.to_numpy(Cdict0)
    if p is None:
        p = unif(C.shape[0])
    else:
        p = nx.to_numpy(p)

    if q is None:
        q = unif(Cdict.shape[-1])
    else:
        q = nx.to_numpy(q)

    T = p[:, None] * q[None, :]
    D = len(Cdict)

    w = unif(D)  # Initialize uniformly the unmixing w
    Cembedded = np.sum(w[:, None, None] * Cdict, axis=0)

    const_q = q[:, None] * q[None, :]
    # Trackers for BCD convergence
    convergence_criterion = np.inf
    current_loss = 10**15
    outer_count = 0

    while (convergence_criterion > tol_outer) and (outer_count < max_iter_outer):
        previous_loss = current_loss
        # 1. Solve GW transport between (C,p) and (\sum_d Cdictionary[d],q) fixing the unmixing w
        T, log = gromov_wasserstein(
            C1=C,
            C2=Cembedded,
            p=p,
            q=q,
            loss_fun="square_loss",
            G0=T,
            max_iter=max_iter_inner,
            tol_rel=tol_inner,
            tol_abs=0.0,
            log=True,
            armijo=False,
            symmetric=symmetric,
            **kwargs,
        )
        current_loss = log["gw_dist"]
        if reg != 0:
            current_loss -= reg * np.sum(w**2)

        # 2. Solve linear unmixing problem over w with a fixed transport plan T
        w, Cembedded, current_loss = _cg_gromov_wasserstein_unmixing(
            C=C,
            Cdict=Cdict,
            Cembedded=Cembedded,
            w=w,
            const_q=const_q,
            T=T,
            starting_loss=current_loss,
            reg=reg,
            tol=tol_inner,
            max_iter=max_iter_inner,
            **kwargs,
        )

        if previous_loss != 0:
            convergence_criterion = abs(previous_loss - current_loss) / abs(
                previous_loss
            )
        else:  # handle numerical issues around 0
            convergence_criterion = abs(previous_loss - current_loss) / 10 ** (-15)
        outer_count += 1

    return (
        nx.from_numpy(w),
        nx.from_numpy(Cembedded),
        nx.from_numpy(T),
        nx.from_numpy(current_loss),
    )


def _cg_gromov_wasserstein_unmixing(
    C,
    Cdict,
    Cembedded,
    w,
    const_q,
    T,
    starting_loss,
    reg=0.0,
    tol=10 ** (-5),
    max_iter=200,
    **kwargs,
):
    r"""
    Returns for a fixed admissible transport plan,
    the linear unmixing w minimizing the Gromov-Wasserstein cost between :math:`(\mathbf{C},\mathbf{p})` and :math:`(\sum_d w[d]*\mathbf{C_{dict}[d]}, \mathbf{q})`

    .. math::
        \min_{\mathbf{w}}  \sum_{ijkl} (C_{i,j} - \sum_{d=1}^D w_d*C_{dict}[d]_{k,l} )^2 T_{i,k}T_{j,l} - reg* \| \mathbf{w}  \|_2^2


    Such that:

        - :math:`\mathbf{w}^\top \mathbf{1}_D = 1`
        - :math:`\mathbf{w} \geq \mathbf{0}_D`

    Where :

    - :math:`\mathbf{C}` is the (ns,ns) pairwise similarity matrix.
    - :math:`\mathbf{C_{dict}}` is a (D, nt, nt) tensor of D pairwise similarity matrices of nt points.
    - :math:`\mathbf{p}` and :math:`\mathbf{q}` are source and target weights.
    - :math:`\mathbf{w}` is the linear unmixing of :math:`(\mathbf{C}, \mathbf{p})` onto :math:`(\sum_d w_d \mathbf{Cdict[d]}, \mathbf{q})`.
    - :math:`\mathbf{T}` is the optimal transport plan conditioned by the current state of :math:`\mathbf{w}`.
    - reg is the regularization coefficient.

    The algorithm used for solving the problem is a Conditional Gradient Descent as discussed in [38]_

    Parameters
    ----------

    C : array-like, shape (ns, ns)
        Metric/Graph cost matrix.
    Cdict : list of D array-like, shape (nt,nt)
        Metric/Graph cost matrices composing the dictionary on which to embed C.
        Each matrix in the dictionary must have the same size (nt,nt).
    Cembedded: array-like, shape (nt,nt)
        Embedded structure :math:`(\sum_d w[d]*Cdict[d],q)` of :math:`(\mathbf{C},\mathbf{p})` onto the dictionary. Used to avoid redundant computations.
    w: array-like, shape (D,)
        Linear unmixing of the input structure onto the dictionary
    const_q: array-like, shape (nt,nt)
        product matrix :math:`\mathbf{q}\mathbf{q}^\top` where q is the target space distribution. Used to avoid redundant computations.
    T: array-like, shape (ns,nt)
        fixed transport plan between the input structure and its representation in the dictionary.
    p : array-like, shape (ns,)
        Distribution in the source space.
    q : array-like, shape (nt,)
        Distribution in the embedding space depicted by the dictionary.
    reg : float, optional.
        Coefficient of the negative quadratic regularization used to promote sparsity of w. Default is 0.

    Returns
    -------
    w: ndarray (D,)
        optimal unmixing of :math:`(\mathbf{C},\mathbf{p})` onto the dictionary span given OT starting from previously optimal unmixing.
    """
    convergence_criterion = np.inf
    current_loss = starting_loss
    count = 0
    const_TCT = np.transpose(C.dot(T)).dot(T)

    while (convergence_criterion > tol) and (count < max_iter):
        previous_loss = current_loss
        # 1) Compute gradient at current point w
        grad_w = 2 * np.sum(
            Cdict
            * (Cembedded[None, :, :] * const_q[None, :, :] - const_TCT[None, :, :]),
            axis=(1, 2),
        )
        grad_w -= 2 * reg * w

        # 2) Conditional gradient direction finding: x= \argmin_x x^T.grad_w
        min_ = np.min(grad_w)
        x = (grad_w == min_).astype(np.float64)
        x /= np.sum(x)

        # 3) Line-search step: solve \argmin_{\gamma \in [0,1]} a*gamma^2 + b*gamma + c
        gamma, a, b, Cembedded_diff = _linesearch_gromov_wasserstein_unmixing(
            w, grad_w, x, Cdict, Cembedded, const_q, const_TCT, reg
        )

        # 4) Updates: w <-- (1-gamma)*w + gamma*x
        w += gamma * (x - w)
        Cembedded += gamma * Cembedded_diff
        current_loss += a * (gamma**2) + b * gamma

        if previous_loss != 0:  # not that the loss can be negative if reg >0
            convergence_criterion = abs(previous_loss - current_loss) / abs(
                previous_loss
            )
        else:  # handle numerical issues around 0
            convergence_criterion = abs(previous_loss - current_loss) / 10 ** (-15)
        count += 1

    return w, Cembedded, current_loss


def _linesearch_gromov_wasserstein_unmixing(
    w, grad_w, x, Cdict, Cembedded, const_q, const_TCT, reg, **kwargs
):
    r"""
    Compute optimal steps for the line search problem of Gromov-Wasserstein linear unmixing
    .. math::
        \min_{\gamma \in [0,1]}  \sum_{ijkl} (C_{i,j} - \sum_{d=1}^D z_d(\gamma)C_{dict}[d]_{k,l} )^2 T_{i,k}T_{j,l} - reg\| \mathbf{z}(\gamma)  \|_2^2


    Such that:

        - :math:`\mathbf{z}(\gamma) = (1- \gamma)\mathbf{w} + \gamma \mathbf{x}`

    Parameters
    ----------

    w : array-like, shape (D,)
        Unmixing.
    grad_w : array-like, shape (D, D)
        Gradient of the reconstruction loss with respect to w.
    x: array-like, shape (D,)
        Conditional gradient direction.
    Cdict : list of D array-like, shape (nt,nt)
        Metric/Graph cost matrices composing the dictionary on which to embed C.
        Each matrix in the dictionary must have the same size (nt,nt).
    Cembedded: array-like, shape (nt,nt)
        Embedded structure :math:`(\sum_d w_dCdict[d],q)` of :math:`(\mathbf{C},\mathbf{p})` onto the dictionary. Used to avoid redundant computations.
    const_q: array-like, shape (nt,nt)
        product matrix :math:`\mathbf{q}\mathbf{q}^\top` where q is the target space distribution. Used to avoid redundant computations.
    const_TCT: array-like, shape (nt, nt)
        :math:`\mathbf{T}^\top \mathbf{C}^\top \mathbf{T}`. Used to avoid redundant computations.
    Returns
    -------
    gamma: float
        Optimal value for the line-search step
    a: float
        Constant factor appearing in the factorization :math:`a \gamma^2 + b \gamma +c` of the reconstruction loss
    b: float
        Constant factor appearing in the factorization :math:`a \gamma^2 + b \gamma +c` of the reconstruction loss
    Cembedded_diff: numpy array, shape (nt, nt)
        Difference between models evaluated in :math:`\mathbf{w}` and in :math:`\mathbf{w}`.
    reg : float, optional.
        Coefficient of the negative quadratic regularization used to promote sparsity of :math:`\mathbf{w}`.
    """

    # 3) Line-search step: solve \argmin_{\gamma \in [0,1]} a*gamma^2 + b*gamma + c
    Cembedded_x = np.sum(x[:, None, None] * Cdict, axis=0)
    Cembedded_diff = Cembedded_x - Cembedded
    trace_diffx = np.sum(Cembedded_diff * Cembedded_x * const_q)
    trace_diffw = np.sum(Cembedded_diff * Cembedded * const_q)
    a = trace_diffx - trace_diffw
    b = 2 * (trace_diffw - np.sum(Cembedded_diff * const_TCT))
    if reg != 0:
        a -= reg * np.sum((x - w) ** 2)
        b -= 2 * reg * np.sum(w * (x - w))

    if a > 0:
        gamma = min(1, max(0, -b / (2 * a)))
    elif a + b < 0:
        gamma = 1
    else:
        gamma = 0

    return gamma, a, b, Cembedded_diff


def fused_gromov_wasserstein_dictionary_learning(
    Cs,
    Ys,
    D,
    nt,
    alpha,
    reg=0.0,
    ps=None,
    q=None,
    epochs=20,
    batch_size=32,
    learning_rate_C=1.0,
    learning_rate_Y=1.0,
    Cdict_init=None,
    Ydict_init=None,
    projection="nonnegative_symmetric",
    use_log=False,
    tol_outer=10 ** (-5),
    tol_inner=10 ** (-5),
    max_iter_outer=20,
    max_iter_inner=200,
    use_adam_optimizer=True,
    verbose=False,
    random_state=None,
    **kwargs,
):
    r"""
    Infer Fused Gromov-Wasserstein linear dictionary :math:`\{ (\mathbf{C_{dict}[d]}, \mathbf{Y_{dict}[d]}, \mathbf{q}) \}_{d \in [D]}`  from the list of S attributed structures :math:`\{ (\mathbf{C_s}, \mathbf{Y_s},\mathbf{p_s}) \}_s`

    .. math::
        \min_{\mathbf{C_{dict}},\mathbf{Y_{dict}}, \{\mathbf{w_s}\}_{s}} \sum_{s=1}^S  FGW_{2,\alpha}(\mathbf{C_s}, \mathbf{Y_s}, \sum_{d=1}^D w_{s,d}\mathbf{C_{dict}[d]},\sum_{d=1}^D w_{s,d}\mathbf{Y_{dict}[d]}, \mathbf{p_s}, \mathbf{q}) \\ - reg\| \mathbf{w_s}  \|_2^2


    Such that :math:`\forall s \leq S` :

    - :math:`\mathbf{w_s}^\top \mathbf{1}_D = 1`
    - :math:`\mathbf{w_s} \geq \mathbf{0}_D`

    Where :

    - :math:`\forall s \leq S, \mathbf{C_s}` is a (ns,ns) pairwise similarity matrix of variable size ns.
    - :math:`\forall s \leq S, \mathbf{Y_s}` is a (ns,d) features matrix of variable size ns and fixed dimension d.
    - :math:`\mathbf{C_{dict}}` is a (D, nt, nt) tensor of D pairwise similarity matrix of fixed size nt.
    - :math:`\mathbf{Y_{dict}}` is a (D, nt, d) tensor of D features matrix of fixed size nt and fixed dimension d.
    - :math:`\forall s \leq S, \mathbf{p_s}` is the source distribution corresponding to :math:`\mathbf{C_s}`
    - :math:`\mathbf{q}` is the target distribution assigned to every structures in the embedding space.
    - :math:`\alpha` is the trade-off parameter of Fused Gromov-Wasserstein
    - reg is the regularization coefficient.


    The stochastic algorithm used for estimating the attributed graph dictionary atoms as proposed in [38]_

    Parameters
    ----------
    Cs : list of S symmetric array-like, shape (ns, ns)
        List of Metric/Graph cost matrices of variable size (ns,ns).
    Ys : list of S array-like, shape (ns, d)
        List of feature matrix of variable size (ns,d) with d fixed.
    D: int
        Number of dictionary atoms to learn
    nt: int
        Number of samples within each dictionary atoms
    alpha : float
        Trade-off parameter of Fused Gromov-Wasserstein
    reg : float, optional
        Coefficient of the negative quadratic regularization used to promote sparsity of w. The default is 0.
    ps : list of S array-like, shape (ns,), optional
        Distribution in each source space C of Cs. Default is None and corresponds to uniform distributions.
    q : array-like, shape (nt,), optional
        Distribution in the embedding space whose structure will be learned. Default is None and corresponds to uniform distributions.
    epochs: int, optional
        Number of epochs used to learn the dictionary. Default is 32.
    batch_size: int, optional
        Batch size for each stochastic gradient update of the dictionary. Set to the dataset size if the provided batch_size is higher than the dataset size. Default is 32.
    learning_rate_C: float, optional
        Learning rate used for the stochastic gradient descent on Cdict. Default is 1.
    learning_rate_Y: float, optional
        Learning rate used for the stochastic gradient descent on Ydict. Default is 1.
    Cdict_init: list of D array-like with shape (nt, nt), optional
        Used to initialize the dictionary structures Cdict.
        If set to None (Default), the dictionary will be initialized randomly.
        Else Cdict must have shape (D, nt, nt) i.e match provided shape features.
    Ydict_init: list of D array-like with shape (nt, d), optional
        Used to initialize the dictionary features Ydict.
        If set to None, the dictionary features will be initialized randomly.
        Else Ydict must have shape (D, nt, d) where d is the features dimension of inputs Ys and also match provided shape features.
    projection: str, optional
        If 'nonnegative' and/or 'symmetric' is in projection, the corresponding projection will be performed at each stochastic update of the dictionary
        Else the set of atoms is :math:`R^{nt * nt}`. Default is 'nonnegative_symmetric'
    log: bool, optional
        If set to True, losses evolution by batches and epochs are tracked. Default is False.
    use_adam_optimizer: bool, optional
        If set to True, adam optimizer with default settings is used as adaptative learning rate strategy.
        Else perform SGD with fixed learning rate. Default is True.
    tol_outer : float, optional
        Solver precision for the BCD algorithm, measured by absolute relative error on consecutive losses. Default is :math:`10^{-5}`.
    tol_inner : float, optional
        Solver precision for the Conjugate Gradient algorithm used to get optimal w at a fixed transport, measured by absolute relative error on consecutive losses. Default is :math:`10^{-5}`.
    max_iter_outer : int, optional
        Maximum number of iterations for the BCD. Default is 20.
    max_iter_inner : int, optional
        Maximum number of iterations for the Conjugate Gradient. Default is 200.
    verbose : bool, optional
        Print the reconstruction loss every epoch. Default is False.
    random_state : int, RandomState instance or None, default=None
        Determines random number generation. Pass an int for reproducible
        output across multiple function calls.

    Returns
    -------

    Cdict_best_state : D array-like, shape (D,nt,nt)
        Metric/Graph cost matrices composing the dictionary.
        The dictionary leading to the best loss over an epoch is saved and returned.
    Ydict_best_state : D array-like, shape (D,nt,d)
        Feature matrices composing the dictionary.
        The dictionary leading to the best loss over an epoch is saved and returned.
    log: dict
        If use_log is True, contains loss evolutions by batches and epochs.

    References
    -------
    .. [38] C. Vincent-Cuaz, T. Vayer, R. Flamary, M. Corneli, N. Courty, Online
        Graph Dictionary Learning, International Conference on Machine Learning
        (ICML), 2021.
    """
    Cs0, Ys0 = Cs, Ys
    nx = get_backend(*Cs0, *Ys0)
    Cs = [nx.to_numpy(C) for C in Cs0]
    Ys = [nx.to_numpy(Y) for Y in Ys0]

    d = Ys[0].shape[-1]
    dataset_size = len(Cs)

    if ps is None:
        ps = [unif(C.shape[0]) for C in Cs]
    else:
        ps = [nx.to_numpy(p) for p in ps]
    if q is None:
        q = unif(nt)
    else:
        q = nx.to_numpy(q)

    rng = check_random_state(random_state)
    if Cdict_init is None:
        # Initialize randomly structures of dictionary atoms based on samples
        dataset_means = [C.mean() for C in Cs]
        Cdict = rng.normal(
            loc=np.mean(dataset_means), scale=np.std(dataset_means), size=(D, nt, nt)
        )
    else:
        Cdict = nx.to_numpy(Cdict_init).copy()
        assert Cdict.shape == (D, nt, nt)
    if Ydict_init is None:
        # Initialize randomly features of dictionary atoms based on samples distribution by feature component
        dataset_feature_means = np.stack([F.mean(axis=0) for F in Ys])
        Ydict = rng.normal(
            loc=dataset_feature_means.mean(axis=0),
            scale=dataset_feature_means.std(axis=0),
            size=(D, nt, d),
        )
    else:
        Ydict = nx.to_numpy(Ydict_init).copy()
        assert Ydict.shape == (D, nt, d)

    if "symmetric" in projection:
        Cdict = 0.5 * (Cdict + Cdict.transpose((0, 2, 1)))
        symmetric = True
    else:
        symmetric = False
    if "nonnegative" in projection:
        Cdict[Cdict < 0.0] = 0.0

    if use_adam_optimizer:
        adam_moments_C = _initialize_adam_optimizer(Cdict)
        adam_moments_Y = _initialize_adam_optimizer(Ydict)

    log = {"loss_batches": [], "loss_epochs": []}
    const_q = q[:, None] * q[None, :]
    diag_q = np.diag(q)
    Cdict_best_state = Cdict.copy()
    Ydict_best_state = Ydict.copy()
    loss_best_state = np.inf
    if batch_size > dataset_size:
        batch_size = dataset_size
    iter_by_epoch = dataset_size // batch_size + int((dataset_size % batch_size) > 0)

    for epoch in range(epochs):
        cumulated_loss_over_epoch = 0.0

        for _ in range(iter_by_epoch):
            # Batch iterations
            batch = rng.choice(range(dataset_size), size=batch_size, replace=False)
            cumulated_loss_over_batch = 0.0
            unmixings = np.zeros((batch_size, D))
            Cs_embedded = np.zeros((batch_size, nt, nt))
            Ys_embedded = np.zeros((batch_size, nt, d))
            Ts = [None] * batch_size

            for batch_idx, C_idx in enumerate(batch):
                # BCD solver for Gromov-Wasserstein linear unmixing used independently on each structure of the sampled batch
                (
                    unmixings[batch_idx],
                    Cs_embedded[batch_idx],
                    Ys_embedded[batch_idx],
                    Ts[batch_idx],
                    current_loss,
                ) = fused_gromov_wasserstein_linear_unmixing(
                    Cs[C_idx],
                    Ys[C_idx],
                    Cdict,
                    Ydict,
                    alpha,
                    reg=reg,
                    p=ps[C_idx],
                    q=q,
                    tol_outer=tol_outer,
                    tol_inner=tol_inner,
                    max_iter_outer=max_iter_outer,
                    max_iter_inner=max_iter_inner,
                    symmetric=symmetric,
                    **kwargs,
                )
                cumulated_loss_over_batch += current_loss
            cumulated_loss_over_epoch += cumulated_loss_over_batch
            if use_log:
                log["loss_batches"].append(cumulated_loss_over_batch)

            # Stochastic projected gradient step over dictionary atoms
            grad_Cdict = np.zeros_like(Cdict)
            grad_Ydict = np.zeros_like(Ydict)

            for batch_idx, C_idx in enumerate(batch):
                shared_term_structures = Cs_embedded[batch_idx] * const_q - (
                    Cs[C_idx].dot(Ts[batch_idx])
                ).T.dot(Ts[batch_idx])
                shared_term_features = diag_q.dot(Ys_embedded[batch_idx]) - Ts[
                    batch_idx
                ].T.dot(Ys[C_idx])
                grad_Cdict += (
                    alpha
                    * unmixings[batch_idx][:, None, None]
                    * shared_term_structures[None, :, :]
                )
                grad_Ydict += (
                    (1 - alpha)
                    * unmixings[batch_idx][:, None, None]
                    * shared_term_features[None, :, :]
                )
            grad_Cdict *= 2 / batch_size
            grad_Ydict *= 2 / batch_size

            if use_adam_optimizer:
                Cdict, adam_moments_C = _adam_stochastic_updates(
                    Cdict, grad_Cdict, learning_rate_C, adam_moments_C
                )
                Ydict, adam_moments_Y = _adam_stochastic_updates(
                    Ydict, grad_Ydict, learning_rate_Y, adam_moments_Y
                )
            else:
                Cdict -= learning_rate_C * grad_Cdict
                Ydict -= learning_rate_Y * grad_Ydict

            if "symmetric" in projection:
                Cdict = 0.5 * (Cdict + Cdict.transpose((0, 2, 1)))
            if "nonnegative" in projection:
                Cdict[Cdict < 0.0] = 0.0

        if use_log:
            log["loss_epochs"].append(cumulated_loss_over_epoch)
        if loss_best_state > cumulated_loss_over_epoch:
            loss_best_state = cumulated_loss_over_epoch
            Cdict_best_state = Cdict.copy()
            Ydict_best_state = Ydict.copy()
        if verbose:
            print(
                "--- epoch: ",
                epoch,
                " cumulated reconstruction error: ",
                cumulated_loss_over_epoch,
            )

    return nx.from_numpy(Cdict_best_state), nx.from_numpy(Ydict_best_state), log


def fused_gromov_wasserstein_linear_unmixing(
    C,
    Y,
    Cdict,
    Ydict,
    alpha,
    reg=0.0,
    p=None,
    q=None,
    tol_outer=10 ** (-5),
    tol_inner=10 ** (-5),
    max_iter_outer=20,
    max_iter_inner=200,
    symmetric=True,
    **kwargs,
):
    r"""
    Returns the Fused Gromov-Wasserstein linear unmixing of :math:`(\mathbf{C},\mathbf{Y},\mathbf{p})` onto the attributed dictionary atoms :math:`\{ (\mathbf{C_{dict}[d]},\mathbf{Y_{dict}[d]}, \mathbf{q}) \}_{d \in [D]}`

    .. math::
        \min_{\mathbf{w}}  FGW_{2,\alpha}(\mathbf{C},\mathbf{Y}, \sum_{d=1}^D w_d\mathbf{C_{dict}[d]},\sum_{d=1}^D w_d\mathbf{Y_{dict}[d]}, \mathbf{p}, \mathbf{q}) - reg \| \mathbf{w}  \|_2^2

    such that, :math:`\forall s \leq S` :

        - :math:`\mathbf{w_s}^\top \mathbf{1}_D = 1`
        - :math:`\mathbf{w_s} \geq \mathbf{0}_D`

    Where :

    - :math:`\mathbf{C}` is a (ns,ns) pairwise similarity matrix of variable size ns.
    - :math:`\mathbf{Y}` is a (ns,d) features matrix of variable size ns and fixed dimension d.
    - :math:`\mathbf{C_{dict}}` is a (D, nt, nt) tensor of D pairwise similarity matrix of fixed size nt.
    - :math:`\mathbf{Y_{dict}}` is a (D, nt, d) tensor of D features matrix of fixed size nt and fixed dimension d.
    - :math:`\mathbf{p}` is the source distribution corresponding to :math:`\mathbf{C_s}`
    - :math:`\mathbf{q}` is the target distribution assigned to every structures in the embedding space.
    - :math:`\alpha` is the trade-off parameter of Fused Gromov-Wasserstein
    - reg is the regularization coefficient.

    The algorithm used for solving the problem is a Block Coordinate Descent as discussed in [38]_, algorithm 6.

    Parameters
    ----------
    C : array-like, shape (ns, ns)
        Metric/Graph cost matrix.
    Y : array-like, shape (ns, d)
        Feature matrix.
    Cdict : D array-like, shape (D,nt,nt)
        Metric/Graph cost matrices composing the dictionary on which to embed (C,Y).
    Ydict : D array-like, shape (D,nt,d)
        Feature matrices composing the dictionary on which to embed (C,Y).
    alpha: float,
        Trade-off parameter of Fused Gromov-Wasserstein.
    reg : float, optional
        Coefficient of the negative quadratic regularization used to promote sparsity of w. The default is 0.
    p : array-like, shape (ns,), optional
        Distribution in the source space C. Default is None and corresponds to uniform distribution.
    q : array-like, shape (nt,), optional
        Distribution in the space depicted by the dictionary. Default is None and corresponds to uniform distribution.
    tol_outer : float, optional
        Solver precision for the BCD algorithm.
    tol_inner : float, optional
        Solver precision for the Conjugate Gradient algorithm used to get optimal w at a fixed transport. Default is :math:`10^{-5}`.
    max_iter_outer : int, optional
        Maximum number of iterations for the BCD. Default is 20.
    max_iter_inner : int, optional
        Maximum number of iterations for the Conjugate Gradient. Default is 200.

    Returns
    -------
    w: array-like, shape (D,)
        fused Gromov-Wasserstein linear unmixing of (C,Y,p) onto the span of the dictionary.
    Cembedded: array-like, shape (nt,nt)
        embedded structure of :math:`(\mathbf{C},\mathbf{Y}, \mathbf{p})` onto the dictionary, :math:`\sum_d w_d\mathbf{C_{dict}[d]}`.
    Yembedded: array-like, shape (nt,d)
        embedded features of :math:`(\mathbf{C},\mathbf{Y}, \mathbf{p})` onto the dictionary, :math:`\sum_d w_d\mathbf{Y_{dict}[d]}`.
    T: array-like (ns,nt)
        Fused Gromov-Wasserstein transport plan between :math:`(\mathbf{C},\mathbf{p})` and :math:`(\sum_d w_d\mathbf{C_{dict}[d]}, \sum_d w_d\mathbf{Y_{dict}[d]},\mathbf{q})`.
    current_loss: float
        reconstruction error
    References
    -------
    .. [38] C. Vincent-Cuaz, T. Vayer, R. Flamary, M. Corneli, N. Courty, Online
        Graph Dictionary Learning, International Conference on Machine Learning
        (ICML), 2021.
    """
    C0, Y0, Cdict0, Ydict0 = C, Y, Cdict, Ydict
    nx = get_backend(C0, Y0, Cdict0, Ydict0)
    C = nx.to_numpy(C0)
    Y = nx.to_numpy(Y0)
    Cdict = nx.to_numpy(Cdict0)
    Ydict = nx.to_numpy(Ydict0)

    if p is None:
        p = unif(C.shape[0])
    else:
        p = nx.to_numpy(p)
    if q is None:
        q = unif(Cdict.shape[-1])
    else:
        q = nx.to_numpy(q)

    T = p[:, None] * q[None, :]
    D = len(Cdict)
    d = Y.shape[-1]
    w = unif(D)  # Initialize with uniform weights
    ns = C.shape[-1]
    nt = Cdict.shape[-1]

    # modeling (C,Y)
    Cembedded = np.sum(w[:, None, None] * Cdict, axis=0)
    Yembedded = np.sum(w[:, None, None] * Ydict, axis=0)

    # constants depending on q
    const_q = q[:, None] * q[None, :]
    diag_q = np.diag(q)
    # Trackers for BCD convergence
    convergence_criterion = np.inf
    current_loss = 10**15
    outer_count = 0
    Ys_constM = (Y**2).dot(
        np.ones((d, nt))
    )  # constant in computing euclidean pairwise feature matrix

    while (convergence_criterion > tol_outer) and (outer_count < max_iter_outer):
        previous_loss = current_loss

        # 1. Solve GW transport between (C,p) and (\sum_d Cdictionary[d],q) fixing the unmixing w
        Yt_varM = (np.ones((ns, d))).dot((Yembedded**2).T)
        M = (
            Ys_constM + Yt_varM - 2 * Y.dot(Yembedded.T)
        )  # euclidean distance matrix between features
        T, log = fused_gromov_wasserstein(
            M,
            C,
            Cembedded,
            p,
            q,
            loss_fun="square_loss",
            alpha=alpha,
            max_iter=max_iter_inner,
            tol_rel=tol_inner,
            tol_abs=0.0,
            armijo=False,
            G0=T,
            log=True,
            symmetric=symmetric,
            **kwargs,
        )
        current_loss = log["fgw_dist"]
        if reg != 0:
            current_loss -= reg * np.sum(w**2)

        # 2. Solve linear unmixing problem over w with a fixed transport plan T
        w, Cembedded, Yembedded, current_loss = _cg_fused_gromov_wasserstein_unmixing(
            C,
            Y,
            Cdict,
            Ydict,
            Cembedded,
            Yembedded,
            w,
            T,
            p,
            q,
            const_q,
            diag_q,
            current_loss,
            alpha,
            reg,
            tol=tol_inner,
            max_iter=max_iter_inner,
            **kwargs,
        )
        if previous_loss != 0:
            convergence_criterion = abs(previous_loss - current_loss) / abs(
                previous_loss
            )
        else:
            convergence_criterion = abs(previous_loss - current_loss) / 10 ** (-12)
        outer_count += 1

    return (
        nx.from_numpy(w),
        nx.from_numpy(Cembedded),
        nx.from_numpy(Yembedded),
        nx.from_numpy(T),
        nx.from_numpy(current_loss),
    )


def _cg_fused_gromov_wasserstein_unmixing(
    C,
    Y,
    Cdict,
    Ydict,
    Cembedded,
    Yembedded,
    w,
    T,
    p,
    q,
    const_q,
    diag_q,
    starting_loss,
    alpha,
    reg,
    tol=10 ** (-6),
    max_iter=200,
    **kwargs,
):
    r"""
    Returns for a fixed admissible transport plan,
    the optimal linear unmixing :math:`\mathbf{w}` minimizing the Fused Gromov-Wasserstein cost between :math:`(\mathbf{C},\mathbf{Y},\mathbf{p})` and :math:`(\sum_d w_d \mathbf{C_{dict}[d]},\sum_d w_d*\mathbf{Y_{dict}[d]}, \mathbf{q})`

    .. math::
        \min_{\mathbf{w}}  \alpha  \sum_{ijkl} (C_{i,j} - \sum_{d=1}^D w_d C_{dict}[d]_{k,l} )^2 T_{i,k}T_{j,l} \\+ (1-\alpha) \sum_{ij} \| \mathbf{Y_i} - \sum_d w_d \mathbf{Y_{dict}[d]_j} \|_2^2 T_{ij}- reg \| \mathbf{w}  \|_2^2

    Such that :

        - :math:`\mathbf{w}^\top \mathbf{1}_D = 1`
        - :math:`\mathbf{w} \geq \mathbf{0}_D`

    Where :

    - :math:`\mathbf{C}` is a (ns,ns) pairwise similarity matrix of variable size ns.
    - :math:`\mathbf{Y}` is a (ns,d) features matrix of variable size ns and fixed dimension d.
    - :math:`\mathbf{C_{dict}}` is a (D, nt, nt) tensor of D pairwise similarity matrix of fixed size nt.
    - :math:`\mathbf{Y_{dict}}` is a (D, nt, d) tensor of D features matrix of fixed size nt and fixed dimension d.
    - :math:`\mathbf{p}` is the source distribution corresponding to :math:`\mathbf{C_s}`
    - :math:`\mathbf{q}` is the target distribution assigned to every structures in the embedding space.
    - :math:`\mathbf{T}` is the optimal transport plan conditioned by the previous state of :math:`\mathbf{w}`
    - :math:`\alpha` is the trade-off parameter of Fused Gromov-Wasserstein
    - reg is the regularization coefficient.

    The algorithm used for solving the problem is a Conditional Gradient Descent as discussed in [38]_, algorithm 7.

    Parameters
    ----------

    C : array-like, shape (ns, ns)
        Metric/Graph cost matrix.
    Y : array-like, shape (ns, d)
        Feature matrix.
    Cdict : list of D array-like, shape (nt,nt)
        Metric/Graph cost matrices composing the dictionary on which to embed (C,Y).
        Each matrix in the dictionary must have the same size (nt,nt).
    Ydict : list of D array-like, shape (nt,d)
        Feature matrices composing the dictionary on which to embed (C,Y).
        Each matrix in the dictionary must have the same size (nt,d).
    Cembedded: array-like, shape (nt,nt)
        Embedded structure of (C,Y) onto the dictionary
    Yembedded: array-like, shape (nt,d)
        Embedded features of (C,Y) onto the dictionary
    w: array-like, shape (n_D,)
        Linear unmixing of (C,Y) onto (Cdict,Ydict)
    const_q: array-like, shape (nt,nt)
        product matrix :math:`\mathbf{qq}^\top` where :math:`\mathbf{q}` is the target space distribution.
    diag_q: array-like, shape (nt,nt)
        diagonal matrix with values of q on the diagonal.
    T: array-like, shape (ns,nt)
        fixed transport plan between (C,Y) and its model
    p : array-like, shape (ns,)
        Distribution in the source space (C,Y).
    q : array-like, shape (nt,)
        Distribution in the embedding space depicted by the dictionary.
    alpha: float,
        Trade-off parameter of Fused Gromov-Wasserstein.
    reg : float, optional
        Coefficient of the negative quadratic regularization used to promote sparsity of w.

    Returns
    -------
    w: ndarray (D,)
        linear unmixing of :math:`(\mathbf{C},\mathbf{Y},\mathbf{p})` onto the span of :math:`(C_{dict},Y_{dict})` given OT corresponding to previous unmixing.
    """
    convergence_criterion = np.inf
    current_loss = starting_loss
    count = 0
    const_TCT = np.transpose(C.dot(T)).dot(T)
    ones_ns_d = np.ones(Y.shape)

    while (convergence_criterion > tol) and (count < max_iter):
        previous_loss = current_loss

        # 1) Compute gradient at current point w
        # structure
        grad_w = alpha * np.sum(
            Cdict
            * (Cembedded[None, :, :] * const_q[None, :, :] - const_TCT[None, :, :]),
            axis=(1, 2),
        )
        # feature
        grad_w += (1 - alpha) * np.sum(
            Ydict * (diag_q.dot(Yembedded)[None, :, :] - T.T.dot(Y)[None, :, :]),
            axis=(1, 2),
        )
        grad_w -= reg * w
        grad_w *= 2

        # 2) Conditional gradient direction finding: x= \argmin_x x^T.grad_w
        min_ = np.min(grad_w)
        x = (grad_w == min_).astype(np.float64)
        x /= np.sum(x)

        # 3) Line-search step: solve \argmin_{\gamma \in [0,1]} a*gamma^2 + b*gamma + c
        gamma, a, b, Cembedded_diff, Yembedded_diff = (
            _linesearch_fused_gromov_wasserstein_unmixing(
                w,
                grad_w,
                x,
                Y,
                Cdict,
                Ydict,
                Cembedded,
                Yembedded,
                T,
                const_q,
                const_TCT,
                ones_ns_d,
                alpha,
                reg,
            )
        )

        # 4) Updates: w <-- (1-gamma)*w + gamma*x
        w += gamma * (x - w)
        Cembedded += gamma * Cembedded_diff
        Yembedded += gamma * Yembedded_diff
        current_loss += a * (gamma**2) + b * gamma

        if previous_loss != 0:
            convergence_criterion = abs(previous_loss - current_loss) / abs(
                previous_loss
            )
        else:
            convergence_criterion = abs(previous_loss - current_loss) / 10 ** (-12)
        count += 1

    return w, Cembedded, Yembedded, current_loss


def _linesearch_fused_gromov_wasserstein_unmixing(
    w,
    grad_w,
    x,
    Y,
    Cdict,
    Ydict,
    Cembedded,
    Yembedded,
    T,
    const_q,
    const_TCT,
    ones_ns_d,
    alpha,
    reg,
    **kwargs,
):
    r"""
    Compute optimal steps for the line search problem of Fused Gromov-Wasserstein linear unmixing
    .. math::
        \min_{\gamma \in [0,1]}  \alpha \sum_{ijkl} (C_{i,j} - \sum_{d=1}^D z_d(\gamma)C_{dict}[d]_{k,l} )^2 T_{i,k}T_{j,l} \\ + (1-\alpha) \sum_{ij} \| \mathbf{Y_i} - \sum_d z_d(\gamma) \mathbf{Y_{dict}[d]_j} \|_2^2 - reg\| \mathbf{z}(\gamma)  \|_2^2


    Such that :

        - :math:`\mathbf{z}(\gamma) = (1- \gamma)\mathbf{w} + \gamma \mathbf{x}`

    Parameters
    ----------

    w : array-like, shape (D,)
        Unmixing.
    grad_w : array-like, shape (D, D)
        Gradient of the reconstruction loss with respect to w.
    x: array-like, shape (D,)
        Conditional gradient direction.
    Y: arrat-like, shape (ns,d)
        Feature matrix of the input space
    Cdict : list of D array-like, shape (nt, nt)
        Metric/Graph cost matrices composing the dictionary on which to embed (C,Y).
        Each matrix in the dictionary must have the same size (nt,nt).
    Ydict : list of D array-like, shape (nt, d)
        Feature matrices composing the dictionary on which to embed (C,Y).
        Each matrix in the dictionary must have the same size (nt,d).
    Cembedded: array-like, shape (nt, nt)
        Embedded structure of (C,Y) onto the dictionary
    Yembedded: array-like, shape (nt, d)
        Embedded features of (C,Y) onto the dictionary
    T: array-like, shape (ns, nt)
        Fixed transport plan between (C,Y) and its current model.
    const_q: array-like, shape (nt,nt)
        product matrix :math:`\mathbf{q}\mathbf{q}^\top` where q is the target space distribution. Used to avoid redundant computations.
    const_TCT: array-like, shape (nt, nt)
        :math:`\mathbf{T}^\top \mathbf{C}^\top \mathbf{T}`. Used to avoid redundant computations.
    ones_ns_d: array-like, shape (ns, d)
        :math:`\mathbf{1}_{ ns \times d}`. Used to avoid redundant computations.
    alpha: float,
        Trade-off parameter of Fused Gromov-Wasserstein.
    reg : float, optional
        Coefficient of the negative quadratic regularization used to promote sparsity of w.

    Returns
    -------
    gamma: float
        Optimal value for the line-search step
    a: float
        Constant factor appearing in the factorization :math:`a \gamma^2 + b \gamma +c` of the reconstruction loss
    b: float
        Constant factor appearing in the factorization :math:`a \gamma^2 + b \gamma +c` of the reconstruction loss
    Cembedded_diff: numpy array, shape (nt, nt)
        Difference between structure matrix of models evaluated in :math:`\mathbf{w}` and in :math:`\mathbf{w}`.
    Yembedded_diff: numpy array, shape (nt, nt)
        Difference between feature matrix of models evaluated in :math:`\mathbf{w}` and in :math:`\mathbf{w}`.
    """
    # polynomial coefficients from quadratic objective (with respect to w) on structures
    Cembedded_x = np.sum(x[:, None, None] * Cdict, axis=0)
    Cembedded_diff = Cembedded_x - Cembedded
    trace_diffx = np.sum(Cembedded_diff * Cembedded_x * const_q)
    trace_diffw = np.sum(Cembedded_diff * Cembedded * const_q)
    # Constant factor appearing in the factorization a*gamma^2 + b*g + c of the Gromov-Wasserstein reconstruction loss
    a_gw = trace_diffx - trace_diffw
    b_gw = 2 * (trace_diffw - np.sum(Cembedded_diff * const_TCT))

    # polynomial coefficient from quadratic objective (with respect to w) on features
    Yembedded_x = np.sum(x[:, None, None] * Ydict, axis=0)
    Yembedded_diff = Yembedded_x - Yembedded
    # Constant factor appearing in the factorization a*gamma^2 + b*g + c of the Gromov-Wasserstein reconstruction loss
    a_w = np.sum(ones_ns_d.dot((Yembedded_diff**2).T) * T)
    b_w = 2 * np.sum(
        T * (ones_ns_d.dot((Yembedded * Yembedded_diff).T) - Y.dot(Yembedded_diff.T))
    )

    a = alpha * a_gw + (1 - alpha) * a_w
    b = alpha * b_gw + (1 - alpha) * b_w
    if reg != 0:
        a -= reg * np.sum((x - w) ** 2)
        b -= 2 * reg * np.sum(w * (x - w))
    if a > 0:
        gamma = min(1, max(0, -b / (2 * a)))
    elif a + b < 0:
        gamma = 1
    else:
        gamma = 0

    return gamma, a, b, Cembedded_diff, Yembedded_diff
