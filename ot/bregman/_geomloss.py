# -*- coding: utf-8 -*-
"""
Wrapper functions for geomloss
"""

# Author: Remi Flamary <remi.flamary@unice.fr>
#
# License: MIT License

import numpy as np

try:
    import geomloss
    from geomloss import SamplesLoss
    import torch
    from torch.autograd import grad
    from ..utils import get_backend, LazyTensor, dist
except ImportError:
    geomloss = False


def get_sinkhorn_geomloss_lazytensor(
    X_a, X_b, f, g, a, b, metric="sqeuclidean", blur=0.1, nx=None
):
    """Get a LazyTensor of sinkhorn solution T = exp((f+g^T-C)/reg)*(ab^T)

    Parameters
    ----------
    X_a : array-like, shape (n_samples_a, dim)
        samples in the source domain
    X_torch: array-like, shape (n_samples_b, dim)
        samples in the target domain
    f : array-like, shape (n_samples_a,)
        First dual potentials (log space)
    g : array-like, shape (n_samples_b,)
        Second dual potentials (log space)
    metric : str, default='sqeuclidean'
        Metric used for the cost matrix computation
    blur : float, default=1e-1
        blur term (blur=sqrt(reg)) >0
    nx : Backend(), default=None
        Numerical backend used


    Returns
    -------
    T : LazyTensor
        Lowrank tensor T = exp((f+g^T-C)/reg)*(ab^T)
    """

    if nx is None:
        nx = get_backend(X_a, X_b, f, g)

    shape = (X_a.shape[0], X_b.shape[0])

    def func(i, j, X_a, X_b, f, g, a, b, metric, blur):
        if metric == "sqeuclidean":
            C = dist(X_a[i], X_b[j], metric=metric) / 2
        else:
            C = dist(X_a[i], X_b[j], metric=metric)
        return nx.exp((f[i, None] + g[None, j] - C) / (blur**2)) * (
            a[i, None] * b[None, j]
        )

    T = LazyTensor(
        shape, func, X_a=X_a, X_b=X_b, f=f, g=g, a=a, b=b, metric=metric, blur=blur
    )

    return T


def empirical_sinkhorn2_geomloss(
    X_s,
    X_t,
    reg,
    a=None,
    b=None,
    metric="sqeuclidean",
    scaling=0.95,
    verbose=False,
    debias=False,
    log=False,
    backend="auto",
):
    r"""Solve the entropic regularization optimal transport problem with geomloss

    The function solves the following optimization problem:

    .. math::
        \gamma = arg\min_\gamma <\gamma,C>_F + reg\cdot\Omega(\gamma)

        s.t. \gamma 1 = a

                \gamma^T 1= b

                \gamma\geq 0

    where :

    - :math:`C` is the cost matrix such that :math:`C_{i,j}=d(x_i^s,x_j^t)` and
      :math:`d` is a metric.
    - :math:`\Omega` is the entropic regularization term
      :math:`\Omega(\gamma)=\sum_{i,j}\gamma_{i,j}\log(\gamma_{i,j})-\gamma_{i,j}+1`
    - :math:`a` and :math:`b` are source and target weights (sum to 1)

    The algorithm used for solving the problem is the Sinkhorn-Knopp matrix
    scaling algorithm as proposed in and computed in log space for
    better stability and epsilon-scaling. The solution is computed in a lazy way
    using the Geomloss [60] and the KeOps library [61].

    Parameters
    ----------
    X_s : array-like, shape (n_samples_a, dim)
        samples in the source domain
    X_t : array-like, shape (n_samples_b, dim)
        samples in the target domain
    reg : float
        Regularization term >0
    a : array-like, shape (n_samples_a,), default=None
        samples weights in the source domain
    b : array-like, shape (n_samples_b,), default=None
        samples weights in the target domain
    metric : str, default='sqeuclidean'
        Metric used for the cost matrix computation Only accepted values are
        'sqeuclidean' and 'euclidean'.
    scaling : float, default=0.95
        Scaling parameter used for epsilon scaling. Value close to one promote
        precision while value close to zero promote speed.
    verbose : bool, default=False
        Print information
    debias : bool, default=False
        Use the debiased version of Sinkhorn algorithm [12]_.
    log : bool, default=False
        Return log dictionary containing all computed objects
    backend : str, default='auto'
        Numerical backend for geomloss. Only 'auto' and 'tensorized' 'online'
        and 'multiscale' are accepted values.

    Returns
    -------
    value : float
        OT value
    log : dict
        Log dictionary return only if log==True in parameters

    References
    ----------

    .. [60] Feydy, J., Roussillon, P., Trouvé, A., & Gori, P. (2019). [Fast
           and scalable optimal transport for brain tractograms. In Medical Image
           Computing and Computer Assisted Intervention–MICCAI 2019: 22nd
           International Conference, Shenzhen, China, October 13–17, 2019,
           Proceedings, Part III 22 (pp. 636-644). Springer International
           Publishing.

    .. [61] Charlier, B., Feydy, J., Glaunes, J. A., Collin, F. D., & Durif, G.
            (2021). Kernel operations on the gpu, with autodiff, without memory
            overflows. The Journal of Machine Learning Research, 22(1), 3457-3462.

    """

    if geomloss:
        nx = get_backend(X_s, X_t, a, b)

        if nx.__name__ not in ["torch", "numpy"]:
            raise ValueError("geomloss only support torch or numpy backend")

        if a is None:
            a = nx.ones(X_s.shape[0], type_as=X_s) / X_s.shape[0]
        if b is None:
            b = nx.ones(X_t.shape[0], type_as=X_t) / X_t.shape[0]

        if nx.__name__ == "numpy":
            X_s_torch = torch.tensor(X_s)
            X_t_torch = torch.tensor(X_t)

            a_torch = torch.tensor(a)
            b_torch = torch.tensor(b)

        else:
            X_s_torch = X_s
            X_t_torch = X_t

            a_torch = a
            b_torch = b

        # after that we are all in torch

        # set blur value and p
        if metric == "sqeuclidean":
            p = 2
            blur = np.sqrt(reg / 2)  # because geomloss divides cost by two
        elif metric == "euclidean":
            p = 1
            blur = np.sqrt(reg)
        else:
            raise ValueError("geomloss only supports sqeuclidean and euclidean metrics")

        # force gradients for computing dual
        a_torch.requires_grad = True
        b_torch.requires_grad = True

        loss = SamplesLoss(
            loss="sinkhorn",
            p=p,
            blur=blur,
            backend=backend,
            debias=debias,
            scaling=scaling,
            verbose=verbose,
        )

        # compute value
        value = loss(
            a_torch, X_s_torch, b_torch, X_t_torch
        )  # linear + entropic/KL reg?

        # get dual potentials
        f, g = grad(value, [a_torch, b_torch])

        if metric == "sqeuclidean":
            value *= 2  # because geomloss divides cost by two

        if nx.__name__ == "numpy":
            f = f.cpu().detach().numpy()
            g = g.cpu().detach().numpy()
            value = value.cpu().detach().numpy()

        if log:
            log = {}
            log["f"] = f
            log["g"] = g
            log["value"] = value

            log["lazy_plan"] = get_sinkhorn_geomloss_lazytensor(
                X_s, X_t, f, g, a, b, metric=metric, blur=blur, nx=nx
            )

            return value, log

        else:
            return value

    else:
        raise ImportError("geomloss not installed")
