"""
Sliced Wasserstein Distance.

"""

# Author: Adrien Corenflos <adrien.corenflos@aalto.fi>
#
# License: MIT License
import math

import torch


def get_random_projections(n_projections, d, seed=None, device=None, dtype=None):
    r"""
    Generates n_projections samples from the uniform on the unit sphere of dimension d-1: :math:`\mathcal{U}(\mathcal{S}^{d-1})`

    Parameters
    ----------
    n_projections : int
        number of samples requested
    d : int
        dimension of the space
    seed: int or torch.Generator, optional
        Seed used for numpy random number generator
    device: torch.device or str, optional
        device on which to instantiate the projections, ignored if a torch.Generator is passed to seed
    dtype: torch.dtype, optional
        output dtype

    Returns
    -------
    out: torch.Tensor (d, n_projections)
        The uniform unit vectors on the sphere

    Examples
    --------
    >>> import numpy as np
    >>> n_projections = 100
    >>> d = 5
    >>> projs = get_random_projections(n_projections, d)
    >>> np.allclose(np.sum(np.square(projs.cpu().numpy()), 0), 1.)  # doctest: +NORMALIZE_WHITESPACE
    True

    """

    if isinstance(seed, torch.Generator):
        gen = seed
        device = gen.device
    elif seed is not None:
        gen = torch.Generator(device)
        gen = gen.manual_seed(seed)
    else:
        gen = None

    projections = torch.empty(d, n_projections, dtype=dtype, device=device).normal_(generator=gen)
    norm = torch.linalg.norm(projections, ord=2, axis=0, keepdims=True)
    projections = projections / norm
    return projections


def ot_loss_sliced(X_s, X_t, a=None, b=None, p=1, n_projections=50, seed=None):
    r"""
    Computes the 1 dimensional OT loss [2] between two (batched) empirical distributions
    ..math:
        ot_{loss} &= \int_0^1 |cdf_u^{-1}(q)  cdf_v^{-1}(q)|^p dq

    It is formally the p-Wasserstein distance raised to the power p.
    We do so in a vectorized way by first building the individual quantile functions then integrating them.
    This has a theoretically higher complexity than the core OT implementation but behaves better with PyTorch

    Parameters
    ----------
    X_s : torch.Tensor (n, d)
        samples in the source domain
    X_t : torch.Tensor (m, d)
        samples in the target domain
    a : torch.Tensor (n,), optional
        samples weights in the source domain
    b : torch.Tensor (m,), optional
        samples weights in the target domain
    p: int, optional
        order of the ground metric used, should be at least 1 (see [2, Chap. 2], default is 1
    n_projections : int, optional
        Number of projections used for the Monte-Carlo approximation
    seed: int or torch.Generator or None, optional
        Used for the random generator, if passing a generator object make sure the device is compatible with the inputs

    Returns
    -------
    cost: torch.Tensor (...,)
        the sliced OT loss

    Examples
    --------

    >>> import ot
    >>> import numpy as np
    >>> import torch
    >>> np.random.seed(0)
    >>> n_source = 7
    >>> n_target = 100
    >>> a = torch.tensor(ot.utils.unif(n_source), requires_grad=True)
    >>> b = torch.tensor(ot.utils.unif(n_target))
    >>> X_source = torch.tensor(np.random.randn(n_source,), requires_grad=True)
    >>> Y_target = torch.tensor(np.random.randn(n_target,))
    >>> loss = ot.torch.lp.ot_loss_1d(X_source, Y_target, a, b)
    >>> torch.autograd.grad(loss, X_source)[0]
    tensor([0.1429, 0.1429, 0.1429, 0.1229, 0.1429, 0.1429, 0.1429],
           dtype=torch.float64)

    References
    ----------

    .. [31] Bonneel, Nicolas, et al. "Sliced and radon wasserstein barycenters of measures." Journal of Mathematical Imaging and Vision 51.1 (2015): 22-45
    """
    from .lp import ot_loss_1d

    X_s = torch.as_tensor(X_s)
    X_t = torch.as_tensor(X_t)

    device = X_s.device
    dtype = X_s.dtype

    n = X_s.shape[0]
    m = X_t.shape[0]

    if X_s.shape[1] != X_t.shape[1]:
        raise ValueError(
            "X_s and X_t must have the same number of dimensions {} and {} respectively given".format(X_s.shape[1],
                                                                                                      X_t.shape[1]))

    if a is None:
        a = torch.full((n,), 1 / n, device=device, dtype=dtype)
    else:
        a = torch.as_tensor(a)

    if b is None:
        b = torch.full((m,), 1 / m, device=device, dtype=dtype)
    else:
        b = torch.as_tensor(b)

    d = X_s.shape[1]

    projections = get_random_projections(n_projections, d, seed, device, dtype)

    X_s_projections = torch.matmul(X_s, projections)
    X_t_projections = torch.matmul(X_t, projections)

    projected_losses = ot_loss_1d(X_s_projections, X_t_projections, a, b, p, True)
    return projected_losses.mean()
