"""

"""

# Author: Remi Flamary <remi.flamary@unice.fr>
#         Adrien Corenflos <adrien.corenflos@gmail.com>
#
# License: MIT License

import numpy as np
import torch
from torch.autograd import Function
from ot import emd
from torch.nn.functional import pad
from ot.torch.utils import quantile_function


# Inherit from Function
class OptimalTransportLossFunction(Function):
    """Return OT Loss for input (a,b,M) """

    @staticmethod
    # bias is an optional argument
    def forward(ctx, a, b, M, num_iter_max=100000):
        # convert to numpy
        a2 = a.detach().cpu().numpy().astype(np.float64)
        b2 = b.detach().cpu().numpy().astype(np.float64)
        M2 = M.detach().cpu().numpy().astype(np.float64)

        # project on simplex for float64 or else numerical errors
        a2 /= a2.sum()
        b2 /= b2.sum()

        G, log = emd(a2, b2, M2, log=True, numItermax=num_iter_max)

        G = torch.from_numpy(G).type_as(M)
        grad_a = torch.from_numpy(log['u']).type_as(a)
        grad_b = torch.from_numpy(log['v']).type_as(b)
        grad_M = G

        ctx.save_for_backward(grad_a, grad_b, grad_M)
        return torch.sum(G * M)

    @staticmethod
    def backward(ctx, grad_output):
        grad_a, grad_b, grad_M = ctx.saved_tensors

        return grad_a, grad_b, grad_M, None  # last one is parameter


def ot_loss(a, b, M, num_iter_max=100000):
    """loss=emd2(a,b,M)"""
    return OptimalTransportLossFunction.apply(a, b, M, num_iter_max)


def ot_solve(a, b, M, num_iter_max=100000, log=False):
    a2 = a.detach().cpu().numpy().astype(np.float64)
    b2 = b.detach().cpu().numpy().astype(np.float64)
    M2 = M.detach().cpu().numpy().astype(np.float64)

    # project on simplex for float64 or else numerical errors
    a2 /= a2.sum()
    b2 /= b2.sum()

    if log:

        G, log = emd(a2, b2, M2, log=False, numItermax=num_iter_max)

        log['u'] = torch.from_numpy(log['u']).type_as(a)
        log['v'] = torch.from_numpy(log['v']).type_as(a)

        return torch.from_numpy(G).type_as(M), log

    else:

        G = emd(a2, b2, M2, log=False, numItermax=num_iter_max)

        return torch.from_numpy(G).type_as(M)


def ot_loss_1d(u_values, v_values, u_weights=None, v_weights=None, p=1, require_sort=True):
    r"""
    Computes the 1 dimensional OT loss [2] between two (batched) empirical distributions
    ..math:
        ot_{loss} &= \int_0^1 |cdf_u^{-1}(q)  cdf_v^{-1}(q)|^p dq

    It is formally the p-Wasserstein distance raised to the power p.
    We do so in a vectorized way by first building the individual quantile functions then integrating them.
    This has a theoretically higher complexity than the core OT implementation but behaves better with PyTorch

    Parameters
    ----------
    u_values: torch.Tensor (n, ...)
        locations of the first empirical distribution
    v_values: torch.Tensor (m, ...)
        locations of the second empirical distribution
    u_weights: torch.Tensor (n, ...), optional
        weights of the first empirical distribution, if None then uniform weights are used
    v_weights: torch.Tensor (m, ...), optional
        weights of the second empirical distribution, if None then uniform weights are used
    p: int, optional
        order of the ground metric used, should be at least 1 (see [2, Chap. 2], default is 1
    require_sort: bool, optional
        sort the distributions atoms locations, if False we will consider they have been sorted prior to being passed to
        the function, default is True

    Returns
    -------
    cost: torch.Tensor (...,)
        the batched EMD

    Examples
    --------
    Simple example:
    >>> import ot
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
    .. [15] Peyré, G., & Cuturi, M. (2018). [Computational Optimal Transport](https://arxiv.org/pdf/1803.00567.pdf).

    """
    assert p >= 1, "The OT loss is only valid for p>=1, {p} was given".format(p=p)

    u_values = torch.as_tensor(u_values)
    v_values = torch.as_tensor(v_values)

    n = u_values.shape[0]
    m = v_values.shape[0]

    device = u_values.device
    dtype = u_values.dtype

    if u_weights is None:
        u_weights = torch.full_like(u_values, 1 / n, dtype=dtype, device=device)
    elif u_weights.ndim != u_values.ndim:
        u_weights = torch.repeat_interleave(u_weights.unsqueeze(-1), u_values.shape[-1], -1)

    if v_weights is None:
        v_weights = torch.full_like(v_values, 1 / m, dtype=dtype, device=device)
    elif v_weights.ndim != v_values.ndim:
        v_weights = torch.repeat_interleave(v_weights.unsqueeze(-1), v_values.shape[-1], -1)

    if require_sort:
        u_values, u_sorter = torch.sort(u_values, 0)
        v_values, v_sorter = torch.sort(v_values, 0)

        u_weights = torch.gather(u_weights, 0, u_sorter)
        v_weights = torch.gather(v_weights, 0, v_sorter)

    u_cumweights = torch.cumsum(u_weights, 0)
    v_cumweights = torch.cumsum(v_weights, 0)

    qs, _ = torch.sort(torch.cat((u_cumweights, v_cumweights), 0), 0)
    u_quantiles = quantile_function(qs, u_cumweights, u_values)
    v_quantiles = quantile_function(qs, v_cumweights, v_values)

    qs = pad(qs, (qs.ndim - 1) * (0, 0) + (1, 0))
    delta = qs[1:, ...] - qs[:-1, ...]
    diff_quantiles = torch.abs(u_quantiles - v_quantiles)

    if p == 1:
        return torch.sum(delta * torch.abs(diff_quantiles), dim=0)
    return torch.sum(delta * torch.pow(diff_quantiles, p), dim=0)
