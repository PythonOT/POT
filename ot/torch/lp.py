"""

"""

import numpy as np
import torch
from torch.autograd import Function
from .. import emd
from torch.nn.functional import pad
from .utils import quantile_function


# Author: Remi Flamary <remi.flamary@unice.fr>
#
# License: MIT License


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

        print(grad_a)

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


def emd1D_loss(u_values, v_values, u_weights=None, v_weights=None, p=1, require_sort=True):
    r"""
    Computes the 1 dimensional earth moving distance between two empirical distributions
    ..math:
        EMD &= \int_0^1 |cdf_u^{-1}(q)  cdf_v^{-1}(q)|^p dq

    We do so in a vectorized way by first building the individual quantile functions then integrating them.
    This has a theoretically higher complexity than the core OT implementation but behaves better with PyTorch

    Parameters
    ----------
    u_values: torch.Tensor (..., n)
        locations of the first empirical distribution
    v_values: torch.Tensor (..., m)
        locations of the second empirical distribution
    u_weights: torch.Tensor (..., n), optional
        weights of the first empirical distribution, if None then uniform weights are used
    v_weights: torch.Tensor (..., n), optional
        weights of the second empirical distribution, if None then uniform weights are used
    p: int, optional
        order of the ground metric used, default is 1
    require_sort: bool, optional
        Are the locations sorted along the last dimension already

    Returns
    -------
    cost: torch.Tensor (...,)
        the batched EMD

    """
    n = u_values.shape[-1]
    m = v_values.shape[-1]

    device = u_values.device
    dtype = u_values.dtype

    if u_weights is None:
        u_weights = torch.full((n,), 1 / n, dtype=dtype, device=device)

    if v_weights is None:
        v_weights = torch.full((m,), 1 / m, dtype=dtype, device=device)

    if require_sort:
        u_values, u_sorter = torch.sort(u_values, -1)
        v_values, v_sorter = torch.sort(v_values, -1)

        u_weights = u_weights[..., u_sorter]
        v_weights = v_weights[..., v_sorter]

    u_cumweights = torch.cumsum(u_weights, -1)
    v_cumweights = torch.cumsum(v_weights, -1)

    qs, _ = torch.sort(torch.cat((u_cumweights, v_cumweights), -1), -1)

    u_quantiles = quantile_function(qs, u_cumweights, u_values)
    v_quantiles = quantile_function(qs, v_cumweights, v_values)

    qs = pad(qs, (1, 0))
    delta = qs[..., 1:] - qs[..., :-1]
    diff_quantiles = torch.abs(u_quantiles - v_quantiles)

    if p == 1:
        return torch.sum(delta * torch.abs(diff_quantiles), dim=-1)
    return torch.sum(delta * torch.pow(diff_quantiles, p), dim=-1)
