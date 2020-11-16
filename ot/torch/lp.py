"""

"""

import numpy as np
import torch
from torch.autograd import Function
from .. import emd


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


def _emd_1d(w_x: torch.Tensor, w_y: torch.Tensor, x: torch.Tensor, y: torch.Tensor, p: int):
    """EMD 1D vectorised along x and y first dimension"""
    k = x.shape[0]  # type: int

    n = x.shape[1]  # type: int
    m = y.shape[1]  # type: int
    w_x = w_x.repeat(k, 1)
    w_y = w_y.repeat(k, 1)

    sorted_x, idx_x = torch.sort(x, dim=1)
    sorted_y, idx_y = torch.sort(y, dim=1)

    sorted_w_x = torch.gather(w_x, 1, idx_x)  # type: torch.Tensor
    sorted_w_y = torch.gather(w_y, 1, idx_y)  # type: torch.Tensor

    sorted_w_x = torch.reshape(sorted_w_x, (-1,))
    sorted_w_y = torch.reshape(sorted_w_y, (-1,))

    finished = torch.zeros(k, dtype=torch.bool)
    cost = torch.zeros(k, dtype=x.dtype)

    i = torch.zeros(k, dtype=torch.int64)  # type: torch.Tensor
    j = torch.zeros(k, dtype=torch.int64)  # type: torch.Tensor
    max_i = torch.tensor(n - 1, dtype=torch.int64)
    max_j = torch.tensor(m - 1, dtype=torch.int64)

    w_i = sorted_w_x[0]  # type: torch.Tensor
    w_j = sorted_w_y[0]  # type: torch.Tensor

    while torch.any(~finished):
        diff = torch.gather(sorted_x, 1, torch.reshape(i, (k, 1))) - torch.gather(sorted_y, 1, torch.reshape(j, (k, 1)))
        m_ij = torch.reshape(torch.abs(diff ** p), (k,))

        update_i = torch.logical_or(torch.lt(w_i, w_j), torch.eq(j, m - 1))

        next_cost = torch.where(update_i, cost + m_ij * w_i, cost + m_ij * w_j)
        next_i = torch.where(update_i, i + 1, i)
        next_j = torch.where(update_i, j, j + 1)
        next_w_i = torch.where(update_i, sorted_w_x[torch.minimum(next_i, max_i)], w_i - w_j)
        next_w_j = torch.where(update_i, w_j - w_i, sorted_w_y[torch.minimum(next_j, max_j)])

        cost = torch.where(finished, cost, next_cost)
        i = torch.where(finished, i, next_i)
        j = torch.where(finished, j, next_j)
        w_i = torch.where(finished, w_i, next_w_i)
        w_j = torch.where(finished, w_j, next_w_j)

        finished = torch.logical_or(torch.eq(i, n), torch.eq(j, m))

    return cost


def emd_1d(w_x: torch.Tensor, w_y: torch.Tensor, x: torch.Tensor, y: torch.Tensor, p: int):
    """EMD 1D"""
    assert w_x.shape[-1] == x.shape[-1]
    assert w_y.shape[-1] == y.shape[-1]
    w_x = w_x / w_x.sum()
    w_y = w_y / w_y.sum()
    return _emd_1d(w_x, w_y, x, y, p)
