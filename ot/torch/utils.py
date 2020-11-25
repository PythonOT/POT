"""

"""

import torch


# Author: Remi Flamary <remi.flamary@unice.fr>
#
# License: MIT License

def unif(n, dtype=None, device=None, requires_grad=False):
    """ returns a uniform histogram of length n (simplex)

    Parameters
    ----------

    n : int
        number of bins in the histogram
    n : int
        number of bins in the histogram

    Returns
    -------
    h : torch.tensor (n,)
        histogram of length n such that h_i=1/n for all i


    """
    return torch.full((n,), 1.0 / n, dtype=dtype, device=device, requires_grad=requires_grad)


def dist(x1, x2, metric="sqeuclidean"):
    """Compute distance between samples in tensors x1 and x2 using torch.cdist

    Parameters
    ----------

    x1 : torch.tensor, shape (n1,d)
        matrix with n1 samples of size d
    x2 : torch.tensor, shape (n2,d), optional
        matrix with n2 samples of size d (if None then x2=x1)
    metric : str | float, optional
        Name of the metric to be computed (full list in the doc of scipy),  If
        a string, the distance function can be 'braycurtis', 'canberra',
        'cityblock', 'euclidean', 'sqeuclidean'. If a float the the metric
        computed is the lp norm with p=metric.


    Returns
    -------

    M : torch.tensor (n1,n2)
        distance matrix computed with given metric

    """

    if x2 is None:
        x2 = x1
    if metric == "sqeuclidean":
        return torch.cdist(x1, x2, p=2) ** 2
    elif metric == "euclidean":
        p = 2
    elif metric == "cityblock":
        p = 1
    elif isinstance(metric, float) or isinstance(metric, int):
        p = metric
    else:
        raise ValueError("metric '{}' is not a valid option.\n".format(metric))
    return torch.cdist(x1, x2, p=p)


def proj_simplex(v, z=1):
    """Orthogonal projection on the simplex along axix 0 """
    n = v.shape[0]
    if v.ndimension() == 1:
        d1 = 1
        v = v[:, None]
    else:
        d1 = 0
    u, indices = torch.sort(v, dim=0, descending=True)
    cssv = torch.cumsum(u, dim=0) - z
    ind = torch.arange(n, device=v.device)[:, None].type_as(v) + 1
    cond = u - cssv / ind > 0
    rho = cond.sum(0)
    theta = cssv[rho - 1, range(v.shape[1])] / (rho)
    w = torch.max(v - theta[None, :], torch.zeros_like(v))
    if d1:
        return w[:, 0]
    else:
        return w


def quantile_function(qs, cws, xs):
    # type: (torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
    r""" Computes the quantile function of an empirical distribution

    Parameters
    ----------
    qs: torch.tensor (n,)
        Quantiles at which the quantile function is evaluated
    cws: torch.tensor (..., m)
        cumulative weights of the 1D empirical distribution, if batched, must be similar to xs
    xs: torch.tensor (..., m)
        locations of the 1D empirical distribution, batched against the `xs.ndim - 1` first dimensions

    Returns
    -------
    q: torch.tensor (..., n)
        The quantiles of the distribution
    """
    n = xs.shape[-1]
    idx = torch.searchsorted(cws, qs)
    return torch.gather(xs, -1, idx.clip(0, n - 1))
