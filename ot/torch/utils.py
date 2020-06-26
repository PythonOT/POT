"""

"""

import torch


# Author: Remi Flamary <remi.flamary@unice.fr>
#
# License: MIT License

def unif(n,dtype=None,device=None,requires_grad=False):
    """ return a uniform histogram of length n (simplex)

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
    return torch.ones(n,dtype=dtype,device=device,requires_grad=requires_grad) / n


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
        return torch.cdist(x1, x2, p=2)**2
    elif metric == "euclidean":
        p = 2
    elif metric == "cityblock":
        p = 1
    elif isinstance(metric, float) or isinstance(metric, int):
        p = metric
    else:
        raise ValueError("metric '{}' is not a valid option.\n".format(metric))
    return torch.cdist(x1, x2, p=p)
