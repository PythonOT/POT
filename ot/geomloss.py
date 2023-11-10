# -*- coding: utf-8 -*-
"""
wrapper functions for geomloss
"""

try:
    import geomloss
    from geomloss import SamplesLoss
    import torch
    from torch.autograd import grad
    from .utils import get_backend, LazyTensor, dist
except:
    geomloss = False

def get_sinkhorn_geomloss_lazytensor(X_a, X_b, f, g, a, b, metric='sqeuclidean', reg=1e-1, nx=None):
    """ Get a LazyTensor of sinkhorn solution T = exp((f+g^T-C)/reg)*(ab^T)

    Parameters
    ----------
    X_a : array-like, shape (n_samples_a, dim)
        samples in the source domain
    X_b : array-like, shape (n_samples_b, dim)
        samples in the target domain
    f : array-like, shape (n_samples_a,)
        First dual potentials (log space)
    g : array-like, shape (n_samples_b,)
        Second dual potentials (log space)
    metric : str, default='sqeuclidean'
        Metric used for the cost matrix computation
    reg : float, default=1e-1
        Regularization term >0
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

    def func(i, j, X_a, X_b, f, g, a, b, metric, reg):
        C = dist(X_a[i], X_b[j], metric=metric)
        return nx.exp((f[i, None] + g[None, j] - C) / reg)* (a[i,None]*b[None,j])

    T = LazyTensor(shape, func, X_a=X_a, X_b=X_b, f=f, g=g, a=a, b=b, metric=metric, reg=reg)

    return T    

def empirical_sinkhorn2_geomloss(X_s, X_t, reg, a=None, b=None, metric='sqeuclidean',
                        numIterMax=10000, stopThr=1e-9,
                        verbose=False, log=False, warn=True, warmstart=None):

    if geomloss:


        nx = get_backend(X_s, X_t, a, b)

        if nx.__name__ != 'torch':
            raise ValueError('geomloss only support torch backend')

        # after that we are all in torch

        if a is None:
            a = torch.ones(X_s.shape[0], dtype=X_s.dtype, device=X_s.device) / X_s.shape[0]
        if b is None:
            b = torch.ones(X_t.shape[0], dtype=X_t.dtype, device=X_t.device) / X_t.shape[0]

        if metric == 'sqeuclidean':
            p=2
            blur = reg/2 # because geomloss divides cost by two
        elif metric == 'euclidean':
            p=1
            blur = reg
        else:
            raise ValueError('geomloss only supports sqeuclidean and euclidean metrics')

        X_s.requires_grad = True
        X_t.requires_grad = True
        a.requires_grad = True
        b.requires_grad = True
        
        loss = SamplesLoss(loss='sinkhorn', p=p, blur=blur, backend='auto', debias=False, verbose=verbose)

        value = loss(a, X_s, b, X_t) # linear + entropic/KL reg?

        if metric == 'sqeuclidean':
            value *= 2  # because geomloss divides cost by two

        f, g = grad(value, [a, b])

        if log:
            log = {}
            log['f'] = f
            log['g'] = g
            log['value'] = value

            log['lazy_tensor'] = get_sinkhorn_geomloss_lazytensor(X_s, X_t, f, g, a, b, metric=metric, reg=reg, nx=nx)

            return value, log

        else:
            return value


    else:
        raise ImportError('geomloss not installed')

    