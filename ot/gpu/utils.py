# -*- coding: utf-8 -*-
"""
Utility functions for GPU
"""

# Author: Remi Flamary <remi.flamary@unice.fr>
#         Nicolas Courty <ncourty@irisa.fr>
#         Leo Gautheron <https://github.com/aje>
#
# License: MIT License

import cupy as np  # np used for matrix computation
import cupy as cp  # cp used for cupy specific operations


def euclidean_distances(a, b, squared=False, to_numpy=True):
    """
    Compute the pairwise euclidean distance between matrices a and b.

    If the input matrix are in numpy format, they will be uploaded to the
    GPU first which can incur significant time overhead.

    Parameters
    ----------
    a : np.ndarray (n, f)
        first matrix
    b : np.ndarray (m, f)
        second matrix
    to_numpy : boolean, optional (default True)
        If true convert back the GPU array result to numpy format.
    squared : boolean, optional (default False)
        if True, return squared euclidean distance matrix

    Returns
    -------
    c : (n x m) np.ndarray or cupy.ndarray
        pairwise euclidean distance distance matrix
    """

    a, b = to_gpu(a, b)

    a2 = np.sum(np.square(a), 1)
    b2 = np.sum(np.square(b), 1)

    c = -2 * np.dot(a, b.T)
    c += a2[:, None]
    c += b2[None, :]

    if not squared:
        np.sqrt(c, out=c)
    if to_numpy:
        return to_np(c)
    else:
        return c


def dist(x1, x2=None, metric='sqeuclidean', to_numpy=True):
    """Compute distance between samples in x1 and x2 on gpu

    Parameters
    ----------

    x1 : np.array (n1,d)
        matrix with n1 samples of size d
    x2 : np.array (n2,d), optional
        matrix with n2 samples of size d (if None then x2=x1)
    metric : str
        Metric from 'sqeuclidean', 'euclidean',


    Returns
    -------

    M : np.array (n1,n2)
        distance matrix computed with given metric

    """
    if x2 is None:
        x2 = x1
    if metric == "sqeuclidean":
        return euclidean_distances(x1, x2, squared=True, to_numpy=to_numpy)
    elif metric == "euclidean":
        return euclidean_distances(x1, x2, squared=False, to_numpy=to_numpy)
    else:
        raise NotImplementedError


def to_gpu(*args):
    """ Upload numpy arrays to GPU and return them"""
    if len(args) > 1:
        return (cp.asarray(x) for x in args)
    else:
        return cp.asarray(args[0])


def to_np(*args):
    """ convert GPU arras to numpy and return them"""
    if len(args) > 1:
        return (cp.asnumpy(x) for x in args)
    else:
        return cp.asnumpy(args[0])
