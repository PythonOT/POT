# -*- coding: utf-8 -*-
"""
Approximation of the Gram matrix for entropic regularized OT
"""

# Author: Titouan Vayer <titouan.vayer@irisa.fr>
#         Remi Flamary <remi.flamary@unice.fr>
#
# License: MIT License

import warnings


from ..utils import dist, get_lowrank_lazytensor
from ..backend import get_backend
import math
import random


def sinkhorn_low_rank_kernel(
    K1,
    K2,
    a=None,
    b=None,
    numItermax=1000,
    stopThr=1e-9,
    verbose=False,
    log=False,
    warn=True,
    warmstart=None,
):
    r"""
    TDO

    where :

    - :math:`\mathbf{K}_1`, `\mathbf{K}_2` are the (`dim_a`, `dim_r`), (`dim_b`, `dim_r`) kernel matrices
    - :math:`\Omega` is the entropic regularization term
      :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are source and target
      weights (histograms, both sum to 1)

    The algorithm used for solving the problem is the Sinkhorn-Knopp
    matrix scaling algorithm as proposed in :ref:`[2] <references-sinkhorn-knopp>`


    Parameters
    ----------
    a : array-like, shape (dim_a,)
        samples weights in the source domain
    b : array-like, shape (dim_b,) or array-like, shape (dim_b, n_hists)
        samples in the target domain, compute sinkhorn with multiple targets
        and fixed :math:`\mathbf{M}` if :math:`\mathbf{b}` is a matrix
        (return OT loss + dual variables in log)
    M : array-like, shape (dim_a, dim_b)
        loss matrix
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshold on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    warn : bool, optional
        if True, raises a warning if the algorithm doesn't convergence.
    warmstart: tuple of arrays, shape (dim_a, dim_b), optional
        Initialization of dual potentials. If provided, the dual potentials should be given
        (that is the logarithm of the u,v sinkhorn scaling vectors)

    Returns
    -------
    gamma : array-like, shape (dim_a, dim_b)
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters

    Examples
    --------

    >>> import ot
    >>> a=[.5, .5]
    >>> b=[.5, .5]
    >>> M=[[0., 1.], [1., 0.]]
    >>> ot.sinkhorn(a, b, M, 1)
    array([[0.36552929, 0.13447071],
           [0.13447071, 0.36552929]])


    .. _references-sinkhorn-knopp:
    References
    ----------

    .. [2] M. Cuturi, Sinkhorn Distances : Lightspeed Computation
        of Optimal Transport, Advances in Neural Information
        Processing Systems (NIPS) 26, 2013


    See Also
    --------
    ot.lp.emd : Unregularized OT
    ot.optim.cg : General regularized OT

    """

    nx = get_backend(K1, K2, a, b)

    if a is None:
        a = nx.full((K1.shape[0],), 1.0 / K1.shape[0], type_as=K1)
    if b is None:
        b = nx.full((K2.shape[0],), 1.0 / K2.shape[0], type_as=K2)

    # init data
    dim_a = len(a)
    dim_b = b.shape[0]

    if len(b.shape) > 1:
        n_hists = b.shape[1]
    else:
        n_hists = 0

    if log:
        dict_log = {"err": []}

    # we assume that no distances are null except those of the diagonal of
    # distances
    if warmstart is None:
        if n_hists:
            u = nx.ones((dim_a, n_hists), type_as=K1) / dim_a
            v = nx.ones((dim_b, n_hists), type_as=K2) / dim_b
        else:
            u = nx.ones(dim_a, type_as=K1) / dim_a
            v = nx.ones(dim_b, type_as=K2) / dim_b
    else:
        u, v = nx.exp(warmstart[0]), nx.exp(warmstart[1])

    err = 1
    for ii in range(numItermax):
        uprev = u
        vprev = v
        KtransposeU = K2 @ (nx.transpose(K1) @ u)
        v = b / KtransposeU
        KV = K1 @ (nx.transpose(K2) @ v)
        u = a / KV

        if (
            nx.any(KtransposeU == 0)
            or nx.any(nx.isnan(u))
            or nx.any(nx.isnan(v))
            or nx.any(nx.isinf(u))
            or nx.any(nx.isinf(v))
        ):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            warnings.warn("Warning: numerical errors at iteration %d" % ii)
            u = uprev
            v = vprev
            break
        if ii % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            if n_hists:
                tmp2 = nx.einsum("ik, ir, jr, jk->jk", u, K1, K2, v)
            else:
                # compute right marginal tmp2= (diag(u)Kdiag(v))^T1
                tmp2 = nx.einsum("i, ir, jr ,j->j", u, K1, K2, v)
            err = nx.norm(tmp2 - b)  # violation of marginal
            if log:
                dict_log["err"].append(err)

            if err < stopThr:
                break
            if verbose:
                if ii % 200 == 0:
                    print("{:5s}|{:12s}".format("It.", "Err") + "\n" + "-" * 19)
                print("{:5d}|{:8e}|".format(ii, err))
    else:
        if warn:
            warnings.warn(
                "Sinkhorn did not converge. You might want to "
                "increase the number of iterations `numItermax` "
                "or the regularization parameter `reg`."
            )

    if log:
        dict_log["niter"] = ii
        dict_log["u"] = u
        dict_log["v"] = v
        dict_log["lazy_plan"] = get_lowrank_lazytensor(
            u.reshape((-1, 1)) * K1, v.reshape((-1, 1)) * K2
        )
        return u, v, dict_log

    else:
        return u, v


def kernel_nystroem(X_s, X_t, rank=50, reg=1.0, random_state=None):
    nx = get_backend(X_s, X_t)

    random.seed(random_state)
    # return left and right factors corresponding to Nystrom on the Gaussian kernel K(x,y) = exp(-\|x-y\|^2/reg)
    n, m = X_s.shape[0], X_t.shape[0]
    n_components_source = min(n, math.ceil(rank / 2))
    n_components_target = min(m, math.ceil(rank / 2))
    # draw n_components/2 points in each distribution
    inds_source = nx.arange(n)
    random.shuffle(inds_source)
    basis_source = X_s[inds_source[:n_components_source]]

    inds_target = nx.arange(m)
    random.shuffle(inds_target)
    basis_target = X_t[inds_target[:n_components_target]]

    basis = nx.concatenate((basis_source, basis_target))

    Mzz = dist(basis, metric="sqeuclidean")  # compute \|z_i - z_j\|_2^2
    basis_kernel = nx.exp(-Mzz / reg)

    normalization = nx.pinv(basis_kernel, hermitian=True)

    Mxz = dist(X_s, basis, metric="sqeuclidean")
    Myz = dist(X_t, basis, metric="sqeuclidean")

    left_factor = nx.exp(-Mxz / reg) @ normalization
    right_factor = nx.exp(-Myz / reg)

    return left_factor, right_factor  # left_factor @ right_factor.T approx K
