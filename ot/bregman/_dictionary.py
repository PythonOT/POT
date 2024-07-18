# -*- coding: utf-8 -*-
"""
Dictionary Learning based on Bregman projections for entropic regularized OT
"""

# Author: Remi Flamary <remi.flamary@unice.fr>
#         Nicolas Courty <ncourty@irisa.fr>
#
# License: MIT License

import warnings


from ..utils import list_to_array
from ..backend import get_backend

from ._utils import projC, projR


def unmix(
    a,
    D,
    M,
    M0,
    h0,
    reg,
    reg0,
    alpha,
    numItermax=1000,
    stopThr=1e-3,
    verbose=False,
    log=False,
    warn=True,
):
    r"""
    Compute the unmixing of an observation with a given dictionary using Wasserstein distance

    The function solve the following optimization problem:

    .. math::

       \mathbf{h} = \mathop{\arg \min}_\mathbf{h} \quad
       (1 - \alpha)  W_{\mathbf{M}, \mathrm{reg}}(\mathbf{a}, \mathbf{Dh}) +
       \alpha W_{\mathbf{M_0}, \mathrm{reg}_0}(\mathbf{h}_0, \mathbf{h})


    where :

    - :math:`W_{M,reg}(\cdot,\cdot)` is the entropic regularized Wasserstein distance
      with :math:`\mathbf{M}` loss matrix (see :py:func:`ot.bregman.sinkhorn`)
    - :math:`\mathbf{D}` is a dictionary of `n_atoms` atoms of dimension `dim_a`,
      its expected shape is `(dim_a, n_atoms)`
    - :math:`\mathbf{h}` is the estimated unmixing of dimension `n_atoms`
    - :math:`\mathbf{a}` is an observed distribution of dimension `dim_a`
    - :math:`\mathbf{h}_0` is a prior on :math:`\mathbf{h}` of dimension `dim_prior`
    - `reg` and :math:`\mathbf{M}` are respectively the regularization term and the
      cost matrix (`dim_a`, `dim_a`) for OT data fitting
    - `reg`:math:`_0` and :math:`\mathbf{M_0}` are respectively the regularization
      term and the cost matrix (`dim_prior`, `n_atoms`) regularization
    - :math:`\alpha` weight data fitting and regularization

    The optimization problem is solved following the algorithm described
    in :ref:`[4] <references-unmix>`


    Parameters
    ----------
    a : array-like, shape (dim_a)
        observed distribution (histogram, sums to 1)
    D : array-like, shape (dim_a, n_atoms)
        dictionary matrix
    M : array-like, shape (dim_a, dim_a)
        loss matrix
    M0 : array-like, shape (n_atoms, dim_prior)
        loss matrix
    h0 : array-like, shape (n_atoms,)
        prior on the estimated unmixing h
    reg : float
        Regularization term >0 (Wasserstein data fitting)
    reg0 : float
        Regularization term >0 (Wasserstein reg with h0)
    alpha : float
        How much should we trust the prior ([0,1])
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

    Returns
    -------
    h : array-like, shape (n_atoms,)
        Wasserstein barycenter
    log : dict
        log dictionary return only if log==True in parameters


    .. _references-unmix:
    References
    ----------

    .. [4] S. Nakhostin, N. Courty, R. Flamary, D. Tuia, T. Corpetti,
        Supervised planetary unmixing with optimal transport, Workshop
        on Hyperspectral Image and Signal Processing :
        Evolution in Remote Sensing (WHISPERS), 2016.
    """

    a, D, M, M0, h0 = list_to_array(a, D, M, M0, h0)

    nx = get_backend(a, D, M, M0, h0)

    # M = M/np.median(M)
    K = nx.exp(-M / reg)

    # M0 = M0/np.median(M0)
    K0 = nx.exp(-M0 / reg0)
    old = h0

    err = 1
    # log = {'niter':0, 'all_err':[]}
    if log:
        log = {"err": []}

    for ii in range(numItermax):
        K = projC(K, a)
        K0 = projC(K0, h0)
        new = nx.sum(K0, axis=1)
        # we recombine the current selection from dictionary
        inv_new = nx.dot(D, new)
        other = nx.sum(K, axis=1)
        # geometric interpolation
        delta = nx.exp(alpha * nx.log(other) + (1 - alpha) * nx.log(inv_new))
        K = projR(K, delta)
        K0 = nx.dot(D.T, delta / inv_new)[:, None] * K0
        err = nx.norm(nx.sum(K0, axis=1) - old)
        old = new
        if log:
            log["err"].append(err)

        if verbose:
            if ii % 200 == 0:
                print("{:5s}|{:12s}".format("It.", "Err") + "\n" + "-" * 19)
            print("{:5d}|{:8e}|".format(ii, err))
        if err < stopThr:
            break
    else:
        if warn:
            warnings.warn(
                "Unmixing algorithm did not converge. You might want to "
                "increase the number of iterations `numItermax` "
                "or the regularization parameter `reg`."
            )
    if log:
        log["niter"] = ii
        return nx.sum(K0, axis=1), log
    else:
        return nx.sum(K0, axis=1)
