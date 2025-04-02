"""
Weak optimal ransport solvers
"""

# Author: Remi Flamary <remi.flamary@polytehnique.edu>
#
# License: MIT License

from .backend import get_backend
from .optim import cg
import numpy as np

__all__ = ["weak_optimal_transport"]


def weak_optimal_transport(
    Xa, Xb, a=None, b=None, verbose=False, log=False, G0=None, **kwargs
):
    r"""Solves the weak optimal transport problem between two empirical distributions


    .. math::
        \gamma = \mathop{\arg \min}_\gamma \quad \sum_i \mathbf{a}_i \left(\mathbf{X^a}_i - \frac{1}{\mathbf{a}_i} \sum_j \gamma_{ij} \mathbf{X^b}_j \right)^2

        s.t. \ \gamma \mathbf{1} = \mathbf{a}

             \gamma^T \mathbf{1} = \mathbf{b}

             \gamma \geq 0

    where :

    - :math:`X^a` and  :math:`X^b`  are the sample matrices.
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are the sample weights


    .. note:: This function is backend-compatible and will work on arrays
        from all compatible backends. But the algorithm uses the C++ CPU backend
        which can lead to copy overhead on GPU arrays.

    Uses the conditional gradient algorithm to solve the problem proposed
    in :ref:`[39] <references-weak>`.

    Parameters
    ----------
    Xa : (ns,d) array-like, float
        Source samples
    Xb : (nt,d) array-like, float
        Target samples
    a : (ns,) array-like, float
        Source histogram (uniform weight if empty list)
    b : (nt,) array-like, float
        Target histogram (uniform weight if empty list))
    G0 : (ns,nt) array-like, float
        initial guess (default is indep joint density)
    numItermax : int, optional
        Max number of iterations
    numItermaxEmd : int, optional
        Max number of iterations for emd
    stopThr : float, optional
        Stop threshold on the relative variation (>0)
    stopThr2 : float, optional
        Stop threshold on the absolute variation (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True


    Returns
    -------
    gamma: array-like, shape (ns, nt)
        Optimal transportation matrix for the given
        parameters
    log: dict, optional
        If input log is true, a dictionary containing the
        cost and dual variables and exit status


    .. _references-weak:
    References
    ----------
    .. [39] Gozlan, N., Roberto, C., Samson, P. M., & Tetali, P. (2017).
        Kantorovich duality for general transport costs and applications.
        Journal of Functional Analysis, 273(11), 3327-3405.

    See Also
    --------
    ot.bregman.sinkhorn : Entropic regularized OT
    ot.optim.cg : General regularized OT
    """

    nx = get_backend(Xa, Xb)

    Xa2 = nx.to_numpy(Xa)
    Xb2 = nx.to_numpy(Xb)

    if a is None:
        a2 = np.ones((Xa.shape[0])) / Xa.shape[0]
    else:
        a2 = nx.to_numpy(a)
    if b is None:
        b2 = np.ones((Xb.shape[0])) / Xb.shape[0]
    else:
        b2 = nx.to_numpy(b)

    # init uniform
    if G0 is None:
        T0 = a2[:, None] * b2[None, :]
    else:
        T0 = nx.to_numpy(G0)

    # weak OT loss
    def f(T):
        return np.dot(a2, np.sum((Xa2 - np.dot(T, Xb2) / a2[:, None]) ** 2, 1))

    # weak OT gradient
    def df(T):
        return -2 * np.dot(Xa2 - np.dot(T, Xb2) / a2[:, None], Xb2.T)

    # solve with conditional gradient and return solution
    if log:
        res, log = cg(a2, b2, 0, 1, f, df, T0, log=log, verbose=verbose, **kwargs)
        log["u"] = nx.from_numpy(log["u"], type_as=Xa)
        log["v"] = nx.from_numpy(log["v"], type_as=Xb)
        return nx.from_numpy(res, type_as=Xa), log
    else:
        return nx.from_numpy(
            cg(a2, b2, 0, 1, f, df, T0, log=log, verbose=verbose, **kwargs), type_as=Xa
        )
