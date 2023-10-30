"""
Factored OT solvers (low rank, cost or OT plan)
"""

# Author: Remi Flamary <remi.flamary@polytehnique.edu>
#
# License: MIT License

from .backend import get_backend
from .utils import dist, get_lowrank_lazytensor
from .lp import emd
from .bregman import sinkhorn

__all__ = ['factored_optimal_transport']


def factored_optimal_transport(Xa, Xb, a=None, b=None, reg=0.0, r=100, X0=None, stopThr=1e-7, numItermax=100, verbose=False, log=False, **kwargs):
    r"""Solves factored OT problem and return OT plans and intermediate distribution

    This function solve the following OT problem [40]_

    .. math::
        \mathop{\arg \min}_\mu \quad  W_2^2(\mu_a,\mu)+ W_2^2(\mu,\mu_b)

    where :

    - :math:`\mu_a` and :math:`\mu_b`  are empirical distributions.
    - :math:`\mu` is an empirical distribution with r samples

    And returns the two OT plans between

    .. note:: This function is backend-compatible and will work on arrays
        from all compatible backends. But the algorithm uses the C++ CPU backend
        which can lead to copy overhead on GPU arrays.

    Uses the conditional gradient algorithm to solve the problem proposed in
    :ref:`[39] <references-weak>`.

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
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshold on the relative variation (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True


    Returns
    -------
    Ga: array-like, shape (ns, r)
        Optimal transportation matrix between source and the intermediate
        distribution
    Gb: array-like, shape (r, nt)
        Optimal transportation matrix between the intermediate and target
        distribution
    X: array-like, shape (r, d)
        Support of the intermediate distribution
    log: dict, optional
        If input log is true, a dictionary containing the cost and dual
        variables and exit status


    .. _references-factored:
    References
    ----------
    .. [40] Forrow, A., Hütter, J. C., Nitzan, M., Rigollet, P., Schiebinger,
        G., & Weed, J. (2019, April). Statistical optimal transport via factored
        couplings. In The 22nd International Conference on Artificial
        Intelligence and Statistics (pp. 2454-2465). PMLR.

    See Also
    --------
    ot.bregman.sinkhorn : Entropic regularized OT
    ot.optim.cg : General regularized OT
    """

    nx = get_backend(Xa, Xb)

    n_a = Xa.shape[0]
    n_b = Xb.shape[0]
    d = Xa.shape[1]

    if a is None:
        a = nx.ones((n_a), type_as=Xa) / n_a
    if b is None:
        b = nx.ones((n_b), type_as=Xb) / n_b

    if X0 is None:
        X = nx.randn(r, d, type_as=Xa)
    else:
        X = X0

    w = nx.ones(r, type_as=Xa) / r

    def solve_ot(X1, X2, w1, w2):
        M = dist(X1, X2)
        if reg > 0:
            G, log = sinkhorn(w1, w2, M, reg, log=True, **kwargs)
            log['cost'] = nx.sum(G * M)
            return G, log
        else:
            return emd(w1, w2, M, log=True, **kwargs)

    norm_delta = []

    # solve the barycenter
    for i in range(numItermax):

        old_X = X

        # solve OT with template
        Ga, loga = solve_ot(Xa, X, a, w)
        Gb, logb = solve_ot(X, Xb, w, b)

        X = 0.5 * (nx.dot(Ga.T, Xa) + nx.dot(Gb, Xb)) * r

        delta = nx.norm(X - old_X)
        if delta < stopThr:
            break
        if log:
            norm_delta.append(delta)

    if log:
        log_dic = {'delta_iter': norm_delta,
                   'ua': loga['u'],
                   'va': loga['v'],
                   'ub': logb['u'],
                   'vb': logb['v'],
                   'costa': loga['cost'],
                   'costb': logb['cost'],
                   'lazy_plan': get_lowrank_lazytensor(Ga * r, Gb.T, nx=nx),
                   }
        return Ga, Gb, X, log_dic

    return Ga, Gb, X
