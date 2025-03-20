# -*- coding: utf-8 -*-
"""
Regularized Unbalanced OT solvers
"""

# Author: Hicham Janati <hicham.janati@inria.fr>
#         Laetitia Chapel <laetitia.chapel@univ-ubs.fr>
#         Quang Huy Tran <quang-huy.tran@univ-ubs.fr>
#
# License: MIT License

from __future__ import division
import warnings

from ..backend import get_backend
from ..utils import list_to_array, get_parameter_pair


def sinkhorn_unbalanced(
    a,
    b,
    M,
    reg,
    reg_m,
    method="sinkhorn",
    reg_type="kl",
    c=None,
    warmstart=None,
    numItermax=1000,
    stopThr=1e-6,
    verbose=False,
    log=False,
    **kwargs,
):
    r"""
    Solve the unbalanced entropic regularization optimal transport problem
    and return the OT plan

    The function solves the following optimization problem:

    .. math::
        W = \arg \min_\gamma \ \langle \gamma, \mathbf{M} \rangle_F +
        \mathrm{reg} \cdot \mathrm{KL}(\gamma, \mathbf{c}) +
        \mathrm{reg_{m1}} \cdot \mathrm{KL}(\gamma \mathbf{1}, \mathbf{a}) +
        \mathrm{reg_{m2}} \cdot \mathrm{KL}(\gamma^T \mathbf{1}, \mathbf{b})

        s.t.
             \gamma \geq 0

    where :

    - :math:`\mathbf{M}` is the (`dim_a`, `dim_b`) metric cost matrix
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are source and target unbalanced distributions
    - :math:`\mathbf{c}` is a reference distribution for the regularization
    - KL is the Kullback-Leibler divergence

    The algorithm used for solving the problem is the generalized
    Sinkhorn-Knopp matrix scaling algorithm as proposed in :ref:`[10, 25]
    <references-sinkhorn-unbalanced>`

    .. warning::
        Starting from version 0.9.5, the default value has been changed to `reg_type='kl'` instead of `reg_type='entropy'`. This makes the function more consistent with the literature
        and the other solvers. If you want to use the entropy regularization, please set `reg_type='entropy'` explicitly.


    Parameters
    ----------
    a : array-like, shape (dim_a,)
        Unnormalized histogram of dimension `dim_a`
        If `a` is an empty list or array ([]),
        then `a` is set to uniform distribution.
    b : array-like, shape (dim_b,)
        One or multiple unnormalized histograms of dimension `dim_b`.
        If `b` is an empty list or array ([]),
        then `b` is set to uniform distribution.
        If many, compute all the OT costs :math:`(\mathbf{a}, \mathbf{b}_i)_i`
    M : array-like, shape (dim_a, dim_b)
        loss matrix
    reg : float
        Entropy regularization term > 0
    reg_m: float or indexable object of length 1 or 2
        Marginal relaxation term.
        If :math:`\mathrm{reg_{m}}` is a scalar or an indexable object of length 1,
        then the same :math:`\mathrm{reg_{m}}` is applied to both marginal relaxations.
        The entropic balanced OT can be recovered using :math:`\mathrm{reg_{m}}=float("inf")`.
        For semi-relaxed case, use either
        :math:`\mathrm{reg_{m}}=(float("inf"), scalar)` or
        :math:`\mathrm{reg_{m}}=(scalar, float("inf"))`.
        If :math:`\mathrm{reg_{m}}` is an array,
        it must have the same backend as input arrays `(a, b, M)`.
    method : str
        method used for the solver either 'sinkhorn', 'sinkhorn_stabilized', 'sinkhorn_translation_invariant' or
        'sinkhorn_reg_scaling', see those function for specific parameters
    reg_type : string, optional
        Regularizer term. Can take two values:

        - Negative entropy: 'entropy':
          :math:`\Omega(\gamma) = \sum_{i,j} \gamma_{i,j} \log(\gamma_{i,j}) - \sum_{i,j} \gamma_{i,j}`.
          This is equivalent (up to a constant) to :math:`\Omega(\gamma) = \text{KL}(\gamma, 1_{dim_a} 1_{dim_b}^T)`.
        - Kullback-Leibler divergence (default): 'kl':
          :math:`\Omega(\gamma) = \text{KL}(\gamma, \mathbf{a} \mathbf{b}^T)`.
    c : array-like, shape (dim_a, dim_b), optional (default=None)
        Reference measure for the regularization.
        If None, then use :math:`\mathbf{c} = \mathbf{a} \mathbf{b}^T`.
        If :math:`\texttt{reg_type}=`'entropy', then :math:`\mathbf{c} = 1_{dim_a} 1_{dim_b}^T`.
    warmstart: tuple of arrays, shape (dim_a, dim_b), optional
        Initialization of dual potentials. If provided, the dual potentials should be given
        (that is the logarithm of the `u`, `v` sinkhorn scaling vectors).
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshold on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record `log` if `True`


    Returns
    -------
    if n_hists == 1:
        - gamma : array-like, shape(dim_a, dim_b)
            Optimal transportation matrix for the given parameters
        - log : dict
            log dictionary returned only if `log` is `True`
    else:
        - ot_distance : array-like, shape (n_hists,)
            the OT distance between :math:`\mathbf{a}` and each of the histograms :math:`\mathbf{b}_i`
        - log : dict
            log dictionary returned only if `log` is `True`

    Examples
    --------

    >>> import ot
    >>> import numpy as np
    >>> a=[.5, .5]
    >>> b=[.5, .5]
    >>> M=[[0., 1.], [1., 0.]]
    >>> np.round(ot.sinkhorn_unbalanced(a, b, M, 1, 1), 7)
    array([[0.3220536, 0.1184769],
           [0.1184769, 0.3220536]])


    .. _references-sinkhorn-unbalanced:
    References
    ----------
    .. [2] M. Cuturi, Sinkhorn Distances : Lightspeed Computation of Optimal
        Transport, Advances in Neural Information Processing Systems
        (NIPS) 26, 2013

    .. [9] Schmitzer, B. (2016). Stabilized Sparse Scaling Algorithms for
        Entropy Regularized Transport Problems. arXiv preprint arXiv:1610.06519.

    .. [10] Chizat, L., Peyré, G., Schmitzer, B., & Vialard, F. X. (2016).
        Scaling algorithms for unbalanced transport problems. arXiv preprint
        arXiv:1607.05816.

    .. [25] Frogner C., Zhang C., Mobahi H., Araya-Polo M., Poggio T. :
        Learning with a Wasserstein Loss,  Advances in Neural Information
        Processing Systems (NIPS) 2015

    .. [73] Séjourné, T., Vialard, F. X., & Peyré, G. (2022).
       Faster unbalanced optimal transport: Translation invariant sinkhorn and 1-d frank-wolfe.
       In International Conference on Artificial Intelligence and Statistics (pp. 4995-5021). PMLR.


    See Also
    --------
    ot.unbalanced.sinkhorn_knopp_unbalanced: Unbalanced Classic Sinkhorn :ref:`[10] <references-sinkhorn-unbalanced>`
    ot.unbalanced.sinkhorn_stabilized_unbalanced:
        Unbalanced Stabilized sinkhorn :ref:`[9, 10] <references-sinkhorn-unbalanced>`
    ot.unbalanced.sinkhorn_reg_scaling_unbalanced:
        Unbalanced Sinkhorn with epsilon scaling :ref:`[9, 10] <references-sinkhorn-unbalanced>`
    ot.unbalanced.sinkhorn_unbalanced_translation_invariant:
        Translation Invariant Unbalanced Sinkhorn :ref:`[73] <references-sinkhorn-unbalanced-translation-invariant>`

    """

    if method.lower() == "sinkhorn":
        return sinkhorn_knopp_unbalanced(
            a,
            b,
            M,
            reg,
            reg_m,
            reg_type,
            c,
            warmstart,
            numItermax=numItermax,
            stopThr=stopThr,
            verbose=verbose,
            log=log,
            **kwargs,
        )

    elif method.lower() == "sinkhorn_stabilized":
        return sinkhorn_stabilized_unbalanced(
            a,
            b,
            M,
            reg,
            reg_m,
            reg_type,
            c,
            warmstart,
            numItermax=numItermax,
            stopThr=stopThr,
            verbose=verbose,
            log=log,
            **kwargs,
        )

    elif method.lower() == "sinkhorn_translation_invariant":
        return sinkhorn_unbalanced_translation_invariant(
            a,
            b,
            M,
            reg,
            reg_m,
            reg_type,
            c,
            warmstart,
            numItermax=numItermax,
            stopThr=stopThr,
            verbose=verbose,
            log=log,
            **kwargs,
        )

    elif method.lower() in ["sinkhorn_reg_scaling"]:
        warnings.warn("Method not implemented yet. Using classic Sinkhorn-Knopp")
        return sinkhorn_knopp_unbalanced(
            a,
            b,
            M,
            reg,
            reg_m,
            reg_type,
            c,
            warmstart,
            numItermax=numItermax,
            stopThr=stopThr,
            verbose=verbose,
            log=log,
            **kwargs,
        )
    else:
        raise ValueError("Unknown method '%s'." % method)


def sinkhorn_unbalanced2(
    a,
    b,
    M,
    reg,
    reg_m,
    method="sinkhorn",
    reg_type="kl",
    c=None,
    warmstart=None,
    returnCost="linear",
    numItermax=1000,
    stopThr=1e-6,
    verbose=False,
    log=False,
    **kwargs,
):
    r"""
    Solve the entropic regularization unbalanced optimal transport problem and
    return the cost

    The function solves the following optimization problem:

    .. math::
        \min_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F +
        \mathrm{reg} \cdot \mathrm{KL}(\gamma, \mathbf{c}) +
        \mathrm{reg_{m1}} \cdot \mathrm{KL}(\gamma \mathbf{1}, \mathbf{a}) +
        \mathrm{reg_{m2}} \cdot \mathrm{KL}(\gamma^T \mathbf{1}, \mathbf{b})

        s.t.
             \gamma\geq 0
    where :

    - :math:`\mathbf{M}` is the (`dim_a`, `dim_b`) metric cost matrix
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are source and target unbalanced distributions
    - :math:`\mathbf{c}` is a reference distribution for the regularization
    - KL is the Kullback-Leibler divergence

    The algorithm used for solving the problem is the generalized
    Sinkhorn-Knopp matrix scaling algorithm as proposed in :ref:`[10, 25]
    <references-sinkhorn-unbalanced2>`

    .. warning::
        Starting from version 0.9.5, the default value has been changed to `reg_type='kl'` instead of `reg_type='entropy'`. This makes the function more consistent with the literature
        and the other solvers. If you want to use the entropy regularization, please set `reg_type='entropy'` explicitly.

    Parameters
    ----------
    a : array-like, shape (dim_a,)
        Unnormalized histogram of dimension `dim_a`
        If `a` is an empty list or array ([]),
        then `a` is set to uniform distribution.
    b : array-like, shape (dim_b,)
        One or multiple unnormalized histograms of dimension `dim_b`.
        If `b` is an empty list or array ([]),
        then `b` is set to uniform distribution.
        If many, compute all the OT costs :math:`(\mathbf{a}, \mathbf{b}_i)_i`
    M : array-like, shape (dim_a, dim_b)
        loss matrix
    reg : float
        Entropy regularization term > 0
    reg_m: float or indexable object of length 1 or 2
        Marginal relaxation term.
        If :math:`\mathrm{reg_{m}}` is a scalar or an indexable object of length 1,
        then the same :math:`\mathrm{reg_{m}}` is applied to both marginal relaxations.
        The entropic balanced OT can be recovered using :math:`\mathrm{reg_{m}}=float("inf")`.
        For semi-relaxed case, use either
        :math:`\mathrm{reg_{m}}=(float("inf"), scalar)` or
        :math:`\mathrm{reg_{m}}=(scalar, float("inf"))`.
        If :math:`\mathrm{reg_{m}}` is an array,
        it must have the same backend as input arrays `(a, b, M)`.
    method : str
        method used for the solver either 'sinkhorn', 'sinkhorn_stabilized', 'sinkhorn_translation_invariant' or
        'sinkhorn_reg_scaling', see those function for specific parameters
    reg_type : string, optional
        Regularizer term. Can take two values:

        - Negative entropy: 'entropy':
          :math:`\Omega(\gamma) = \sum_{i,j} \gamma_{i,j} \log(\gamma_{i,j}) - \sum_{i,j} \gamma_{i,j}`.
          This is equivalent (up to a constant) to :math:`\Omega(\gamma) = \text{KL}(\gamma, 1_{dim_a} 1_{dim_b}^T)`.
        - Kullback-Leibler divergence: 'kl':
          :math:`\Omega(\gamma) = \text{KL}(\gamma, \mathbf{a} \mathbf{b}^T)`.
    c : array-like, shape (dim_a, dim_b), optional (default=None)
        Reference measure for the regularization.
        If None, then use :math:`\mathbf{c} = \mathbf{a} \mathbf{b}^T`.
        If :math:`\texttt{reg_type}=`'entropy', then :math:`\mathbf{c} = 1_{dim_a} 1_{dim_b}^T`.
    warmstart: tuple of arrays, shape (dim_a, dim_b), optional
        Initialization of dual potentials. If provided, the dual potentials should be given
        (that is the logarithm of the u,v sinkhorn scaling vectors).
    returnCost: string, optional (default = "linear")
        If `returnCost` = "linear", then return the linear part of the unbalanced OT loss.
        If `returnCost` = "total", then return the total unbalanced OT loss.
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshold on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record `log` if `True`


    Returns
    -------
    ot_cost : array-like, shape (n_hists,)
        the OT cost between :math:`\mathbf{a}` and each of the histograms :math:`\mathbf{b}_i`
    log : dict
        log dictionary returned only if `log` is `True`

    Examples
    --------

    >>> import ot
    >>> import numpy as np
    >>> a=[.5, .10]
    >>> b=[.5, .5]
    >>> M=[[0., 1.],[1., 0.]]
    >>> np.round(ot.unbalanced.sinkhorn_unbalanced2(a, b, M, 1., 1.), 8)
    0.19600125


    .. _references-sinkhorn-unbalanced2:
    References
    ----------
    .. [2] M. Cuturi, Sinkhorn Distances : Lightspeed Computation of Optimal
        Transport, Advances in Neural Information Processing Systems
        (NIPS) 26, 2013

    .. [9] Schmitzer, B. (2016). Stabilized Sparse Scaling Algorithms for
        Entropy Regularized Transport Problems. arXiv preprint arXiv:1610.06519.

    .. [10] Chizat, L., Peyré, G., Schmitzer, B., & Vialard, F. X. (2016).
        Scaling algorithms for unbalanced transport problems. arXiv preprint
        arXiv:1607.05816.

    .. [25] Frogner C., Zhang C., Mobahi H., Araya-Polo M., Poggio T. :
        Learning with a Wasserstein Loss,  Advances in Neural Information
        Processing Systems (NIPS) 2015

    .. [73] Séjourné, T., Vialard, F. X., & Peyré, G. (2022).
        Faster unbalanced optimal transport: Translation invariant sinkhorn and 1-d frank-wolfe.
        In International Conference on Artificial Intelligence and Statistics (pp. 4995-5021). PMLR.

    See Also
    --------
    ot.unbalanced.sinkhorn_knopp: Unbalanced Classic Sinkhorn :ref:`[10] <references-sinkhorn-unbalanced2>`
    ot.unbalanced.sinkhorn_stabilized: Unbalanced Stabilized sinkhorn :ref:`[9, 10] <references-sinkhorn-unbalanced2>`
    ot.unbalanced.sinkhorn_reg_scaling: Unbalanced Sinkhorn with epsilon scaling :ref:`[9, 10] <references-sinkhorn-unbalanced2>`
    ot.unbalanced.sinkhorn_unbalanced_translation_invariant: Translation Invariant Unbalanced Sinkhorn :ref:`[73] <references-sinkhorn-unbalanced2>`

    """
    M, a, b = list_to_array(M, a, b)

    if len(b.shape) < 2:
        if method.lower() == "sinkhorn":
            res = sinkhorn_knopp_unbalanced(
                a,
                b,
                M,
                reg,
                reg_m,
                reg_type,
                c,
                warmstart,
                numItermax=numItermax,
                stopThr=stopThr,
                verbose=verbose,
                log=True,
                **kwargs,
            )

        elif method.lower() == "sinkhorn_stabilized":
            res = sinkhorn_stabilized_unbalanced(
                a,
                b,
                M,
                reg,
                reg_m,
                reg_type,
                c,
                warmstart,
                numItermax=numItermax,
                stopThr=stopThr,
                verbose=verbose,
                log=True,
                **kwargs,
            )

        elif method.lower() == "sinkhorn_translation_invariant":
            res = sinkhorn_unbalanced_translation_invariant(
                a,
                b,
                M,
                reg,
                reg_m,
                reg_type,
                c,
                warmstart,
                numItermax=numItermax,
                stopThr=stopThr,
                verbose=verbose,
                log=True,
                **kwargs,
            )

        elif method.lower() in ["sinkhorn_reg_scaling"]:
            warnings.warn("Method not implemented yet. Using classic Sinkhorn-Knopp")
            res = sinkhorn_knopp_unbalanced(
                a,
                b,
                M,
                reg,
                reg_m,
                reg_type,
                c,
                warmstart,
                numItermax=numItermax,
                stopThr=stopThr,
                verbose=verbose,
                log=True,
                **kwargs,
            )
        else:
            raise ValueError("Unknown method %s." % method)

        if returnCost == "linear":
            cost = res[1]["cost"]
        elif returnCost == "total":
            cost = res[1]["total_cost"]
        else:
            raise ValueError("Unknown returnCost = {}".format(returnCost))

        if log:
            return cost, res[1]
        else:
            return cost

    else:
        if reg_type == "kl":
            warnings.warn("Reg_type not implemented yet. Use entropy.")

        if method.lower() == "sinkhorn":
            return sinkhorn_knopp_unbalanced(
                a,
                b,
                M,
                reg,
                reg_m,
                reg_type,
                c,
                warmstart,
                numItermax=numItermax,
                stopThr=stopThr,
                verbose=verbose,
                log=log,
                **kwargs,
            )

        elif method.lower() == "sinkhorn_stabilized":
            return sinkhorn_stabilized_unbalanced(
                a,
                b,
                M,
                reg,
                reg_m,
                reg_type,
                c,
                warmstart,
                numItermax=numItermax,
                stopThr=stopThr,
                verbose=verbose,
                log=log,
                **kwargs,
            )

        elif method.lower() == "sinkhorn_translation_invariant":
            return sinkhorn_unbalanced_translation_invariant(
                a,
                b,
                M,
                reg,
                reg_m,
                reg_type,
                c,
                warmstart,
                numItermax=numItermax,
                stopThr=stopThr,
                verbose=verbose,
                log=log,
                **kwargs,
            )

        elif method.lower() in ["sinkhorn_reg_scaling"]:
            warnings.warn("Method not implemented yet. Using classic Sinkhorn-Knopp")
            return sinkhorn_knopp_unbalanced(
                a,
                b,
                M,
                reg,
                reg_m,
                reg_type,
                c,
                warmstart,
                numItermax=numItermax,
                stopThr=stopThr,
                verbose=verbose,
                log=log,
                **kwargs,
            )
        else:
            raise ValueError("Unknown method %s." % method)


def sinkhorn_knopp_unbalanced(
    a,
    b,
    M,
    reg,
    reg_m,
    reg_type="kl",
    c=None,
    warmstart=None,
    numItermax=1000,
    stopThr=1e-6,
    verbose=False,
    log=False,
    **kwargs,
):
    r"""
    Solve the entropic regularization unbalanced optimal transport problem and
    return the OT plan

    The function solves the following optimization problem:

    .. math::
        W = \arg \min_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F +
        \mathrm{reg} \cdot \mathrm{KL}(\gamma, \mathbf{c}) +
        \mathrm{reg_{m1}} \cdot \mathrm{KL}(\gamma \mathbf{1}, \mathbf{a}) +
        \mathrm{reg_{m2}} \cdot \mathrm{KL}(\gamma^T \mathbf{1}, \mathbf{b})

        s.t.
             \gamma \geq 0

    where :

    - :math:`\mathbf{M}` is the (`dim_a`, `dim_b`) metric cost matrix
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are source and target unbalanced distributions
    - :math:`\mathbf{c}` is a reference distribution for the regularization
    - KL is the Kullback-Leibler divergence

    The algorithm used for solving the problem is the generalized Sinkhorn-Knopp matrix scaling algorithm as proposed in :ref:`[10, 25] <references-sinkhorn-knopp-unbalanced>`

    .. warning::
        Starting from version 0.9.5, the default value has been changed to `reg_type='kl'` instead of `reg_type='entropy'`. This makes the function more consistent with the literature
        and the other solvers. If you want to use the entropy regularization, please set `reg_type='entropy'` explicitly.


    Parameters
    ----------
    a : array-like, shape (dim_a,)
        Unnormalized histogram of dimension `dim_a`
        If `a` is an empty list or array ([]),
        then `a` is set to uniform distribution.
    b : array-like, shape (dim_b,)
        One or multiple unnormalized histograms of dimension `dim_b`.
        If `b` is an empty list or array ([]),
        then `b` is set to uniform distribution.
        If many, compute all the OT costs :math:`(\mathbf{a}, \mathbf{b}_i)_i`
    M : array-like, shape (dim_a, dim_b)
        loss matrix
    reg : float
        Entropy regularization term > 0
    reg_m: float or indexable object of length 1 or 2
        Marginal relaxation term.
        If :math:`\mathrm{reg_{m}}` is a scalar or an indexable object of length 1,
        then the same :math:`\mathrm{reg_{m}}` is applied to both marginal relaxations.
        The entropic balanced OT can be recovered using :math:`\mathrm{reg_{m}}=float("inf")`.
        For semi-relaxed case, use either
        :math:`\mathrm{reg_{m}}=(float("inf"), scalar)` or
        :math:`\mathrm{reg_{m}}=(scalar, float("inf"))`.
        If :math:`\mathrm{reg_{m}}` is an array,
        it must have the same backend as input arrays `(a, b, M)`.
    reg_type : string, optional
        Regularizer term. Can take two values:

        - Negative entropy: 'entropy':
          :math:`\Omega(\gamma) = \sum_{i,j} \gamma_{i,j} \log(\gamma_{i,j}) - \sum_{i,j} \gamma_{i,j}`.
          This is equivalent (up to a constant) to :math:`\Omega(\gamma) = \text{KL}(\gamma, 1_{dim_a} 1_{dim_b}^T)`.
        - Kullback-Leibler divergence: 'kl':
          :math:`\Omega(\gamma) = \text{KL}(\gamma, \mathbf{a} \mathbf{b}^T)`.
    c : array-like, shape (dim_a, dim_b), optional (default=None)
        Reference measure for the regularization.
        If None, then use :math:`\mathbf{c} = \mathbf{a} \mathbf{b}^T`.
        If :math:`\texttt{reg_type}=`'entropy', then :math:`\mathbf{c} = 1_{dim_a} 1_{dim_b}^T`.
    warmstart: tuple of arrays, shape (dim_a, dim_b), optional
        Initialization of dual potentials. If provided, the dual potentials should be given
        (that is the logarithm of the `u`, `v` sinkhorn scaling vectors).
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshold on error (> 0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record `log` if `True`


    Returns
    -------
    if n_hists == 1:
        - gamma : array-like, shape (dim_a, dim_b)
            Optimal transportation matrix for the given parameters
        - log : dict
            log dictionary returned only if `log` is `True`
    else:
        - ot_cost : array-like, shape (n_hists,)
            the OT cost between :math:`\mathbf{a}` and each of the histograms :math:`\mathbf{b}_i`
        - log : dict
            log dictionary returned only if `log` is `True`

    Examples
    --------

    >>> import ot
    >>> import numpy as np
    >>> a=[.5, .5]
    >>> b=[.5, .5]
    >>> M=[[0., 1.],[1., 0.]]
    >>> np.round(ot.unbalanced.sinkhorn_knopp_unbalanced(a, b, M, 1., 1.), 7)
    array([[0.3220536, 0.1184769],
           [0.1184769, 0.3220536]])


    .. _references-sinkhorn-knopp-unbalanced:
    References
    ----------
    .. [10] Chizat, L., Peyré, G., Schmitzer, B., & Vialard, F. X. (2016).
        Scaling algorithms for unbalanced transport problems. arXiv preprint
        arXiv:1607.05816.

    .. [25] Frogner C., Zhang C., Mobahi H., Araya-Polo M., Poggio T. :
        Learning with a Wasserstein Loss,  Advances in Neural Information
        Processing Systems (NIPS) 2015

    See Also
    --------
    ot.lp.emd : Unregularized OT
    ot.optim.cg : General regularized OT

    """
    M, a, b = list_to_array(M, a, b)
    nx = get_backend(M, a, b)

    dim_a, dim_b = M.shape

    if len(a) == 0:
        a = nx.ones(dim_a, type_as=M) / dim_a
    if len(b) == 0:
        b = nx.ones(dim_b, type_as=M) / dim_b

    if len(b.shape) > 1:
        n_hists = b.shape[1]
    else:
        n_hists = 0

    reg_m1, reg_m2 = get_parameter_pair(reg_m)

    if log:
        dict_log = {"err": []}

    # we assume that no distances are null except those of the diagonal of
    # distances
    if warmstart is None:
        if n_hists:
            u = nx.ones((dim_a, 1), type_as=M)
            v = nx.ones((dim_b, n_hists), type_as=M)
            a = a.reshape(dim_a, 1)
        else:
            u = nx.ones(dim_a, type_as=M)
            v = nx.ones(dim_b, type_as=M)
    else:
        u, v = nx.exp(warmstart[0]), nx.exp(warmstart[1])

    if reg_type.lower() == "entropy":
        warnings.warn(
            "If reg_type = entropy, then the matrix c is overwritten by the one matrix."
        )
        c = nx.ones((dim_a, dim_b), type_as=M)

    if n_hists:
        K = nx.exp(-M / reg)
    else:
        c = a[:, None] * b[None, :] if c is None else c
        K = nx.exp(-M / reg) * c

    fi_1 = reg_m1 / (reg_m1 + reg) if reg_m1 != float("inf") else 1
    fi_2 = reg_m2 / (reg_m2 + reg) if reg_m2 != float("inf") else 1

    err = 1.0

    for i in range(numItermax):
        uprev = u
        vprev = v

        Kv = nx.dot(K, v)
        u = (a / Kv) ** fi_1
        Ktu = nx.dot(K.T, u)
        v = (b / Ktu) ** fi_2

        if (
            nx.any(Ktu == 0.0)
            or nx.any(nx.isnan(u))
            or nx.any(nx.isnan(v))
            or nx.any(nx.isinf(u))
            or nx.any(nx.isinf(v))
        ):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            warnings.warn("Numerical errors at iteration %s" % i)
            u = uprev
            v = vprev
            break

        err_u = nx.max(nx.abs(u - uprev)) / max(
            nx.max(nx.abs(u)), nx.max(nx.abs(uprev)), 1.0
        )
        err_v = nx.max(nx.abs(v - vprev)) / max(
            nx.max(nx.abs(v)), nx.max(nx.abs(vprev)), 1.0
        )
        err = 0.5 * (err_u + err_v)
        if log:
            dict_log["err"].append(err)
            if verbose:
                if i % 50 == 0:
                    print("{:5s}|{:12s}".format("It.", "Err") + "\n" + "-" * 19)
                print("{:5d}|{:8e}|".format(i, err))
        if err < stopThr:
            break

    if log:
        dict_log["logu"] = nx.log(u + 1e-300)
        dict_log["logv"] = nx.log(v + 1e-300)

    if n_hists:  # return only loss
        res = nx.einsum("ik,ij,jk,ij->k", u, K, v, M)
        if log:
            return res, dict_log
        else:
            return res

    else:  # return OT matrix
        plan = u[:, None] * K * v[None, :]

        if log:
            linear_cost = nx.sum(plan * M)
            dict_log["cost"] = linear_cost

            total_cost = linear_cost + reg * nx.kl_div(plan, c)
            if reg_m1 != float("inf"):
                total_cost = total_cost + reg_m1 * nx.kl_div(nx.sum(plan, 1), a)
            if reg_m2 != float("inf"):
                total_cost = total_cost + reg_m2 * nx.kl_div(nx.sum(plan, 0), b)
            dict_log["total_cost"] = total_cost

            return plan, dict_log
        else:
            return plan


def sinkhorn_stabilized_unbalanced(
    a,
    b,
    M,
    reg,
    reg_m,
    reg_type="kl",
    c=None,
    warmstart=None,
    tau=1e5,
    numItermax=1000,
    stopThr=1e-6,
    verbose=False,
    log=False,
    **kwargs,
):
    r"""
    Solve the entropic regularization unbalanced optimal transport
    problem and return the loss

    The function solves the following optimization problem using log-domain
    stabilization as proposed in :ref:`[10] <references-sinkhorn-stabilized-unbalanced>`:

    .. math::
        W = \arg \min_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F +
        \mathrm{reg} \cdot \mathrm{KL}(\gamma, \mathbf{c}) +
        \mathrm{reg_{m1}} \cdot \mathrm{KL}(\gamma \mathbf{1}, \mathbf{a}) +
        \mathrm{reg_{m2}} \cdot \mathrm{KL}(\gamma^T \mathbf{1}, \mathbf{b})

        s.t.
             \gamma \geq 0

    where :

    - :math:`\mathbf{M}` is the (`dim_a`, `dim_b`) metric cost matrix
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are source and target unbalanced distributions
    - :math:`\mathbf{c}` is a reference distribution for the regularization
    - KL is the Kullback-Leibler divergence

    The algorithm used for solving the problem is the generalized
    Sinkhorn-Knopp matrix scaling algorithm as proposed in :ref:`[10, 25] <references-sinkhorn-stabilized-unbalanced>`


    Parameters
    ----------
    a : array-like, shape (dim_a,)
        Unnormalized histogram of dimension `dim_a`
        If `a` is an empty list or array ([]),
        then `a` is set to uniform distribution.
    b : array-like, shape (dim_b,)
        One or multiple unnormalized histograms of dimension `dim_b`.
        If `b` is an empty list or array ([]),
        then `b` is set to uniform distribution.
        If many, compute all the OT costs :math:`(\mathbf{a}, \mathbf{b}_i)_i`
    M : array-like, shape (dim_a, dim_b)
        loss matrix
    reg : float
        Entropy regularization term > 0
    reg_m: float or indexable object of length 1 or 2
        Marginal relaxation term.
        If :math:`\mathrm{reg_{m}}` is a scalar or an indexable object of length 1,
        then the same :math:`\mathrm{reg_{m}}` is applied to both marginal relaxations.
        The entropic balanced OT can be recovered using :math:`\mathrm{reg_{m}}=float("inf")`.
        For semi-relaxed case, use either
        :math:`\mathrm{reg_{m}}=(float("inf"), scalar)` or
        :math:`\mathrm{reg_{m}}=(scalar, float("inf"))`.
        If :math:`\mathrm{reg_{m}}` is an array,
        it must have the same backend as input arrays `(a, b, M)`.
    method : str
        method used for the solver either 'sinkhorn', 'sinkhorn_stabilized' or
        'sinkhorn_reg_scaling', see those function for specific parameters
    reg_type : string, optional
        Regularizer term. Can take two values:

        - Negative entropy: 'entropy':
          :math:`\Omega(\gamma) = \sum_{i,j} \gamma_{i,j} \log(\gamma_{i,j}) - \sum_{i,j} \gamma_{i,j}`.
          This is equivalent (up to a constant) to :math:`\Omega(\gamma) = \text{KL}(\gamma, 1_{dim_a} 1_{dim_b}^T)`.
        - Kullback-Leibler divergence: 'kl':
          :math:`\Omega(\gamma) = \text{KL}(\gamma, \mathbf{a} \mathbf{b}^T)`.
    c : array-like, shape (dim_a, dim_b), optional (default=None)
        Reference measure for the regularization.
        If None, then use :math:`\mathbf{c} = \mathbf{a} \mathbf{b}^T`.
        If :math:`\texttt{reg_type}=`'entropy', then :math:`\mathbf{c} = 1_{dim_a} 1_{dim_b}^T`.
    warmstart: tuple of arrays, shape (dim_a, dim_b), optional
        Initialization of dual potentials. If provided, the dual potentials should be given
        (that is the logarithm of the `u`, `v` sinkhorn scaling vectors).
    tau : float
        threshold for max value in `u` or `v` for log scaling
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshold on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record `log` if `True`

    .. warning::
        Starting from version 0.9.5, the default value has been changed to `reg_type='kl'` instead of `reg_type='entropy'`. This makes the function more consistent with the literature
        and the other solvers. If you want to use the entropy regularization, please set `reg_type='entropy'` explicitly.


    Returns
    -------
    if n_hists == 1:
        - gamma : array-like, shape (dim_a, dim_b)
            Optimal transportation matrix for the given parameters
        - log : dict
            log dictionary returned only if `log` is `True`
    else:
        - ot_cost : array-like, shape (n_hists,)
            the OT cost between :math:`\mathbf{a}` and each of the histograms :math:`\mathbf{b}_i`
        - log : dict
            log dictionary returned only if `log` is `True`
    Examples
    --------

    >>> import ot
    >>> import numpy as np
    >>> a=[.5, .5]
    >>> b=[.5, .5]
    >>> M=[[0., 1.],[1., 0.]]
    >>> np.round(ot.unbalanced.sinkhorn_stabilized_unbalanced(a, b, M, 1., 1.), 7)
    array([[0.3220536, 0.1184769],
           [0.1184769, 0.3220536]])


    .. _references-sinkhorn-stabilized-unbalanced:
    References
    ----------
    .. [10] Chizat, L., Peyré, G., Schmitzer, B., & Vialard, F. X. (2016).
        Scaling algorithms for unbalanced transport problems. arXiv preprint arXiv:1607.05816.

    .. [25] Frogner C., Zhang C., Mobahi H., Araya-Polo M., Poggio T. :
        Learning with a Wasserstein Loss,  Advances in Neural Information
        Processing Systems (NIPS) 2015

    See Also
    --------
    ot.lp.emd : Unregularized OT
    ot.optim.cg : General regularized OT

    """
    a, b, M = list_to_array(a, b, M)
    nx = get_backend(M, a, b)

    dim_a, dim_b = M.shape

    if len(a) == 0:
        a = nx.ones(dim_a, type_as=M) / dim_a
    if len(b) == 0:
        b = nx.ones(dim_b, type_as=M) / dim_b

    if len(b.shape) > 1:
        n_hists = b.shape[1]
    else:
        n_hists = 0

    reg_m1, reg_m2 = get_parameter_pair(reg_m)

    if log:
        dict_log = {"err": []}

    # we assume that no distances are null except those of the diagonal of
    # distances
    if warmstart is None:
        if n_hists:
            u = nx.ones((dim_a, n_hists), type_as=M)
            v = nx.ones((dim_b, n_hists), type_as=M)
            a = a.reshape(dim_a, 1)
        else:
            u = nx.ones(dim_a, type_as=M)
            v = nx.ones(dim_b, type_as=M)
    else:
        u, v = nx.exp(warmstart[0]), nx.exp(warmstart[1])

    if reg_type == "entropy":
        warnings.warn(
            "If reg_type = entropy, then the matrix c is overwritten by the one matrix."
        )
        c = nx.ones((dim_a, dim_b), type_as=M)

    if n_hists:
        M0 = M
    else:
        c = a[:, None] * b[None, :] if c is None else c
        M0 = M - reg * nx.log(c)
    K = nx.exp(-M0 / reg)

    fi_1 = reg_m1 / (reg_m1 + reg) if reg_m1 != float("inf") else 1
    fi_2 = reg_m2 / (reg_m2 + reg) if reg_m2 != float("inf") else 1

    cpt = 0
    err = 1.0
    alpha = nx.zeros(dim_a, type_as=M)
    beta = nx.zeros(dim_b, type_as=M)
    ones_a = nx.ones(dim_a, type_as=M)
    ones_b = nx.ones(dim_b, type_as=M)

    while err > stopThr and cpt < numItermax:
        uprev = u
        vprev = v

        Kv = nx.dot(K, v)
        f_alpha = nx.exp(-alpha / (reg + reg_m1)) if reg_m1 != float("inf") else ones_a
        f_beta = nx.exp(-beta / (reg + reg_m2)) if reg_m2 != float("inf") else ones_b

        if n_hists:
            f_alpha = f_alpha[:, None]
            f_beta = f_beta[:, None]
        u = ((a / (Kv + 1e-16)) ** fi_1) * f_alpha
        Ktu = nx.dot(K.T, u)
        v = ((b / (Ktu + 1e-16)) ** fi_2) * f_beta
        absorbing = False
        if nx.any(u > tau) or nx.any(v > tau):
            absorbing = True
            if n_hists:
                alpha = alpha + reg * nx.log(nx.max(u, 1))
                beta = beta + reg * nx.log(nx.max(v, 1))
            else:
                alpha = alpha + reg * nx.log(nx.max(u))
                beta = beta + reg * nx.log(nx.max(v))
            K = nx.exp((alpha[:, None] + beta[None, :] - M0) / reg)
            v = nx.ones(v.shape, type_as=v)
        Kv = nx.dot(K, v)

        if (
            nx.any(Ktu == 0.0)
            or nx.any(nx.isnan(u))
            or nx.any(nx.isnan(v))
            or nx.any(nx.isinf(u))
            or nx.any(nx.isinf(v))
        ):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            warnings.warn("Numerical errors at iteration %s" % cpt)
            u = uprev
            v = vprev
            break
        if (cpt % 10 == 0 and not absorbing) or cpt == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            err = nx.max(nx.abs(u - uprev)) / max(
                nx.max(nx.abs(u)), nx.max(nx.abs(uprev)), 1.0
            )
            if log:
                dict_log["err"].append(err)
            if verbose:
                if cpt % 200 == 0:
                    print("{:5s}|{:12s}".format("It.", "Err") + "\n" + "-" * 19)
                print("{:5d}|{:8e}|".format(cpt, err))
        cpt = cpt + 1

    if err > stopThr:
        warnings.warn(
            "Stabilized Unbalanced Sinkhorn did not converge."
            + "Try a larger entropy `reg` or a lower mass `reg_m`."
            + "Or a larger absorption threshold `tau`."
        )
    if n_hists:
        logu = alpha[:, None] / reg + nx.log(u)
        logv = beta[:, None] / reg + nx.log(v)
    else:
        logu = alpha / reg + nx.log(u)
        logv = beta / reg + nx.log(v)
    if log:
        dict_log["logu"] = logu
        dict_log["logv"] = logv
    if n_hists:  # return only loss
        res = nx.logsumexp(
            nx.log(M + 1e-100)[:, :, None]
            + logu[:, None, :]
            + logv[None, :, :]
            - M0[:, :, None] / reg,
            axis=(0, 1),
        )
        res = nx.exp(res)
        if log:
            return res, dict_log
        else:
            return res

    else:  # return OT matrix
        plan = nx.exp(logu[:, None] + logv[None, :] - M0 / reg)
        if log:
            linear_cost = nx.sum(plan * M)
            dict_log["cost"] = linear_cost

            total_cost = linear_cost + reg * nx.kl_div(plan, c)
            if reg_m1 != float("inf"):
                total_cost = total_cost + reg_m1 * nx.kl_div(nx.sum(plan, 1), a)
            if reg_m2 != float("inf"):
                total_cost = total_cost + reg_m2 * nx.kl_div(nx.sum(plan, 0), b)
            dict_log["total_cost"] = total_cost

            return plan, dict_log
        else:
            return plan


def sinkhorn_unbalanced_translation_invariant(
    a,
    b,
    M,
    reg,
    reg_m,
    reg_type="kl",
    c=None,
    warmstart=None,
    numItermax=1000,
    stopThr=1e-6,
    verbose=False,
    log=False,
    **kwargs,
):
    r"""
    Solve the entropic regularization unbalanced optimal transport problem and
    return the OT plan

    The function solves the following optimization problem:

    .. math::
        W = \arg \min_\gamma \ \langle \gamma, \mathbf{M} \rangle_F +
        \mathrm{reg} \cdot \mathrm{KL}(\gamma, \mathbf{c}) +
        \mathrm{reg_{m1}} \cdot \mathrm{KL}(\gamma \mathbf{1}, \mathbf{a}) +
        \mathrm{reg_{m2}} \cdot \mathrm{KL}(\gamma^T \mathbf{1}, \mathbf{b})

        s.t.
             \gamma \geq 0

    where :

    - :math:`\mathbf{M}` is the (`dim_a`, `dim_b`) metric cost matrix
    - :math:`\Omega` is the entropic regularization term,KL divergence
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are source and target unbalanced distributions
    - KL is the Kullback-Leibler divergence

    The algorithm used for solving the problem is the translation invariant Sinkhorn algorithm as proposed in :ref:`[73] <references-sinkhorn-unbalanced-translation-invariant>`

    Parameters
    ----------
    a : array-like, shape (dim_a,)
        Unnormalized histogram of dimension `dim_a`
    b : array-like, shape (dim_b,) or (dim_b, n_hists)
        One or multiple unnormalized histograms of dimension `dim_b`
        If many, compute all the OT distances (a, b_i)
    M : array-like, shape (dim_a, dim_b)
        loss matrix
    reg : float
        Entropy regularization term > 0
    reg_m: float or indexable object of length 1 or 2
        Marginal relaxation term.
        If reg_m is a scalar or an indexable object of length 1,
        then the same reg_m is applied to both marginal relaxations.
        The entropic balanced OT can be recovered using `reg_m=float("inf")`.
        For semi-relaxed case, use either
        `reg_m=(float("inf"), scalar)` or `reg_m=(scalar, float("inf"))`.
        If reg_m is an array, it must have the same backend as input arrays (a, b, M).
    reg_type : string, optional
        Regularizer term. Can take two values:
        'entropy' (negative entropy)
        :math:`\Omega(\gamma) = \sum_{i,j} \gamma_{i,j} \log(\gamma_{i,j}) - \sum_{i,j} \gamma_{i,j}`, or
        'kl' (Kullback-Leibler)
        :math:`\Omega(\gamma) = \text{KL}(\gamma, \mathbf{a} \mathbf{b}^T)`.
    c : array-like, shape (dim_a, dim_b), optional (default=None)
        Reference measure for the regularization.
        If None, then use :math:`\mathbf{c} = \mathbf{a} \mathbf{b}^T`.
        If :math:`\texttt{reg_type}=`'entropy', then :math:`\mathbf{c} = 1_{dim_a} 1_{dim_b}^T`.
    warmstart: tuple of arrays, shape (dim_a, dim_b), optional
        Initialization of dual potentials. If provided, the dual potentials should be given
        (that is the logarithm of the u,v sinkhorn scaling vectors).
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshold on error (> 0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True


    Returns
    -------
    if n_hists == 1:
        - gamma : array-like, shape (dim_a, dim_b)
            Optimal transportation matrix for the given parameters
        - log : dict
            log dictionary returned only if `log` is `True`
    else:
        - ot_distance : array-like, shape (n_hists,)
            the OT distance between :math:`\mathbf{a}` and each of the histograms :math:`\mathbf{b}_i`
        - log : dict
            log dictionary returned only if `log` is `True`

    Examples
    --------

    >>> import ot
    >>> a=[.5, .5]
    >>> b=[.5, .5]
    >>> M=[[0., 1.],[1., 0.]]
    >>> ot.unbalanced.sinkhorn_unbalanced_translation_invariant(a, b, M, 1., 1.)
    array([[0.32205357, 0.11847689],
           [0.11847689, 0.32205357]])


    .. _references-sinkhorn-unbalanced-translation-invariant:
    References
    ----------
    .. [73] Séjourné, T., Vialard, F. X., & Peyré, G. (2022).
        Faster unbalanced optimal transport: Translation invariant sinkhorn and 1-d frank-wolfe.
        In International Conference on Artificial Intelligence and Statistics (pp. 4995-5021). PMLR.
    """

    M, a, b = list_to_array(M, a, b)
    nx = get_backend(M, a, b)

    dim_a, dim_b = M.shape

    if len(a) == 0:
        a = nx.ones(dim_a, type_as=M) / dim_a
    if len(b) == 0:
        b = nx.ones(dim_b, type_as=M) / dim_b

    if len(b.shape) > 1:
        n_hists = b.shape[1]
    else:
        n_hists = 0

    reg_m1, reg_m2 = get_parameter_pair(reg_m)

    if log:
        dict_log = {"err": []}

    # we assume that no distances are null except those of the diagonal of
    # distances
    if warmstart is None:
        if n_hists:
            u = nx.ones((dim_a, 1), type_as=M)
            v = nx.ones((dim_b, n_hists), type_as=M)
            a = a.reshape(dim_a, 1)
        else:
            u = nx.ones(dim_a, type_as=M)
            v = nx.ones(dim_b, type_as=M)
    else:
        u, v = nx.exp(warmstart[0]), nx.exp(warmstart[1])

    u_, v_ = u, v

    if reg_type == "entropy":
        warnings.warn(
            "If reg_type = entropy, then the matrix c is overwritten by the one matrix."
        )
        c = nx.ones((dim_a, dim_b), type_as=M)

    if n_hists:
        M0 = M
    else:
        c = a[:, None] * b[None, :] if c is None else c
        M0 = M - reg * nx.log(c)
    K = nx.exp(-M0 / reg)

    fi_1 = reg_m1 / (reg_m1 + reg) if reg_m1 != float("inf") else 1
    fi_2 = reg_m2 / (reg_m2 + reg) if reg_m2 != float("inf") else 1

    k1 = (
        reg * reg_m1 / ((reg + reg_m1) * (reg_m1 + reg_m2))
        if reg_m1 != float("inf")
        else 0
    )
    k2 = (
        reg * reg_m2 / ((reg + reg_m2) * (reg_m1 + reg_m2))
        if reg_m2 != float("inf")
        else 0
    )

    k_rho1 = k1 * reg_m1 / reg if reg_m1 != float("inf") else 0
    k_rho2 = k2 * reg_m2 / reg if reg_m2 != float("inf") else 0

    if reg_m1 == float("inf") and reg_m2 == float("inf"):
        xi1, xi2 = 0, 0
        fi_12 = 1
    elif reg_m1 == float("inf"):
        xi1 = 0
        xi2 = reg / reg_m2
        fi_12 = reg_m2
    elif reg_m2 == float("inf"):
        xi1 = reg / reg_m1
        xi2 = 0
        fi_12 = reg_m1
    else:
        xi1 = (reg_m2 * reg) / (reg_m1 * (reg + reg_m1 + reg_m2))
        xi2 = (reg_m1 * reg) / (reg_m2 * (reg + reg_m1 + reg_m2))
        fi_12 = reg_m1 * reg_m2 / (reg_m1 + reg_m2)

    xi_rho1 = xi1 * reg_m1 / reg if reg_m1 != float("inf") else 0
    xi_rho2 = xi2 * reg_m2 / reg if reg_m2 != float("inf") else 0

    reg_ratio1 = reg / reg_m1 if reg_m1 != float("inf") else 0
    reg_ratio2 = reg / reg_m2 if reg_m2 != float("inf") else 0

    err = 1.0

    for i in range(numItermax):
        uprev = u
        vprev = v

        Kv = nx.dot(K, v_)
        u_hat = (a / Kv) ** fi_1 * nx.sum(b * v_**reg_ratio2) ** k_rho2
        u_ = u_hat * nx.sum(a * u_hat ** (-reg_ratio1)) ** (-xi_rho1)

        Ktu = nx.dot(K.T, u_)
        v_hat = (b / Ktu) ** fi_2 * nx.sum(a * u_ ** (-reg_ratio1)) ** k_rho1
        v_ = v_hat * nx.sum(b * v_hat ** (-reg_ratio2)) ** (-xi_rho2)

        if (
            nx.any(Ktu == 0.0)
            or nx.any(nx.isnan(u_))
            or nx.any(nx.isnan(v_))
            or nx.any(nx.isinf(u_))
            or nx.any(nx.isinf(v_))
        ):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            warnings.warn("Numerical errors at iteration %s" % i)
            u = uprev
            v = vprev
            break

        t = (nx.sum(a * u_ ** (-reg_ratio1)) / nx.sum(b * v_ ** (-reg_ratio2))) ** (
            fi_12 / reg
        )

        u = u_ * t
        v = v_ / t

        err_u = nx.max(nx.abs(u - uprev)) / max(
            nx.max(nx.abs(u)), nx.max(nx.abs(uprev)), 1.0
        )
        err_v = nx.max(nx.abs(v - vprev)) / max(
            nx.max(nx.abs(v)), nx.max(nx.abs(vprev)), 1.0
        )
        err = 0.5 * (err_u + err_v)
        if log:
            dict_log["err"].append(err)
            if verbose:
                if i % 50 == 0:
                    print("{:5s}|{:12s}".format("It.", "Err") + "\n" + "-" * 19)
                print("{:5d}|{:8e}|".format(i, err))

        if err < stopThr:
            break

    if log:
        dict_log["logu"] = nx.log(u + 1e-300)
        dict_log["logv"] = nx.log(v + 1e-300)

    if n_hists:  # return only loss
        res = nx.einsum("ik,ij,jk,ij->k", u, K, v, M)
        if log:
            return res, dict_log
        else:
            return res

    else:  # return OT matrix
        plan = u[:, None] * K * v[None, :]

        if log:
            linear_cost = nx.sum(plan * M)
            dict_log["cost"] = linear_cost

            total_cost = linear_cost + reg * nx.kl_div(plan, c)
            if reg_m1 != float("inf"):
                total_cost = total_cost + reg_m1 * nx.kl_div(nx.sum(plan, 1), a)
            if reg_m2 != float("inf"):
                total_cost = total_cost + reg_m2 * nx.kl_div(nx.sum(plan, 0), b)
            dict_log["total_cost"] = total_cost

            return plan, dict_log
        else:
            return plan


def barycenter_unbalanced_stabilized(
    A,
    M,
    reg,
    reg_m,
    weights=None,
    tau=1e3,
    numItermax=1000,
    stopThr=1e-6,
    verbose=False,
    log=False,
):
    r"""Compute the entropic unbalanced wasserstein barycenter of :math:`\mathbf{A}` with stabilization.

     The function solves the following optimization problem:

    .. math::
       \mathbf{a} = \mathop{\arg \min}_\mathbf{a} \quad \sum_i W_{u_{reg}}(\mathbf{a},\mathbf{a}_i)

    where :

    - :math:`W_{u_{reg}}(\cdot,\cdot)` is the unbalanced entropic regularized Wasserstein distance (see :py:func:`ot.unbalanced.sinkhorn_unbalanced`)
    - :math:`\mathbf{a}_i` are training distributions in the columns of matrix :math:`\mathbf{A}`
    - reg and :math:`\mathbf{M}` are respectively the regularization term and the cost matrix for OT
    - reg_mis the marginal relaxation hyperparameter

    The algorithm used for solving the problem is the generalized
    Sinkhorn-Knopp matrix scaling algorithm as proposed in :ref:`[10] <references-barycenter-unbalanced-stabilized>`

    Parameters
    ----------
    A : array-like, shape (dim, n_hists)
        `n_hists` training distributions :math:`\mathbf{a}_i` of dimension `dim`
    M : array-like, shape (dim, dim)
        ground metric matrix for OT.
    reg : float
        Entropy regularization term > 0
    reg_m : float
        Marginal relaxation term > 0
    tau : float
        Stabilization threshold for log domain absorption.
    weights : array-like, shape (n_hists,) optional
        Weight of each distribution (barycentric coordinates)
        If None, uniform weights are used.
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshold on error (> 0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record `log` if `True`


    Returns
    -------
    a : array-like, shape (dim,)
        Unbalanced Wasserstein barycenter
    log : dict
        log dictionary return only if :math:`log==True` in parameters


    .. _references-barycenter-unbalanced-stabilized:
    References
    ----------
    .. [3] Benamou, J. D., Carlier, G., Cuturi, M., Nenna, L., & Peyré,
        G. (2015). Iterative Bregman projections for regularized transportation
        problems. SIAM Journal on Scientific Computing, 37(2), A1111-A1138.
    .. [10] Chizat, L., Peyré, G., Schmitzer, B., & Vialard, F. X. (2016).
        Scaling algorithms for unbalanced transport problems. arXiv preprint
        arXiv:1607.05816.


    """
    A, M = list_to_array(A, M)
    nx = get_backend(A, M)

    dim, n_hists = A.shape
    if weights is None:
        weights = nx.ones(n_hists, type_as=A) / n_hists
    else:
        assert len(weights) == A.shape[1]

    if log:
        log = {"err": []}

    fi = reg_m / (reg_m + reg)

    u = nx.ones((dim, n_hists), type_as=A) / dim
    v = nx.ones((dim, n_hists), type_as=A) / dim

    # print(reg)
    K = nx.exp(-M / reg)

    fi = reg_m / (reg_m + reg)

    cpt = 0
    err = 1.0
    alpha = nx.zeros(dim, type_as=A)
    beta = nx.zeros(dim, type_as=A)
    q = nx.ones(dim, type_as=A) / dim
    for i in range(numItermax):
        qprev = nx.copy(q)
        Kv = nx.dot(K, v)
        f_alpha = nx.exp(-alpha / (reg + reg_m))
        f_beta = nx.exp(-beta / (reg + reg_m))
        f_alpha = f_alpha[:, None]
        f_beta = f_beta[:, None]
        u = ((A / (Kv + 1e-16)) ** fi) * f_alpha
        Ktu = nx.dot(K.T, u)
        q = (Ktu ** (1 - fi)) * f_beta
        q = nx.dot(q, weights) ** (1 / (1 - fi))
        Q = q[:, None]
        v = ((Q / (Ktu + 1e-16)) ** fi) * f_beta
        absorbing = False
        if nx.any(u > tau) or nx.any(v > tau):
            absorbing = True
            alpha = alpha + reg * nx.log(nx.max(u, 1))
            beta = beta + reg * nx.log(nx.max(v, 1))
            K = nx.exp((alpha[:, None] + beta[None, :] - M) / reg)
            v = nx.ones(v.shape, type_as=v)
        Kv = nx.dot(K, v)
        if (
            nx.any(Ktu == 0.0)
            or nx.any(nx.isnan(u))
            or nx.any(nx.isnan(v))
            or nx.any(nx.isinf(u))
            or nx.any(nx.isinf(v))
        ):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            warnings.warn("Numerical errors at iteration %s" % cpt)
            q = qprev
            break
        if (i % 10 == 0 and not absorbing) or i == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            err = nx.max(nx.abs(q - qprev)) / max(
                nx.max(nx.abs(q)), nx.max(nx.abs(qprev)), 1.0
            )
            if log:
                log["err"].append(err)
            if verbose:
                if i % 50 == 0:
                    print("{:5s}|{:12s}".format("It.", "Err") + "\n" + "-" * 19)
                print("{:5d}|{:8e}|".format(i, err))
            if err < stopThr:
                break

    if err > stopThr:
        warnings.warn(
            "Stabilized Unbalanced Sinkhorn did not converge."
            + "Try a larger entropy `reg` or a lower mass `reg_m`."
            + "Or a larger absorption threshold `tau`."
        )
    if log:
        log["niter"] = i
        log["logu"] = nx.log(u + 1e-300)
        log["logv"] = nx.log(v + 1e-300)
        return q, log
    else:
        return q


def barycenter_unbalanced_sinkhorn(
    A,
    M,
    reg,
    reg_m,
    weights=None,
    numItermax=1000,
    stopThr=1e-6,
    verbose=False,
    log=False,
):
    r"""Compute the entropic unbalanced wasserstein barycenter of :math:`\mathbf{A}`.

     The function solves the following optimization problem with :math:`\mathbf{a}`

    .. math::
       \mathbf{a} = \mathop{\arg \min}_\mathbf{a} \quad \sum_i W_{u_{reg}}(\mathbf{a},\mathbf{a}_i)

    where :

    - :math:`W_{u_{reg}}(\cdot,\cdot)` is the unbalanced entropic regularized Wasserstein distance (see :py:func:`ot.unbalanced.sinkhorn_unbalanced`)
    - :math:`\mathbf{a}_i` are training distributions in the columns of matrix :math:`\mathbf{A}`
    - reg and :math:`\mathbf{M}` are respectively the regularization term and the cost matrix for OT
    - reg_mis the marginal relaxation hyperparameter

    The algorithm used for solving the problem is the generalized
    Sinkhorn-Knopp matrix scaling algorithm as proposed in :ref:`[10] <references-barycenter-unbalanced-sinkhorn>`

    Parameters
    ----------
    A : array-like, shape (dim, n_hists)
        `n_hists` training distributions :math:`\mathbf{a}_i` of dimension `dim`
    M : array-like, shape (dim, dim)
        ground metric matrix for OT.
    reg : float
        Entropy regularization term > 0
    reg_m: float
        Marginal relaxation term > 0
    weights : array-like, shape (n_hists,) optional
        Weight of each distribution (barycentric coordinates)
        If None, uniform weights are used.
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshold on error (> 0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record `log` if `True`


    Returns
    -------
    a : array-like, shape (dim,)
        Unbalanced Wasserstein barycenter
    log : dict
        log dictionary return only if :math:`log==True` in parameters


    .. _references-barycenter-unbalanced-sinkhorn:
    References
    ----------
    .. [3] Benamou, J. D., Carlier, G., Cuturi, M., Nenna, L., & Peyré, G.
        (2015). Iterative Bregman projections for regularized transportation
        problems. SIAM Journal on Scientific Computing, 37(2), A1111-A1138.
    .. [10] Chizat, L., Peyré, G., Schmitzer, B., & Vialard, F. X. (2016).
        Scaling algorithms for unbalanced transport problems. arXiv preprin
        arXiv:1607.05816.


    """
    A, M = list_to_array(A, M)
    nx = get_backend(A, M)

    dim, n_hists = A.shape
    if weights is None:
        weights = nx.ones(n_hists, type_as=A) / n_hists
    else:
        assert len(weights) == A.shape[1]

    if log:
        log = {"err": []}

    K = nx.exp(-M / reg)

    fi = reg_m / (reg_m + reg)

    v = nx.ones((dim, n_hists), type_as=A)
    u = nx.ones((dim, 1), type_as=A)
    q = nx.ones(dim, type_as=A)
    err = 1.0

    for i in range(numItermax):
        uprev = nx.copy(u)
        vprev = nx.copy(v)
        qprev = nx.copy(q)

        Kv = nx.dot(K, v)
        u = (A / Kv) ** fi
        Ktu = nx.dot(K.T, u)
        q = nx.dot(Ktu ** (1 - fi), weights)
        q = q ** (1 / (1 - fi))
        Q = q[:, None]
        v = (Q / Ktu) ** fi

        if (
            nx.any(Ktu == 0.0)
            or nx.any(nx.isnan(u))
            or nx.any(nx.isnan(v))
            or nx.any(nx.isinf(u))
            or nx.any(nx.isinf(v))
        ):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            warnings.warn("Numerical errors at iteration %s" % i)
            u = uprev
            v = vprev
            q = qprev
            break
            # compute change in barycenter
        err = nx.max(nx.abs(q - qprev)) / max(
            nx.max(nx.abs(q)), nx.max(nx.abs(qprev)), 1.0
        )
        if log:
            log["err"].append(err)
        # if barycenter did not change + at least 10 iterations - stop
        if err < stopThr and i > 10:
            break

        if verbose:
            if i % 10 == 0:
                print("{:5s}|{:12s}".format("It.", "Err") + "\n" + "-" * 19)
            print("{:5d}|{:8e}|".format(i, err))

    if log:
        log["niter"] = i
        log["logu"] = nx.log(u + 1e-300)
        log["logv"] = nx.log(v + 1e-300)
        return q, log
    else:
        return q


def barycenter_unbalanced(
    A,
    M,
    reg,
    reg_m,
    method="sinkhorn",
    weights=None,
    numItermax=1000,
    stopThr=1e-6,
    verbose=False,
    log=False,
    **kwargs,
):
    r"""Compute the entropic unbalanced wasserstein barycenter of :math:`\mathbf{A}`.

     The function solves the following optimization problem with :math:`\mathbf{a}`

    .. math::
       \mathbf{a} = \mathop{\arg \min}_\mathbf{a} \quad \sum_i W_{u_{reg}}(\mathbf{a},\mathbf{a}_i)

    where :

    - :math:`W_{u_{reg}}(\cdot,\cdot)` is the unbalanced entropic regularized Wasserstein distance (see :py:func:`ot.unbalanced.sinkhorn_unbalanced`)
    - :math:`\mathbf{a}_i` are training distributions in the columns of matrix :math:`\mathbf{A}`
    - reg and :math:`\mathbf{M}` are respectively the regularization term and the cost matrix for OT
    - reg_mis the marginal relaxation hyperparameter

    The algorithm used for solving the problem is the generalized
    Sinkhorn-Knopp matrix scaling algorithm as proposed in :ref:`[10] <references-barycenter-unbalanced>`

    Parameters
    ----------
    A : array-like, shape (dim, n_hists)
        `n_hists` training distributions :math:`\mathbf{a}_i` of dimension `dim`
    M : array-like, shape (dim, dim)
        ground metric matrix for OT.
    reg : float
        Entropy regularization term > 0
    reg_m: float
        Marginal relaxation term > 0
    weights : array-like, shape (n_hists,) optional
        Weight of each distribution (barycentric coordinates)
        If None, uniform weights are used.
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshold on error (> 0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True


    Returns
    -------
    a : array-like, shape (dim,)
        Unbalanced Wasserstein barycenter
    log : dict
        log dictionary return only if log==True in parameters


    .. _references-barycenter-unbalanced:
    References
    ----------
    .. [3] Benamou, J. D., Carlier, G., Cuturi, M., Nenna, L., & Peyré, G.
        (2015). Iterative Bregman projections for regularized transportation
        problems. SIAM Journal on Scientific Computing, 37(2), A1111-A1138.
    .. [10] Chizat, L., Peyré, G., Schmitzer, B., & Vialard, F. X. (2016).
        Scaling algorithms for unbalanced transport problems. arXiv preprin
        arXiv:1607.05816.

    """

    if method.lower() == "sinkhorn":
        return barycenter_unbalanced_sinkhorn(
            A,
            M,
            reg,
            reg_m,
            weights=weights,
            numItermax=numItermax,
            stopThr=stopThr,
            verbose=verbose,
            log=log,
            **kwargs,
        )

    elif method.lower() == "sinkhorn_stabilized":
        return barycenter_unbalanced_stabilized(
            A,
            M,
            reg,
            reg_m,
            weights=weights,
            numItermax=numItermax,
            stopThr=stopThr,
            verbose=verbose,
            log=log,
            **kwargs,
        )
    elif method.lower() in ["sinkhorn_reg_scaling", "sinkhorn_translation_invariant"]:
        warnings.warn("Method not implemented yet. Using classic Sinkhorn Knopp")
        return barycenter_unbalanced(
            A,
            M,
            reg,
            reg_m,
            weights=weights,
            numItermax=numItermax,
            stopThr=stopThr,
            verbose=verbose,
            log=log,
            **kwargs,
        )
    else:
        raise ValueError("Unknown method '%s'." % method)
