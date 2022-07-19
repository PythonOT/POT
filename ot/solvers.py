# -*- coding: utf-8 -*-
"""
General OT solvers with unified API
"""

# Author: Remi Flamary <remi.flamary@polytechnique.edu>
#
# License: MIT License

from .utils import OTResult
from .lp import emd2
from .backend import get_backend
from .unbalanced import mm_unbalanced
from .bregman import sinkhorn_log


def solve(M, a=None, b=None, reg=0, reg_type="KL", unbalanced=None,
          unbalanced_type='KL', n_threads=1, max_iter=None, plan_init=None,
          potentials_init=None,
          tol=None, verbose=False):

    # detect backend
    arr = [M]
    if a is not None:
        arr.append(a)
    if b is not None:
        arr.append(b)
    nx = get_backend(*arr)

    # create uniform weights if not given
    if a is None:
        a = nx.ones(M.shape[0], type_as=M) / M.shape[0]
    if b is None:
        b = nx.ones(M.shape[1], type_as=M) / M.shape[1]

    # default values for solution
    potentials = None
    value = None
    value_linear = None
    plan = None
    status = None

    if reg == 0:  # exact OT

        if unbalanced is None:  # Exact balanced OT

            # default values for EMD solver
            if max_iter is None:
                max_iter = 1000000

            value_linear, log = emd2(a, b, M, log=True, return_matrix=True, numThreads=max_iter)

            value = value_linear
            potentials = (log['u'], log['v'])
            plan = log['G']
            status = log["warning"] if log["warning"] is not None else 'Converged'

        elif unbalanced_type.lower() in ['kl', 'l2']:  # unbalanced exact OT

            # default values for exact unbalanced OT
            if max_iter is None:
                max_iter = 1000
            if tol is None:
                tol = 1e-12

            plan, log = mm_unbalanced(a, b, M, reg_m=unbalanced,
                                      div=unbalanced_type.lower(), numItermax=max_iter,
                                      stopThr=tol, log=True,
                                      verbose=verbose, G0=plan_init)

            value = log['cost']

        # TODO partial OT (as unbalanced type but not backend compatible yet)

        else:
            raise(NotImplementedError('Unknown unbalanced_type parameter "{}"'.format(unbalanced_type)))

    else:  # regularized OT

        if unbalanced is None:  # Balanced regularized OT

            if reg_type.lower() in ['entropy', 'kl']:

                # default values for sinkhorn
                if max_iter is None:
                    max_iter = 1000
                if tol is None:
                    tol = 1e-9

                plan, log = sinkhorn_log(a, b, M, reg=reg, numItermax=max_iter,
                                         stopThr=tol, log=True,
                                         verbose=verbose)

                value_linear = nx.sum(M * plan)

                if reg_type.lower() == 'entropy':
                    value = value_linear + reg * nx.sum(plan * nx.log(plan + 1e-16))
                else:
                    value = value_linear + reg * nx.sum(plan * nx.log(plan / (a[:, None] * b[None, :]) + 1e-16))

                potentials = (log['log_u'], log['log_v'])

            # TODO L2 regularization (smooth OT not backend compatible yet)

            else:
                raise(NotImplementedError('Not implemented reg_type "{}" and '.format(reg_type)))

    res = OTResult(potentials=potentials, value=value,
                   value_linear=value_linear, plan=plan, status=status, backend=nx)

    return res
