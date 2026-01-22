# -*- coding: utf-8 -*-
"""
Cython linker with C solver
"""

# Author: Remi Flamary <remi.flamary@unice.fr>
#
# License: MIT License

import numpy as np
cimport numpy as np

from ..utils import dist

cimport cython
cimport libc.math as math
from libc.stdint cimport uint64_t, int64_t

import warnings


cdef extern from "EMD.h":
    int EMD_wrap(int n1,int n2, double *X, double *Y,double *D, double *G, double* alpha, double* beta, double *cost, uint64_t maxIter) nogil
    int EMD_wrap_omp(int n1,int n2, double *X, double *Y,double *D, double *G, double* alpha, double* beta, double *cost, uint64_t maxIter, int numThreads) nogil
    int EMD_wrap_sparse(int n1, int n2, double *X, double *Y, uint64_t n_edges, uint64_t *edge_sources, uint64_t *edge_targets, double *edge_costs, uint64_t *flow_sources_out, uint64_t *flow_targets_out, double *flow_values_out, uint64_t *n_flows_out, double *alpha, double *beta, double *cost, uint64_t maxIter) nogil
    int EMD_wrap_lazy(int n1, int n2, double *X, double *Y, double *coords_a, double *coords_b, int dim, int metric, double *G, double* alpha, double* beta, double *cost, uint64_t maxIter) nogil
    cdef enum ProblemType: INFEASIBLE, OPTIMAL, UNBOUNDED, MAX_ITER_REACHED


def check_result(result_code):
    if result_code == OPTIMAL:
        return None

    if result_code == INFEASIBLE:
        message = "Problem infeasible. Check that a and b are in the simplex"
    elif result_code == UNBOUNDED:
        message = "Problem unbounded"
    elif result_code == MAX_ITER_REACHED:
        message = "numItermax reached before optimality. Try to increase numItermax."
    warnings.warn(message)
    return message
 
@cython.boundscheck(False)
@cython.wraparound(False)
def emd_c(np.ndarray[double, ndim=1, mode="c"] a, np.ndarray[double, ndim=1, mode="c"]  b, np.ndarray[double, ndim=2, mode="c"]  M, uint64_t max_iter, int numThreads):
    """
        Solves the Earth Movers distance problem and returns the optimal transport matrix

        gamm=emd(a,b,M)

    .. math::
        \gamma = arg\min_\gamma <\gamma,M>_F

        s.t. \gamma 1 = a

             \gamma^T 1= b

             \gamma\geq 0
    where :

    - M is the metric cost matrix
    - a and b are the sample weights

    .. warning::
        Note that the M matrix needs to be a C-order :py.cls:`numpy.array`

    .. warning::
        The C++ solver discards all samples in the distributions with 
        zeros weights. This means that while the primal variable (transport 
        matrix) is exact, the solver only returns feasible dual potentials
        on the samples with weights different from zero. 

    Parameters
    ----------
    a : (ns,) numpy.ndarray, float64
        source histogram
    b : (nt,) numpy.ndarray, float64
        target histogram
    M : (ns,nt) numpy.ndarray, float64
        loss matrix
    max_iter : uint64_t
        The maximum number of iterations before stopping the optimization
        algorithm if it has not converged.

    Returns
    -------
    gamma: (ns x nt) numpy.ndarray
        Optimal transportation matrix for the given parameters

    """
    cdef int n1= M.shape[0]
    cdef int n2= M.shape[1]
    cdef int nmax=n1+n2-1
    cdef int result_code = 0
    cdef int nG=0

    cdef double cost=0
    cdef np.ndarray[double, ndim=1, mode="c"] alpha=np.zeros(n1)
    cdef np.ndarray[double, ndim=1, mode="c"] beta=np.zeros(n2)

    cdef np.ndarray[double, ndim=2, mode="c"] G=np.zeros([0, 0])

    cdef np.ndarray[double, ndim=1, mode="c"] Gv=np.zeros(0)

    if not len(a):
        a=np.ones((n1,))/n1

    if not len(b):
        b=np.ones((n2,))/n2

    # init OT matrix
    G=np.zeros([n1, n2])

    # calling the function
    with nogil:
        if numThreads == 1:
            result_code = EMD_wrap(n1, n2, <double*> a.data, <double*> b.data, <double*> M.data, <double*> G.data, <double*> alpha.data, <double*> beta.data, <double*> &cost, max_iter)
        else:
            result_code = EMD_wrap_omp(n1, n2, <double*> a.data, <double*> b.data, <double*> M.data, <double*> G.data, <double*> alpha.data, <double*> beta.data, <double*> &cost, max_iter, numThreads)
    return G, cost, alpha, beta, result_code


@cython.boundscheck(False)
@cython.wraparound(False)
def emd_1d_sorted(np.ndarray[double, ndim=1, mode="c"] u_weights,
                  np.ndarray[double, ndim=1, mode="c"] v_weights,
                  np.ndarray[double, ndim=1, mode="c"] u,
                  np.ndarray[double, ndim=1, mode="c"] v,
                  str metric='sqeuclidean',
                  double p=1.):
    r"""
    Solves the Earth Movers distance problem between sorted 1d measures and
    returns the OT matrix and the associated cost

    Parameters
    ----------
    u_weights : (ns,) ndarray, float64
        Source histogram
    v_weights : (nt,) ndarray, float64
        Target histogram
    u : (ns,) ndarray, float64
        Source dirac locations (on the real line)
    v : (nt,) ndarray, float64
        Target dirac locations (on the real line)
    metric: str, optional (default='sqeuclidean')
        Metric to be used. Only works with either of the strings
        `'sqeuclidean'`, `'minkowski'`, `'cityblock'`,  or `'euclidean'`.
    p: float, optional (default=1.0)
         The p-norm to apply for if metric='minkowski'

    Returns
    -------
    gamma: (n, ) ndarray, float64
        Values in the Optimal transportation matrix
    indices: (n, 2) ndarray, int64
        Indices of the values stored in gamma for the Optimal transportation
        matrix
    cost
        cost associated to the optimal transportation
    """
    cdef double cost = 0.
    cdef Py_ssize_t n = u_weights.shape[0]
    cdef Py_ssize_t m = v_weights.shape[0]

    cdef Py_ssize_t i = 0
    cdef double w_i = u_weights[0]
    cdef Py_ssize_t j = 0
    cdef double w_j = v_weights[0]

    cdef double m_ij = 0.

    cdef np.ndarray[double, ndim=1, mode="c"] G = np.zeros((n + m - 1, ),
                                                           dtype=np.float64)
    cdef np.ndarray[long long, ndim=2, mode="c"] indices = np.zeros((n + m - 1, 2),
                                                              dtype=np.int64)
    cdef Py_ssize_t cur_idx = 0
    while True:
        if metric == 'sqeuclidean':
            m_ij = (u[i] - v[j]) * (u[i] - v[j])
        elif metric == 'cityblock' or metric == 'euclidean':
            m_ij = math.fabs(u[i] - v[j])
        elif metric == 'minkowski':
            m_ij = math.pow(math.fabs(u[i] - v[j]), p)
        else:
            raise ValueError("Solver for EMD in 1d only supports metrics " +
                             "from the following list: " +
                             "`['sqeuclidean', 'minkowski', 'cityblock', 'euclidean']`")
        if w_i < w_j or j == m - 1:
            cost += m_ij * w_i
            G[cur_idx] = w_i
            indices[cur_idx, 0] = i
            indices[cur_idx, 1] = j
            i += 1
            if i == n:
                break
            w_j -= w_i
            w_i = u_weights[i]
        else:
            cost += m_ij * w_j
            G[cur_idx] = w_j
            indices[cur_idx, 0] = i
            indices[cur_idx, 1] = j
            j += 1
            if j == m:
                break
            w_i -= w_j
            w_j = v_weights[j]
        cur_idx += 1
    cur_idx += 1
    return G[:cur_idx], indices[:cur_idx], cost

@cython.boundscheck(False)
@cython.wraparound(False)
def emd_c_sparse(np.ndarray[double, ndim=1, mode="c"] a,
                np.ndarray[double, ndim=1, mode="c"] b,
                np.ndarray[uint64_t, ndim=1, mode="c"] edge_sources,
                np.ndarray[uint64_t, ndim=1, mode="c"] edge_targets,
                np.ndarray[double, ndim=1, mode="c"] edge_costs,
                uint64_t max_iter):
    """
    Sparse EMD solver using cost matrix in COO (Coordinate) sparse format.
    
    The cost matrix is passed as three parallel arrays representing non-zero
    entries in COO format: (edge_sources[i], edge_targets[i]) -> edge_costs[i].
    Only edges explicitly provided will be considered by the solver.

    Parameters
    ----------
    a : (n1,) array, float64
        Source histogram
    b : (n2,) array, float64
        Target histogram
    edge_sources : (k,) array, uint64
        Source indices for each edge (row indices in COO format)
    edge_targets : (k,) array, uint64
        Target indices for each edge (column indices in COO format)
    edge_costs : (k,) array, float64
        Cost for each edge (non-zero values in COO format)
    max_iter : uint64_t
        Maximum number of iterations

    Returns
    -------
    flow_sources : (n_flows,) array, uint64
        Source indices of non-zero flows
    flow_targets : (n_flows,) array, uint64
        Target indices of non-zero flows
    flow_values : (n_flows,) array, float64
        Flow values
    cost : float
        Total cost
    alpha : (n1,) array
        Dual variables for sources
    beta : (n2,) array
        Dual variables for targets
    result_code : int
        Result status
    """
    cdef int n1 = a.shape[0]
    cdef int n2 = b.shape[0]
    cdef uint64_t n_edges = edge_sources.shape[0]
    cdef uint64_t n_flows_out = 0
    cdef int result_code = 0
    cdef double cost = 0

    # Allocate output arrays (max size = n_edges)
    cdef np.ndarray[uint64_t, ndim=1, mode="c"] flow_sources = np.zeros(n_edges, dtype=np.uint64)
    cdef np.ndarray[uint64_t, ndim=1, mode="c"] flow_targets = np.zeros(n_edges, dtype=np.uint64)
    cdef np.ndarray[double, ndim=1, mode="c"] flow_values = np.zeros(n_edges, dtype=np.float64)
    cdef np.ndarray[double, ndim=1, mode="c"] alpha = np.zeros(n1)
    cdef np.ndarray[double, ndim=1, mode="c"] beta = np.zeros(n2)

    with nogil:
        result_code = EMD_wrap_sparse(
            n1, n2,
            <double*> a.data, <double*> b.data,
            n_edges,
            <uint64_t*> edge_sources.data, <uint64_t*> edge_targets.data, <double*> edge_costs.data,
            <uint64_t*> flow_sources.data, <uint64_t*> flow_targets.data, <double*> flow_values.data,
            &n_flows_out,
            <double*> alpha.data, <double*> beta.data, &cost, max_iter
        )

    # Trim to actual number of flows
    flow_sources = flow_sources[:n_flows_out]
    flow_targets = flow_targets[:n_flows_out]
    flow_values = flow_values[:n_flows_out]

    return flow_sources, flow_targets, flow_values, cost, alpha, beta, result_code


@cython.boundscheck(False)
@cython.wraparound(False)
def emd_c_lazy(np.ndarray[double, ndim=1, mode="c"] a, np.ndarray[double, ndim=1, mode="c"] b, np.ndarray[double, ndim=2, mode="c"] coords_a, np.ndarray[double, ndim=2, mode="c"] coords_b, str metric='sqeuclidean', uint64_t max_iter=100000):
    """Solves the Earth Movers distance problem with lazy cost computation from coordinates."""
    cdef int n1 = coords_a.shape[0]
    cdef int n2 = coords_b.shape[0]
    cdef int dim = coords_a.shape[1]
    cdef int result_code = 0
    cdef double cost = 0
    cdef int metric_code
    
    # Validate dimension consistency
    if coords_b.shape[1] != dim:
        raise ValueError(f"Coordinate dimension mismatch: coords_a has {dim} dimensions but coords_b has {coords_b.shape[1]}")
    
    metric_map = {
        'sqeuclidean': 0,
        'euclidean': 1,
        'cityblock': 2
    }
    
    try:
        metric_code = metric_map[metric]
    except KeyError:
        raise ValueError(f"Unknown metric: '{metric}'. Supported metrics are: {list(metric_map.keys())}")
        
    cdef np.ndarray[double, ndim=1, mode="c"] alpha = np.zeros(n1)
    cdef np.ndarray[double, ndim=1, mode="c"] beta = np.zeros(n2)
    cdef np.ndarray[double, ndim=2, mode="c"] G = np.zeros([n1, n2])
    if not len(a):
        a = np.ones((n1,)) / n1
    if not len(b):
        b = np.ones((n2,)) / n2
    with nogil:
        result_code = EMD_wrap_lazy(n1, n2, <double*> a.data, <double*> b.data, <double*> coords_a.data, <double*> coords_b.data, dim, metric_code, <double*> G.data, <double*> alpha.data, <double*> beta.data, <double*> &cost, max_iter)
    return G, cost, alpha, beta, result_code
