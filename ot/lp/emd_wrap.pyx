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
    int EMD_wrap(int n1, int n2, double *X, double *Y, double *D, double *G, 
                 double* alpha, double* beta, double *cost, uint64_t maxIter,
                 int resume_mode, int return_checkpoint,
                 double* flow_state, double* pi_state, signed char* state_state,
                 int* parent_state, int64_t* pred_state,
                 int* thread_state, int* rev_thread_state,
                 int* succ_num_state, int* last_succ_state,
                 signed char* forward_state,
                 int64_t* search_arc_num_out, int64_t* all_arc_num_out) nogil
    int EMD_wrap_omp(int n1,int n2, double *X, double *Y,double *D, double *G, double* alpha, double* beta, double *cost, uint64_t maxIter, int numThreads) nogil
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
def emd_c(np.ndarray[double, ndim=1, mode="c"] a, 
          np.ndarray[double, ndim=1, mode="c"] b, 
          np.ndarray[double, ndim=2, mode="c"] M, 
          uint64_t max_iter, 
          int numThreads,
          checkpoint_in=None,
          int return_checkpoint=0):
    """
        Solves the Earth Movers distance problem and returns the optimal transport matrix
        with optional checkpoint support for pause/resume.

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
    numThreads : int
        Number of threads for parallel computation (1 = no OpenMP)
    checkpoint_in : dict or None
        Checkpoint data to resume from. Should contain flow, pi, state, parent,
        pred, thread, rev_thread, succ_num, last_succ, forward arrays.
    return_checkpoint : int
        If 1, returns checkpoint data; if 0, returns None for checkpoint.

    Returns
    -------
    gamma: (ns x nt) numpy.ndarray
        Optimal transportation matrix for the given parameters
    cost : float
        Optimal transport cost
    alpha : (ns,) numpy.ndarray
        Source dual potentials
    beta : (nt,) numpy.ndarray
        Target dual potentials
    result_code : int
        Result code (OPTIMAL, INFEASIBLE, UNBOUNDED, MAX_ITER_REACHED)
    checkpoint_out : dict or None
        Checkpoint data if return_checkpoint=1, None otherwise

    """
    cdef int n1 = M.shape[0]
    cdef int n2 = M.shape[1]
    cdef int all_nodes = n1 + n2 + 1
    cdef int64_t max_arcs = n1 * n2 + 2 * (n1 + n2)
    cdef int result_code = 0
    cdef double cost = 0
    cdef int64_t search_arc_num = 0
    cdef int64_t all_arc_num = 0
    
    cdef np.ndarray[double, ndim=1, mode="c"] alpha = np.zeros(n1)
    cdef np.ndarray[double, ndim=1, mode="c"] beta = np.zeros(n2)
    cdef np.ndarray[double, ndim=2, mode="c"] G = np.zeros([n1, n2])

    # Checkpoint arrays (for both input and output)
    cdef np.ndarray[double, ndim=1, mode="c"] flow_state
    cdef np.ndarray[double, ndim=1, mode="c"] pi_state
    cdef np.ndarray[signed char, ndim=1, mode="c"] state_state
    cdef np.ndarray[int, ndim=1, mode="c"] parent_state
    cdef np.ndarray[int64_t, ndim=1, mode="c"] pred_state
    cdef np.ndarray[int, ndim=1, mode="c"] thread_state
    cdef np.ndarray[int, ndim=1, mode="c"] rev_thread_state
    cdef np.ndarray[int, ndim=1, mode="c"] succ_num_state
    cdef np.ndarray[int, ndim=1, mode="c"] last_succ_state
    cdef np.ndarray[signed char, ndim=1, mode="c"] forward_state
    
    cdef int resume_mode = 0

    if not len(a):
        a = np.ones((n1,)) / n1

    if not len(b):
        b = np.ones((n2,)) / n2
    
    # Prepare checkpoint arrays
    if checkpoint_in is not None:
        resume_mode = 1
        flow_state = np.asarray(checkpoint_in['flow'], dtype=np.float64, order='C')
        pi_state = np.asarray(checkpoint_in['pi'], dtype=np.float64, order='C')
        state_state = np.asarray(checkpoint_in['state'], dtype=np.int8, order='C')
        parent_state = np.asarray(checkpoint_in['parent'], dtype=np.int32, order='C')
        pred_state = np.asarray(checkpoint_in['pred'], dtype=np.int64, order='C')
        thread_state = np.asarray(checkpoint_in['thread'], dtype=np.int32, order='C')
        rev_thread_state = np.asarray(checkpoint_in['rev_thread'], dtype=np.int32, order='C')
        
        # Sanity check: array sizes must match expected sizes
        if flow_state.shape[0] != max_arcs or pi_state.shape[0] != all_nodes:
            raise ValueError(
                f"Checkpoint size mismatch: expected flow={max_arcs}, pi={all_nodes}, "
                f"got flow={flow_state.shape[0]}, pi={pi_state.shape[0]}"
            )
        succ_num_state = np.asarray(checkpoint_in['succ_num'], dtype=np.int32, order='C')
        last_succ_state = np.asarray(checkpoint_in['last_succ'], dtype=np.int32, order='C')
        forward_state = np.asarray(checkpoint_in['forward'], dtype=np.int8, order='C')
        
        # Extract the arc counts
        search_arc_num = checkpoint_in['search_arc_num']
        all_arc_num = checkpoint_in['all_arc_num']
    else:
        # Allocate empty arrays (will be filled if return_checkpoint=1)
        flow_state = np.zeros(max_arcs, dtype=np.float64)
        pi_state = np.zeros(all_nodes, dtype=np.float64)
        state_state = np.zeros(max_arcs, dtype=np.int8)
        parent_state = np.zeros(all_nodes, dtype=np.int32)
        pred_state = np.zeros(all_nodes, dtype=np.int64)
        thread_state = np.zeros(all_nodes, dtype=np.int32)
        rev_thread_state = np.zeros(all_nodes, dtype=np.int32)
        succ_num_state = np.zeros(all_nodes, dtype=np.int32)
        last_succ_state = np.zeros(all_nodes, dtype=np.int32)
        forward_state = np.zeros(all_nodes, dtype=np.int8)
    
    # Call C++ function with checkpoint support
    with nogil:
        if numThreads == 1:
            result_code = EMD_wrap(
                n1, n2, 
                <double*> a.data, <double*> b.data, <double*> M.data,
                <double*> G.data, <double*> alpha.data, <double*> beta.data,
                <double*> &cost, max_iter,
                resume_mode, return_checkpoint,
                <double*> flow_state.data,
                <double*> pi_state.data,
                <signed char*> state_state.data,
                <int*> parent_state.data,
                <int64_t*> pred_state.data,
                <int*> thread_state.data,
                <int*> rev_thread_state.data,
                <int*> succ_num_state.data,
                <int*> last_succ_state.data,
                <signed char*> forward_state.data,
                &search_arc_num,
                &all_arc_num
            )
        else:
            # For now, OpenMP version falls back to regular (not implemented yet)
            result_code = EMD_wrap_omp(n1, n2, <double*> a.data, <double*> b.data, <double*> M.data, 
                                      <double*> G.data, <double*> alpha.data, <double*> beta.data, 
                                      <double*> &cost, max_iter, numThreads)
    
    # Build checkpoint output dict if requested
    checkpoint_out = None
    if return_checkpoint:
        checkpoint_out = {
            'flow': flow_state,
            'pi': pi_state,
            'state': state_state,
            'parent': parent_state,
            'pred': pred_state,
            'thread': thread_state,
            'rev_thread': rev_thread_state,
            'succ_num': succ_num_state,
            'last_succ': last_succ_state,
            'forward': forward_state,
            'search_arc_num': search_arc_num,
            'all_arc_num': all_arc_num,
        }
    
    return G, cost, alpha, beta, result_code, checkpoint_out


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
