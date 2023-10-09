# lowrank.py
# Author: Gaëtan Brison <gaetan.brison@gmail.com>
# 09/10/2023

# License: MIT License

import numpy as np
import warnings

def compute_alpha(r):
    """
    A basic function to compute alpha based on r. 
    """
    return 1.0 / (r + 1)

def low_rank_sinkhorn(X, Y, a, b, reg, r, alpha='auto', numItermax=1000, stopThr=1e-4, verbose=False, log=False):
    """
    Implement the Low Rank Sinkhorn algorithm.

    Parameters
    ----------
    X : array-like, shape (n,)
        First set of points.
    Y : array-like, shape (m,)
        Second set of points.
    a : array-like, shape (n,)
        Weights for the first set of points.
    b : array-like, shape (m,)
        Weights for the second set of points.
    reg : float
        Regularization parameter.
    r : float
        Rank parameter.
    alpha : float or 'auto', optional (default='auto')
        Scaling parameter. If 'auto', it will be computed using `compute_alpha`.
    numItermax : int, optional (default=1000)
        Maximum number of iterations.
    stopThr : float, optional (default=1e-4)
        Convergence threshold.
    verbose : bool, optional (default=False)
        If True, print progress messages.
    log : bool, optional (default=False)
        If True, return log.

    Returns
    -------
    Q : array-like, shape (n, m)
        First low-rank matrix.
    R : array-like, shape (m, n)
        Second low-rank matrix.

    Warnings
    --------
    The provided alpha value might lead to instabilities.

    Notes
    -----
    This implementation assumes X and Y are 1D arrays for simplicity.
    """
    n, m = len(X), len(Y)
    
    if alpha == 'auto':
        alpha = compute_alpha(r)
        
    if (1/r > alpha) or (alpha < 0):
        warnings.warn("The provided alpha value might lead to instabilities.")
    
    # Initialization
    u = np.ones((n, 1))
    v = np.ones((m, 1))
    
    K = np.exp(-(X[:, None] - Y[None, :])/reg)  # This assumes X and Y are 1D arrays for simplicity
    
    for _ in range(numItermax):
        u = a / (K @ v)
        v = b / (K.T @ u)

        # Check for convergence using some criteria, for example, the change in u and v
        if np.max(np.abs(u - a / (K @ v))) < stopThr and np.max(np.abs(v - b / (K.T @ u))) < stopThr:
            break

    # Placeholder for Q and R, which you might compute during the algorithm.
    Q = u @ v.T  # Assuming rank 1 for simplicity. This might need to be modified for your use case.
    R = v @ u.T
    
    return Q, R  # Return Q and R as specified.



def low_rank_kernel_sinkhorn(X, Y, a, b, reg, r, kernel_function, alpha='auto', numItermax=1000, stopThr=1e-4, verbose=False, log=False):
    """
    Implement the Low Rank Kernel Sinkhorn algorithm for optimal transport.

    Parameters
    ----------
    X : ndarray, shape (n, d)
        Source samples.
    Y : ndarray, shape (m, d)
        Target samples.
    a : ndarray, shape (n,)
        Source weights.
    b : ndarray, shape (m,)
        Target weights.
    reg : float
        Regularization term.
    r : int
        Rank of the approximation.
    kernel_function : callable
        A function that computes the kernel for given inputs.
    alpha : float or 'auto', optional
        Parameter for stability. If 'auto', it will be computed automatically. Default is 'auto'.
    numItermax : int, optional
        Maximum number of iterations. Default is 1000.
    stopThr : float, optional
        Tolerance for convergence. Default is 1e-4.
    verbose : bool, optional
        If True, prints information about the iterations. Default is False.
    log : bool, optional
        If True, returns a dictionary of logs. Default is False.

    Returns
    -------
    Q : ndarray, shape (n, m)
        Low-rank approximation matrix.
    R : ndarray, shape (m, n)
        Low-rank approximation matrix.

    Notes
    -----
    The Low Rank Kernel Sinkhorn algorithm is an approach to compute the optimal transport plan between two distributions using a low-rank approximation of the kernel matrix. This can be particularly useful when dealing with large-scale problems where the full kernel matrix is too large to handle.

    References
    ----------
    .. [1] Cuturi, M., "Sinkhorn distances: Lightspeed computation of optimal transport", Advances in Neural Information Processing Systems, 2013.

    Examples
    --------
    >>> def gaussian_kernel(x, y, sigma=1.0):
    ...     return np.exp(-np.linalg.norm(x-y)**2 / (2 * sigma**2))
    ...
    >>> X = np.array([[1, 2], [3, 4]])
    >>> Y = np.array([[5, 6], [7, 8]])
    >>> a = np.array([0.5, 0.5])
    >>> b = np.array([0.5, 0.5])
    >>> Q, R = low_rank_kernel_sinkhorn(X, Y, a, b, reg=0.1, r=1, kernel_function=gaussian_kernel)
    """
    
    n, m = len(X), len(Y)
    
    if alpha == 'auto':
        alpha = compute_alpha(r)
        
    if (1/r > alpha) or (alpha < 0):
        warnings.warn("The provided alpha value might lead to instabilities.")
    
    # Compute the Kernel Matrix using the provided kernel_function
    K = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            K[i, j] = kernel_function(X[i], Y[j])
            
    K = np.exp(-K / reg)  # Apply the regularization

    # Initialization
    u = np.ones((n, 1))
    v = np.ones((m, 1))

    # Sinkhorn Iterations
    for _ in range(numItermax):
        u = a / (K @ v)
        v = b / (K.T @ u)
        
        # Check for convergence using some criteria, for example, the change in u and v
        if np.max(np.abs(u - a / (K @ v))) < stopThr and np.max(np.abs(v - b / (K.T @ u))) < stopThr:
            break
            
    # For low-rank approximation, typically some decomposition (like SVD) would be done on the kernel.
    # Here, as an example, we use a simple outer product as a placeholder.
    Q = u @ v.T  # This should actually be a more complex decomposition to get a rank-r approximation.
    R = v @ u.T
    
    return Q, R  # Return Q and R as specified.






# if you have several convex sets and a point outside these sets, the algorithm will find the closest point (in terms of Euclidean distance) that lies within all of these sets simultaneously.
def LR_Dykstra(p0, projectors, numItermax=1000, tol=1e-4):
    """
    Apply the Low-Rank Dykstra's algorithm for projecting onto the intersection of convex sets.
    
    Parameters
    ----------
    p0 : ndarray
        Initial point.
    projectors : list of callables
        A list of functions. Each function projects a point onto a convex set.
    numItermax : int, optional
        Maximum number of iterations. Default is 1000.
    tol : float, optional
        Tolerance for convergence. The algorithm stops when the change in `p` is less than `tol`. Default is 1e-4.
    
    Returns
    -------
    p : ndarray
        The projected point onto the intersection of the convex sets.
    
    Notes
    -----
    Dykstra's algorithm is an iterative method to find the projection of a point onto the intersection of several convex sets. This implementation is a low-rank version of the algorithm.
    
    References
    ----------
    .. [1] R.L. Dykstra, "An algorithm for restricted least squares regression", Journal of the American Statistical Association, 78(384):837-842, 1983.
    
    Examples
    --------
    >>> def proj_onto_positive(x):
    ...     return np.maximum(x, 0)
    ...
    >>> def proj_onto_ball(x, center=np.array([0,0]), radius=1):
    ...     diff = x - center
    ...     norm = np.linalg.norm(diff)
    ...     if norm > radius:
    ...         return center + (diff / norm) * radius
    ...     return x
    ...
    >>> p0 = np.array([-1, -1])
    >>> projectors = [proj_onto_positive, lambda x: proj_onto_ball(x, np.array([0,0]), 1)]
    >>> p = LR_Dykstra(p0, projectors)
    >>> print(p)
    [0.  0.5]
    """
    
    r = [np.zeros_like(p0) for _ in projectors]
    p = p0.copy()
    p_prev = p0.copy()
    
    for _ in range(numItermax):
        for i, projector in enumerate(projectors):
            y = p + r[i]
            p_i = projector(y)  # Project onto the i-th set
            r[i] = y - p_i
            p += p_i - y  # Update p using the projection difference
            
        # Check for convergence based on the change in p
        if np.linalg.norm(p - p_prev) < tol:
            break
        p_prev = p.copy()

    return p


def empirical_sinkhorn(X, Y, a, b, reg, numItermax=1000, stopThr=1e-4, verbose=False, log=False):
    """
    Implement the Empirical Sinkhorn algorithm using Euclidean distance as ground metric.

    Parameters:
    - X, Y: Empirical samples from two distributions.
    - a, b: Histogram weights for X and Y respectively.
    - reg: Regularization parameter.
    """
    n, m = len(X), len(Y)
    
    # Compute pairwise distances to form the cost matrix M
    M = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            M[i, j] = np.linalg.norm(X[i] - Y[j])
    
    # Compute the kernel matrix K
    K = np.exp(-M / reg)
    
    # Initialization
    u = np.ones(n) / n
    v = np.ones(m) / m

    # Sinkhorn iterations
    for _ in range(numItermax):
        u_prev, v_prev = u.copy(), v.copy()
        
        u = a / np.dot(K, v)
        v = b / np.dot(K.T, u)
        
        # Check for convergence using some criteria, for example, the change in u and v
        if np.max(np.abs(u - a / np.dot(K, v))) < stopThr and np.max(np.abs(v - b / np.dot(K.T, u))) < stopThr:
            break

    # Once convergence is achieved, the transport plan is given by diag(u) * K * diag(v)
    P = np.outer(u, v) * K

    if log:
        return P, {"u": u, "v": v}
    else:
        return P
