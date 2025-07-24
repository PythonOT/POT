from contextlib import contextmanager
from ot.backend import get_backend, TorchBackend
import torch


def bmm(A, B, nx):
    """
    Batched matrix multiplication for tensors A and B.
    """
    return nx.einsum("bij,bjk->bik", A, B)


def bmv(A, b, nx):
    """
    Batched matrix-vector multiplication for tensor A and vector b.
    """
    return nx.einsum("bij,bj->bi", A, b)


def bop(a, b, nx):
    """
    Batched outer product for vectors a and b.
    """
    return nx.einsum("bi,bj->bij", a, b)


@contextmanager
def grad_enabled(nx, enabled):
    """
    Context manager to enable/disable gradient tracking.
    Useful to save memory when gradients are not needed.
    """
    if isinstance(nx, TorchBackend):
        prev = torch.is_grad_enabled()
        torch.set_grad_enabled(enabled)
        try:
            yield
        finally:
            torch.set_grad_enabled(prev)
    else:
        yield


def bregman_batch(K, a=None, b=None, nx=None, max_iter=10000, tol=1e-5, grad="detach"):
    r"""
    Apply Bregman projection to a batch of affinity matrices K.

    The function solves the following optimization problem:

    .. math::
        T = \mathop{\arg \max}_T \quad \langle T, \log(K) \rangle_F 

        s.t. \ T \mathbf{1} &= \mathbf{a}

             T^T \mathbf{1} &= \mathbf{b}

             T &\geq 0
             
    Which is equivalent to solving:
    
    .. math::
        T = \mathop{\arg \min}_T \quad KL(T || K)
        s.t. \ T \mathbf{1} &= \mathbf{a}

             T^T \mathbf{1} &= \mathbf{b}

             T &\geq 0
             
    We know that the optimal has the form :math: `T = \text{diag}(f) K \text{diag}(g)`,
    where the dual variables :math: `f` and :math: `g` can be found iteratively.
    
    .. math::
        f_{k+1} = \frac{a}{K g_{k}} \quad \text{and} \
        g_{k+1} = \frac{b}{f_{k} K}
        
    Parameters
    ----------
    K : array-like, shape (B, n, m)
        Affinity matrix
    a : array-like, shape (B, n), optional
        Source distribution (optional). If None, uniform distribution is used.
    b : array-like, shape (B, m), optional
        Target distribution (optional). If None, uniform distribution is used.
    nx : Backend, optional
        Backend to use for computations. If None, the default backend is used.
    max_iter : int, optional
        Maximum number of iterations to run the solver.
    tol : float, optional
        Tolerance for convergence. The solver stops when the maximum change in `f` and `g` is below this value.
    grad : str, optional
        Type of gradient computation, either 'detach', 'autodiff', 'last_step' or 'envelope'.
        
    Returns
    -------
    dict
        A dictionary containing:
        - 'T': the transport plan
        - 'n_iters': number of iterations performed
    """
    if nx is None:
        nx = get_backend(a, b, K)

    B, n, m = K.shape

    if a is None:
        a = nx.ones((B, n)) / n
    if b is None:
        b = nx.ones((B, m)) / m

    f = nx.ones((B, n))  # a / nx.sum(K, axis=2)
    g = nx.ones((B, m))  # b / nx.sum(K, axis=1)

    with grad_enabled(nx, grad == "autodiff"):
        for n_iters in range(max_iter):
            f = a / nx.sum(K * g[:, None, :], axis=2)
            g = b / nx.sum(K * f[:, :, None], axis=1)
            if n_iters % 10 == 0:
                T = K * f[:, :, None] * g[:, None, :]
                marginal = nx.sum(T, axis=2)
                err = nx.max(nx.abs(marginal - a))
                if err < tol:
                    break

    if grad == "last_step":
        summand_f = nx.sum(K * g[:, None, :], axis=2)
        f = a / summand_f

        summand_g = nx.sum(K * f[:, :, None], axis=1)
        g = b / summand_g

    T = K * f[:, :, None] * g[:, None, :]

    out = {"T": T, "n_iters": n_iters}

    return out


def bregman_log_batch(
    K, a=None, b=None, nx=None, max_iter=10000, tol=1e-5, grad="detach"
):
    r"""
    Apply Bregman projection to a batch of affinity matrices :math:`\exp(K)`.

    The function solves the following optimization problem:

    .. math::
        T = \mathop{\arg \max}_T \quad \langle T, K) \rangle_F 

        s.t. \ T \mathbf{1} &= \mathbf{a}

             T^T \mathbf{1} &= \mathbf{b}

             T &\geq 0
             
    Which is equivalent to solving:
    
    .. math::
        T = \mathop{\arg \min}_T \quad KL(T || \exp(K))
        s.t. \ T \mathbf{1} &= \mathbf{a}

             T^T \mathbf{1} &= \mathbf{b}

             T &\geq 0

    We know that the optimal has the form :math: `T = \text{diag}(\exp(u)) \exp(K) \text{diag}(\exp(v))`,
    where the dual variables :math: `u` and :math: `v` can be found iteratively.
    
    .. math::
        u_{k+1} = \log(a) - \log(\exp(K) \exp(v_{k})) \quad \text{and} \
        v_{k+1} = \log(b) - \log(\exp(K) \exp(u_{k}))

    The iterations are performed using the log-sum-exp trick to avoid numerical issues.

    Parameters
    ----------
    K : array-like, shape (B, n, m)
        Affinity matrix
    a : array-like, shape (B, n), optional
        Source distribution (optional). If None, uniform distribution is used.
    b : array-like, shape (B, m), optional
        Target distribution (optional). If None, uniform distribution is used.
    nx : Backend, optional
        Backend to use for computations. If None, the default backend is used.
    max_iter : int, optional
        Maximum number of iterations to run the solver.
    tol : float, optional
        Tolerance for convergence. The solver stops when the maximum change in `f` and `g` is below this value.
    grad : str, optional
        Type of gradient computation, either 'detach', 'autodiff', 'last_step' or 'envelope'.
        
    Returns
    -------
    dict
        A dictionary containing:
        - 'T': the transport plan
        - 'n_iters': number of iterations performed
    """

    """
    Apply Bregman projection to a batch of affinity matrices exp(K)
    Meaning:
    Solve argmin_{T transport plan} KL(T||exp(K)) = argmax_T <T,K>
    
    We know that the optimal has the form T = diag(exp(u)) exp(K) diag(exp(v)) (where u = log f and v = log g)
    Given v the optimal u is given by u = log(a) - log( exp(K) exp(v) ) 
    this can be computed with the log-sum-exp trick.
    Idem for v given u.
    
    The solver iteratively updates f and g until convergence.
    """
    if nx is None:
        nx = get_backend(a, b, K)

    B, n, m = K.shape

    if a is None:
        a = nx.ones((B, n)) / n
    if b is None:
        b = nx.ones((B, m)) / m

    u = nx.zeros((B, n), type_as=K)  # u = nx.log(a) - nx.logsumexp(K, axis=2).squeeze()
    v = nx.zeros((B, m), type_as=K)  # v = nx.log(b) - nx.logsumexp(K, axis=1).squeeze()

    with grad_enabled(nx, grad == "autodiff"):
        for n_iters in range(max_iter):
            u = nx.log(a) - nx.logsumexp(K + v[:, None, :], axis=2).squeeze()
            v = nx.log(b) - nx.logsumexp(K + u[:, :, None], axis=1).squeeze()

            # Check convergence once every 10 iterations
            if n_iters % 10 == 0:
                T = nx.exp(K + u[:, :, None] + v[:, None, :])
                marginal = nx.sum(T, axis=2)
                err = nx.max(nx.abs(marginal - a))
                if err < tol:
                    break

    if grad == "last_step":
        summand_u = K + v[:, None, :]
        u = nx.log(a) - nx.logsumexp(summand_u, axis=2).squeeze()

        summand_v = K + u[:, :, None]
        v = nx.log(b) - nx.logsumexp(summand_v, axis=1).squeeze()

    log_T = K + u[:, :, None] + v[:, None, :]
    T = nx.exp(log_T)

    return {"T": T, "log_T": log_T, "n_iters": n_iters}
