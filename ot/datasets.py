"""
Simple example datasets
"""

# Author: Remi Flamary <remi.flamary@unice.fr>
#
# License: MIT License

import numpy as np
import scipy as sp
from scipy.stats import ortho_group, multivariate_normal
from .utils import check_random_state, deprecated


def make_1D_gauss(n, m, s):
    """return a 1D histogram for a gaussian distribution (`n` bins, mean `m` and std `s`)

    Parameters
    ----------
    n : int
        number of bins in the histogram
    m : float
        mean value of the gaussian distribution
    s : float
        standard deviation of the gaussian distribution

    Returns
    -------
    h : ndarray (`n`,)
        1D histogram for a gaussian distribution
    """
    x = np.arange(n, dtype=np.float64)
    h = np.exp(-((x - m) ** 2) / (2 * s**2))
    return h / h.sum()


@deprecated()
def get_1D_gauss(n, m, sigma):
    """Deprecated see  make_1D_gauss"""
    return make_1D_gauss(n, m, sigma)


def make_2D_samples_gauss(n, m, sigma, random_state=None):
    r"""Return `n` samples drawn from 2D gaussian :math:`\mathcal{N}(m, \sigma)`

    Parameters
    ----------
    n : int
        number of samples to make
    m : ndarray, shape (2,)
        mean value of the gaussian distribution
    sigma : ndarray, shape (2, 2)
        covariance matrix of the gaussian distribution
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Returns
    -------
    X : ndarray, shape (`n`, 2)
        n samples drawn from :math:`\mathcal{N}(m, \sigma)`.
    """

    generator = check_random_state(random_state)
    if np.isscalar(sigma):
        sigma = np.array(
            [
                sigma,
            ]
        )
    if len(sigma) > 1:
        P = sp.linalg.sqrtm(sigma)
        res = generator.randn(n, 2).dot(P) + m
    else:
        res = generator.randn(n, 2) * np.sqrt(sigma) + m
    return res


@deprecated()
def get_2D_samples_gauss(n, m, sigma, random_state=None):
    """Deprecated see  make_2D_samples_gauss"""
    return make_2D_samples_gauss(n, m, sigma, random_state=None)


def make_data_classif(dataset, n, nz=0.5, theta=0, p=0.5, random_state=None, **kwargs):
    """Dataset generation for classification problems

    Parameters
    ----------
    dataset : str
        type of classification problem (see code)
    n : int
        number of training samples
    nz : float
        noise level (>0)
    p : float
        proportion of one class in the binary setting
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Returns
    -------
    X : ndarray, shape (n, d)
        `n` observation of size `d`
    y : ndarray, shape (n,)
        labels of the samples.
    """
    generator = check_random_state(random_state)

    if dataset.lower() == "3gauss":
        y = np.floor((np.arange(n) * 1.0 / n * 3)) + 1
        x = np.zeros((n, 2))
        # class 1
        x[y == 1, 0] = -1.0
        x[y == 1, 1] = -1.0
        x[y == 2, 0] = -1.0
        x[y == 2, 1] = 1.0
        x[y == 3, 0] = 1.0
        x[y == 3, 1] = 0

        x[y != 3, :] += 1.5 * nz * generator.randn(sum(y != 3), 2)
        x[y == 3, :] += 2 * nz * generator.randn(sum(y == 3), 2)

    elif dataset.lower() == "3gauss2":
        y = np.floor((np.arange(n) * 1.0 / n * 3)) + 1
        x = np.zeros((n, 2))
        y[y == 4] = 3
        # class 1
        x[y == 1, 0] = -2.0
        x[y == 1, 1] = -2.0
        x[y == 2, 0] = -2.0
        x[y == 2, 1] = 2.0
        x[y == 3, 0] = 2.0
        x[y == 3, 1] = 0

        x[y != 3, :] += nz * generator.randn(sum(y != 3), 2)
        x[y == 3, :] += 2 * nz * generator.randn(sum(y == 3), 2)

    elif dataset.lower() == "gaussrot":
        rot = np.array(
            [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]
        )
        m1 = np.array([-1, 1])
        m2 = np.array([1, -1])
        y = np.floor((np.arange(n) * 1.0 / n * 2)) + 1
        n1 = np.sum(y == 1)
        n2 = np.sum(y == 2)
        x = np.zeros((n, 2))

        x[y == 1, :] = make_2D_samples_gauss(n1, m1, nz, random_state=generator)
        x[y == 2, :] = make_2D_samples_gauss(n2, m2, nz, random_state=generator)

        x = x.dot(rot)

    elif dataset.lower() == "2gauss_prop":
        y = np.concatenate((np.ones(int(p * n)), np.zeros(int((1 - p) * n))))
        x = np.hstack((0 * y[:, None] - 0, 1 - 2 * y[:, None])) + nz * generator.randn(
            len(y), 2
        )

        if ("bias" not in kwargs) and ("b" not in kwargs):
            kwargs["bias"] = np.array([0, 2])

        x[:, 0] += kwargs["bias"][0]
        x[:, 1] += kwargs["bias"][1]

    else:
        x = np.array(0)
        y = np.array(0)
        print("unknown dataset")

    return x, y.astype(int)


@deprecated()
def get_data_classif(dataset, n, nz=0.5, theta=0, random_state=None, **kwargs):
    """Deprecated see  make_data_classif"""
    return make_data_classif(dataset, n, nz=0.5, theta=0, random_state=None, **kwargs)


def make_gauss_hd(
    ns, nt, p=100, dim=5, m_diff=3, a=(10, 15), b=(3, 3), sub_the_same=False
):
    """Generation of source and target domains from Gaussian HD distributions

    Parameters
    ----------
    ns      : int
            number of samples (source)
    nt      : int
            number of samples (target)
    p       : int
            dimension of the ambient space the data live in
    dim     : (int,int) or int
            the intrinsic dimensions of the source and target Gaussian HD distriutions. If a single int the intrinsic dimension is assumed to be the same
    m_diff  : float
            the shift in the first coordinate of the means of the Gaussian HD distributions, i.e. ms_0 and mt_0, respectively (see code)
    a       : (float, float)
            positive floating numbers corresponding to the isotropic variances in the principal subspace, for the source and target distributions, respectively. The same as \delta in :ref:`[1] <references-make_gauss-hd>`, Proposition 2.2
    b       : (float, float)
            positive floating numbers corresponding to the isotropic variance outside the principal subspace for the source and target distributions, respectively.
    sub_the_same : bool
              should the source/target Gaussian HD distributions live in the same principal subspace?

    Returns
    -------
    Xs   : ndarray, shape (ns, p)
        `ns` observations of size `p` (source)
    Xt   : ndarray, shape (nt, p)
        `nt` observations of size `p` (destination)
    pmts : list
         a list containing the parameters of the Gaussian HD distributions

    .. _references-make_gauss_hd:
    References
    ----------

    .. [1] Bouveyron, C. & Corneli, M. ("Scaling Optimal Transport to High-Dimensional Gaussian Distributions")

    """
    d = (dim, dim) if isinstance(dim, int) else dim
    mu = np.zeros((2, p))
    S = []
    mu[1, 0] = m_diff
    Q = [ortho_group.rvs(p) for _ in range(2)]

    if sub_the_same:
        Q[1] = Q[0]

    S.append(
        Q[0]
        @ np.diag(np.hstack((np.full(d[0], a[0]), np.full(p - d[0], b[0]))))
        @ Q[0].T
    )
    S.append(
        Q[1]
        @ np.diag(np.hstack((np.full(d[1], a[1]), np.full(p - d[1], b[1]))))
        @ Q[1].T
    )

    Xs = multivariate_normal.rvs(mean=mu[0], cov=S[0], size=ns)
    Xt = multivariate_normal.rvs(mean=mu[1], cov=S[1], size=ns)

    ms = mu[0]
    mt = mu[1]
    ds = d[0]
    dt = d[1]
    sigma2_s = np.array(b[0])
    sigma2_t = np.array(b[1])
    ls = np.repeat(a[0], ds) - sigma2_s
    lt = np.repeat(a[1], dt) - sigma2_t
    Us = Q[0][:, :ds]
    Ut = Q[1][:, :dt]
    ds = np.array([ds])
    dt = np.array([dt])

    prmts = {
        "ms": ms,
        "mt": mt,
        "sigma2_s": sigma2_s,
        "sigma2_t": sigma2_t,
        "ls": ls,
        "lt": lt,
        "Us": Us,
        "Ut": Ut,
        "ds": ds,
        "dt": dt,
        "Cs": S[0],
        "Ct": S[1],
    }

    return Xs, Xt, prmts
