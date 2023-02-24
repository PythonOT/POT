# -*- coding: utf-8 -*-
"""
Various useful functions
"""

# Author: Remi Flamary <remi.flamary@unice.fr>
#
# License: MIT License

from functools import reduce
import time

import numpy as np
from scipy.spatial.distance import cdist
import sys
import warnings
from inspect import signature
from .backend import get_backend, Backend, NumpyBackend

__time_tic_toc = time.time()


def tic():
    r""" Python implementation of Matlab tic() function """
    global __time_tic_toc
    __time_tic_toc = time.time()


def toc(message='Elapsed time : {} s'):
    r""" Python implementation of Matlab toc() function """
    t = time.time()
    print(message.format(t - __time_tic_toc))
    return t - __time_tic_toc


def toq():
    r""" Python implementation of Julia toc() function """
    t = time.time()
    return t - __time_tic_toc


def kernel(x1, x2, method='gaussian', sigma=1, **kwargs):
    r"""Compute kernel matrix"""

    nx = get_backend(x1, x2)

    if method.lower() in ['gaussian', 'gauss', 'rbf']:
        K = nx.exp(-dist(x1, x2) / (2 * sigma**2))
    return K


def laplacian(x):
    r"""Compute Laplacian matrix"""
    nx = get_backend(x)
    L = nx.diag(nx.sum(x, axis=0)) - x
    return L


def list_to_array(*lst):
    r""" Convert a list if in numpy format """
    if len(lst) > 1:
        return [np.array(a) if isinstance(a, list) else a for a in lst]
    else:
        return np.array(lst[0]) if isinstance(lst[0], list) else lst[0]


def proj_simplex(v, z=1):
    r"""Compute the closest point (orthogonal projection) on the
    generalized `(n-1)`-simplex of a vector :math:`\mathbf{v}` wrt. to the Euclidean
    distance, thus solving:

    .. math::
        \mathcal{P}(w) \in \mathop{\arg \min}_\gamma \| \gamma - \mathbf{v} \|_2

        s.t. \ \gamma^T \mathbf{1} = z

             \gamma \geq 0

    If :math:`\mathbf{v}` is a 2d array, compute all the projections wrt. axis 0

    .. note:: This function is backend-compatible and will work on arrays
        from all compatible backends.

    Parameters
    ----------
    v : {array-like}, shape (n, d)
    z : int, optional
        'size' of the simplex (each vectors sum to z, 1 by default)

    Returns
    -------
    h : ndarray, shape (`n`, `d`)
        Array of projections on the simplex
    """
    nx = get_backend(v)
    n = v.shape[0]
    if v.ndim == 1:
        d1 = 1
        v = v[:, None]
    else:
        d1 = 0
    d = v.shape[1]

    # sort u in ascending order
    u = nx.sort(v, axis=0)
    # take the descending order
    u = nx.flip(u, 0)
    cssv = nx.cumsum(u, axis=0) - z
    ind = nx.arange(n, type_as=v)[:, None] + 1
    cond = u - cssv / ind > 0
    rho = nx.sum(cond, 0)
    theta = cssv[rho - 1, nx.arange(d)] / rho
    w = nx.maximum(v - theta[None, :], nx.zeros(v.shape, type_as=v))
    if d1:
        return w[:, 0]
    else:
        return w


def unif(n, type_as=None):
    r"""
    Return a uniform histogram of length `n` (simplex).

    Parameters
    ----------
    n : int
        number of bins in the histogram
    type_as : array_like
        array of the same type of the expected output (numpy/pytorch/jax)

    Returns
    -------
    h : array_like (`n`,)
        histogram of length `n` such that :math:`\forall i, \mathbf{h}_i = \frac{1}{n}`
    """
    if type_as is None:
        return np.ones((n,)) / n
    else:
        nx = get_backend(type_as)
        return nx.ones((n,), type_as=type_as) / n


def clean_zeros(a, b, M):
    r""" Remove all components with zeros weights in :math:`\mathbf{a}` and :math:`\mathbf{b}`
    """
    M2 = M[a > 0, :][:, b > 0].copy()  # copy force c style matrix (froemd)
    a2 = a[a > 0]
    b2 = b[b > 0]
    return a2, b2, M2


def euclidean_distances(X, Y, squared=False):
    r"""
    Considering the rows of :math:`\mathbf{X}` (and :math:`\mathbf{Y} = \mathbf{X}`) as vectors, compute the
    distance matrix between each pair of vectors.

    .. note:: This function is backend-compatible and will work on arrays
        from all compatible backends.

    Parameters
    ----------
    X : array-like, shape (n_samples_1, n_features)
    Y : array-like, shape (n_samples_2, n_features)
    squared : boolean, optional
        Return squared Euclidean distances.

    Returns
    -------
    distances : array-like, shape (`n_samples_1`, `n_samples_2`)
    """

    nx = get_backend(X, Y)

    a2 = nx.einsum('ij,ij->i', X, X)
    b2 = nx.einsum('ij,ij->i', Y, Y)

    c = -2 * nx.dot(X, Y.T)
    c += a2[:, None]
    c += b2[None, :]

    c = nx.maximum(c, 0)

    if not squared:
        c = nx.sqrt(c)

    if X is Y:
        c = c * (1 - nx.eye(X.shape[0], type_as=c))

    return c


def dist(x1, x2=None, metric='sqeuclidean', p=2, w=None):
    r"""Compute distance between samples in :math:`\mathbf{x_1}` and :math:`\mathbf{x_2}`

    .. note:: This function is backend-compatible and will work on arrays
        from all compatible backends.

    Parameters
    ----------

    x1 : array-like, shape (n1,d)
        matrix with `n1` samples of size `d`
    x2 : array-like, shape (n2,d), optional
        matrix with `n2` samples of size `d` (if None then :math:`\mathbf{x_2} = \mathbf{x_1}`)
    metric : str | callable, optional
        'sqeuclidean' or 'euclidean' on all backends. On numpy the function also
        accepts  from the scipy.spatial.distance.cdist function : 'braycurtis',
        'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice',
        'euclidean', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis',
        'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
        'sokalmichener', 'sokalsneath', 'sqeuclidean', 'wminkowski', 'yule'.
    p : float, optional
        p-norm for the Minkowski and the Weighted Minkowski metrics. Default value is 2.
    w : array-like, rank 1
        Weights for the weighted metrics.


    Returns
    -------

    M : array-like, shape (`n1`, `n2`)
        distance matrix computed with given metric

    """
    if x2 is None:
        x2 = x1
    if metric == "sqeuclidean":
        return euclidean_distances(x1, x2, squared=True)
    elif metric == "euclidean":
        return euclidean_distances(x1, x2, squared=False)
    else:
        if not get_backend(x1, x2).__name__ == 'numpy':
            raise NotImplementedError()
        else:
            if isinstance(metric, str) and metric.endswith("minkowski"):
                return cdist(x1, x2, metric=metric, p=p, w=w)
            if w is not None:
                return cdist(x1, x2, metric=metric, w=w)
            return cdist(x1, x2, metric=metric)


def dist0(n, method='lin_square'):
    r"""Compute standard cost matrices of size (`n`, `n`) for OT problems

    Parameters
    ----------
    n : int
        Size of the cost matrix.
    method : str, optional
        Type of loss matrix chosen from:

        * 'lin_square' : linear sampling between 0 and `n-1`, quadratic loss

    Returns
    -------
    M : ndarray, shape (`n1`, `n2`)
        Distance matrix computed with given metric.
    """
    res = 0
    if method == 'lin_square':
        x = np.arange(n, dtype=np.float64).reshape((n, 1))
        res = dist(x, x)
    return res


def cost_normalization(C, norm=None):
    r""" Apply normalization to the loss matrix

    Parameters
    ----------
    C : ndarray, shape (n1, n2)
        The cost matrix to normalize.
    norm : str
        Type of normalization from 'median', 'max', 'log', 'loglog'. Any
        other value do not normalize.

    Returns
    -------
    C : ndarray, shape (`n1`, `n2`)
        The input cost matrix normalized according to given norm.
    """

    if norm is None:
        pass
    elif norm == "median":
        C /= float(np.median(C))
    elif norm == "max":
        C /= float(np.max(C))
    elif norm == "log":
        C = np.log(1 + C)
    elif norm == "loglog":
        C = np.log1p(np.log1p(C))
    else:
        raise ValueError('Norm %s is not a valid option.\n'
                         'Valid options are:\n'
                         'median, max, log, loglog' % norm)
    return C


def dots(*args):
    r""" dots function for multiple matrix multiply """
    nx = get_backend(*args)
    return reduce(nx.dot, args)


def label_normalization(y, start=0):
    r""" Transform labels to start at a given value

    Parameters
    ----------
    y : array-like, shape (n, )
        The vector of labels to be normalized.
    start : int
        Desired value for the smallest label in :math:`\mathbf{y}` (default=0)

    Returns
    -------
    y : array-like, shape (`n1`, )
        The input vector of labels normalized according to given start value.
    """
    nx = get_backend(y)

    diff = nx.min(nx.unique(y)) - start
    if diff != 0:
        y -= diff
    return y


def parmap(f, X, nprocs="default"):
    r""" parallel map for multiprocessing.
    The function has been deprecated and only performs a regular map.
    """
    return list(map(f, X))


def check_params(**kwargs):
    r"""check_params: check whether some parameters are missing
    """

    missing_params = []
    check = True

    for param in kwargs:
        if kwargs[param] is None:
            missing_params.append(param)

    if len(missing_params) > 0:
        print("POT - Warning: following necessary parameters are missing")
        for p in missing_params:
            print("\n", p)

        check = False

    return check


def check_random_state(seed):
    r"""Turn `seed` into a np.random.RandomState instance

    Parameters
    ----------
    seed : None | int | instance of RandomState
        If `seed` is None, return the RandomState singleton used by np.random.
        If `seed` is an int, return a new RandomState instance seeded with `seed`.
        If `seed` is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (int, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('{} cannot be used to seed a numpy.random.RandomState'
                     ' instance'.format(seed))


def get_coordinate_circle(x):
    r"""For :math:`x\in S^1 \subset \mathbb{R}^2`, returns the coordinates in
    turn (in [0,1[).

    .. math::
        u = \frac{\pi + \mathrm{atan2}(-x_2,-x_1)}{2\pi}

    Parameters
    ----------
    x: ndarray, shape (n, 2)
        Samples on the circle with ambient coordinates

    Returns
    -------
    x_t: ndarray, shape (n,)
        Coordinates on [0,1[

    Examples
    --------
    >>> u = np.array([[0.2,0.5,0.8]]) * (2 * np.pi)
    >>> x1, y1 = np.cos(u), np.sin(u)
    >>> x = np.concatenate([x1, y1]).T
    >>> get_coordinate_circle(x)
    array([0.2, 0.5, 0.8])
    """
    nx = get_backend(x)
    x_t = (nx.atan2(-x[:, 1], -x[:, 0]) + np.pi) / (2 * np.pi)
    return x_t


class deprecated(object):
    r"""Decorator to mark a function or class as deprecated.

    deprecated class from scikit-learn package
    https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/utils/deprecation.py
    Issue a warning when the function is called/the class is instantiated and
    adds a warning to the docstring.
    The optional extra argument will be appended to the deprecation message
    and the docstring.

    .. note::
        To use this with the default value for extra, use empty parentheses:

        >>> from ot.deprecation import deprecated  # doctest: +SKIP
        >>> @deprecated()  # doctest: +SKIP
        ... def some_function(): pass  # doctest: +SKIP

    Parameters
    ----------
    extra : str
        To be added to the deprecation messages.
    """

    # Adapted from http://wiki.python.org/moin/PythonDecoratorLibrary,
    # but with many changes.

    def __init__(self, extra=''):
        self.extra = extra

    def __call__(self, obj):
        r"""Call method
        Parameters
        ----------
        obj : object
        """
        if isinstance(obj, type):
            return self._decorate_class(obj)
        else:
            return self._decorate_fun(obj)

    def _decorate_class(self, cls):
        msg = "Class %s is deprecated" % cls.__name__
        if self.extra:
            msg += "; %s" % self.extra

        # FIXME: we should probably reset __new__ for full generality
        init = cls.__init__

        def wrapped(*args, **kwargs):
            warnings.warn(msg, category=DeprecationWarning)
            return init(*args, **kwargs)

        cls.__init__ = wrapped

        wrapped.__name__ = '__init__'
        wrapped.__doc__ = self._update_doc(init.__doc__)
        wrapped.deprecated_original = init

        return cls

    def _decorate_fun(self, fun):
        r"""Decorate function fun"""

        msg = "Function %s is deprecated" % fun.__name__
        if self.extra:
            msg += "; %s" % self.extra

        def wrapped(*args, **kwargs):
            warnings.warn(msg, category=DeprecationWarning)
            return fun(*args, **kwargs)

        wrapped.__name__ = fun.__name__
        wrapped.__dict__ = fun.__dict__
        wrapped.__doc__ = self._update_doc(fun.__doc__)

        return wrapped

    def _update_doc(self, olddoc):
        newdoc = "DEPRECATED"
        if self.extra:
            newdoc = "%s: %s" % (newdoc, self.extra)
        if olddoc:
            newdoc = "%s\n\n%s" % (newdoc, olddoc)
        return newdoc


def _is_deprecated(func):
    r"""Helper to check if func is wraped by our deprecated decorator"""
    if sys.version_info < (3, 5):
        raise NotImplementedError("This is only available for python3.5 "
                                  "or above")
    closures = getattr(func, '__closure__', [])
    if closures is None:
        closures = []
    is_deprecated = ('deprecated' in ''.join([c.cell_contents
                                              for c in closures
                                              if isinstance(c.cell_contents, str)]))
    return is_deprecated


class BaseEstimator(object):
    r"""Base class for most objects in POT

    Code adapted from sklearn BaseEstimator class

    Notes
    -----
    All estimators should specify all the parameters that can be set
    at the class level in their ``__init__`` as explicit keyword
    arguments (no ``*args`` or ``**kwargs``).
    """

    nx: Backend = None

    def _get_backend(self, *arrays):
        nx = get_backend(
            *[input_ for input_ in arrays if input_ is not None]
        )
        if nx.__name__ in ("jax", "tf"):
            raise TypeError(
                """JAX or TF arrays have been received but domain
                adaptation does not support those backend.""")
        self.nx = nx
        return nx

    @classmethod
    def _get_param_names(cls):
        r"""Get parameter names for the estimator"""

        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [p for p in init_signature.parameters.values()
                      if p.name != 'self' and p.kind != p.VAR_KEYWORD]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError("POT estimators should always "
                                   "specify their parameters in the signature"
                                   " of their __init__ (no varargs)."
                                   " %s with constructor %s doesn't "
                                   " follow this convention."
                                   % (cls, init_signature))
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

    def get_params(self, deep=True):
        r"""Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            # We need deprecation warnings to always be on in order to
            # catch deprecated param values.
            # This is set in utils/__init__.py but it gets overwritten
            # when running under python3 somehow.
            warnings.simplefilter("always", DeprecationWarning)
            try:
                with warnings.catch_warnings(record=True) as w:
                    value = getattr(self, key, None)
                if len(w) and w[0].category == DeprecationWarning:
                    # if the parameter is deprecated, don't show it
                    continue
            finally:
                warnings.filters.pop(0)

            # XXX: should we rather test if instance of estimator?
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        r"""Set the parameters of this estimator.

        The method works on simple estimators as well as on nested objects
        (such as pipelines). The latter have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.

        Returns
        -------
        self
        """
        if not params:
            # Simple optimisation to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)
        # for key, value in iteritems(params):
        for key, value in params.items():
            split = key.split('__', 1)
            if len(split) > 1:
                # nested objects case
                name, sub_name = split
                if name not in valid_params:
                    raise ValueError('Invalid parameter %s for estimator %s. '
                                     'Check the list of available parameters '
                                     'with `estimator.get_params().keys()`.' %
                                     (name, self))
                sub_object = valid_params[name]
                sub_object.set_params(**{sub_name: value})
            else:
                # simple objects case
                if key not in valid_params:
                    raise ValueError('Invalid parameter %s for estimator %s. '
                                     'Check the list of available parameters '
                                     'with `estimator.get_params().keys()`.' %
                                     (key, self.__class__.__name__))
                setattr(self, key, value)
        return self


class UndefinedParameter(Exception):
    r"""
    Aim at raising an Exception when a undefined parameter is called

    """
    pass


class OTResult:
    def __init__(self, potentials=None, value=None, value_linear=None, plan=None, log=None, backend=None, sparse_plan=None, lazy_plan=None, status=None):

        self._potentials = potentials
        self._value = value
        self._value_linear = value_linear
        self._plan = plan
        self._log = log
        self._sparse_plan = sparse_plan
        self._lazy_plan = lazy_plan
        self._backend = backend if backend is not None else NumpyBackend()
        self._status = status

        # I assume that other solvers may return directly
        # some primal objects?
        # In the code below, let's define the main quantities
        # that may be of interest to users.
        # An OT solver returns an object that inherits from OTResult
        # (e.g. SinkhornOTResult) and implements the relevant
        # methods (e.g. "plan" and "lazy_plan" but not "sparse_plan", etc.).
        # log is a dictionary containing potential information about the solver

    # Dual potentials --------------------------------------------

    def __repr__(self):
        s = 'OTResult('
        if self._value is not None:
            s += 'value={},'.format(self._value)
        if self._value_linear is not None:
            s += 'value_linear={},'.format(self._value_linear)
        if self._plan is not None:
            s += 'plan={}(shape={}),'.format(self._plan.__class__.__name__, self._plan.shape)

        if s[-1] != '(':
            s = s[:-1] + ')'
        else:
            s = s + ')'
        return s

    @property
    def potentials(self):
        """Dual potentials, i.e. Lagrange multipliers for the marginal constraints.

        This pair of arrays has the same shape, numerical type
        and properties as the input weights "a" and "b".
        """
        if self._potentials is not None:
            return self._potentials
        else:
            raise NotImplementedError()

    @property
    def potential_a(self):
        """First dual potential, associated to the "source" measure "a"."""
        if self._potentials is not None:
            return self._potentials[0]
        else:
            raise NotImplementedError()

    @property
    def potential_b(self):
        """Second dual potential, associated to the "target" measure "b"."""
        if self._potentials is not None:
            return self._potentials[1]
        else:
            raise NotImplementedError()

    # Transport plan -------------------------------------------
    @property
    def plan(self):
        """Transport plan, encoded as a dense array."""
        # N.B.: We may catch out-of-memory errors and suggest
        # the use of lazy_plan or sparse_plan when appropriate.

        if self._plan is not None:
            return self._plan
        else:
            raise NotImplementedError()

    @property
    def sparse_plan(self):
        """Transport plan, encoded as a sparse array."""
        if self._sparse_plan is not None:
            return self._sparse_plan
        elif self._plan is not None:
            return self._backend.tocsr(self._plan)
        else:
            raise NotImplementedError()

    @property
    def lazy_plan(self):
        """Transport plan, encoded as a symbolic KeOps LazyTensor."""
        raise NotImplementedError()

    # Loss values --------------------------------

    @property
    def value(self):
        """Full transport cost, including possible regularization terms."""
        if self._value is not None:
            return self._value
        else:
            raise NotImplementedError()

    @property
    def value_linear(self):
        """The "minimal" transport cost, i.e. the product between the transport plan and the cost."""
        if self._value_linear is not None:
            return self._value_linear
        else:
            raise NotImplementedError()

    # Marginal constraints -------------------------
    @property
    def marginals(self):
        """Marginals of the transport plan: should be very close to "a" and "b"
        for balanced OT."""
        if self._plan is not None:
            return self.marginal_a, self.marginal_b
        else:
            raise NotImplementedError()

    @property
    def marginal_a(self):
        """First marginal of the transport plan, with the same shape as "a"."""
        if self._plan is not None:
            return self._backend.sum(self._plan, 1)
        else:
            raise NotImplementedError()

    @property
    def marginal_b(self):
        """Second marginal of the transport plan, with the same shape as "b"."""
        if self._plan is not None:
            return self._backend.sum(self._plan, 0)
        else:
            raise NotImplementedError()

    @property
    def status(self):
        """Optimization status of the solver."""
        if self._status is not None:
            return self._status
        else:
            raise NotImplementedError()

    # Barycentric mappings -------------------------
    # Return the displacement vectors as an array
    # that has the same shape as "xa"/"xb" (for samples)
    # or "a"/"b" * D (for images)?

    @property
    def a_to_b(self):
        """Displacement vectors from the first to the second measure."""
        raise NotImplementedError()

    @property
    def b_to_a(self):
        """Displacement vectors from the second to the first measure."""
        raise NotImplementedError()

    # # Wasserstein barycenters ----------------------
    # @property
    # def masses(self):
    #     """Masses for the Wasserstein barycenter."""
    #     raise NotImplementedError()

    # @property
    # def samples(self):
    #     """Sample locations for the Wasserstein barycenter."""
    #     raise NotImplementedError()

    # Miscellaneous --------------------------------

    @property
    def citation(self):
        """Appropriate citation(s) for this result, in plain text and BibTex formats."""

        # The string below refers to the POT library:
        # successor methods may concatenate the relevant references
        # to the original definitions, solvers and underlying numerical backends.
        return """POT library:

            POT Python Optimal Transport library, Journal of Machine Learning Research, 22(78):1−8, 2021.
            Website: https://pythonot.github.io/
            Rémi Flamary, Nicolas Courty, Alexandre Gramfort, Mokhtar Z. Alaya, Aurélie Boisbunon, Stanislas Chambon, Laetitia Chapel, Adrien Corenflos, Kilian Fatras, Nemo Fournier, Léo Gautheron, Nathalie T.H. Gayraud, Hicham Janati, Alain Rakotomamonjy, Ievgen Redko, Antoine Rolet, Antony Schutz, Vivien Seguy, Danica J. Sutherland, Romain Tavenard, Alexander Tong, Titouan Vayer;

            @article{flamary2021pot,
              author  = {R{\'e}mi Flamary and Nicolas Courty and Alexandre Gramfort and Mokhtar Z. Alaya and Aur{\'e}lie Boisbunon and Stanislas Chambon and Laetitia Chapel and Adrien Corenflos and Kilian Fatras and Nemo Fournier and L{\'e}o Gautheron and Nathalie T.H. Gayraud and Hicham Janati and Alain Rakotomamonjy and Ievgen Redko and Antoine Rolet and Antony Schutz and Vivien Seguy and Danica J. Sutherland and Romain Tavenard and Alexander Tong and Titouan Vayer},
              title   = {{POT}: {Python} {Optimal} {Transport}},
              journal = {Journal of Machine Learning Research},
              year    = {2021},
              volume  = {22},
              number  = {78},
              pages   = {1-8},
              url     = {http://jmlr.org/papers/v22/20-451.html}
            }
        """
