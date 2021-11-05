# -*- coding: utf-8 -*-
"""
Multi-lib backend for POT

The goal is to write backend-agnostic code. Whether you're using Numpy, PyTorch,
or Jax, POT code should work nonetheless.
To achieve that, POT provides backend classes which implements functions in their respective backend
imitating Numpy API. As a convention, we use nx instead of np to refer to the backend.

Examples
--------

>>> from ot.utils import list_to_array
>>> from ot.backend import get_backend
>>> def f(a, b):  # the function does not know which backend to use
...     a, b = list_to_array(a, b)  # if a list in given, make it an array
...     nx = get_backend(a, b)  # infer the backend from the arguments
...     c = nx.dot(a, b)  # now use the backend to do any calculation
...     return c
"""

# Author: Remi Flamary <remi.flamary@polytechnique.edu>
#         Nicolas Courty <ncourty@irisa.fr>
#
# License: MIT License

import numpy as np
import scipy.special as scipy
from scipy.sparse import issparse, coo_matrix, csr_matrix

try:
    import torch
    torch_type = torch.Tensor
except ImportError:
    torch = False
    torch_type = float

try:
    import jax
    import jax.numpy as jnp
    import jax.scipy.special as jscipy
    jax_type = jax.numpy.ndarray
except ImportError:
    jax = False
    jax_type = float

str_type_error = "All array should be from the same type/backend. Current types are : {}"


def get_backend_list():
    """Returns the list of available backends"""
    lst = [NumpyBackend(), ]

    if torch:
        lst.append(TorchBackend())

    if jax:
        lst.append(JaxBackend())

    return lst


def get_backend(*args):
    """Returns the proper backend for a list of input arrays

        Also raises TypeError if all arrays are not from the same backend
    """
    # check that some arrays given
    if not len(args) > 0:
        raise ValueError(" The function takes at least one parameter")
    # check all same type
    if not len(set(type(a) for a in args)) == 1:
        raise ValueError(str_type_error.format([type(a) for a in args]))

    if isinstance(args[0], np.ndarray):
        return NumpyBackend()
    elif isinstance(args[0], torch_type):
        return TorchBackend()
    elif isinstance(args[0], jax_type):
        return JaxBackend()
    else:
        raise ValueError("Unknown type of non implemented backend.")


def to_numpy(*args):
    """Returns numpy arrays from any compatible backend"""

    if len(args) == 1:
        return get_backend(args[0]).to_numpy(args[0])
    else:
        return [get_backend(a).to_numpy(a) for a in args]


class Backend():
    """
    Backend abstract class.
    Implementations: :py:class:`JaxBackend`, :py:class:`NumpyBackend`, :py:class:`TorchBackend`

    - The `__name__` class attribute refers to the name of the backend.
    - The `__type__` class attribute refers to the data structure used by the backend.
    """

    __name__ = None
    __type__ = None
    __type_list__ = None

    rng_ = None

    def __str__(self):
        return self.__name__

    # convert to numpy
    def to_numpy(self, a):
        """Returns the numpy version of a tensor"""
        raise NotImplementedError()

    # convert from numpy
    def from_numpy(self, a, type_as=None):
        """Creates a tensor cloning a numpy array, with the given precision (defaulting to input's precision) and the given device (in case of GPUs)"""
        raise NotImplementedError()

    def set_gradients(self, val, inputs, grads):
        """Define the gradients for the value val wrt the inputs """
        raise NotImplementedError()

    def zeros(self, shape, type_as=None):
        r"""
        Creates a tensor full of zeros.

        This function follows the api from :any:`numpy.zeros`

        See: https://numpy.org/doc/stable/reference/generated/numpy.zeros.html
        """
        raise NotImplementedError()

    def ones(self, shape, type_as=None):
        r"""
        Creates a tensor full of ones.

        This function follows the api from :any:`numpy.ones`

        See: https://numpy.org/doc/stable/reference/generated/numpy.ones.html
        """
        raise NotImplementedError()

    def arange(self, stop, start=0, step=1, type_as=None):
        r"""
        Returns evenly spaced values within a given interval.

        This function follows the api from :any:`numpy.arange`

        See: https://numpy.org/doc/stable/reference/generated/numpy.arange.html
        """
        raise NotImplementedError()

    def full(self, shape, fill_value, type_as=None):
        r"""
        Creates a tensor with given shape, filled with given value.

        This function follows the api from :any:`numpy.full`

        See: https://numpy.org/doc/stable/reference/generated/numpy.full.html
        """
        raise NotImplementedError()

    def eye(self, N, M=None, type_as=None):
        r"""
        Creates the identity matrix of given size.

        This function follows the api from :any:`numpy.eye`

        See: https://numpy.org/doc/stable/reference/generated/numpy.eye.html
        """
        raise NotImplementedError()

    def sum(self, a, axis=None, keepdims=False):
        r"""
        Sums tensor elements over given dimensions.

        This function follows the api from :any:`numpy.sum`

        See: https://numpy.org/doc/stable/reference/generated/numpy.sum.html
        """
        raise NotImplementedError()

    def cumsum(self, a, axis=None):
        r"""
        Returns the cumulative sum of tensor elements over given dimensions.

        This function follows the api from :any:`numpy.cumsum`

        See: https://numpy.org/doc/stable/reference/generated/numpy.cumsum.html
        """
        raise NotImplementedError()

    def max(self, a, axis=None, keepdims=False):
        r"""
        Returns the maximum of an array or maximum along given dimensions.

        This function follows the api from :any:`numpy.amax`

        See: https://numpy.org/doc/stable/reference/generated/numpy.amax.html
        """
        raise NotImplementedError()

    def min(self, a, axis=None, keepdims=False):
        r"""
        Returns the maximum of an array or maximum along given dimensions.

        This function follows the api from :any:`numpy.amin`

        See: https://numpy.org/doc/stable/reference/generated/numpy.amin.html
        """
        raise NotImplementedError()

    def maximum(self, a, b):
        r"""
        Returns element-wise maximum of array elements.

        This function follows the api from :any:`numpy.maximum`

        See: https://numpy.org/doc/stable/reference/generated/numpy.maximum.html
        """
        raise NotImplementedError()

    def minimum(self, a, b):
        r"""
        Returns element-wise minimum of array elements.

        This function follows the api from :any:`numpy.minimum`

        See: https://numpy.org/doc/stable/reference/generated/numpy.minimum.html
        """
        raise NotImplementedError()

    def dot(self, a, b):
        r"""
        Returns the dot product of two tensors.

        This function follows the api from :any:`numpy.dot`

        See: https://numpy.org/doc/stable/reference/generated/numpy.dot.html
        """
        raise NotImplementedError()

    def abs(self, a):
        r"""
        Computes the absolute value element-wise.

        This function follows the api from :any:`numpy.absolute`

        See: https://numpy.org/doc/stable/reference/generated/numpy.absolute.html
        """
        raise NotImplementedError()

    def exp(self, a):
        r"""
        Computes the exponential value element-wise.

        This function follows the api from :any:`numpy.exp`

        See: https://numpy.org/doc/stable/reference/generated/numpy.exp.html
        """
        raise NotImplementedError()

    def log(self, a):
        r"""
        Computes the natural logarithm, element-wise.

        This function follows the api from :any:`numpy.log`

        See: https://numpy.org/doc/stable/reference/generated/numpy.log.html
        """
        raise NotImplementedError()

    def sqrt(self, a):
        r"""
        Returns the non-ngeative square root of a tensor, element-wise.

        This function follows the api from :any:`numpy.sqrt`

        See: https://numpy.org/doc/stable/reference/generated/numpy.sqrt.html
        """
        raise NotImplementedError()

    def power(self, a, exponents):
        r"""
        First tensor elements raised to powers from second tensor, element-wise.

        This function follows the api from :any:`numpy.power`

        See: https://numpy.org/doc/stable/reference/generated/numpy.power.html
        """
        raise NotImplementedError()

    def norm(self, a):
        r"""
        Computes the matrix frobenius norm.

        This function follows the api from :any:`numpy.linalg.norm`

        See: https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html
        """
        raise NotImplementedError()

    def any(self, a):
        r"""
        Tests whether any tensor element along given dimensions evaluates to True.

        This function follows the api from :any:`numpy.any`

        See: https://numpy.org/doc/stable/reference/generated/numpy.any.html
        """
        raise NotImplementedError()

    def isnan(self, a):
        r"""
        Tests element-wise for NaN and returns result as a boolean tensor.

        This function follows the api from :any:`numpy.isnan`

        See: https://numpy.org/doc/stable/reference/generated/numpy.isnan.html
        """
        raise NotImplementedError()

    def isinf(self, a):
        r"""
        Tests element-wise for positive or negative infinity and returns result as a boolean tensor.

        This function follows the api from :any:`numpy.isinf`

        See: https://numpy.org/doc/stable/reference/generated/numpy.isinf.html
        """
        raise NotImplementedError()

    def einsum(self, subscripts, *operands):
        r"""
        Evaluates the Einstein summation convention on the operands.

        This function follows the api from :any:`numpy.einsum`

        See: https://numpy.org/doc/stable/reference/generated/numpy.einsum.html
        """
        raise NotImplementedError()

    def sort(self, a, axis=-1):
        r"""
        Returns a sorted copy of a tensor.

        This function follows the api from :any:`numpy.sort`

        See: https://numpy.org/doc/stable/reference/generated/numpy.sort.html
        """
        raise NotImplementedError()

    def argsort(self, a, axis=None):
        r"""
        Returns the indices that would sort a tensor.

        This function follows the api from :any:`numpy.argsort`

        See: https://numpy.org/doc/stable/reference/generated/numpy.argsort.html
        """
        raise NotImplementedError()

    def searchsorted(self, a, v, side='left'):
        r"""
        Finds indices where elements should be inserted to maintain order in given tensor.

        This function follows the api from :any:`numpy.searchsorted`

        See: https://numpy.org/doc/stable/reference/generated/numpy.searchsorted.html
        """
        raise NotImplementedError()

    def flip(self, a, axis=None):
        r"""
        Reverses the order of elements in a tensor along given dimensions.

        This function follows the api from :any:`numpy.flip`

        See: https://numpy.org/doc/stable/reference/generated/numpy.flip.html
        """
        raise NotImplementedError()

    def clip(self, a, a_min, a_max):
        """
        Limits the values in a tensor.

        This function follows the api from :any:`numpy.clip`

        See: https://numpy.org/doc/stable/reference/generated/numpy.clip.html
        """
        raise NotImplementedError()

    def repeat(self, a, repeats, axis=None):
        r"""
        Repeats elements of a tensor.

        This function follows the api from :any:`numpy.repeat`

        See: https://numpy.org/doc/stable/reference/generated/numpy.repeat.html
        """
        raise NotImplementedError()

    def take_along_axis(self, arr, indices, axis):
        r"""
        Gathers elements of a tensor along given dimensions.

        This function follows the api from :any:`numpy.take_along_axis`

        See: https://numpy.org/doc/stable/reference/generated/numpy.take_along_axis.html
        """
        raise NotImplementedError()

    def concatenate(self, arrays, axis=0):
        r"""
        Joins a sequence of tensors along an existing dimension.

        This function follows the api from :any:`numpy.concatenate`

        See: https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html
        """
        raise NotImplementedError()

    def zero_pad(self, a, pad_width):
        r"""
        Pads a tensor.

        This function follows the api from :any:`numpy.pad`

        See: https://numpy.org/doc/stable/reference/generated/numpy.pad.html
        """
        raise NotImplementedError()

    def argmax(self, a, axis=None):
        r"""
        Returns the indices of the maximum values of a tensor along given dimensions.

        This function follows the api from :any:`numpy.argmax`

        See: https://numpy.org/doc/stable/reference/generated/numpy.argmax.html
        """
        raise NotImplementedError()

    def mean(self, a, axis=None):
        r"""
        Computes the arithmetic mean of a tensor along given dimensions.

        This function follows the api from :any:`numpy.mean`

        See: https://numpy.org/doc/stable/reference/generated/numpy.mean.html
        """
        raise NotImplementedError()

    def std(self, a, axis=None):
        r"""
        Computes the standard deviation of a tensor along given dimensions.

        This function follows the api from :any:`numpy.std`

        See: https://numpy.org/doc/stable/reference/generated/numpy.std.html
        """
        raise NotImplementedError()

    def linspace(self, start, stop, num):
        r"""
        Returns a specified number of evenly spaced values over a given interval.

        This function follows the api from :any:`numpy.linspace`

        See: https://numpy.org/doc/stable/reference/generated/numpy.linspace.html
        """
        raise NotImplementedError()

    def meshgrid(self, a, b):
        r"""
        Returns coordinate matrices from coordinate vectors (Numpy convention).

        This function follows the api from :any:`numpy.meshgrid`

        See: https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html
        """
        raise NotImplementedError()

    def diag(self, a, k=0):
        r"""
        Extracts or constructs a diagonal tensor.

        This function follows the api from :any:`numpy.diag`

        See: https://numpy.org/doc/stable/reference/generated/numpy.diag.html
        """
        raise NotImplementedError()

    def unique(self, a):
        r"""
        Finds unique elements of given tensor.

        This function follows the api from :any:`numpy.unique`

        See: https://numpy.org/doc/stable/reference/generated/numpy.unique.html
        """
        raise NotImplementedError()

    def logsumexp(self, a, axis=None):
        r"""
        Computes the log of the sum of exponentials of input elements.

        This function follows the api from :any:`scipy.special.logsumexp`

        See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.logsumexp.html
        """
        raise NotImplementedError()

    def stack(self, arrays, axis=0):
        r"""
        Joins a sequence of tensors along a new dimension.

        This function follows the api from :any:`numpy.stack`

        See: https://numpy.org/doc/stable/reference/generated/numpy.stack.html
        """
        raise NotImplementedError()

    def outer(self, a, b):
        r"""
        Computes the outer product between two vectors.

        This function follows the api from :any:`numpy.outer`

        See: https://numpy.org/doc/stable/reference/generated/numpy.outer.html
        """
        raise NotImplementedError()

    def reshape(self, a, shape):
        r"""
        Gives a new shape to a tensor without changing its data.

        This function follows the api from :any:`numpy.reshape`

        See: https://numpy.org/doc/stable/reference/generated/numpy.reshape.html
        """
        raise NotImplementedError()

    def seed(self, seed=None):
        r"""
        Sets the seed for the random generator.

        This function follows the api from :any:`numpy.random.seed`

        See: https://numpy.org/doc/stable/reference/generated/numpy.random.seed.html
        """
        raise NotImplementedError()

    def rand(self, *size, type_as=None):
        r"""
        Generate uniform random numbers.

        This function follows the api from :any:`numpy.random.rand`

        See: https://numpy.org/doc/stable/reference/generated/numpy.random.rand.html
        """
        raise NotImplementedError()

    def randn(self, *size, type_as=None):
        r"""
        Generate normal Gaussian random numbers.

        This function follows the api from :any:`numpy.random.rand`

        See: https://numpy.org/doc/stable/reference/generated/numpy.random.rand.html
        """
        raise NotImplementedError()

    def coo_matrix(self, data, rows, cols, shape=None, type_as=None):
        r"""
        Creates a sparse tensor in COOrdinate format.

        This function follows the api from :any:`scipy.sparse.coo_matrix`

        See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html
        """
        raise NotImplementedError()

    def issparse(self, a):
        r"""
        Checks whether or not the input tensor is a sparse tensor.

        This function follows the api from :any:`scipy.sparse.issparse`

        See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.issparse.html
        """
        raise NotImplementedError()

    def tocsr(self, a):
        r"""
        Converts this matrix to Compressed Sparse Row format.

        This function follows the api from :any:`scipy.sparse.coo_matrix.tocsr`

        See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.tocsr.html
        """
        raise NotImplementedError()

    def eliminate_zeros(self, a, threshold=0.):
        r"""
        Removes entries smaller than the given threshold from the sparse tensor.

        This function follows the api from :any:`scipy.sparse.csr_matrix.eliminate_zeros`

        See: https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.sparse.csr_matrix.eliminate_zeros.html
        """
        raise NotImplementedError()

    def todense(self, a):
        r"""
        Converts a sparse tensor to a dense tensor.

        This function follows the api from :any:`scipy.sparse.csr_matrix.toarray`

        See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.toarray.html
        """
        raise NotImplementedError()

    def where(self, condition, x, y):
        r"""
        Returns elements chosen from x or y depending on condition.

        This function follows the api from :any:`numpy.where`

        See: https://numpy.org/doc/stable/reference/generated/numpy.where.html
        """
        raise NotImplementedError()

    def copy(self, a):
        r"""
        Returns a copy of the given tensor.

        This function follows the api from :any:`numpy.copy`

        See: https://numpy.org/doc/stable/reference/generated/numpy.copy.html
        """
        raise NotImplementedError()

    def allclose(self, a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
        r"""
        Returns True if two arrays are element-wise equal within a tolerance.

        This function follows the api from :any:`numpy.allclose`

        See: https://numpy.org/doc/stable/reference/generated/numpy.allclose.html
        """
        raise NotImplementedError()

    def dtype_device(self, a):
        r"""
        Returns the dtype and the device of the given tensor.
        """
        raise NotImplementedError()

    def assert_same_dtype_device(self, a, b):
        r"""
        Checks whether or not the two given inputs have the same dtype as well as the same device
        """
        raise NotImplementedError()


class NumpyBackend(Backend):
    """
    NumPy implementation of the backend

    - `__name__` is "numpy"
    - `__type__` is np.ndarray
    """

    __name__ = 'numpy'
    __type__ = np.ndarray
    __type_list__ = [np.array(1, dtype=np.float32),
                     np.array(1, dtype=np.float64)]

    rng_ = np.random.RandomState()

    def to_numpy(self, a):
        return a

    def from_numpy(self, a, type_as=None):
        if type_as is None:
            return a
        elif isinstance(a, float):
            return a
        else:
            return a.astype(type_as.dtype)

    def set_gradients(self, val, inputs, grads):
        # No gradients for numpy
        return val

    def zeros(self, shape, type_as=None):
        if type_as is None:
            return np.zeros(shape)
        else:
            return np.zeros(shape, dtype=type_as.dtype)

    def ones(self, shape, type_as=None):
        if type_as is None:
            return np.ones(shape)
        else:
            return np.ones(shape, dtype=type_as.dtype)

    def arange(self, stop, start=0, step=1, type_as=None):
        return np.arange(start, stop, step)

    def full(self, shape, fill_value, type_as=None):
        if type_as is None:
            return np.full(shape, fill_value)
        else:
            return np.full(shape, fill_value, dtype=type_as.dtype)

    def eye(self, N, M=None, type_as=None):
        if type_as is None:
            return np.eye(N, M)
        else:
            return np.eye(N, M, dtype=type_as.dtype)

    def sum(self, a, axis=None, keepdims=False):
        return np.sum(a, axis, keepdims=keepdims)

    def cumsum(self, a, axis=None):
        return np.cumsum(a, axis)

    def max(self, a, axis=None, keepdims=False):
        return np.max(a, axis, keepdims=keepdims)

    def min(self, a, axis=None, keepdims=False):
        return np.min(a, axis, keepdims=keepdims)

    def maximum(self, a, b):
        return np.maximum(a, b)

    def minimum(self, a, b):
        return np.minimum(a, b)

    def dot(self, a, b):
        return np.dot(a, b)

    def abs(self, a):
        return np.abs(a)

    def exp(self, a):
        return np.exp(a)

    def log(self, a):
        return np.log(a)

    def sqrt(self, a):
        return np.sqrt(a)

    def power(self, a, exponents):
        return np.power(a, exponents)

    def norm(self, a):
        return np.sqrt(np.sum(np.square(a)))

    def any(self, a):
        return np.any(a)

    def isnan(self, a):
        return np.isnan(a)

    def isinf(self, a):
        return np.isinf(a)

    def einsum(self, subscripts, *operands):
        return np.einsum(subscripts, *operands)

    def sort(self, a, axis=-1):
        return np.sort(a, axis)

    def argsort(self, a, axis=-1):
        return np.argsort(a, axis)

    def searchsorted(self, a, v, side='left'):
        if a.ndim == 1:
            return np.searchsorted(a, v, side)
        else:
            # this is a not very efficient way to make numpy
            # searchsorted work on 2d arrays
            ret = np.empty(v.shape, dtype=int)
            for i in range(a.shape[0]):
                ret[i, :] = np.searchsorted(a[i, :], v[i, :], side)
            return ret

    def flip(self, a, axis=None):
        return np.flip(a, axis)

    def outer(self, a, b):
        return np.outer(a, b)

    def clip(self, a, a_min, a_max):
        return np.clip(a, a_min, a_max)

    def repeat(self, a, repeats, axis=None):
        return np.repeat(a, repeats, axis)

    def take_along_axis(self, arr, indices, axis):
        return np.take_along_axis(arr, indices, axis)

    def concatenate(self, arrays, axis=0):
        return np.concatenate(arrays, axis)

    def zero_pad(self, a, pad_width):
        return np.pad(a, pad_width)

    def argmax(self, a, axis=None):
        return np.argmax(a, axis=axis)

    def mean(self, a, axis=None):
        return np.mean(a, axis=axis)

    def std(self, a, axis=None):
        return np.std(a, axis=axis)

    def linspace(self, start, stop, num):
        return np.linspace(start, stop, num)

    def meshgrid(self, a, b):
        return np.meshgrid(a, b)

    def diag(self, a, k=0):
        return np.diag(a, k)

    def unique(self, a):
        return np.unique(a)

    def logsumexp(self, a, axis=None):
        return scipy.logsumexp(a, axis=axis)

    def stack(self, arrays, axis=0):
        return np.stack(arrays, axis)

    def reshape(self, a, shape):
        return np.reshape(a, shape)

    def seed(self, seed=None):
        if seed is not None:
            self.rng_.seed(seed)

    def rand(self, *size, type_as=None):
        return self.rng_.rand(*size)

    def randn(self, *size, type_as=None):
        return self.rng_.randn(*size)

    def coo_matrix(self, data, rows, cols, shape=None, type_as=None):
        if type_as is None:
            return coo_matrix((data, (rows, cols)), shape=shape)
        else:
            return coo_matrix((data, (rows, cols)), shape=shape, dtype=type_as.dtype)

    def issparse(self, a):
        return issparse(a)

    def tocsr(self, a):
        if self.issparse(a):
            return a.tocsr()
        else:
            return csr_matrix(a)

    def eliminate_zeros(self, a, threshold=0.):
        if threshold > 0:
            if self.issparse(a):
                a.data[self.abs(a.data) <= threshold] = 0
            else:
                a[self.abs(a) <= threshold] = 0
        if self.issparse(a):
            a.eliminate_zeros()
        return a

    def todense(self, a):
        if self.issparse(a):
            return a.toarray()
        else:
            return a

    def where(self, condition, x, y):
        return np.where(condition, x, y)

    def copy(self, a):
        return a.copy()

    def allclose(self, a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
        return np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)

    def dtype_device(self, a):
        if hasattr(a, "dtype"):
            return a.dtype, "cpu"
        else:
            return type(a), "cpu"

    def assert_same_dtype_device(self, a, b):
        # numpy has implicit type conversion so we automatically validate the test
        pass


class JaxBackend(Backend):
    """
    JAX implementation of the backend

    - `__name__` is "jax"
    - `__type__` is jax.numpy.ndarray
    """

    __name__ = 'jax'
    __type__ = jax_type
    __type_list__ = None

    rng_ = None

    def __init__(self):
        self.rng_ = jax.random.PRNGKey(42)

        for d in jax.devices():
            self.__type_list__ = [jax.device_put(jnp.array(1, dtype=jnp.float32), d),
                                  jax.device_put(jnp.array(1, dtype=jnp.float64), d)]

    def to_numpy(self, a):
        return np.array(a)

    def _change_device(self, a, type_as):
        return jax.device_put(a, type_as.device_buffer.device())

    def from_numpy(self, a, type_as=None):
        if type_as is None:
            return jnp.array(a)
        else:
            return self._change_device(jnp.array(a).astype(type_as.dtype), type_as)

    def set_gradients(self, val, inputs, grads):
        from jax.flatten_util import ravel_pytree
        val, = jax.lax.stop_gradient((val,))

        ravelled_inputs, _ = ravel_pytree(inputs)
        ravelled_grads, _ = ravel_pytree(grads)

        aux = jnp.sum(ravelled_inputs * ravelled_grads) / 2
        aux = aux - jax.lax.stop_gradient(aux)

        val, = jax.tree_map(lambda z: z + aux, (val,))
        return val

    def zeros(self, shape, type_as=None):
        if type_as is None:
            return jnp.zeros(shape)
        else:
            return self._change_device(jnp.zeros(shape, dtype=type_as.dtype), type_as)

    def ones(self, shape, type_as=None):
        if type_as is None:
            return jnp.ones(shape)
        else:
            return self._change_device(jnp.ones(shape, dtype=type_as.dtype), type_as)

    def arange(self, stop, start=0, step=1, type_as=None):
        return jnp.arange(start, stop, step)

    def full(self, shape, fill_value, type_as=None):
        if type_as is None:
            return jnp.full(shape, fill_value)
        else:
            return self._change_device(jnp.full(shape, fill_value, dtype=type_as.dtype), type_as)

    def eye(self, N, M=None, type_as=None):
        if type_as is None:
            return jnp.eye(N, M)
        else:
            return self._change_device(jnp.eye(N, M, dtype=type_as.dtype), type_as)

    def sum(self, a, axis=None, keepdims=False):
        return jnp.sum(a, axis, keepdims=keepdims)

    def cumsum(self, a, axis=None):
        return jnp.cumsum(a, axis)

    def max(self, a, axis=None, keepdims=False):
        return jnp.max(a, axis, keepdims=keepdims)

    def min(self, a, axis=None, keepdims=False):
        return jnp.min(a, axis, keepdims=keepdims)

    def maximum(self, a, b):
        return jnp.maximum(a, b)

    def minimum(self, a, b):
        return jnp.minimum(a, b)

    def dot(self, a, b):
        return jnp.dot(a, b)

    def abs(self, a):
        return jnp.abs(a)

    def exp(self, a):
        return jnp.exp(a)

    def log(self, a):
        return jnp.log(a)

    def sqrt(self, a):
        return jnp.sqrt(a)

    def power(self, a, exponents):
        return jnp.power(a, exponents)

    def norm(self, a):
        return jnp.sqrt(jnp.sum(jnp.square(a)))

    def any(self, a):
        return jnp.any(a)

    def isnan(self, a):
        return jnp.isnan(a)

    def isinf(self, a):
        return jnp.isinf(a)

    def einsum(self, subscripts, *operands):
        return jnp.einsum(subscripts, *operands)

    def sort(self, a, axis=-1):
        return jnp.sort(a, axis)

    def argsort(self, a, axis=-1):
        return jnp.argsort(a, axis)

    def searchsorted(self, a, v, side='left'):
        if a.ndim == 1:
            return jnp.searchsorted(a, v, side)
        else:
            # this is a not very efficient way to make jax numpy
            # searchsorted work on 2d arrays
            return jnp.array([jnp.searchsorted(a[i, :], v[i, :], side) for i in range(a.shape[0])])

    def flip(self, a, axis=None):
        return jnp.flip(a, axis)

    def outer(self, a, b):
        return jnp.outer(a, b)

    def clip(self, a, a_min, a_max):
        return jnp.clip(a, a_min, a_max)

    def repeat(self, a, repeats, axis=None):
        return jnp.repeat(a, repeats, axis)

    def take_along_axis(self, arr, indices, axis):
        return jnp.take_along_axis(arr, indices, axis)

    def concatenate(self, arrays, axis=0):
        return jnp.concatenate(arrays, axis)

    def zero_pad(self, a, pad_width):
        return jnp.pad(a, pad_width)

    def argmax(self, a, axis=None):
        return jnp.argmax(a, axis=axis)

    def mean(self, a, axis=None):
        return jnp.mean(a, axis=axis)

    def std(self, a, axis=None):
        return jnp.std(a, axis=axis)

    def linspace(self, start, stop, num):
        return jnp.linspace(start, stop, num)

    def meshgrid(self, a, b):
        return jnp.meshgrid(a, b)

    def diag(self, a, k=0):
        return jnp.diag(a, k)

    def unique(self, a):
        return jnp.unique(a)

    def logsumexp(self, a, axis=None):
        return jscipy.logsumexp(a, axis=axis)

    def stack(self, arrays, axis=0):
        return jnp.stack(arrays, axis)

    def reshape(self, a, shape):
        return jnp.reshape(a, shape)

    def seed(self, seed=None):
        if seed is not None:
            self.rng_ = jax.random.PRNGKey(seed)

    def rand(self, *size, type_as=None):
        self.rng_, subkey = jax.random.split(self.rng_)
        if type_as is not None:
            return jax.random.uniform(subkey, shape=size, dtype=type_as.dtype)
        else:
            return jax.random.uniform(subkey, shape=size)

    def randn(self, *size, type_as=None):
        self.rng_, subkey = jax.random.split(self.rng_)
        if type_as is not None:
            return jax.random.normal(subkey, shape=size, dtype=type_as.dtype)
        else:
            return jax.random.normal(subkey, shape=size)

    def coo_matrix(self, data, rows, cols, shape=None, type_as=None):
        # Currently, JAX does not support sparse matrices
        data = self.to_numpy(data)
        rows = self.to_numpy(rows)
        cols = self.to_numpy(cols)
        nx = NumpyBackend()
        coo_matrix = nx.coo_matrix(data, rows, cols, shape=shape, type_as=type_as)
        matrix = nx.todense(coo_matrix)
        return self.from_numpy(matrix)

    def issparse(self, a):
        # Currently, JAX does not support sparse matrices
        return False

    def tocsr(self, a):
        # Currently, JAX does not support sparse matrices
        return a

    def eliminate_zeros(self, a, threshold=0.):
        # Currently, JAX does not support sparse matrices
        if threshold > 0:
            return self.where(
                self.abs(a) <= threshold,
                self.zeros((1,), type_as=a),
                a
            )
        return a

    def todense(self, a):
        # Currently, JAX does not support sparse matrices
        return a

    def where(self, condition, x, y):
        return jnp.where(condition, x, y)

    def copy(self, a):
        # No need to copy, JAX arrays are immutable
        return a

    def allclose(self, a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
        return jnp.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)

    def dtype_device(self, a):
        return a.dtype, a.device_buffer.device()

    def assert_same_dtype_device(self, a, b):
        a_dtype, a_device = self.dtype_device(a)
        b_dtype, b_device = self.dtype_device(b)

        assert a_dtype == b_dtype, "Dtype discrepancy"
        assert a_device == b_device, f"Device discrepancy. First input is on {str(a_device)}, whereas second input is on {str(b_device)}"


class TorchBackend(Backend):
    """
    PyTorch implementation of the backend

    - `__name__` is "torch"
    - `__type__` is torch.Tensor
    """

    __name__ = 'torch'
    __type__ = torch_type
    __type_list__ = None

    rng_ = None

    def __init__(self):

        self.rng_ = torch.Generator()
        self.rng_.seed()

        self.__type_list__ = [torch.tensor(1, dtype=torch.float32),
                              torch.tensor(1, dtype=torch.float64)]

        if torch.cuda.is_available():
            self.__type_list__.append(torch.tensor(1, dtype=torch.float32, device='cuda'))
            self.__type_list__.append(torch.tensor(1, dtype=torch.float64, device='cuda'))

        from torch.autograd import Function

        # define a function that takes inputs val and grads
        # ad returns a val tensor with proper gradients
        class ValFunction(Function):

            @staticmethod
            def forward(ctx, val, grads, *inputs):
                ctx.grads = grads
                return val

            @staticmethod
            def backward(ctx, grad_output):
                # the gradients are grad
                return (None, None) + ctx.grads

        self.ValFunction = ValFunction

    def to_numpy(self, a):
        return a.cpu().detach().numpy()

    def from_numpy(self, a, type_as=None):
        if isinstance(a, float):
            a = np.array(a)
        if type_as is None:
            return torch.from_numpy(a)
        else:
            return torch.as_tensor(a, dtype=type_as.dtype, device=type_as.device)

    def set_gradients(self, val, inputs, grads):

        Func = self.ValFunction()

        res = Func.apply(val, grads, *inputs)

        return res

    def zeros(self, shape, type_as=None):
        if isinstance(shape, int):
            shape = (shape,)
        if type_as is None:
            return torch.zeros(shape)
        else:
            return torch.zeros(shape, dtype=type_as.dtype, device=type_as.device)

    def ones(self, shape, type_as=None):
        if isinstance(shape, int):
            shape = (shape,)
        if type_as is None:
            return torch.ones(shape)
        else:
            return torch.ones(shape, dtype=type_as.dtype, device=type_as.device)

    def arange(self, stop, start=0, step=1, type_as=None):
        if type_as is None:
            return torch.arange(start, stop, step)
        else:
            return torch.arange(start, stop, step, device=type_as.device)

    def full(self, shape, fill_value, type_as=None):
        if isinstance(shape, int):
            shape = (shape,)
        if type_as is None:
            return torch.full(shape, fill_value)
        else:
            return torch.full(shape, fill_value, dtype=type_as.dtype, device=type_as.device)

    def eye(self, N, M=None, type_as=None):
        if M is None:
            M = N
        if type_as is None:
            return torch.eye(N, m=M)
        else:
            return torch.eye(N, m=M, dtype=type_as.dtype, device=type_as.device)

    def sum(self, a, axis=None, keepdims=False):
        if axis is None:
            return torch.sum(a)
        else:
            return torch.sum(a, axis, keepdim=keepdims)

    def cumsum(self, a, axis=None):
        if axis is None:
            return torch.cumsum(a.flatten(), 0)
        else:
            return torch.cumsum(a, axis)

    def max(self, a, axis=None, keepdims=False):
        if axis is None:
            return torch.max(a)
        else:
            return torch.max(a, axis, keepdim=keepdims)[0]

    def min(self, a, axis=None, keepdims=False):
        if axis is None:
            return torch.min(a)
        else:
            return torch.min(a, axis, keepdim=keepdims)[0]

    def maximum(self, a, b):
        if isinstance(a, int) or isinstance(a, float):
            a = torch.tensor([float(a)], dtype=b.dtype, device=b.device)
        if isinstance(b, int) or isinstance(b, float):
            b = torch.tensor([float(b)], dtype=a.dtype, device=a.device)
        if hasattr(torch, "maximum"):
            return torch.maximum(a, b)
        else:
            return torch.max(torch.stack(torch.broadcast_tensors(a, b)), axis=0)[0]

    def minimum(self, a, b):
        if isinstance(a, int) or isinstance(a, float):
            a = torch.tensor([float(a)], dtype=b.dtype, device=b.device)
        if isinstance(b, int) or isinstance(b, float):
            b = torch.tensor([float(b)], dtype=a.dtype, device=a.device)
        if hasattr(torch, "minimum"):
            return torch.minimum(a, b)
        else:
            return torch.min(torch.stack(torch.broadcast_tensors(a, b)), axis=0)[0]

    def dot(self, a, b):
        return torch.matmul(a, b)

    def abs(self, a):
        return torch.abs(a)

    def exp(self, a):
        return torch.exp(a)

    def log(self, a):
        return torch.log(a)

    def sqrt(self, a):
        return torch.sqrt(a)

    def power(self, a, exponents):
        return torch.pow(a, exponents)

    def norm(self, a):
        return torch.sqrt(torch.sum(torch.square(a)))

    def any(self, a):
        return torch.any(a)

    def isnan(self, a):
        return torch.isnan(a)

    def isinf(self, a):
        return torch.isinf(a)

    def einsum(self, subscripts, *operands):
        return torch.einsum(subscripts, *operands)

    def sort(self, a, axis=-1):
        sorted0, indices = torch.sort(a, dim=axis)
        return sorted0

    def argsort(self, a, axis=-1):
        sorted, indices = torch.sort(a, dim=axis)
        return indices

    def searchsorted(self, a, v, side='left'):
        right = (side != 'left')
        return torch.searchsorted(a, v, right=right)

    def flip(self, a, axis=None):
        if axis is None:
            return torch.flip(a, tuple(i for i in range(len(a.shape))))
        if isinstance(axis, int):
            return torch.flip(a, (axis,))
        else:
            return torch.flip(a, dims=axis)

    def outer(self, a, b):
        return torch.outer(a, b)

    def clip(self, a, a_min, a_max):
        return torch.clamp(a, a_min, a_max)

    def repeat(self, a, repeats, axis=None):
        return torch.repeat_interleave(a, repeats, dim=axis)

    def take_along_axis(self, arr, indices, axis):
        return torch.gather(arr, axis, indices)

    def concatenate(self, arrays, axis=0):
        return torch.cat(arrays, dim=axis)

    def zero_pad(self, a, pad_width):
        from torch.nn.functional import pad
        # pad_width is an array of ndim tuples indicating how many 0 before and after
        # we need to add. We first need to make it compliant with torch syntax, that
        # starts with the last dim, then second last, etc.
        how_pad = tuple(element for tupl in pad_width[::-1] for element in tupl)
        return pad(a, how_pad)

    def argmax(self, a, axis=None):
        return torch.argmax(a, dim=axis)

    def mean(self, a, axis=None):
        if axis is not None:
            return torch.mean(a, dim=axis)
        else:
            return torch.mean(a)

    def std(self, a, axis=None):
        if axis is not None:
            return torch.std(a, dim=axis, unbiased=False)
        else:
            return torch.std(a, unbiased=False)

    def linspace(self, start, stop, num):
        return torch.linspace(start, stop, num, dtype=torch.float64)

    def meshgrid(self, a, b):
        X, Y = torch.meshgrid(a, b)
        return X.T, Y.T

    def diag(self, a, k=0):
        return torch.diag(a, diagonal=k)

    def unique(self, a):
        return torch.unique(a)

    def logsumexp(self, a, axis=None):
        if axis is not None:
            return torch.logsumexp(a, dim=axis)
        else:
            return torch.logsumexp(a, dim=tuple(range(len(a.shape))))

    def stack(self, arrays, axis=0):
        return torch.stack(arrays, dim=axis)

    def reshape(self, a, shape):
        return torch.reshape(a, shape)

    def seed(self, seed=None):
        if isinstance(seed, int):
            self.rng_.manual_seed(seed)
        elif isinstance(seed, torch.Generator):
            self.rng_ = seed
        else:
            raise ValueError("Non compatible seed : {}".format(seed))

    def rand(self, *size, type_as=None):
        if type_as is not None:
            return torch.rand(size=size, generator=self.rng_, dtype=type_as.dtype, device=type_as.device)
        else:
            return torch.rand(size=size, generator=self.rng_)

    def randn(self, *size, type_as=None):
        if type_as is not None:
            return torch.randn(size=size, dtype=type_as.dtype, generator=self.rng_, device=type_as.device)
        else:
            return torch.randn(size=size, generator=self.rng_)

    def coo_matrix(self, data, rows, cols, shape=None, type_as=None):
        if type_as is None:
            return torch.sparse_coo_tensor(torch.stack([rows, cols]), data, size=shape)
        else:
            return torch.sparse_coo_tensor(
                torch.stack([rows, cols]), data, size=shape,
                dtype=type_as.dtype, device=type_as.device
            )

    def issparse(self, a):
        return getattr(a, "is_sparse", False) or getattr(a, "is_sparse_csr", False)

    def tocsr(self, a):
        # Versions older than 1.9 do not support CSR tensors. PyTorch 1.9 and 1.10 offer a very limited support
        return self.todense(a)

    def eliminate_zeros(self, a, threshold=0.):
        if self.issparse(a):
            if threshold > 0:
                mask = self.abs(a) <= threshold
                mask = ~mask
                mask = mask.nonzero()
            else:
                mask = a._values().nonzero()
            nv = a._values().index_select(0, mask.view(-1))
            ni = a._indices().index_select(1, mask.view(-1))
            return self.coo_matrix(nv, ni[0], ni[1], shape=a.shape, type_as=a)
        else:
            if threshold > 0:
                a[self.abs(a) <= threshold] = 0
            return a

    def todense(self, a):
        if self.issparse(a):
            return a.to_dense()
        else:
            return a

    def where(self, condition, x, y):
        return torch.where(condition, x, y)

    def copy(self, a):
        return torch.clone(a)

    def allclose(self, a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
        return torch.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)

    def dtype_device(self, a):
        return a.dtype, a.device

    def assert_same_dtype_device(self, a, b):
        a_dtype, a_device = self.dtype_device(a)
        b_dtype, b_device = self.dtype_device(b)

        assert a_dtype == b_dtype, "Dtype discrepancy"
        assert a_device == b_device, f"Device discrepancy. First input is on {str(a_device)}, whereas second input is on {str(b_device)}"
