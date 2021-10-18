# -*- coding: utf-8 -*-
"""
Multi-lib backend for POT
"""

# Author: Remi Flamary <remi.flamary@polytechnique.edu>
#         Nicolas Courty <ncourty@irisa.fr>
#
# License: MIT License

import numpy as np
import scipy.special as scipy

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

        Numpy equivalent: https://numpy.org/doc/stable/reference/generated/numpy.zeros.html
        """
        raise NotImplementedError()

    def ones(self, shape, type_as=None):
        r"""
        Creates a tensor full of ones.

        Numpy equivalent: https://numpy.org/doc/stable/reference/generated/numpy.ones.html
        """
        raise NotImplementedError()

    def arange(self, stop, start=0, step=1, type_as=None):
        r"""
        Returns evenly spaced values within a given interval.

        Numpy equivalent: https://numpy.org/doc/stable/reference/generated/numpy.arange.html
        """
        raise NotImplementedError()

    def full(self, shape, fill_value, type_as=None):
        r"""
        Creates a tensor with given shape, filled with given value.

        Numpy equivalent: https://numpy.org/doc/stable/reference/generated/numpy.full.html
        """
        raise NotImplementedError()

    def eye(self, N, M=None, type_as=None):
        r"""
        Creates the identity matrix of given size.

        Numpy equivalent: https://numpy.org/doc/stable/reference/generated/numpy.eye.html
        """
        raise NotImplementedError()

    def sum(self, a, axis=None, keepdims=False):
        r"""
        Sums tensor elements over given dimensions.

        Numpy equivalent: https://numpy.org/doc/stable/reference/generated/numpy.sum.html
        """
        raise NotImplementedError()

    def cumsum(self, a, axis=None):
        r"""
        Returns the cumulative sum of tensor elements over given dimensions.

        Numpy equivalent: https://numpy.org/doc/stable/reference/generated/numpy.cumsum.html
        """
        raise NotImplementedError()

    def max(self, a, axis=None, keepdims=False):
        r"""
        Returns the maximum of an array or maximum along given dimensions.

        Numpy equivalent: https://numpy.org/doc/stable/reference/generated/numpy.amax.html
        """
        raise NotImplementedError()

    def min(self, a, axis=None, keepdims=False):
        r"""
        Returns the maximum of an array or maximum along given dimensions.

        Numpy equivalent: https://numpy.org/doc/stable/reference/generated/numpy.amin.html
        """
        raise NotImplementedError()

    def maximum(self, a, b):
        r"""
        Returns element-wise maximum of array elements.

        Numpy equivalent: https://numpy.org/doc/stable/reference/generated/numpy.maximum.html
        """
        raise NotImplementedError()

    def minimum(self, a, b):
        r"""
        Returns element-wise minimum of array elements.

        Numpy equivalent: https://numpy.org/doc/stable/reference/generated/numpy.minimum.html
        """
        raise NotImplementedError()

    def dot(self, a, b):
        r"""
        Returns the dot product of two tensors.

        Numpy equivalent: https://numpy.org/doc/stable/reference/generated/numpy.dot.html
        """
        raise NotImplementedError()

    def abs(self, a):
        r"""
        Computes the absolute value element-wise.

        Numpy equivalent: https://numpy.org/doc/stable/reference/generated/numpy.absolute.html
        """
        raise NotImplementedError()

    def exp(self, a):
        r"""
        Computes the exponential value element-wise.

        Numpy equivalent: https://numpy.org/doc/stable/reference/generated/numpy.exp.html
        """
        raise NotImplementedError()

    def log(self, a):
        r"""
        Computes the natural logarithm, element-wise.

        Numpy equivalent: https://numpy.org/doc/stable/reference/generated/numpy.log.html
        """
        raise NotImplementedError()

    def sqrt(self, a):
        r"""
        Returns the non-ngeative square root of a tensor, element-wise.

        Numpy equivalent: https://numpy.org/doc/stable/reference/generated/numpy.sqrt.html
        """
        raise NotImplementedError()

    def power(self, a, exponents):
        r"""
        First tensor elements raised to powers from second tensor, element-wise.

        Numpy equivalent: https://numpy.org/doc/stable/reference/generated/numpy.sqrt.html
        """
        raise NotImplementedError()

    def norm(self, a):
        r"""
        Computes the matrix frobenius norm.

        Numpy equivalent: https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html
        """
        raise NotImplementedError()

    def any(self, a):
        r"""
        Tests whether any tensor element along given dimensions evaluates to True.

        Numpy equivalent: https://numpy.org/doc/stable/reference/generated/numpy.any.html
        """
        raise NotImplementedError()

    def isnan(self, a):
        r"""
        Tests element-wise for NaN and returns result as a boolean tensor.

        Numpy equivalent: https://numpy.org/doc/stable/reference/generated/numpy.isnan.html
        """
        raise NotImplementedError()

    def isinf(self, a):
        r"""
        Tests element-wise for positive or negative infinity and returns result as a boolean tensor.

        Numpy equivalent: https://numpy.org/doc/stable/reference/generated/numpy.isinf.html
        """
        raise NotImplementedError()

    def einsum(self, subscripts, *operands):
        r"""
        Evaluates the Einstein summation convention on the operands.

        Numpy equivalent: https://numpy.org/doc/stable/reference/generated/numpy.einsum.html
        """
        raise NotImplementedError()

    def sort(self, a, axis=-1):
        r"""
        Returns a sorted copy of a tensor.

        Numpy equivalent: https://numpy.org/doc/stable/reference/generated/numpy.sort.html
        """
        raise NotImplementedError()

    def argsort(self, a, axis=None):
        r"""
        Returns the indices that would sort a tensor.

        Numpy equivalent: https://numpy.org/doc/stable/reference/generated/numpy.argsort.html
        """
        raise NotImplementedError()

    def searchsorted(self, a, v, side='left'):
        r"""
        Finds indices where elements should be inserted to maintain order in given tensor.

        Numpy equivalent: https://numpy.org/doc/stable/reference/generated/numpy.searchsorted.html
        """
        raise NotImplementedError()

    def flip(self, a, axis=None):
        r"""
        Reverses the order of elements in a tensor along given dimensions.

        Numpy equivalent: https://numpy.org/doc/stable/reference/generated/numpy.flip.html
        """
        raise NotImplementedError()

    def clip(self, a, a_min, a_max):
        """
        Limits the values in a tensor.

        Numpy equivalent: https://numpy.org/doc/stable/reference/generated/numpy.clip.html
        """
        raise NotImplementedError()

    def repeat(self, a, repeats, axis=None):
        r"""
        Repeats elements of a tensor.

        Numpy equivalent: https://numpy.org/doc/stable/reference/generated/numpy.repeat.html
        """
        raise NotImplementedError()

    def take_along_axis(self, arr, indices, axis):
        r"""
        Gathers elements of a tensor along given dimensions.

        Numpy equivalent: https://numpy.org/doc/stable/reference/generated/numpy.take_along_axis.html
        """
        raise NotImplementedError()

    def concatenate(self, arrays, axis=0):
        r"""
        Joins a sequence of tensors along an existing dimension.

        Numpy equivalent: https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html
        """
        raise NotImplementedError()

    def zero_pad(self, a, pad_width):
        r"""
        Pads a tensor.

        Numpy equivalent: https://numpy.org/doc/stable/reference/generated/numpy.pad.html
        """
        raise NotImplementedError()

    def argmax(self, a, axis=None):
        r"""
        Returns the indices of the maximum values of a tensor along given dimensions.

        Numpy equivalent: https://numpy.org/doc/stable/reference/generated/numpy.argmax.html
        """
        raise NotImplementedError()

    def mean(self, a, axis=None):
        r"""
        Computes the arithmetic mean of a tensor along given dimensions.

        Numpy equivalent: https://numpy.org/doc/stable/reference/generated/numpy.mean.html
        """
        raise NotImplementedError()

    def std(self, a, axis=None):
        r"""
        Computes the standard deviation of a tensor along given dimensions.

        Numpy equivalent: https://numpy.org/doc/stable/reference/generated/numpy.std.html
        """
        raise NotImplementedError()

    def linspace(self, start, stop, num):
        r"""
        Returns a specified number of evenly spaced values over a given interval.

        Numpy equivalent: https://numpy.org/doc/stable/reference/generated/numpy.linspace.html
        """
        raise NotImplementedError()

    def meshgrid(self, a, b):
        r"""
        Returns coordinate matrices from coordinate vectors (Numpy convention).

        Numpy equivalent: https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html
        """
        raise NotImplementedError()

    def diag(self, a, k=0):
        r"""
        Extracts or constructs a diagonal tensor.

        Numpy equivalent: https://numpy.org/doc/stable/reference/generated/numpy.diag.html
        """
        raise NotImplementedError()

    def unique(self, a):
        r"""
        Finds unique elements of given tensor.

        Numpy equivalent: https://numpy.org/doc/stable/reference/generated/numpy.unique.html
        """
        raise NotImplementedError()

    def logsumexp(self, a, axis=None):
        r"""
        Computes the log of the sum of exponentials of input elements.

        Numpy equivalent: https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.logsumexp.html
        """
        raise NotImplementedError()

    def stack(self, arrays, axis=0):
        r"""
        Joins a sequence of tensors along a new dimension.

        Numpy equivalent: https://numpy.org/doc/stable/reference/generated/numpy.stack.html
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

    def to_numpy(self, a):
        """:meta private:"""
        return a

    def from_numpy(self, a, type_as=None):
        """:meta private:"""
        if type_as is None:
            return a
        elif isinstance(a, float):
            return a
        else:
            return a.astype(type_as.dtype)

    def set_gradients(self, val, inputs, grads):
        """:meta private:
        No gradients for numpy
        """
        return val

    def zeros(self, shape, type_as=None):
        """:meta private:"""
        if type_as is None:
            return np.zeros(shape)
        else:
            return np.zeros(shape, dtype=type_as.dtype)

    def ones(self, shape, type_as=None):
        """:meta private:"""
        if type_as is None:
            return np.ones(shape)
        else:
            return np.ones(shape, dtype=type_as.dtype)

    def arange(self, stop, start=0, step=1, type_as=None):
        """:meta private:"""
        return np.arange(start, stop, step)

    def full(self, shape, fill_value, type_as=None):
        """:meta private:"""
        if type_as is None:
            return np.full(shape, fill_value)
        else:
            return np.full(shape, fill_value, dtype=type_as.dtype)

    def eye(self, N, M=None, type_as=None):
        """:meta private:"""
        if type_as is None:
            return np.eye(N, M)
        else:
            return np.eye(N, M, dtype=type_as.dtype)

    def sum(self, a, axis=None, keepdims=False):
        """:meta private:"""
        return np.sum(a, axis, keepdims=keepdims)

    def cumsum(self, a, axis=None):
        """:meta private:"""
        return np.cumsum(a, axis)

    def max(self, a, axis=None, keepdims=False):
        """:meta private:"""
        return np.max(a, axis, keepdims=keepdims)

    def min(self, a, axis=None, keepdims=False):
        """:meta private:"""
        return np.min(a, axis, keepdims=keepdims)

    def maximum(self, a, b):
        """:meta private:"""
        return np.maximum(a, b)

    def minimum(self, a, b):
        """:meta private:"""
        return np.minimum(a, b)

    def dot(self, a, b):
        """:meta private:"""
        return np.dot(a, b)

    def abs(self, a):
        """:meta private:"""
        return np.abs(a)

    def exp(self, a):
        """:meta private:"""
        return np.exp(a)

    def log(self, a):
        """:meta private:"""
        return np.log(a)

    def sqrt(self, a):
        """:meta private:"""
        return np.sqrt(a)

    def power(self, a, exponents):
        """:meta private:"""
        return np.power(a, exponents)

    def norm(self, a):
        """:meta private:"""
        return np.sqrt(np.sum(np.square(a)))

    def any(self, a):
        """:meta private:"""
        return np.any(a)

    def isnan(self, a):
        """:meta private:"""
        return np.isnan(a)

    def isinf(self, a):
        """:meta private:"""
        return np.isinf(a)

    def einsum(self, subscripts, *operands):
        """:meta private:"""
        return np.einsum(subscripts, *operands)

    def sort(self, a, axis=-1):
        """:meta private:"""
        return np.sort(a, axis)

    def argsort(self, a, axis=-1):
        """:meta private:"""
        return np.argsort(a, axis)

    def searchsorted(self, a, v, side='left'):
        """:meta private:"""
        if a.ndim == 1:
            return np.searchsorted(a, v, side)
        else:
            # this is a not very efficient way to make numpy
            # searchsorted work on 2d arrays
            return np.array([np.searchsorted(a[i, :], v[i, :], side) for i in range(a.shape[0])])

    def flip(self, a, axis=None):
        """:meta private:"""
        return np.flip(a, axis)

    def clip(self, a, a_min, a_max):
        """:meta private:"""
        return np.clip(a, a_min, a_max)

    def repeat(self, a, repeats, axis=None):
        """:meta private:"""
        return np.repeat(a, repeats, axis)

    def take_along_axis(self, arr, indices, axis):
        """:meta private:"""
        return np.take_along_axis(arr, indices, axis)

    def concatenate(self, arrays, axis=0):
        """:meta private:"""
        return np.concatenate(arrays, axis)

    def zero_pad(self, a, pad_width):
        """:meta private:"""
        return np.pad(a, pad_width)

    def argmax(self, a, axis=None):
        """:meta private:"""
        return np.argmax(a, axis=axis)

    def mean(self, a, axis=None):
        """:meta private:"""
        return np.mean(a, axis=axis)

    def std(self, a, axis=None):
        """:meta private:"""
        return np.std(a, axis=axis)

    def linspace(self, start, stop, num):
        """:meta private:"""
        return np.linspace(start, stop, num)

    def meshgrid(self, a, b):
        """:meta private:"""
        return np.meshgrid(a, b)

    def diag(self, a, k=0):
        """:meta private:"""
        return np.diag(a, k)

    def unique(self, a):
        """:meta private:"""
        return np.unique(a)

    def logsumexp(self, a, axis=None):
        """:meta private:"""
        return scipy.logsumexp(a, axis=axis)

    def stack(self, arrays, axis=0):
        """:meta private:"""
        return np.stack(arrays, axis)


class JaxBackend(Backend):
    """
    JAX implementation of the backend

    - `__name__` is "jax"
    - `__type__` is jax.numpy.ndarray
    """

    __name__ = 'jax'
    __type__ = jax_type

    def to_numpy(self, a):
        """:meta private:"""
        return np.array(a)

    def from_numpy(self, a, type_as=None):
        """:meta private:"""
        if type_as is None:
            return jnp.array(a)
        else:
            return jnp.array(a).astype(type_as.dtype)

    def set_gradients(self, val, inputs, grads):
        """:meta private:"""
        # no gradients for jax because it is functional

        # does not work
        # from jax import custom_jvp
        # @custom_jvp
        # def f(*inputs):
        #     return val
        # f.defjvps(*grads)
        # return f(*inputs)

        return val

    def zeros(self, shape, type_as=None):
        """:meta private:"""
        if type_as is None:
            return jnp.zeros(shape)
        else:
            return jnp.zeros(shape, dtype=type_as.dtype)

    def ones(self, shape, type_as=None):
        """:meta private:"""
        if type_as is None:
            return jnp.ones(shape)
        else:
            return jnp.ones(shape, dtype=type_as.dtype)

    def arange(self, stop, start=0, step=1, type_as=None):
        """:meta private:"""
        return jnp.arange(start, stop, step)

    def full(self, shape, fill_value, type_as=None):
        """:meta private:"""
        if type_as is None:
            return jnp.full(shape, fill_value)
        else:
            return jnp.full(shape, fill_value, dtype=type_as.dtype)

    def eye(self, N, M=None, type_as=None):
        """:meta private:"""
        if type_as is None:
            return jnp.eye(N, M)
        else:
            return jnp.eye(N, M, dtype=type_as.dtype)

    def sum(self, a, axis=None, keepdims=False):
        """:meta private:"""
        return jnp.sum(a, axis, keepdims=keepdims)

    def cumsum(self, a, axis=None):
        """:meta private:"""
        return jnp.cumsum(a, axis)

    def max(self, a, axis=None, keepdims=False):
        """:meta private:"""
        return jnp.max(a, axis, keepdims=keepdims)

    def min(self, a, axis=None, keepdims=False):
        """:meta private:"""
        return jnp.min(a, axis, keepdims=keepdims)

    def maximum(self, a, b):
        """:meta private:"""
        return jnp.maximum(a, b)

    def minimum(self, a, b):
        """:meta private:"""
        return jnp.minimum(a, b)

    def dot(self, a, b):
        """:meta private:"""
        return jnp.dot(a, b)

    def abs(self, a):
        """:meta private:"""
        return jnp.abs(a)

    def exp(self, a):
        """:meta private:"""
        return jnp.exp(a)

    def log(self, a):
        """:meta private:"""
        return jnp.log(a)

    def sqrt(self, a):
        """:meta private:"""
        return jnp.sqrt(a)

    def power(self, a, exponents):
        """:meta private:"""
        return jnp.power(a, exponents)

    def norm(self, a):
        """:meta private:"""
        return jnp.sqrt(jnp.sum(jnp.square(a)))

    def any(self, a):
        """:meta private:"""
        return jnp.any(a)

    def isnan(self, a):
        """:meta private:"""
        return jnp.isnan(a)

    def isinf(self, a):
        """:meta private:"""
        return jnp.isinf(a)

    def einsum(self, subscripts, *operands):
        """:meta private:"""
        return jnp.einsum(subscripts, *operands)

    def sort(self, a, axis=-1):
        """:meta private:"""
        return jnp.sort(a, axis)

    def argsort(self, a, axis=-1):
        """:meta private:"""
        return jnp.argsort(a, axis)

    def searchsorted(self, a, v, side='left'):
        """:meta private:"""
        if a.ndim == 1:
            return jnp.searchsorted(a, v, side)
        else:
            # this is a not very efficient way to make jax numpy
            # searchsorted work on 2d arrays
            return jnp.array([jnp.searchsorted(a[i, :], v[i, :], side) for i in range(a.shape[0])])

    def flip(self, a, axis=None):
        """:meta private:"""
        return jnp.flip(a, axis)

    def clip(self, a, a_min, a_max):
        """:meta private:"""
        return jnp.clip(a, a_min, a_max)

    def repeat(self, a, repeats, axis=None):
        """:meta private:"""
        return jnp.repeat(a, repeats, axis)

    def take_along_axis(self, arr, indices, axis):
        """:meta private:"""
        return jnp.take_along_axis(arr, indices, axis)

    def concatenate(self, arrays, axis=0):
        """:meta private:"""
        return jnp.concatenate(arrays, axis)

    def zero_pad(self, a, pad_width):
        """:meta private:"""
        return jnp.pad(a, pad_width)

    def argmax(self, a, axis=None):
        """:meta private:"""
        return jnp.argmax(a, axis=axis)

    def mean(self, a, axis=None):
        """:meta private:"""
        return jnp.mean(a, axis=axis)

    def std(self, a, axis=None):
        """:meta private:"""
        return jnp.std(a, axis=axis)

    def linspace(self, start, stop, num):
        """:meta private:"""
        return jnp.linspace(start, stop, num)

    def meshgrid(self, a, b):
        """:meta private:"""
        return jnp.meshgrid(a, b)

    def diag(self, a, k=0):
        """:meta private:"""
        return jnp.diag(a, k)

    def unique(self, a):
        """:meta private:"""
        return jnp.unique(a)

    def logsumexp(self, a, axis=None):
        """:meta private:"""
        return jscipy.logsumexp(a, axis=axis)

    def stack(self, arrays, axis=0):
        """:meta private:"""
        return jnp.stack(arrays, axis)


class TorchBackend(Backend):
    """
    PyTorch implementation of the backend

    - `__name__` is "torch"
    - `__type__` is torch.Tensor
    """

    __name__ = 'torch'
    __type__ = torch_type

    def __init__(self):
        """:meta private:"""

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
        """:meta private:"""
        return a.cpu().detach().numpy()

    def from_numpy(self, a, type_as=None):
        """:meta private:"""
        if type_as is None:
            return torch.from_numpy(a)
        else:
            return torch.as_tensor(a, dtype=type_as.dtype, device=type_as.device)

    def set_gradients(self, val, inputs, grads):
        """:meta private:"""

        Func = self.ValFunction()

        res = Func.apply(val, grads, *inputs)

        return res

    def zeros(self, shape, type_as=None):
        """:meta private:"""
        if type_as is None:
            return torch.zeros(shape)
        else:
            return torch.zeros(shape, dtype=type_as.dtype, device=type_as.device)

    def ones(self, shape, type_as=None):
        """:meta private:"""
        if type_as is None:
            return torch.ones(shape)
        else:
            return torch.ones(shape, dtype=type_as.dtype, device=type_as.device)

    def arange(self, stop, start=0, step=1, type_as=None):
        """:meta private:"""
        if type_as is None:
            return torch.arange(start, stop, step)
        else:
            return torch.arange(start, stop, step, device=type_as.device)

    def full(self, shape, fill_value, type_as=None):
        """:meta private:"""
        if type_as is None:
            return torch.full(shape, fill_value)
        else:
            return torch.full(shape, fill_value, dtype=type_as.dtype, device=type_as.device)

    def eye(self, N, M=None, type_as=None):
        """:meta private:"""
        if M is None:
            M = N
        if type_as is None:
            return torch.eye(N, m=M)
        else:
            return torch.eye(N, m=M, dtype=type_as.dtype, device=type_as.device)

    def sum(self, a, axis=None, keepdims=False):
        """:meta private:"""
        if axis is None:
            return torch.sum(a)
        else:
            return torch.sum(a, axis, keepdim=keepdims)

    def cumsum(self, a, axis=None):
        """:meta private:"""
        if axis is None:
            return torch.cumsum(a.flatten(), 0)
        else:
            return torch.cumsum(a, axis)

    def max(self, a, axis=None, keepdims=False):
        """:meta private:"""
        if axis is None:
            return torch.max(a)
        else:
            return torch.max(a, axis, keepdim=keepdims)[0]

    def min(self, a, axis=None, keepdims=False):
        """:meta private:"""
        if axis is None:
            return torch.min(a)
        else:
            return torch.min(a, axis, keepdim=keepdims)[0]

    def maximum(self, a, b):
        """:meta private:"""
        if isinstance(a, int) or isinstance(a, float):
            a = torch.tensor([float(a)], dtype=b.dtype, device=b.device)
        if isinstance(b, int) or isinstance(b, float):
            b = torch.tensor([float(b)], dtype=a.dtype, device=a.device)
        if torch.__version__ >= '1.7.0':
            return torch.maximum(a, b)
        else:
            return torch.max(torch.stack(torch.broadcast_tensors(a, b)), axis=0)[0]

    def minimum(self, a, b):
        """:meta private:"""
        if isinstance(a, int) or isinstance(a, float):
            a = torch.tensor([float(a)], dtype=b.dtype, device=b.device)
        if isinstance(b, int) or isinstance(b, float):
            b = torch.tensor([float(b)], dtype=a.dtype, device=a.device)
        if torch.__version__ >= '1.7.0':
            return torch.minimum(a, b)
        else:
            return torch.min(torch.stack(torch.broadcast_tensors(a, b)), axis=0)[0]

    def dot(self, a, b):
        """:meta private:"""
        return torch.matmul(a, b)

    def abs(self, a):
        """:meta private:"""
        return torch.abs(a)

    def exp(self, a):
        """:meta private:"""
        return torch.exp(a)

    def log(self, a):
        """:meta private:"""
        return torch.log(a)

    def sqrt(self, a):
        """:meta private:"""
        return torch.sqrt(a)

    def power(self, a, exponents):
        """:meta private:"""
        return torch.pow(a, exponents)

    def norm(self, a):
        """:meta private:"""
        return torch.sqrt(torch.sum(torch.square(a)))

    def any(self, a):
        """:meta private:"""
        return torch.any(a)

    def isnan(self, a):
        """:meta private:"""
        return torch.isnan(a)

    def isinf(self, a):
        """:meta private:"""
        return torch.isinf(a)

    def einsum(self, subscripts, *operands):
        """:meta private:"""
        return torch.einsum(subscripts, *operands)

    def sort(self, a, axis=-1):
        """:meta private:"""
        sorted0, indices = torch.sort(a, dim=axis)
        return sorted0

    def argsort(self, a, axis=-1):
        """:meta private:"""
        sorted, indices = torch.sort(a, dim=axis)
        return indices

    def searchsorted(self, a, v, side='left'):
        """:meta private:"""
        right = (side != 'left')
        return torch.searchsorted(a, v, right=right)

    def flip(self, a, axis=None):
        """:meta private:"""
        if axis is None:
            return torch.flip(a, tuple(i for i in range(len(a.shape))))
        if isinstance(axis, int):
            return torch.flip(a, (axis,))
        else:
            return torch.flip(a, dims=axis)

    def clip(self, a, a_min, a_max):
        """:meta private:"""
        return torch.clamp(a, a_min, a_max)

    def repeat(self, a, repeats, axis=None):
        """:meta private:"""
        return torch.repeat_interleave(a, repeats, dim=axis)

    def take_along_axis(self, arr, indices, axis):
        """:meta private:"""
        return torch.gather(arr, axis, indices)

    def concatenate(self, arrays, axis=0):
        """:meta private:"""
        return torch.cat(arrays, dim=axis)

    def zero_pad(self, a, pad_width):
        """:meta private:"""
        from torch.nn.functional import pad
        # pad_width is an array of ndim tuples indicating how many 0 before and after
        # we need to add. We first need to make it compliant with torch syntax, that
        # starts with the last dim, then second last, etc.
        how_pad = tuple(element for tupl in pad_width[::-1] for element in tupl)
        return pad(a, how_pad)

    def argmax(self, a, axis=None):
        """:meta private:"""
        return torch.argmax(a, dim=axis)

    def mean(self, a, axis=None):
        """:meta private:"""
        if axis is not None:
            return torch.mean(a, dim=axis)
        else:
            return torch.mean(a)

    def std(self, a, axis=None):
        """:meta private:"""
        if axis is not None:
            return torch.std(a, dim=axis, unbiased=False)
        else:
            return torch.std(a, unbiased=False)

    def linspace(self, start, stop, num):
        """:meta private:"""
        return torch.linspace(start, stop, num, dtype=torch.float64)

    def meshgrid(self, a, b):
        """:meta private:"""
        X, Y = torch.meshgrid(a, b)
        return X.T, Y.T

    def diag(self, a, k=0):
        """:meta private:"""
        return torch.diag(a, diagonal=k)

    def unique(self, a):
        """:meta private:"""
        return torch.unique(a)

    def logsumexp(self, a, axis=None):
        """:meta private:"""
        if axis is not None:
            return torch.logsumexp(a, dim=axis)
        else:
            return torch.logsumexp(a, dim=tuple(range(len(a.shape))))

    def stack(self, arrays, axis=0):
        """:meta private:"""
        return torch.stack(arrays, dim=axis)
