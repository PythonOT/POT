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
    from jax.config import config
    config.update("jax_enable_x64", True)
except ImportError:
    jax = False
    jax_type = float

str_type_error = "All array should be from the same type/backend. Current types are : {}"


def get_backend_list():
    """ returns the list of available backends)"""
    lst = [NumpyBackend(), ]

    if torch:
        lst.append(TorchBackend())

    if jax:
        lst.append(JaxBackend())

    return lst


def get_backend(*args):
    """returns the proper backend for a list of input arrays

        Also raises TypeError if all arrays are not from the same backend
    """
    # check that some arrays given
    if not len(args) > 0:
        raise ValueError(" The function takes at least one parameter")
    # check all same type

    if isinstance(args[0], np.ndarray):
        if not len(set(type(a) for a in args)) == 1:
            raise ValueError(str_type_error.format([type(a) for a in args]))
        return NumpyBackend()
    elif torch and isinstance(args[0], torch_type):
        if not len(set(type(a) for a in args)) == 1:
            raise ValueError(str_type_error.format([type(a) for a in args]))
        return TorchBackend()
    elif isinstance(args[0], jax_type):
        return JaxBackend()
    else:
        raise ValueError("Unknown type of non implemented backend.")


def to_numpy(*args):
    """returns numpy arrays from any compatible backend"""

    if len(args) == 1:
        return get_backend(args[0]).to_numpy(args[0])
    else:
        return [get_backend(a).to_numpy(a) for a in args]


class Backend():

    __name__ = None
    __type__ = None

    def __str__(self):
        return self.__name__

    # convert to numpy
    def to_numpy(self, a):
        raise NotImplementedError()

    # convert from numpy
    def from_numpy(self, a, type_as=None):
        raise NotImplementedError()

    def set_gradients(self, val, inputs, grads):
        """ define the gradients for the value val wrt the inputs """
        raise NotImplementedError()

    def zeros(self, shape, type_as=None):
        raise NotImplementedError()

    def ones(self, shape, type_as=None):
        raise NotImplementedError()

    def arange(self, stop, start=0, step=1, type_as=None):
        raise NotImplementedError()

    def full(self, shape, fill_value, type_as=None):
        raise NotImplementedError()

    def eye(self, N, M=None, type_as=None):
        raise NotImplementedError()

    def sum(self, a, axis=None, keepdims=False):
        raise NotImplementedError()

    def cumsum(self, a, axis=None):
        raise NotImplementedError()

    def max(self, a, axis=None, keepdims=False):
        raise NotImplementedError()

    def min(self, a, axis=None, keepdims=False):
        raise NotImplementedError()

    def maximum(self, a, b):
        raise NotImplementedError()

    def minimum(self, a, b):
        raise NotImplementedError()

    def dot(self, a, b):
        raise NotImplementedError()

    def abs(self, a):
        raise NotImplementedError()

    def exp(self, a):
        raise NotImplementedError()

    def log(self, a):
        raise NotImplementedError()

    def sqrt(self, a):
        raise NotImplementedError()

    def power(self, a, exponents):
        raise NotImplementedError()

    def norm(self, a):
        raise NotImplementedError()

    def any(self, a):
        raise NotImplementedError()

    def isnan(self, a):
        raise NotImplementedError()

    def isinf(self, a):
        raise NotImplementedError()

    def einsum(self, subscripts, *operands):
        raise NotImplementedError()

    def sort(self, a, axis=-1):
        raise NotImplementedError()

    def argsort(self, a, axis=None):
        raise NotImplementedError()

    def searchsorted(self, a, v, side='left'):
        raise NotImplementedError()

    def flip(self, a, axis=None):
        raise NotImplementedError()

    def clip(self, a, a_min, a_max):
        raise NotImplementedError()

    def repeat(self, a, repeats, axis=None):
        raise NotImplementedError()

    def take_along_axis(self, arr, indices, axis):
        raise NotImplementedError()

    def concatenate(self, arrays, axis=0):
        raise NotImplementedError()

    def zero_pad(self, a, pad_with):
        raise NotImplementedError()

    def argmax(self, a, axis=None):
        raise NotImplementedError()

    def mean(self, a, axis=None):
        raise NotImplementedError()

    def std(self, a, axis=None):
        raise NotImplementedError()

    def linspace(self, start, stop, num):
        raise NotImplementedError()

    def meshgrid(self, a, b):
        raise NotImplementedError()

    def diag(self, a, k=0):
        raise NotImplementedError()

    def unique(self, a):
        raise NotImplementedError()

    def logsumexp(self, a, axis=None):
        raise NotImplementedError()

    def stack(self, arrays, axis=0):
        raise NotImplementedError()


class NumpyBackend(Backend):

    __name__ = 'numpy'
    __type__ = np.ndarray

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
        # no gradients for numpy
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
            return np.array([np.searchsorted(a[i, :], v[i, :], side) for i in range(a.shape[0])])

    def flip(self, a, axis=None):
        return np.flip(a, axis)

    def clip(self, a, a_min, a_max):
        return np.clip(a, a_min, a_max)

    def repeat(self, a, repeats, axis=None):
        return np.repeat(a, repeats, axis)

    def take_along_axis(self, arr, indices, axis):
        return np.take_along_axis(arr, indices, axis)

    def concatenate(self, arrays, axis=0):
        return np.concatenate(arrays, axis)

    def zero_pad(self, a, pad_with):
        return np.pad(a, pad_with)

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


class JaxBackend(Backend):

    __name__ = 'jax'
    __type__ = jax_type

    def to_numpy(self, a):
        return np.array(a)

    def from_numpy(self, a, type_as=None):
        if type_as is None:
            return jnp.array(a)
        else:
            return jnp.array(a).astype(type_as.dtype)

    def set_gradients(self, val, inputs, grads):
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
        if type_as is None:
            return jnp.zeros(shape)
        else:
            return jnp.zeros(shape, dtype=type_as.dtype)

    def ones(self, shape, type_as=None):
        if type_as is None:
            return jnp.ones(shape)
        else:
            return jnp.ones(shape, dtype=type_as.dtype)

    def arange(self, stop, start=0, step=1, type_as=None):
        return jnp.arange(start, stop, step)

    def full(self, shape, fill_value, type_as=None):
        if type_as is None:
            return jnp.full(shape, fill_value)
        else:
            return jnp.full(shape, fill_value, dtype=type_as.dtype)

    def eye(self, N, M=None, type_as=None):
        if type_as is None:
            return jnp.eye(N, M)
        else:
            return jnp.eye(N, M, dtype=type_as.dtype)

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

    def clip(self, a, a_min, a_max):
        return jnp.clip(a, a_min, a_max)

    def repeat(self, a, repeats, axis=None):
        return jnp.repeat(a, repeats, axis)

    def take_along_axis(self, arr, indices, axis):
        return jnp.take_along_axis(arr, indices, axis)

    def concatenate(self, arrays, axis=0):
        return jnp.concatenate(arrays, axis)

    def zero_pad(self, a, pad_with):
        return jnp.pad(a, pad_with)

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


class TorchBackend(Backend):

    __name__ = 'torch'
    __type__ = torch_type

    def __init__(self):

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
        if type_as is None:
            return torch.from_numpy(a)
        else:
            return torch.as_tensor(a, dtype=type_as.dtype, device=type_as.device)

    def set_gradients(self, val, inputs, grads):

        Func = self.ValFunction()

        res = Func.apply(val, grads, *inputs)

        return res

    def zeros(self, shape, type_as=None):
        if type_as is None:
            return torch.zeros(shape)
        else:
            return torch.zeros(shape, dtype=type_as.dtype, device=type_as.device)

    def ones(self, shape, type_as=None):
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
        if torch.__version__ >= '1.7.0':
            return torch.maximum(a, b)
        else:
            return torch.max(torch.stack(torch.broadcast_tensors(a, b)), axis=0)[0]

    def minimum(self, a, b):
        if isinstance(a, int) or isinstance(a, float):
            a = torch.tensor([float(a)], dtype=b.dtype, device=b.device)
        if isinstance(b, int) or isinstance(b, float):
            b = torch.tensor([float(b)], dtype=a.dtype, device=a.device)
        if torch.__version__ >= '1.7.0':
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

    def clip(self, a, a_min, a_max):
        return torch.clamp(a, a_min, a_max)

    def repeat(self, a, repeats, axis=None):
        return torch.repeat_interleave(a, repeats, dim=axis)

    def take_along_axis(self, arr, indices, axis):
        return torch.gather(arr, axis, indices)

    def concatenate(self, arrays, axis=0):
        return torch.cat(arrays, dim=axis)

    def zero_pad(self, a, pad_with):
        from torch.nn.functional import pad
        # pad_with is an array of ndim tuples indicating how many 0 before and after
        # we need to add. We first need to make it compliant with torch syntax, that
        # starts with the last dim, then second last, etc.
        how_pad = tuple(element for tupl in pad_with[::-1] for element in tupl)
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
