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

    def flip(self, a, axis=None):
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

    def concatenate(self, list_a, axis=0):
        raise NotImplementedError()

    def logsumexp(self, a, axis=None):
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

    def flip(self, a, axis=None):
        return np.flip(a, axis)

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

    def concatenate(self, list_a, axis=0):
        return np.concatenate(list_a, axis=axis)

    def logsumexp(self, a, axis=None):
        return scipy.logsumexp(a, axis=axis)


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

    def flip(self, a, axis=None):
        return jnp.flip(a, axis)

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

    def concatenate(self, list_a, axis=0):
        return jnp.concatenate(list_a, axis=axis)

    def logsumexp(self, a, axis=None):
        return jscipy.logsumexp(a, axis=axis)


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
        return torch.maximum(a, b)

    def minimum(self, a, b):
        if isinstance(a, int) or isinstance(a, float):
            a = torch.tensor([float(a)], dtype=b.dtype, device=b.device)
        if isinstance(b, int) or isinstance(b, float):
            b = torch.tensor([float(b)], dtype=a.dtype, device=a.device)
        return torch.minimum(a, b)

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

    def flip(self, a, axis=None):
        if axis is None:
            return torch.flip(a, tuple(i for i in range(len(a.shape))))
        if isinstance(axis, int):
            return torch.flip(a, (axis,))
        else:
            return torch.flip(a, dims=axis)

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

    def concatenate(self, list_a, axis=0):
        return torch.cat(list_a, dim=axis)

    def logsumexp(self, a, axis=None):
        if axis is not None:
            return torch.logsumexp(a, dim=axis)
        else:
            return torch.logsumexp(a, dim=tuple(range(len(a.shape))))
