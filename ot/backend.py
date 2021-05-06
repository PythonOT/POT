# -*- coding: utf-8 -*-
"""
Multi-lib backend for POT
"""

# Author: Remi Flamary <remi.flamary@polytechnique.edu>
#
# License: MIT License

import numpy as np

try:
    import torch
    torch_type = torch.Tensor
except ImportError:
    torch = False
    torch_type = float

try:
    import jax
    import jax.numpy as jnp
    jax_type = jax.numpy.ndarray
except ImportError:
    jax = False
    jax_type = float


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
            raise ValueError("All array should be from the same type/backend. Current types are : {}".format([type(a) for a in args]))
        return NumpyBackend()
    elif torch and isinstance(args[0], torch_type):
        if not len(set(type(a) for a in args)) == 1:
            raise ValueError("All array should be from the same type/backend. Current types are : {}".format([type(a) for a in args]))
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

    # convert from and to numpy
    def to_numpy(self, a):
        raise NotImplementedError()

    def from_numpy(self, a, type_as=None):
        raise NotImplementedError()

    def set_gradients(self, val, inputs, grads):
        """ define the gradients for the value val wrt the inputs """
        raise NotImplementedError()

    def zeros(self, shape, type_as=None):
        raise NotImplementedError()

    def ones(self, shape, type_as=None):
        raise NotImplementedError()

    def full(self, shape, fill_value, type_as=None):
        raise NotImplementedError()

    def sum(self, a, axis=None, keepdims=False):
        raise NotImplementedError()

    def max(self, a, axis=None, keepdims=False):
        raise NotImplementedError()

    def min(self, a, axis=None, keepdims=False):
        raise NotImplementedError()

    def dot(self, a, b):
        raise NotImplementedError()

    def abs(self, a):
        raise NotImplementedError()

    def exp(self, a):
        raise NotImplementedError()

    def log(self, a):
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

    def full(self, shape, fill_value, type_as=None):
        if type_as is None:
            return np.full(shape, fill_value)
        else:
            return np.full(shape, fill_value, dtype=type_as.dtype)

    def sum(self, a, axis=None, keepdims=False):
        return np.sum(a, axis, keepdims=keepdims)

    def max(self, a, axis=None, keepdims=False):
        return np.max(a, axis, keepdims=keepdims)

    def min(self, a, axis=None, keepdims=False):
        return np.min(a, axis, keepdims=keepdims)

    def dot(self, a, b):
        return np.dot(a, b)

    def abs(self, a):
        return np.abs(a)

    def exp(self, a):
        return np.exp(a)

    def log(self, a):
        return np.log(a)

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
        # no gradients for numpy

        from jax import custom_jvp

        @custom_jvp
        def f(*inputs):
            return val

        f.defjvps(*grads)

        return f(*inputs)

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

    def full(self, shape, fill_value, type_as=None):
        if type_as is None:
            return jnp.full(shape, fill_value)
        else:
            return jnp.full(shape, fill_value, dtype=type_as.dtype)

    def sum(self, a, axis=None, keepdims=False):
        return jnp.sum(a, axis, keepdims=keepdims)

    def max(self, a, axis=None, keepdims=False):
        return jnp.max(a, axis, keepdims=keepdims)

    def min(self, a, axis=None, keepdims=False):
        return jnp.min(a, axis, keepdims=keepdims)

    def dot(self, a, b):
        return jnp.dot(a, b)

    def abs(self, a):
        return jnp.abs(a)

    def exp(self, a):
        return jnp.exp(a)

    def log(self, a):
        return jnp.log(a)

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


class TorchBackend(Backend):

    __name__ = 'torch'
    __type__ = torch_type

    def to_numpy(self, a):
        return a.cpu().detach().numpy()

    def from_numpy(self, a, type_as=None):
        if type_as is None:
            return torch.from_numpy(a)
        else:
            return torch.as_tensor(a, dtype=type_as.dtype, device=type_as.device)

    def set_gradients(self, val, inputs, grads):
        from torch.autograd import Function

        # define a function that takes inputs and return val
        class ValFunction(Function):
            @staticmethod
            def forward(ctx, *inputs):
                return val

            @staticmethod
            def backward(ctx, grad_output):
                # the gradients are grad
                return grads

        return ValFunction.apply(*inputs)

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

    def full(self, shape, fill_value, type_as=None):
        if type_as is None:
            return torch.full(shape, fill_value)
        else:
            return torch.full(shape, fill_value, dtype=type_as.dtype, device=type_as.device)

    def sum(self, a, axis=None, keepdims=False):
        if axis is None:
            return torch.sum(a)
        else:
            return torch.sum(a, axis, keepdim=keepdims)

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

    def dot(self, a, b):
        if len(a.shape) == len(b.shape) == 1:
            return torch.dot(a, b)
        elif len(a.shape) == 2 and len(b.shape) == 1:
            return torch.mv(a, b)
        else:
            return torch.mm(a, b)

    def abs(self, a):
        return torch.abs(a)

    def exp(self, a):
        return torch.exp(a)

    def log(self, a):
        return torch.log(a)

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
