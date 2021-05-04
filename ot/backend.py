# -*- coding: utf-8 -*-
"""
Multi-lib backend for POT
"""

# Author: Remi Flamary <remi.flamary@polytechnique.edu>
#
# License: MIT License

import numpy as np
import scipy.spatial

try:
    import torch
except ImportError:
    torch = False


def get_backend_list():
    """ returns the list of available backends)"""
    lst = [NumpyBackend(), ]

    if torch:
        lst.append(TorchBackend())

    return lst


def get_backend(*args):
    """returns the proper backend for a list of input arrays

        Also raises TypeError if all arrays are not from the same backend
    """
    # check that some arrays given
    if not len(args) > 0:
        raise ValueError(" The function takes at least one parameter")
    # check all same type
    if not len(set(type(a) for a in args)) == 1:
        raise ValueError("All array should be from the same type/backend. Current types are : {}".format(args))
    if isinstance(args[0], np.ndarray):
        return NumpyBackend()
    elif torch and isinstance(args[0], torch.Tensor):
        return torch
    #elif isinstance(args[0], jax.numpy.ndarray):
    #    return jax.numpy
    else:
        raise ValueError("Unknown type of non implemented backend.")


def to_numpy(*lst):
    """returns numpy arrays from any compatible backend"""

    return (get_backend(a).to_numpy(a) for a in lst)


class Backend():

    __name__ = 'defautl'

    # convert from and to numpy
    def to_numpy(self, a):
        raise NotImplementedError()

    def from_numpy(self, a, type_as=None):
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


class NumpyBackend(Backend):

    __name__ = 'numpy'

    def to_numpy(self, a):
        return a

    def from_numpy(self, a, type_as=None):
        if type_as is None:
            return a
        else:
            return a.astype(type_as.dtype)

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
        return np.sum(a, axis, keepdims)

    def max(self, a, axis=None, keepdims=False):
        return np.max(a, axis, keepdims)

    def min(self, a, axis=None, keepdims=False):
        return np.min(a, axis, keepdims)

    def dot(self, a, b):
        return np.dot(a, b)

    def abs(self, a):
        return np.abs(a)

    def exp(self, a):
        return np.exp(a)

    def log(self, a):
        return np.log(a)


class TorchBackend(Backend):

    __name__ = 'torch'

    def to_numpy(self, a):
        return a.cpu().numpy()

    def from_numpy(self, a, type_as=None):
        if type_as is None:
            return torch.from_numpy(a)
        else:
            return torch.as_tensor(a, dtype=type_as.dtype, device=type_as.device)

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
        return torch.sum(a, axis, keepdims)

    def max(self, a, axis=None, keepdims=False):
        return torch.max(a, axis, keepdims)

    def min(self, a, axis=None, keepdims=False):
        return torch.min(a, axis, keepdims)

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
