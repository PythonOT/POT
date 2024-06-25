# -*- coding: utf-8 -*-
"""
Multi-lib backend for POT

The goal is to write backend-agnostic code. Whether you're using Numpy, PyTorch,
Jax, Cupy, or Tensorflow, POT code should work nonetheless.
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

.. warning::
    Tensorflow only works with the Numpy API. To activate it, please run the following:

    .. code-block::

        from tensorflow.python.ops.numpy_ops import np_config
        np_config.enable_numpy_behavior()

Performance
-----------

- CPU: Intel(R) Xeon(R) Gold 6248 CPU @ 2.50GHz
- GPU: Tesla V100-SXM2-32GB
- Date of the benchmark: December 8th, 2021
- Commit of benchmark: PR #316, https://github.com/PythonOT/POT/pull/316

.. raw:: html

    <style>
    #perftable {
        width: 100%;
        margin-bottom: 1em;
    }

    #perftable table{
        border-collapse: collapse;
        table-layout: fixed;
        width: 100%;
    }

    #perftable th, #perftable td {
        border: 1px solid #ddd;
        padding: 8px;
        font-size: smaller;
    }
    </style>

    <div id="perftable">
    <table>
    <tr><th align="center" colspan="8">Sinkhorn Knopp - Averaged on 100 runs</th></tr>
    <tr><th align="center">Bitsize</th><th align="center" colspan="7">32 bits</th></tr>
    <tr><th align="center">Device</th><th align="center" colspan="3.0"">CPU</th><th align="center" colspan="4.0">GPU</tr>
    <tr><th align="center">Sample size</th><th align="center">Numpy</th><th align="center">Pytorch</th><th align="center">Tensorflow</th><th align="center">Cupy</th><th align="center">Jax</th><th align="center">Pytorch</th><th align="center">Tensorflow</th></tr>
    <tr><td align="center">50</td><td align="center">0.0008</td><td align="center">0.0022</td><td align="center">0.0151</td><td align="center">0.0095</td><td align="center">0.0193</td><td align="center">0.0051</td><td align="center">0.0293</td></tr>
    <tr><td align="center">100</td><td align="center">0.0005</td><td align="center">0.0013</td><td align="center">0.0097</td><td align="center">0.0057</td><td align="center">0.0115</td><td align="center">0.0029</td><td align="center">0.0173</td></tr>
    <tr><td align="center">500</td><td align="center">0.0009</td><td align="center">0.0016</td><td align="center">0.0110</td><td align="center">0.0058</td><td align="center">0.0115</td><td align="center">0.0029</td><td align="center">0.0166</td></tr>
    <tr><td align="center">1000</td><td align="center">0.0021</td><td align="center">0.0021</td><td align="center">0.0145</td><td align="center">0.0056</td><td align="center">0.0118</td><td align="center">0.0029</td><td align="center">0.0168</td></tr>
    <tr><td align="center">2000</td><td align="center">0.0069</td><td align="center">0.0043</td><td align="center">0.0278</td><td align="center">0.0059</td><td align="center">0.0118</td><td align="center">0.0030</td><td align="center">0.0165</td></tr>
    <tr><td align="center">5000</td><td align="center">0.0707</td><td align="center">0.0314</td><td align="center">0.1395</td><td align="center">0.0074</td><td align="center">0.0125</td><td align="center">0.0035</td><td align="center">0.0198</td></tr>
    <tr><td colspan="8">&nbsp;</td></tr>
    <tr><th align="center">Bitsize</th><th align="center" colspan="7">64 bits</th></tr>
    <tr><th align="center">Device</th><th align="center" colspan="3.0"">CPU</th><th align="center" colspan="4.0">GPU</tr>
    <tr><th align="center">Sample size</th><th align="center">Numpy</th><th align="center">Pytorch</th><th align="center">Tensorflow</th><th align="center">Cupy</th><th align="center">Jax</th><th align="center">Pytorch</th><th align="center">Tensorflow</th></tr>
    <tr><td align="center">50</td><td align="center">0.0008</td><td align="center">0.0020</td><td align="center">0.0154</td><td align="center">0.0093</td><td align="center">0.0191</td><td align="center">0.0051</td><td align="center">0.0328</td></tr>
    <tr><td align="center">100</td><td align="center">0.0005</td><td align="center">0.0013</td><td align="center">0.0094</td><td align="center">0.0056</td><td align="center">0.0114</td><td align="center">0.0029</td><td align="center">0.0169</td></tr>
    <tr><td align="center">500</td><td align="center">0.0013</td><td align="center">0.0017</td><td align="center">0.0120</td><td align="center">0.0059</td><td align="center">0.0116</td><td align="center">0.0029</td><td align="center">0.0168</td></tr>
    <tr><td align="center">1000</td><td align="center">0.0034</td><td align="center">0.0027</td><td align="center">0.0177</td><td align="center">0.0058</td><td align="center">0.0118</td><td align="center">0.0029</td><td align="center">0.0167</td></tr>
    <tr><td align="center">2000</td><td align="center">0.0146</td><td align="center">0.0075</td><td align="center">0.0436</td><td align="center">0.0059</td><td align="center">0.0120</td><td align="center">0.0029</td><td align="center">0.0165</td></tr>
    <tr><td align="center">5000</td><td align="center">0.1467</td><td align="center">0.0568</td><td align="center">0.2468</td><td align="center">0.0077</td><td align="center">0.0146</td><td align="center">0.0045</td><td align="center">0.0204</td></tr>
    </table>
    </div>
"""

# Author: Remi Flamary <remi.flamary@polytechnique.edu>
#         Nicolas Courty <ncourty@irisa.fr>
#
# License: MIT License

import os
import time
import warnings

import numpy as np
import scipy
import scipy.linalg
import scipy.special as special
from scipy.sparse import coo_matrix, csr_matrix, issparse

DISABLE_TORCH_KEY = 'POT_BACKEND_DISABLE_PYTORCH'
DISABLE_JAX_KEY = 'POT_BACKEND_DISABLE_JAX'
DISABLE_CUPY_KEY = 'POT_BACKEND_DISABLE_CUPY'
DISABLE_TF_KEY = 'POT_BACKEND_DISABLE_TENSORFLOW'


if not os.environ.get(DISABLE_TORCH_KEY, False):
    try:
        import torch
        torch_type = torch.Tensor
    except ImportError:
        torch = False
        torch_type = float
else:
    torch = False
    torch_type = float

if not os.environ.get(DISABLE_JAX_KEY, False):
    try:
        import jax
        import jax.numpy as jnp
        import jax.scipy.special as jspecial
        from jax.lib import xla_bridge
        jax_type = jax.numpy.ndarray
        jax_new_version = float('.'.join(jax.__version__.split('.')[1:])) > 4.24
    except ImportError:
        jax = False
        jax_type = float
else:
    jax = False
    jax_type = float

if not os.environ.get(DISABLE_CUPY_KEY, False):
    try:
        import cupy as cp
        import cupyx
        cp_type = cp.ndarray
    except ImportError:
        cp = False
        cp_type = float
else:
    cp = False
    cp_type = float

if not os.environ.get(DISABLE_TF_KEY, False):
    try:
        import tensorflow as tf
        import tensorflow.experimental.numpy as tnp
        tf_type = tf.Tensor
    except ImportError:
        tf = False
        tf_type = float
else:
    tf = False
    tf_type = float


str_type_error = "All array should be from the same type/backend. Current types are : {}"


# Mapping between argument types and the existing backend
_BACKEND_IMPLEMENTATIONS = []
_BACKENDS = {}


def _register_backend_implementation(backend_impl):
    _BACKEND_IMPLEMENTATIONS.append(backend_impl)


def _get_backend_instance(backend_impl):
    if backend_impl.__name__ not in _BACKENDS:
        _BACKENDS[backend_impl.__name__] = backend_impl()
    return _BACKENDS[backend_impl.__name__]


def _check_args_backend(backend_impl, args):
    is_instance = set(isinstance(arg, backend_impl.__type__) for arg in args)
    # check that all arguments matched or not the type
    if len(is_instance) == 1:
        return is_instance.pop()

    # Otherwise return an error
    raise ValueError(str_type_error.format([type(arg) for arg in args]))


def get_backend_list():
    """Returns instances of all available backends.

    Note that the function forces all detected implementations
    to be instantiated even if specific backend was not use before.
    Be careful as instantiation of the backend might lead to side effects,
    like GPU memory pre-allocation. See the documentation for more details.
    If you only need to know which implementations are available,
    use `:py:func:`ot.backend.get_available_backend_implementations`,
    which does not force instance of the backend object to be created.
    """
    return [
        _get_backend_instance(backend_impl)
        for backend_impl
        in get_available_backend_implementations()
    ]


def get_available_backend_implementations():
    """Returns the list of available backend implementations."""
    return _BACKEND_IMPLEMENTATIONS


def get_backend(*args):
    """Returns the proper backend for a list of input arrays

        Accepts None entries in the arguments, and ignores them

        Also raises TypeError if all arrays are not from the same backend
    """
    args = [arg for arg in args if arg is not None]  # exclude None entries

    # check that some arrays given
    if not len(args) > 0:
        raise ValueError(" The function takes at least one (non-None) parameter")

    for backend_impl in _BACKEND_IMPLEMENTATIONS:
        if _check_args_backend(backend_impl, args):
            return _get_backend_instance(backend_impl)

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
    Implementations: :py:class:`JaxBackend`, :py:class:`NumpyBackend`, :py:class:`TorchBackend`,
    :py:class:`CupyBackend`, :py:class:`TensorflowBackend`

    - The `__name__` class attribute refers to the name of the backend.
    - The `__type__` class attribute refers to the data structure used by the backend.
    """

    __name__ = None
    __type__ = None
    __type_list__ = None

    rng_ = None

    def __str__(self):
        return self.__name__

    # convert batch of tensors to numpy
    def to_numpy(self, *arrays):
        """Returns the numpy version of tensors"""
        if len(arrays) == 1:
            return self._to_numpy(arrays[0])
        else:
            return [self._to_numpy(array) for array in arrays]

    # convert a tensor to numpy
    def _to_numpy(self, a):
        """Returns the numpy version of a tensor"""
        raise NotImplementedError()

    # convert batch of arrays from numpy
    def from_numpy(self, *arrays, type_as=None):
        """Creates tensors cloning a numpy array, with the given precision (defaulting to input's precision) and the given device (in case of GPUs)"""
        if len(arrays) == 1:
            return self._from_numpy(arrays[0], type_as=type_as)
        else:
            return [self._from_numpy(array, type_as=type_as) for array in arrays]

    # convert an array from numpy
    def _from_numpy(self, a, type_as=None):
        """Creates a tensor cloning a numpy array, with the given precision (defaulting to input's precision) and the given device (in case of GPUs)"""
        raise NotImplementedError()

    def set_gradients(self, val, inputs, grads):
        """Define the gradients for the value val wrt the inputs """
        raise NotImplementedError()

    def detach(self, *arrays):
        """Detach the tensors from the computation graph

        See: https://pytorch.org/docs/stable/generated/torch.Tensor.detach.html"""
        if len(arrays) == 1:
            return self._detach(arrays[0])
        else:
            return [self._detach(array) for array in arrays]

    def _detach(self, a):
        """Detach the tensor from the computation graph"""
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

    def sign(self, a):
        r""" Returns an element-wise indication of the sign of a number.

        This function follows the api from :any:`numpy.sign`

        See: https://numpy.org/doc/stable/reference/generated/numpy.sign.html
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

    def norm(self, a, axis=None, keepdims=False):
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

    def zero_pad(self, a, pad_width, value=0):
        r"""
        Pads a tensor with a given value (0 by default).

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

    def argmin(self, a, axis=None):
        r"""
        Returns the indices of the minimum values of a tensor along given dimensions.

        This function follows the api from :any:`numpy.argmin`

        See: https://numpy.org/doc/stable/reference/generated/numpy.argmin.html
        """
        raise NotImplementedError()

    def mean(self, a, axis=None):
        r"""
        Computes the arithmetic mean of a tensor along given dimensions.

        This function follows the api from :any:`numpy.mean`

        See: https://numpy.org/doc/stable/reference/generated/numpy.mean.html
        """
        raise NotImplementedError()

    def median(self, a, axis=None):
        r"""
        Computes the median of a tensor along given dimensions.

        This function follows the api from :any:`numpy.median`

        See: https://numpy.org/doc/stable/reference/generated/numpy.median.html
        """
        raise NotImplementedError()

    def std(self, a, axis=None):
        r"""
        Computes the standard deviation of a tensor along given dimensions.

        This function follows the api from :any:`numpy.std`

        See: https://numpy.org/doc/stable/reference/generated/numpy.std.html
        """
        raise NotImplementedError()

    def linspace(self, start, stop, num, type_as=None):
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

    def unique(self, a, return_inverse=False):
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

        See: https://numpy.org/doc/stable/reference/random/generated/numpy.random.seed.html
        """
        raise NotImplementedError()

    def rand(self, *size, type_as=None):
        r"""
        Generate uniform random numbers.

        This function follows the api from :any:`numpy.random.rand`

        See: https://numpy.org/doc/stable/reference/random/generated/numpy.random.rand.html
        """
        raise NotImplementedError()

    def randn(self, *size, type_as=None):
        r"""
        Generate normal Gaussian random numbers.

        This function follows the api from :any:`numpy.random.rand`

        See: https://numpy.org/doc/stable/reference/random/generated/numpy.random.rand.html
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

    def squeeze(self, a, axis=None):
        r"""
        Remove axes of length one from a.

        This function follows the api from :any:`numpy.squeeze`.

        See: https://numpy.org/doc/stable/reference/generated/numpy.squeeze.html
        """
        raise NotImplementedError()

    def bitsize(self, type_as):
        r"""
        Gives the number of bits used by the data type of the given tensor.
        """
        raise NotImplementedError()

    def device_type(self, type_as):
        r"""
        Returns CPU or GPU depending on the device where the given tensor is located.
        """
        raise NotImplementedError()

    def _bench(self, callable, *args, n_runs=1, warmup_runs=1):
        r"""
        Executes a benchmark of the given callable with the given arguments.
        """
        raise NotImplementedError()

    def solve(self, a, b):
        r"""
        Solves a linear matrix equation, or system of linear scalar equations.

        This function follows the api from :any:`numpy.linalg.solve`.

        See: https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html
        """
        raise NotImplementedError()

    def trace(self, a):
        r"""
        Returns the sum along diagonals of the array.

        This function follows the api from :any:`numpy.trace`.

        See: https://numpy.org/doc/stable/reference/generated/numpy.trace.html
        """
        raise NotImplementedError()

    def inv(self, a):
        r"""
        Computes the inverse of a matrix.

        This function follows the api from :any:`scipy.linalg.inv`.

        See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.inv.html
        """
        raise NotImplementedError()

    def sqrtm(self, a):
        r"""
        Computes the matrix square root. Requires input to be definite positive.

        This function follows the api from :any:`scipy.linalg.sqrtm`.

        See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.sqrtm.html
        """
        raise NotImplementedError()

    def eigh(self, a):
        r"""
        Computes the eigenvalues and eigenvectors of a symmetric tensor.

        This function follows the api from :any:`scipy.linalg.eigh`.

        See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eigh.html
        """
        raise NotImplementedError()

    def kl_div(self, p, q, eps=1e-16):
        r"""
        Computes the Kullback-Leibler divergence.

        This function follows the api from :any:`scipy.stats.entropy`.

        Parameter eps is used to avoid numerical errors and is added in the log.

        .. math::
             KL(p,q) = \sum_i p(i) \log (\frac{p(i)}{q(i)}+\epsilon)

        See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html
        """
        raise NotImplementedError()

    def isfinite(self, a):
        r"""
        Tests element-wise for finiteness (not infinity and not Not a Number).

        This function follows the api from :any:`numpy.isfinite`.

        See: https://numpy.org/doc/stable/reference/generated/numpy.isfinite.html
        """
        raise NotImplementedError()

    def array_equal(self, a, b):
        r"""
        True if two arrays have the same shape and elements, False otherwise.

        This function follows the api from :any:`numpy.array_equal`.

        See: https://numpy.org/doc/stable/reference/generated/numpy.array_equal.html
        """
        raise NotImplementedError()

    def is_floating_point(self, a):
        r"""
        Returns whether or not the input consists of floats
        """
        raise NotImplementedError()

    def tile(self, a, reps):
        r"""
        Construct an array by repeating a the number of times given by reps

        See: https://numpy.org/doc/stable/reference/generated/numpy.tile.html
        """
        raise NotImplementedError()

    def floor(self, a):
        r"""
        Return the floor of the input element-wise

        See: https://numpy.org/doc/stable/reference/generated/numpy.floor.html
        """
        raise NotImplementedError()

    def prod(self, a, axis=None):
        r"""
        Return the product of all elements.

        See: https://numpy.org/doc/stable/reference/generated/numpy.prod.html
        """
        raise NotImplementedError()

    def sort2(self, a, axis=None):
        r"""
        Return the sorted array and the indices to sort the array

        See: https://pytorch.org/docs/stable/generated/torch.sort.html
        """
        raise NotImplementedError()

    def qr(self, a):
        r"""
        Return the QR factorization

        See: https://numpy.org/doc/stable/reference/generated/numpy.linalg.qr.html
        """
        raise NotImplementedError()

    def atan2(self, a, b):
        r"""
        Element wise arctangent

        See: https://numpy.org/doc/stable/reference/generated/numpy.arctan2.html
        """
        raise NotImplementedError()

    def transpose(self, a, axes=None):
        r"""
        Returns a tensor that is a transposed version of a. The given dimensions dim0 and dim1 are swapped.

        See: https://numpy.org/doc/stable/reference/generated/numpy.transpose.html
        """
        raise NotImplementedError()

    def matmul(self, a, b):
        r"""
        Matrix product of two arrays.

        See: https://numpy.org/doc/stable/reference/generated/numpy.matmul.html#numpy.matmul
        """
        raise NotImplementedError()

    def nan_to_num(self, x, copy=True, nan=0.0, posinf=None, neginf=None):
        r"""
        Replace NaN with zero and infinity with large finite numbers or with the numbers defined by the user.

        See: https://numpy.org/doc/stable/reference/generated/numpy.nan_to_num.html#numpy.nan_to_num
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

    def _to_numpy(self, a):
        return a

    def _from_numpy(self, a, type_as=None):
        if type_as is None:
            return a
        elif isinstance(a, float):
            return a
        else:
            return a.astype(type_as.dtype)

    def set_gradients(self, val, inputs, grads):
        # No gradients for numpy
        return val

    def _detach(self, a):
        # No gradients for numpy
        return a

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

    def sign(self, a):
        return np.sign(a)

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

    def norm(self, a, axis=None, keepdims=False):
        return np.linalg.norm(a, axis=axis, keepdims=keepdims)

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

    def zero_pad(self, a, pad_width, value=0):
        return np.pad(a, pad_width, constant_values=value)

    def argmax(self, a, axis=None):
        return np.argmax(a, axis=axis)

    def argmin(self, a, axis=None):
        return np.argmin(a, axis=axis)

    def mean(self, a, axis=None):
        return np.mean(a, axis=axis)

    def median(self, a, axis=None):
        return np.median(a, axis=axis)

    def std(self, a, axis=None):
        return np.std(a, axis=axis)

    def linspace(self, start, stop, num, type_as=None):
        if type_as is None:
            return np.linspace(start, stop, num)
        else:
            return np.linspace(start, stop, num, dtype=type_as.dtype)

    def meshgrid(self, a, b):
        return np.meshgrid(a, b)

    def diag(self, a, k=0):
        return np.diag(a, k)

    def unique(self, a, return_inverse=False):
        return np.unique(a, return_inverse=return_inverse)

    def logsumexp(self, a, axis=None):
        return special.logsumexp(a, axis=axis)

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

    def where(self, condition, x=None, y=None):
        if x is None and y is None:
            return np.where(condition)
        else:
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

    def squeeze(self, a, axis=None):
        return np.squeeze(a, axis=axis)

    def bitsize(self, type_as):
        return type_as.itemsize * 8

    def device_type(self, type_as):
        return "CPU"

    def _bench(self, callable, *args, n_runs=1, warmup_runs=1):
        results = dict()
        for type_as in self.__type_list__:
            inputs = [self.from_numpy(arg, type_as=type_as) for arg in args]
            for _ in range(warmup_runs):
                callable(*inputs)
            t0 = time.perf_counter()
            for _ in range(n_runs):
                callable(*inputs)
            t1 = time.perf_counter()
            key = ("Numpy", self.device_type(type_as), self.bitsize(type_as))
            results[key] = (t1 - t0) / n_runs
        return results

    def solve(self, a, b):
        return np.linalg.solve(a, b)

    def trace(self, a):
        return np.trace(a)

    def inv(self, a):
        return scipy.linalg.inv(a)

    def sqrtm(self, a):
        L, V = np.linalg.eigh(a)
        return (V * np.sqrt(L)[None, :]) @ V.T

    def eigh(self, a):
        return np.linalg.eigh(a)

    def kl_div(self, p, q, eps=1e-16):
        return np.sum(p * np.log(p / q + eps))

    def isfinite(self, a):
        return np.isfinite(a)

    def array_equal(self, a, b):
        return np.array_equal(a, b)

    def is_floating_point(self, a):
        return a.dtype.kind == "f"

    def tile(self, a, reps):
        return np.tile(a, reps)

    def floor(self, a):
        return np.floor(a)

    def prod(self, a, axis=0):
        return np.prod(a, axis=axis)

    def sort2(self, a, axis=-1):
        return self.sort(a, axis), self.argsort(a, axis)

    def qr(self, a):
        np_version = tuple([int(k) for k in np.__version__.split(".")])
        if np_version < (1, 22, 0):
            M, N = a.shape[-2], a.shape[-1]
            K = min(M, N)

            if len(a.shape) >= 3:
                n = a.shape[0]

                qs, rs = np.zeros((n, M, K)), np.zeros((n, K, N))

                for i in range(a.shape[0]):
                    qs[i], rs[i] = np.linalg.qr(a[i])

            else:
                return np.linalg.qr(a)

            return qs, rs
        return np.linalg.qr(a)

    def atan2(self, a, b):
        return np.arctan2(a, b)

    def transpose(self, a, axes=None):
        return np.transpose(a, axes)

    def matmul(self, a, b):
        return np.matmul(a, b)

    def nan_to_num(self, x, copy=True, nan=0.0, posinf=None, neginf=None):
        return np.nan_to_num(x, copy=copy, nan=nan, posinf=posinf, neginf=neginf)


_register_backend_implementation(NumpyBackend)


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

        self.__type_list__ = []
        # available_devices = jax.devices("cpu")
        available_devices = []
        if xla_bridge.get_backend().platform == "gpu":
            available_devices += jax.devices("gpu")
        for d in available_devices:
            self.__type_list__ += [
                jax.device_put(jnp.array(1, dtype=jnp.float32), d),
                jax.device_put(jnp.array(1, dtype=jnp.float64), d)
            ]

        self.jax_new_version = jax_new_version

    def _to_numpy(self, a):
        return np.array(a)

    def _get_device(self, a):
        if self.jax_new_version:
            return list(a.devices())[0]
        else:
            return a.device_buffer.device()

    def _change_device(self, a, type_as):
        return jax.device_put(a, self._get_device(type_as))

    def _from_numpy(self, a, type_as=None):
        if isinstance(a, float):
            a = np.array(a)
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

    def _detach(self, a):
        return jax.lax.stop_gradient(a)

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

    def sign(self, a):
        return jnp.sign(a)

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

    def norm(self, a, axis=None, keepdims=False):
        return jnp.linalg.norm(a, axis=axis, keepdims=keepdims)

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

    def zero_pad(self, a, pad_width, value=0):
        return jnp.pad(a, pad_width, constant_values=value)

    def argmax(self, a, axis=None):
        return jnp.argmax(a, axis=axis)

    def argmin(self, a, axis=None):
        return jnp.argmin(a, axis=axis)

    def mean(self, a, axis=None):
        return jnp.mean(a, axis=axis)

    def median(self, a, axis=None):
        return jnp.median(a, axis=axis)

    def std(self, a, axis=None):
        return jnp.std(a, axis=axis)

    def linspace(self, start, stop, num, type_as=None):
        if type_as is None:
            return jnp.linspace(start, stop, num)
        else:
            return self._change_device(jnp.linspace(start, stop, num, dtype=type_as.dtype), type_as)

    def meshgrid(self, a, b):
        return jnp.meshgrid(a, b)

    def diag(self, a, k=0):
        return jnp.diag(a, k)

    def unique(self, a, return_inverse=False):
        return jnp.unique(a, return_inverse=return_inverse)

    def logsumexp(self, a, axis=None):
        return jspecial.logsumexp(a, axis=axis)

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

    def where(self, condition, x=None, y=None):
        if x is None and y is None:
            return jnp.where(condition)
        else:
            return jnp.where(condition, x, y)

    def copy(self, a):
        # No need to copy, JAX arrays are immutable
        return a

    def allclose(self, a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
        return jnp.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)

    def dtype_device(self, a):
        if self.jax_new_version:
            return a.dtype, list(a.devices())[0]
        else:
            return a.dtype, a.device_buffer.device()

    def assert_same_dtype_device(self, a, b):
        a_dtype, a_device = self.dtype_device(a)
        b_dtype, b_device = self.dtype_device(b)

        assert a_dtype == b_dtype, "Dtype discrepancy"
        assert a_device == b_device, f"Device discrepancy. First input is on {str(a_device)}, whereas second input is on {str(b_device)}"

    def squeeze(self, a, axis=None):
        return jnp.squeeze(a, axis=axis)

    def bitsize(self, type_as):
        return type_as.dtype.itemsize * 8

    def device_type(self, type_as):
        return self.dtype_device(type_as)[1].platform.upper()

    def _bench(self, callable, *args, n_runs=1, warmup_runs=1):
        results = dict()

        for type_as in self.__type_list__:
            inputs = [self.from_numpy(arg, type_as=type_as) for arg in args]
            for _ in range(warmup_runs):
                a = callable(*inputs)
            a.block_until_ready()
            t0 = time.perf_counter()
            for _ in range(n_runs):
                a = callable(*inputs)
            a.block_until_ready()
            t1 = time.perf_counter()
            key = ("Jax", self.device_type(type_as), self.bitsize(type_as))
            results[key] = (t1 - t0) / n_runs
        return results

    def solve(self, a, b):
        return jnp.linalg.solve(a, b)

    def trace(self, a):
        return jnp.trace(a)

    def inv(self, a):
        return jnp.linalg.inv(a)

    def sqrtm(self, a):
        L, V = jnp.linalg.eigh(a)
        return (V * jnp.sqrt(L)[None, :]) @ V.T

    def eigh(self, a):
        return jnp.linalg.eigh(a)

    def kl_div(self, p, q, eps=1e-16):
        return jnp.sum(p * jnp.log(p / q + eps))

    def isfinite(self, a):
        return jnp.isfinite(a)

    def array_equal(self, a, b):
        return jnp.array_equal(a, b)

    def is_floating_point(self, a):
        return a.dtype.kind == "f"

    def tile(self, a, reps):
        return jnp.tile(a, reps)

    def floor(self, a):
        return jnp.floor(a)

    def prod(self, a, axis=0):
        return jnp.prod(a, axis=axis)

    def sort2(self, a, axis=-1):
        return self.sort(a, axis), self.argsort(a, axis)

    def qr(self, a):
        return jnp.linalg.qr(a)

    def atan2(self, a, b):
        return jnp.arctan2(a, b)

    def transpose(self, a, axes=None):
        return jnp.transpose(a, axes)

    def matmul(self, a, b):
        return jnp.matmul(a, b)

    def nan_to_num(self, x, copy=True, nan=0.0, posinf=None, neginf=None):
        return jnp.nan_to_num(x, copy=copy, nan=nan, posinf=posinf, neginf=neginf)


if jax:
    # Only register jax backend if it is installed
    _register_backend_implementation(JaxBackend)


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

        self.rng_ = torch.Generator("cpu")
        self.rng_.seed()

        self.__type_list__ = [torch.tensor(1, dtype=torch.float32),
                              torch.tensor(1, dtype=torch.float64)]

        if torch.cuda.is_available():
            self.rng_cuda_ = torch.Generator("cuda")
            self.rng_cuda_.seed()
            self.__type_list__.append(torch.tensor(1, dtype=torch.float32, device='cuda'))
            self.__type_list__.append(torch.tensor(1, dtype=torch.float64, device='cuda'))
        else:
            self.rng_cuda_ = torch.Generator("cpu")

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
                return (None, None) + tuple(g * grad_output for g in ctx.grads)

        self.ValFunction = ValFunction

    def _to_numpy(self, a):
        if isinstance(a, float) or isinstance(a, int) or isinstance(a, np.ndarray):
            return np.array(a)
        return a.cpu().detach().numpy()

    def _from_numpy(self, a, type_as=None):
        if isinstance(a, float) or isinstance(a, int):
            a = np.array(a)
        if type_as is None:
            return torch.from_numpy(a)
        else:
            return torch.as_tensor(a, dtype=type_as.dtype, device=type_as.device)

    def set_gradients(self, val, inputs, grads):

        Func = self.ValFunction

        res = Func.apply(val, grads, *inputs)

        return res

    def _detach(self, a):
        return a.detach()

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

    def sign(self, a):
        return torch.sign(a)

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

    def norm(self, a, axis=None, keepdims=False):
        return torch.linalg.norm(a, dim=axis, keepdims=keepdims)

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

    def zero_pad(self, a, pad_width, value=0):
        from torch.nn.functional import pad

        # pad_width is an array of ndim tuples indicating how many 0 before and after
        # we need to add. We first need to make it compliant with torch syntax, that
        # starts with the last dim, then second last, etc.
        how_pad = tuple(element for tupl in pad_width[::-1] for element in tupl)
        return pad(a, how_pad, value=value)

    def argmax(self, a, axis=None):
        return torch.argmax(a, dim=axis)

    def argmin(self, a, axis=None):
        return torch.argmin(a, dim=axis)

    def mean(self, a, axis=None):
        if axis is not None:
            return torch.mean(a, dim=axis)
        else:
            return torch.mean(a)

    def median(self, a, axis=None):
        from packaging import version

        # Since version 1.11.0, interpolation is available
        if version.parse(torch.__version__) >= version.parse("1.11.0"):
            if axis is not None:
                return torch.quantile(a, 0.5, interpolation="midpoint", dim=axis)
            else:
                return torch.quantile(a, 0.5, interpolation="midpoint")

        # Else, use numpy
        warnings.warn("The median is being computed using numpy and the array has been detached "
                      "in the Pytorch backend.")
        a_ = self.to_numpy(a)
        a_median = np.median(a_, axis=axis)
        return self.from_numpy(a_median, type_as=a)

    def std(self, a, axis=None):
        if axis is not None:
            return torch.std(a, dim=axis, unbiased=False)
        else:
            return torch.std(a, unbiased=False)

    def linspace(self, start, stop, num, type_as=None):
        if type_as is None:
            return torch.linspace(start, stop, num)
        else:
            return torch.linspace(start, stop, num, dtype=type_as.dtype, device=type_as.device)

    def meshgrid(self, a, b):
        try:
            return torch.meshgrid(a, b, indexing="xy")
        except TypeError:
            X, Y = torch.meshgrid(a, b)
            return X.T, Y.T

    def diag(self, a, k=0):
        return torch.diag(a, diagonal=k)

    def unique(self, a, return_inverse=False):
        return torch.unique(a, return_inverse=return_inverse)

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
            self.rng_cuda_.manual_seed(seed)
        elif isinstance(seed, torch.Generator):
            if self.device_type(seed) == "GPU":
                self.rng_cuda_ = seed
            else:
                self.rng_ = seed
        else:
            raise ValueError("Non compatible seed : {}".format(seed))

    def rand(self, *size, type_as=None):
        if type_as is not None:
            generator = self.rng_cuda_ if self.device_type(type_as) == "GPU" else self.rng_
            return torch.rand(size=size, generator=generator, dtype=type_as.dtype, device=type_as.device)
        else:
            return torch.rand(size=size, generator=self.rng_)

    def randn(self, *size, type_as=None):
        if type_as is not None:
            generator = self.rng_cuda_ if self.device_type(type_as) == "GPU" else self.rng_
            return torch.randn(size=size, dtype=type_as.dtype, generator=generator, device=type_as.device)
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

    def where(self, condition, x=None, y=None):
        if x is None and y is None:
            return torch.where(condition)
        else:
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

    def squeeze(self, a, axis=None):
        if axis is None:
            return torch.squeeze(a)
        else:
            return torch.squeeze(a, dim=axis)

    def bitsize(self, type_as):
        return torch.finfo(type_as.dtype).bits

    def device_type(self, type_as):
        return type_as.device.type.replace("cuda", "gpu").upper()

    def _bench(self, callable, *args, n_runs=1, warmup_runs=1):
        results = dict()
        for type_as in self.__type_list__:
            inputs = [self.from_numpy(arg, type_as=type_as) for arg in args]
            for _ in range(warmup_runs):
                callable(*inputs)
            if self.device_type(type_as) == "GPU":  # pragma: no cover
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
            else:
                start = time.perf_counter()
            for _ in range(n_runs):
                callable(*inputs)
            if self.device_type(type_as) == "GPU":  # pragma: no cover
                end.record()
                torch.cuda.synchronize()
                duration = start.elapsed_time(end) / 1000.
            else:
                end = time.perf_counter()
                duration = end - start
            key = ("Pytorch", self.device_type(type_as), self.bitsize(type_as))
            results[key] = duration / n_runs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return results

    def solve(self, a, b):
        return torch.linalg.solve(a, b)

    def trace(self, a):
        return torch.trace(a)

    def inv(self, a):
        return torch.linalg.inv(a)

    def sqrtm(self, a):
        L, V = torch.linalg.eigh(a)
        return (V * torch.sqrt(L)[None, :]) @ V.T

    def eigh(self, a):
        return torch.linalg.eigh(a)

    def kl_div(self, p, q, eps=1e-16):
        return torch.sum(p * torch.log(p / q + eps))

    def isfinite(self, a):
        return torch.isfinite(a)

    def array_equal(self, a, b):
        return torch.equal(a, b)

    def is_floating_point(self, a):
        return a.dtype.is_floating_point

    def tile(self, a, reps):
        return a.repeat(reps)

    def floor(self, a):
        return torch.floor(a)

    def prod(self, a, axis=0):
        return torch.prod(a, dim=axis)

    def sort2(self, a, axis=-1):
        return torch.sort(a, axis)

    def qr(self, a):
        return torch.linalg.qr(a)

    def atan2(self, a, b):
        return torch.atan2(a, b)

    def transpose(self, a, axes=None):
        if axes is None:
            axes = tuple(range(a.ndim)[::-1])
        return a.permute(axes)

    def matmul(self, a, b):
        return torch.matmul(a, b)

    def nan_to_num(self, x, copy=True, nan=0.0, posinf=None, neginf=None):
        out = None if copy else x
        return torch.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf, out=out)


if torch:
    # Only register torch backend if it is installed
    _register_backend_implementation(TorchBackend)


class CupyBackend(Backend):  # pragma: no cover
    """
    CuPy implementation of the backend

    - `__name__` is "cupy"
    - `__type__` is cp.ndarray
    """

    __name__ = 'cupy'
    __type__ = cp_type
    __type_list__ = None

    rng_ = None

    def __init__(self):
        self.rng_ = cp.random.RandomState()

        self.__type_list__ = [
            cp.array(1, dtype=cp.float32),
            cp.array(1, dtype=cp.float64)
        ]

    def _to_numpy(self, a):
        return cp.asnumpy(a)

    def _from_numpy(self, a, type_as=None):
        if isinstance(a, float):
            a = np.array(a)
        if type_as is None:
            return cp.asarray(a)
        else:
            with cp.cuda.Device(type_as.device):
                return cp.asarray(a, dtype=type_as.dtype)

    def set_gradients(self, val, inputs, grads):
        # No gradients for cupy
        return val

    def _detach(self, a):
        return a

    def zeros(self, shape, type_as=None):
        if isinstance(shape, (list, tuple)):
            shape = tuple(int(i) for i in shape)
        if type_as is None:
            return cp.zeros(shape)
        else:
            with cp.cuda.Device(type_as.device):
                return cp.zeros(shape, dtype=type_as.dtype)

    def ones(self, shape, type_as=None):
        if isinstance(shape, (list, tuple)):
            shape = tuple(int(i) for i in shape)
        if type_as is None:
            return cp.ones(shape)
        else:
            with cp.cuda.Device(type_as.device):
                return cp.ones(shape, dtype=type_as.dtype)

    def arange(self, stop, start=0, step=1, type_as=None):
        return cp.arange(start, stop, step)

    def full(self, shape, fill_value, type_as=None):
        if isinstance(shape, (list, tuple)):
            shape = tuple(int(i) for i in shape)
        if type_as is None:
            return cp.full(shape, fill_value)
        else:
            with cp.cuda.Device(type_as.device):
                return cp.full(shape, fill_value, dtype=type_as.dtype)

    def eye(self, N, M=None, type_as=None):
        if type_as is None:
            return cp.eye(N, M)
        else:
            with cp.cuda.Device(type_as.device):
                return cp.eye(N, M, dtype=type_as.dtype)

    def sum(self, a, axis=None, keepdims=False):
        return cp.sum(a, axis, keepdims=keepdims)

    def cumsum(self, a, axis=None):
        return cp.cumsum(a, axis)

    def max(self, a, axis=None, keepdims=False):
        return cp.max(a, axis, keepdims=keepdims)

    def min(self, a, axis=None, keepdims=False):
        return cp.min(a, axis, keepdims=keepdims)

    def maximum(self, a, b):
        return cp.maximum(a, b)

    def minimum(self, a, b):
        return cp.minimum(a, b)

    def sign(self, a):
        return cp.sign(a)

    def abs(self, a):
        return cp.abs(a)

    def exp(self, a):
        return cp.exp(a)

    def log(self, a):
        return cp.log(a)

    def sqrt(self, a):
        return cp.sqrt(a)

    def power(self, a, exponents):
        return cp.power(a, exponents)

    def dot(self, a, b):
        return cp.dot(a, b)

    def norm(self, a, axis=None, keepdims=False):
        return cp.linalg.norm(a, axis=axis, keepdims=keepdims)

    def any(self, a):
        return cp.any(a)

    def isnan(self, a):
        return cp.isnan(a)

    def isinf(self, a):
        return cp.isinf(a)

    def einsum(self, subscripts, *operands):
        return cp.einsum(subscripts, *operands)

    def sort(self, a, axis=-1):
        return cp.sort(a, axis)

    def argsort(self, a, axis=-1):
        return cp.argsort(a, axis)

    def searchsorted(self, a, v, side='left'):
        if a.ndim == 1:
            return cp.searchsorted(a, v, side)
        else:
            # this is a not very efficient way to make numpy
            # searchsorted work on 2d arrays
            ret = cp.empty(v.shape, dtype=int)
            for i in range(a.shape[0]):
                ret[i, :] = cp.searchsorted(a[i, :], v[i, :], side)
            return ret

    def flip(self, a, axis=None):
        return cp.flip(a, axis)

    def outer(self, a, b):
        return cp.outer(a, b)

    def clip(self, a, a_min, a_max):
        return cp.clip(a, a_min, a_max)

    def repeat(self, a, repeats, axis=None):
        return cp.repeat(a, repeats, axis)

    def take_along_axis(self, arr, indices, axis):
        return cp.take_along_axis(arr, indices, axis)

    def concatenate(self, arrays, axis=0):
        return cp.concatenate(arrays, axis)

    def zero_pad(self, a, pad_width, value=0):
        return cp.pad(a, pad_width, constant_values=value)

    def argmax(self, a, axis=None):
        return cp.argmax(a, axis=axis)

    def argmin(self, a, axis=None):
        return cp.argmin(a, axis=axis)

    def mean(self, a, axis=None):
        return cp.mean(a, axis=axis)

    def median(self, a, axis=None):
        return cp.median(a, axis=axis)

    def std(self, a, axis=None):
        return cp.std(a, axis=axis)

    def linspace(self, start, stop, num, type_as=None):
        if type_as is None:
            return cp.linspace(start, stop, num)
        else:
            with cp.cuda.Device(type_as.device):
                return cp.linspace(start, stop, num, dtype=type_as.dtype)

    def meshgrid(self, a, b):
        return cp.meshgrid(a, b)

    def diag(self, a, k=0):
        return cp.diag(a, k)

    def unique(self, a, return_inverse=False):
        return cp.unique(a, return_inverse=return_inverse)

    def logsumexp(self, a, axis=None):
        # Taken from
        # https://github.com/scipy/scipy/blob/v1.7.1/scipy/special/_logsumexp.py#L7-L127
        a_max = cp.amax(a, axis=axis, keepdims=True)

        if a_max.ndim > 0:
            a_max[~cp.isfinite(a_max)] = 0
        elif not cp.isfinite(a_max):
            a_max = 0

        tmp = cp.exp(a - a_max)
        s = cp.sum(tmp, axis=axis)
        out = cp.log(s)
        a_max = cp.squeeze(a_max, axis=axis)
        out += a_max
        return out

    def stack(self, arrays, axis=0):
        return cp.stack(arrays, axis)

    def reshape(self, a, shape):
        return cp.reshape(a, shape)

    def seed(self, seed=None):
        if seed is not None:
            self.rng_.seed(seed)

    def rand(self, *size, type_as=None):
        if type_as is None:
            return self.rng_.rand(*size)
        else:
            with cp.cuda.Device(type_as.device):
                return self.rng_.rand(*size, dtype=type_as.dtype)

    def randn(self, *size, type_as=None):
        if type_as is None:
            return self.rng_.randn(*size)
        else:
            with cp.cuda.Device(type_as.device):
                return self.rng_.randn(*size, dtype=type_as.dtype)

    def coo_matrix(self, data, rows, cols, shape=None, type_as=None):
        data = self.from_numpy(data)
        rows = self.from_numpy(rows)
        cols = self.from_numpy(cols)
        if type_as is None:
            return cupyx.scipy.sparse.coo_matrix(
                (data, (rows, cols)), shape=shape
            )
        else:
            with cp.cuda.Device(type_as.device):
                return cupyx.scipy.sparse.coo_matrix(
                    (data, (rows, cols)), shape=shape, dtype=type_as.dtype
                )

    def issparse(self, a):
        return cupyx.scipy.sparse.issparse(a)

    def tocsr(self, a):
        if self.issparse(a):
            return a.tocsr()
        else:
            return cupyx.scipy.sparse.csr_matrix(a)

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

    def where(self, condition, x=None, y=None):
        if x is None and y is None:
            return cp.where(condition)
        else:
            return cp.where(condition, x, y)

    def copy(self, a):
        return a.copy()

    def allclose(self, a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
        return cp.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)

    def dtype_device(self, a):
        return a.dtype, a.device

    def assert_same_dtype_device(self, a, b):
        a_dtype, a_device = self.dtype_device(a)
        b_dtype, b_device = self.dtype_device(b)

        # cupy has implicit type conversion so
        # we automatically validate the test for type
        assert a_device == b_device, f"Device discrepancy. First input is on {str(a_device)}, whereas second input is on {str(b_device)}"

    def squeeze(self, a, axis=None):
        return cp.squeeze(a, axis=axis)

    def bitsize(self, type_as):
        return type_as.itemsize * 8

    def device_type(self, type_as):
        return "GPU"

    def _bench(self, callable, *args, n_runs=1, warmup_runs=1):
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()

        results = dict()
        for type_as in self.__type_list__:
            inputs = [self.from_numpy(arg, type_as=type_as) for arg in args]
            start_gpu = cp.cuda.Event()
            end_gpu = cp.cuda.Event()
            for _ in range(warmup_runs):
                callable(*inputs)
            start_gpu.synchronize()
            start_gpu.record()
            for _ in range(n_runs):
                callable(*inputs)
            end_gpu.record()
            end_gpu.synchronize()
            key = ("Cupy", self.device_type(type_as), self.bitsize(type_as))
            t_gpu = cp.cuda.get_elapsed_time(start_gpu, end_gpu) / 1000.
            results[key] = t_gpu / n_runs
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
        return results

    def solve(self, a, b):
        return cp.linalg.solve(a, b)

    def trace(self, a):
        return cp.trace(a)

    def inv(self, a):
        return cp.linalg.inv(a)

    def sqrtm(self, a):
        L, V = cp.linalg.eigh(a)
        return (V * cp.sqrt(L)[None, :]) @ V.T

    def eigh(self, a):
        return cp.linalg.eigh(a)

    def kl_div(self, p, q, eps=1e-16):
        return cp.sum(p * cp.log(p / q + eps))

    def isfinite(self, a):
        return cp.isfinite(a)

    def array_equal(self, a, b):
        return cp.array_equal(a, b)

    def is_floating_point(self, a):
        return a.dtype.kind == "f"

    def tile(self, a, reps):
        return cp.tile(a, reps)

    def floor(self, a):
        return cp.floor(a)

    def prod(self, a, axis=0):
        return cp.prod(a, axis=axis)

    def sort2(self, a, axis=-1):
        return self.sort(a, axis), self.argsort(a, axis)

    def qr(self, a):
        return cp.linalg.qr(a)

    def atan2(self, a, b):
        return cp.arctan2(a, b)

    def transpose(self, a, axes=None):
        return cp.transpose(a, axes)

    def matmul(self, a, b):
        return cp.matmul(a, b)

    def nan_to_num(self, x, copy=True, nan=0.0, posinf=None, neginf=None):
        return cp.nan_to_num(x, copy=copy, nan=nan, posinf=posinf, neginf=neginf)


if cp:
    # Only register cp backend if it is installed
    _register_backend_implementation(CupyBackend)


class TensorflowBackend(Backend):

    __name__ = "tf"
    __type__ = tf_type
    __type_list__ = None

    rng_ = None

    def __init__(self):
        self.seed(None)

        self.__type_list__ = [
            tf.convert_to_tensor([1], dtype=tf.float32),
            tf.convert_to_tensor([1], dtype=tf.float64)
        ]

        tmp = self.randn(15, 10)
        try:
            tmp.reshape((150, 1))
        except AttributeError:
            warnings.warn(
                "To use TensorflowBackend, you need to activate the tensorflow "
                "numpy API. You can activate it by running: \n"
                "from tensorflow.python.ops.numpy_ops import np_config\n"
                "np_config.enable_numpy_behavior()",
                stacklevel=2
            )

    def _to_numpy(self, a):
        if isinstance(a, float) or isinstance(a, int) or isinstance(a, np.ndarray):
            return np.array(a)
        return a.numpy()

    def _from_numpy(self, a, type_as=None):
        if isinstance(a, float):
            a = np.array(a)
        if not isinstance(a, self.__type__):
            if type_as is None:
                return tf.convert_to_tensor(a)
            else:
                return tf.convert_to_tensor(a, dtype=type_as.dtype)
        else:
            if type_as is None:
                return a
            else:
                return tf.cast(a, dtype=type_as.dtype)

    def set_gradients(self, val, inputs, grads):
        @tf.custom_gradient
        def tmp(input):
            def grad(upstream):
                return grads
            return val, grad
        return tmp(inputs)

    def _detach(self, a):
        return tf.stop_gradient(a)

    def zeros(self, shape, type_as=None):
        if type_as is None:
            return tnp.zeros(shape)
        else:
            return tnp.zeros(shape, dtype=type_as.dtype)

    def ones(self, shape, type_as=None):
        if type_as is None:
            return tnp.ones(shape)
        else:
            return tnp.ones(shape, dtype=type_as.dtype)

    def arange(self, stop, start=0, step=1, type_as=None):
        return tnp.arange(start, stop, step)

    def full(self, shape, fill_value, type_as=None):
        if type_as is None:
            return tnp.full(shape, fill_value)
        else:
            return tnp.full(shape, fill_value, dtype=type_as.dtype)

    def eye(self, N, M=None, type_as=None):
        if type_as is None:
            return tnp.eye(N, M)
        else:
            return tnp.eye(N, M, dtype=type_as.dtype)

    def sum(self, a, axis=None, keepdims=False):
        return tnp.sum(a, axis, keepdims=keepdims)

    def cumsum(self, a, axis=None):
        return tnp.cumsum(a, axis)

    def max(self, a, axis=None, keepdims=False):
        return tnp.max(a, axis, keepdims=keepdims)

    def min(self, a, axis=None, keepdims=False):
        return tnp.min(a, axis, keepdims=keepdims)

    def maximum(self, a, b):
        return tnp.maximum(a, b)

    def minimum(self, a, b):
        return tnp.minimum(a, b)

    def sign(self, a):
        return tnp.sign(a)

    def dot(self, a, b):
        if len(b.shape) == 1:
            if len(a.shape) == 1:
                # inner product
                return tf.reduce_sum(tf.multiply(a, b))
            else:
                # matrix vector
                return tf.linalg.matvec(a, b)
        else:
            if len(a.shape) == 1:
                return tf.linalg.matvec(b.T, a.T).T
            else:
                return tf.matmul(a, b)

    def abs(self, a):
        return tnp.abs(a)

    def exp(self, a):
        return tnp.exp(a)

    def log(self, a):
        return tnp.log(a)

    def sqrt(self, a):
        return tnp.sqrt(a)

    def power(self, a, exponents):
        return tnp.power(a, exponents)

    def norm(self, a, axis=None, keepdims=False):
        return tf.math.reduce_euclidean_norm(a, axis=axis, keepdims=keepdims)

    def any(self, a):
        return tnp.any(a)

    def isnan(self, a):
        return tnp.isnan(a)

    def isinf(self, a):
        return tnp.isinf(a)

    def einsum(self, subscripts, *operands):
        return tnp.einsum(subscripts, *operands)

    def sort(self, a, axis=-1):
        return tnp.sort(a, axis)

    def argsort(self, a, axis=-1):
        return tnp.argsort(a, axis)

    def searchsorted(self, a, v, side='left'):
        return tf.searchsorted(a, v, side=side)

    def flip(self, a, axis=None):
        return tnp.flip(a, axis)

    def outer(self, a, b):
        return tnp.outer(a, b)

    def clip(self, a, a_min, a_max):
        return tnp.clip(a, a_min, a_max)

    def repeat(self, a, repeats, axis=None):
        return tnp.repeat(a, repeats, axis)

    def take_along_axis(self, arr, indices, axis):
        return tnp.take_along_axis(arr, indices, axis)

    def concatenate(self, arrays, axis=0):
        return tnp.concatenate(arrays, axis)

    def zero_pad(self, a, pad_width, value=0):
        return tnp.pad(a, pad_width, mode="constant", constant_values=value)

    def argmax(self, a, axis=None):
        return tnp.argmax(a, axis=axis)

    def argmin(self, a, axis=None):
        return tnp.argmin(a, axis=axis)

    def mean(self, a, axis=None):
        return tnp.mean(a, axis=axis)

    def median(self, a, axis=None):
        warnings.warn("The median is being computed using numpy and the array has been detached "
                      "in the Tensorflow backend.")
        a_ = self.to_numpy(a)
        a_median = np.median(a_, axis=axis)
        return self.from_numpy(a_median, type_as=a)

    def std(self, a, axis=None):
        return tnp.std(a, axis=axis)

    def linspace(self, start, stop, num, type_as=None):
        if type_as is None:
            return tnp.linspace(start, stop, num)
        else:
            return tnp.linspace(start, stop, num, dtype=type_as.dtype)

    def meshgrid(self, a, b):
        return tnp.meshgrid(a, b)

    def diag(self, a, k=0):
        return tnp.diag(a, k)

    def unique(self, a, return_inverse=False):
        y, idx = tf.unique(tf.reshape(a, [-1]))
        sort_idx = tf.argsort(y)
        y_prime = tf.gather(y, sort_idx)
        if return_inverse:
            inv_sort_idx = tf.math.invert_permutation(sort_idx)
            return y_prime, tf.gather(inv_sort_idx, idx)
        else:
            return y_prime

    def logsumexp(self, a, axis=None):
        return tf.math.reduce_logsumexp(a, axis=axis)

    def stack(self, arrays, axis=0):
        return tnp.stack(arrays, axis)

    def reshape(self, a, shape):
        return tnp.reshape(a, shape)

    def seed(self, seed=None):
        if isinstance(seed, int):
            self.rng_ = tf.random.Generator.from_seed(seed)
        elif isinstance(seed, tf.random.Generator):
            self.rng_ = seed
        elif seed is None:
            self.rng_ = tf.random.Generator.from_non_deterministic_state()
        else:
            raise ValueError("Non compatible seed : {}".format(seed))

    def rand(self, *size, type_as=None):
        if type_as is None:
            return self.rng_.uniform(size, minval=0., maxval=1.)
        else:
            return self.rng_.uniform(
                size, minval=0., maxval=1., dtype=type_as.dtype
            )

    def randn(self, *size, type_as=None):
        if type_as is None:
            return self.rng_.normal(size)
        else:
            return self.rng_.normal(size, dtype=type_as.dtype)

    def _convert_to_index_for_coo(self, tensor):
        if isinstance(tensor, self.__type__):
            return int(self.max(tensor)) + 1
        else:
            return int(np.max(tensor)) + 1

    def coo_matrix(self, data, rows, cols, shape=None, type_as=None):
        if shape is None:
            shape = (
                self._convert_to_index_for_coo(rows),
                self._convert_to_index_for_coo(cols)
            )
        if type_as is not None:
            data = self.from_numpy(data, type_as=type_as)

        sparse_tensor = tf.sparse.SparseTensor(
            indices=tnp.stack([rows, cols]).T,
            values=data,
            dense_shape=shape
        )
        # if type_as is not None:
        #     sparse_tensor = self.from_numpy(sparse_tensor, type_as=type_as)
        # SparseTensor are not subscriptable so we use dense tensors
        return self.todense(sparse_tensor)

    def issparse(self, a):
        return isinstance(a, tf.sparse.SparseTensor)

    def tocsr(self, a):
        return a

    def eliminate_zeros(self, a, threshold=0.):
        if self.issparse(a):
            values = a.values
            if threshold > 0:
                mask = self.abs(values) <= threshold
            else:
                mask = values == 0
            return tf.sparse.retain(a, ~mask)
        else:
            if threshold > 0:
                a = tnp.where(self.abs(a) > threshold, a, 0.)
            return a

    def todense(self, a):
        if self.issparse(a):
            return tf.sparse.to_dense(tf.sparse.reorder(a))
        else:
            return a

    def where(self, condition, x=None, y=None):
        if x is None and y is None:
            return tnp.where(condition)
        else:
            return tnp.where(condition, x, y)

    def copy(self, a):
        return tf.identity(a)

    def allclose(self, a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
        return tnp.allclose(
            a, b, rtol=rtol, atol=atol, equal_nan=equal_nan
        )

    def dtype_device(self, a):
        return a.dtype, a.device.split("device:")[1]

    def assert_same_dtype_device(self, a, b):
        a_dtype, a_device = self.dtype_device(a)
        b_dtype, b_device = self.dtype_device(b)

        assert a_dtype == b_dtype, "Dtype discrepancy"
        assert a_device == b_device, f"Device discrepancy. First input is on {str(a_device)}, whereas second input is on {str(b_device)}"

    def squeeze(self, a, axis=None):
        return tnp.squeeze(a, axis=axis)

    def bitsize(self, type_as):
        return type_as.dtype.size * 8

    def device_type(self, type_as):
        return self.dtype_device(type_as)[1].split(":")[0]

    def _bench(self, callable, *args, n_runs=1, warmup_runs=1):
        results = dict()
        device_contexts = [tf.device("/CPU:0")]
        if len(tf.config.list_physical_devices('GPU')) > 0:  # pragma: no cover
            device_contexts.append(tf.device("/GPU:0"))

        for device_context in device_contexts:
            with device_context:
                for type_as in self.__type_list__:
                    inputs = [self.from_numpy(arg, type_as=type_as) for arg in args]
                    for _ in range(warmup_runs):
                        callable(*inputs)
                    t0 = time.perf_counter()
                    for _ in range(n_runs):
                        res = callable(*inputs)
                    _ = res.numpy()
                    t1 = time.perf_counter()
                    key = (
                        "Tensorflow",
                        self.device_type(inputs[0]),
                        self.bitsize(type_as)
                    )
                    results[key] = (t1 - t0) / n_runs

        return results

    def solve(self, a, b):
        return tf.linalg.solve(a, b)

    def trace(self, a):
        return tf.linalg.trace(a)

    def inv(self, a):
        return tf.linalg.inv(a)

    def sqrtm(self, a):
        L, V = tf.linalg.eigh(a)
        return (V * tf.sqrt(L)[None, :]) @ V.T

    def eigh(self, a):
        return tf.linalg.eigh(a)

    def kl_div(self, p, q, eps=1e-16):
        return tnp.sum(p * tnp.log(p / q + eps))

    def isfinite(self, a):
        return tnp.isfinite(a)

    def array_equal(self, a, b):
        return tnp.array_equal(a, b)

    def is_floating_point(self, a):
        return a.dtype.is_floating

    def tile(self, a, reps):
        return tnp.tile(a, reps)

    def floor(self, a):
        return tf.floor(a)

    def prod(self, a, axis=0):
        return tnp.prod(a, axis=axis)

    def sort2(self, a, axis=-1):
        return self.sort(a, axis), self.argsort(a, axis)

    def qr(self, a):
        return tf.linalg.qr(a)

    def atan2(self, a, b):
        return tf.math.atan2(a, b)

    def transpose(self, a, axes=None):
        return tf.transpose(a, perm=axes)

    def matmul(self, a, b):
        return tnp.matmul(a, b)

    # todo(okachaiev): replace this with a more reasonable implementation
    def nan_to_num(self, x, copy=True, nan=0.0, posinf=None, neginf=None):
        x = self.to_numpy(x)
        x = np.nan_to_num(x, copy=copy, nan=nan, posinf=posinf, neginf=neginf)
        return self.from_numpy(x)


if tf:
    # Only register tensorflow backend if it is installed
    _register_backend_implementation(TensorflowBackend)
