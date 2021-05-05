"""Tests for backend module """

# Author: Remi Flamary <remi.flamary@polytechnique.edu>
#
# License: MIT License

import ot
import ot.backend
from ot.backend import torch

import pytest

import numpy as np
from numpy.testing import assert_array_almost_equal_nulp

from ot.backend import get_backend, get_backend_list, to_numpy


def test_get_backend_list():

    lst = get_backend_list()

    assert len(lst) > 0
    assert isinstance(lst[0], ot.backend.NumpyBackend)


def test_to_numpy():

    for nx in get_backend_list():

        v = nx.zeros(10)
        M = nx.ones((10, 10))

        v2 = to_numpy(v)
        assert isinstance(v2, np.ndarray)

        v2, M2 = to_numpy(v, M)


def test_get_backend():

    A = np.zeros((3, 2))
    B = np.zeros((3, 1))

    nx = get_backend(A)
    assert nx.__name__ == 'numpy'

    nx = get_backend(A, B)
    assert nx.__name__ == 'numpy'

    # error if no parameters
    with pytest.raises(ValueError):
        get_backend()

    # error if unknown types
    with pytest.raises(ValueError):
        get_backend(1, 2.0)

    # test torch
    if torch:

        A2 = torch.from_numpy(A)
        B2 = torch.from_numpy(B)

        nx = get_backend(A2)
        assert nx.__name__ == 'torch'

        nx = get_backend(A2, B2)
        assert nx.__name__ == 'torch'

        # test not unique types in input
        with pytest.raises(ValueError):
            get_backend(A, B2)


def test_convert_between_backends():

    A = np.zeros((3, 2))
    B = np.zeros((3, 1))

    for nx in get_backend_list()[1:]:

        A2 = nx.from_numpy(A)
        B2 = nx.from_numpy(B)

        assert isinstance(A2, nx.__type__)
        assert isinstance(B2, nx.__type__)

        nx2 = get_backend(A2, B2)

        assert nx2.__name__ == nx.__name__

        assert_array_almost_equal_nulp(nx.to_numpy(A2), A)
        assert_array_almost_equal_nulp(nx.to_numpy(B2), B)


def test_empty_backend():

    rnd = np.random.RandomState(0)
    M = rnd.randn(10, 3)
    v = rnd.randn(3)

    nx = ot.backend.Backend()

    with pytest.raises(NotImplementedError):
        nx.from_numpy(M)
    with pytest.raises(NotImplementedError):
        nx.zeros((10, 3))
    with pytest.raises(NotImplementedError):
        nx.ones((10, 3))
    with pytest.raises(NotImplementedError):
        nx.full((10, 3), 3.14)
    with pytest.raises(NotImplementedError):
        nx.sum(M)
    with pytest.raises(NotImplementedError):
        nx.max(M)
    with pytest.raises(NotImplementedError):
        nx.min(M)
    with pytest.raises(NotImplementedError):
        nx.abs(M)
    with pytest.raises(NotImplementedError):
        nx.log(M)
    with pytest.raises(NotImplementedError):
        nx.exp(M)
    with pytest.raises(NotImplementedError):
        nx.dot(v, v)
    with pytest.raises(NotImplementedError):
        nx.exp(M)
    with pytest.raises(NotImplementedError):
        nx.any(M)
    with pytest.raises(NotImplementedError):
        nx.isnan(M)
    with pytest.raises(NotImplementedError):
        nx.isinf(M)
    with pytest.raises(NotImplementedError):
        nx.einsum('ij->i', M)


def test_func_backends():

    rnd = np.random.RandomState(0)
    M = rnd.randn(10, 3)
    v = rnd.randn(3)

    lst_tot = []

    for nx in get_backend_list():

        print('Backend: ', nx.__name__)

        lst_b = []

        Mb = nx.from_numpy(M)
        vb = nx.from_numpy(v)

        A = nx.zeros((10, 3))
        A = nx.zeros((10, 3), type_as=Mb)
        lst_b.append(nx.to_numpy(A))

        A = nx.ones((10, 3))
        A = nx.ones((10, 3), type_as=Mb)
        lst_b.append(nx.to_numpy(A))

        A = nx.full((10, 3), 3.14)
        A = nx.full((10, 3), 3.14, type_as=Mb)
        lst_b.append(nx.to_numpy(A))

        A = nx.sum(Mb)
        lst_b.append(nx.to_numpy(A))

        A = nx.sum(Mb, axis=1, keepdims=True)
        lst_b.append(nx.to_numpy(A))

        A = nx.max(Mb)
        lst_b.append(nx.to_numpy(A))

        A = nx.max(Mb, axis=1, keepdims=True)
        lst_b.append(nx.to_numpy(A))

        A = nx.min(Mb)
        lst_b.append(nx.to_numpy(A))

        A = nx.min(Mb, axis=1, keepdims=True)
        lst_b.append(nx.to_numpy(A))

        A = nx.abs(Mb)
        lst_b.append(nx.to_numpy(A))

        A = nx.log(A)
        lst_b.append(nx.to_numpy(A))

        A = nx.exp(Mb)
        lst_b.append(nx.to_numpy(A))

        A = nx.dot(vb, vb)
        lst_b.append(nx.to_numpy(A))

        A = nx.dot(Mb, vb)
        lst_b.append(nx.to_numpy(A))

        A = nx.dot(Mb, Mb.T)
        lst_b.append(nx.to_numpy(A))

        A = nx.norm(vb)
        lst_b.append(nx.to_numpy(A))

        A = nx.any(vb > 0)
        lst_b.append(nx.to_numpy(A))

        A = nx.isnan(vb)
        lst_b.append(nx.to_numpy(A))

        A = nx.isinf(vb)
        lst_b.append(nx.to_numpy(A))

        A = nx.einsum('ij->i', Mb)
        lst_b.append(nx.to_numpy(A))

        A = nx.einsum('ij,j->i', Mb, vb)
        lst_b.append(nx.to_numpy(A))

        lst_tot.append(lst_b)

    lst_np = lst_tot[0]
    for lst_b in lst_tot[1:]:

        for a1, a2 in zip(lst_np, lst_b):

            assert np.allclose(a1, a2)
