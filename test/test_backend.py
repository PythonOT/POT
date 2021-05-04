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

from ot.backend import get_backend, get_backend_list


def test_get_backend_list():

    lst = get_backend_list()

    assert len(lst) > 0
    assert isinstance(lst[0], ot.backend.NumpyBackend)


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

    for b in get_backend_list()[1:]:

        A2 = b.from_numpy(A)
        B2 = b.from_numpy(B)

        nx = get_backend(A2, B2)

        assert nx.__name__ == b.__name__

        assert_array_almost_equal_nulp(b.to_numpy(A2), A)
        assert_array_almost_equal_nulp(b.to_numpy(B2), B)
