"""Tests for ot solvers"""

# Author: Remi Flamary <remi.flamary@polytechnique.edu>
#
# License: MIT License


import itertools
import numpy as np
import pytest

import ot


lst_reg = [None, 1.0]
lst_reg_type = ['KL', 'entropy', 'L2']
lst_unbalanced = [None, 0.9]
lst_unbalanced_type = ['KL', 'L2', 'TV']


def assert_allclose_sol(sol1, sol2):

    lst_attr = ['value', 'value_linear', 'plan',
                'potential_a', 'potential_b', 'marginal_a', 'marginal_b']

    nx1 = sol1._backend if sol1._backend is not None else ot.backend.NumpyBackend()
    nx2 = sol2._backend if sol2._backend is not None else ot.backend.NumpyBackend()

    for attr in lst_attr:
        try:
            np.allclose(nx1.to_numpy(getattr(sol1, attr)), nx2.to_numpy(getattr(sol2, attr)))
        except NotImplementedError:
            pass


def test_solve(nx):
    n_samples_s = 10
    n_samples_t = 7
    n_features = 2
    rng = np.random.RandomState(0)

    x = rng.randn(n_samples_s, n_features)
    y = rng.randn(n_samples_t, n_features)
    a = ot.utils.unif(n_samples_s)
    b = ot.utils.unif(n_samples_t)

    M = ot.dist(x, y)

    # solve unif weights
    sol0 = ot.solve(M)

    print(sol0)

    # solve signe weights
    sol = ot.solve(M, a, b)

    # check some attributes
    sol.potentials
    sol.sparse_plan
    sol.marginals
    sol.status

    assert_allclose_sol(sol0, sol)

    # solve in backend
    ab, bb, Mb = nx.from_numpy(a, b, M)
    solb = ot.solve(M, a, b)

    assert_allclose_sol(sol, solb)

    # test not implemented unbalanced and check raise
    with pytest.raises(NotImplementedError):
        sol0 = ot.solve(M, unbalanced=1, unbalanced_type='cryptic divergence')

    # test not implemented reg_type and check raise
    with pytest.raises(NotImplementedError):
        sol0 = ot.solve(M, reg=1, reg_type='cryptic divergence')


@pytest.mark.parametrize("reg,reg_type,unbalanced,unbalanced_type", itertools.product(lst_reg, lst_reg_type, lst_unbalanced, lst_unbalanced_type))
def test_solve_grid(nx, reg, reg_type, unbalanced, unbalanced_type):
    n_samples_s = 10
    n_samples_t = 7
    n_features = 2
    rng = np.random.RandomState(0)

    x = rng.randn(n_samples_s, n_features)
    y = rng.randn(n_samples_t, n_features)
    a = ot.utils.unif(n_samples_s)
    b = ot.utils.unif(n_samples_t)

    M = ot.dist(x, y)

    try:

        # solve unif weights
        sol0 = ot.solve(M, reg=reg, reg_type=reg_type, unbalanced=unbalanced, unbalanced_type=unbalanced_type)

        # solve signe weights
        sol = ot.solve(M, a, b, reg=reg, reg_type=reg_type, unbalanced=unbalanced, unbalanced_type=unbalanced_type)

        assert_allclose_sol(sol0, sol)

        # solve in backend
        ab, bb, Mb = nx.from_numpy(a, b, M)
        solb = ot.solve(M, a, b, reg=reg, reg_type=reg_type, unbalanced=unbalanced, unbalanced_type=unbalanced_type)

        assert_allclose_sol(sol, solb)
    except NotImplementedError:
        pass


def test_solve_not_implemented(nx):

    n_samples_s = 10
    n_samples_t = 7
    n_features = 2
    rng = np.random.RandomState(0)

    x = rng.randn(n_samples_s, n_features)
    y = rng.randn(n_samples_t, n_features)

    M = ot.dist(x, y)

    # test not implemented and check raise
    with pytest.raises(NotImplementedError):
        ot.solve(M, reg=1.0, reg_type='cryptic divergence')
    with pytest.raises(NotImplementedError):
        ot.solve(M, unbalanced=1.0, unbalanced_type='cryptic divergence')

    # pairs of incompatible divergences
    with pytest.raises(NotImplementedError):
        ot.solve(M, reg=1.0, reg_type='kl', unbalanced=1.0, unbalanced_type='tv')
