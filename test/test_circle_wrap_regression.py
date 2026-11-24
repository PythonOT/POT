"""Regression for the all-negative CDF branch in circular Sinkhorn costs."""

import numpy as np

from ot.lp.solver_circle import binary_search_circle, ot_cost_on_circle


def _case():
    x = np.array(
        [
            0.16941984065866045, 0.1836324134419436, 0.44653045879629283,
            0.4820980124097798, 0.4917087310586765, 0.8126037842950916,
            0.8510944169432116, 0.9797740750713527,
        ]
    )
    y = np.array(
        [
            0.27379110343532287, 0.3658703056370922, 0.471481879564172,
            0.4991513312359799, 0.634675836166241, 0.7040614541610074,
            0.8349212395014908,
        ]
    )
    a = np.array(
        [
            0.11527813452574018, 0.13388447349681654, 0.1598842189785102,
            0.06981890027127201, 0.16354173746775985, 0.1749747944285811,
            0.037675572756166016, 0.14494216807515392,
        ]
    )
    b = np.array(
        [
            0.043465847478736226, 0.07830347135588803, 0.15577476895197248,
            0.058202097877683455, 0.2908482240722164, 0.2924309055386121,
            0.08097468472489115,
        ]
    )
    return x, y, a, b


def test_tiny_negative_theta_is_continuous():
    x, y, a, b = _case()
    uc = np.cumsum(a)[None, :]
    vc = np.cumsum(b)[None, :]
    assert uc[0, -1] < 1.0 and vc[0, -1] < 1.0  # float accumulation in this case
    theta0 = np.zeros((1, len(y)))
    theta_neg = np.full((1, len(y)), -1e-16)
    c0 = ot_cost_on_circle(theta0, x[None, :], y[None, :], uc, vc, 2)[0]
    cneg = ot_cost_on_circle(theta_neg, x[None, :], y[None, :], uc, vc, 2)[0]
    np.testing.assert_allclose(cneg, c0, atol=1e-10, rtol=1e-10)


def test_binary_search_matches_independent_optimum():
    x, y, a, b = _case()
    got = np.asarray(binary_search_circle(x, y, a, b, p=2, eps=1e-6))[0]
    # SciPy linprog / exact Fraction breakpoint enumeration value.
    np.testing.assert_allclose(got, 0.03158514371890842, atol=1e-9, rtol=1e-9)


def test_batch_composition_invariance():
    """An all-negative batch must not depend on another batch's sign mask."""
    from ot.lp.solver_circle import derivative_cost_on_circle

    theta = -1e-17
    xA = np.array([0.0, 0.125, 0.25])
    y = np.array([0.0, 0.125, 0.25, 0.875])
    aA = np.array([2.0, 3.0, 1.0]) / 6
    bA = np.array([1.0, 1.0, 3.0, 2.0]) / 7
    xB = np.array([0.5, 0.625, 0.75])
    aB = np.ones(3) / 3
    bB = np.ones(4) / 4

    def value(x, a, b):
        X = x[None, :] if x.ndim == 1 else x
        A = a[None, :] if a.ndim == 1 else a
        B = b[None, :] if b.ndim == 1 else b
        Y = np.repeat(y[None, :], X.shape[0], axis=0)
        B = np.repeat(B, X.shape[0], axis=0) if B.shape[0] == 1 else B
        th = np.full((X.shape[0], Y.shape[1]), theta)
        c = np.asarray(ot_cost_on_circle(th, X, Y, np.cumsum(A, 1), np.cumsum(B, 1), 2))
        dp, dm = derivative_cost_on_circle(th, X, Y, np.cumsum(A, 1), np.cumsum(B, 1), 2)
        return c, dp, dm

    a_alone = value(xA, aA, bA)[0][0]
    a_with_b = value(np.stack([xA, xB]), np.stack([aA, aB]), np.stack([bA, bB]))[0][0]
    a_with_ba = value(np.stack([xB, xA]), np.stack([aB, aA]), np.stack([bB, bA]))[0][1]
    a_repeated = value(np.stack([xA, xA]), np.stack([aA, aA]), np.stack([bA, bA]))[0][0]
    np.testing.assert_allclose(a_with_b, a_alone, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(a_with_ba, a_alone, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(a_repeated, a_alone, rtol=1e-12, atol=1e-12)
