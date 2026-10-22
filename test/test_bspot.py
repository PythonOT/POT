"""Tests for module ot.bsp"""

# Author: Baptiste Genest <baptistegenest@gmail.com>
#
# License: MIT License

import hashlib
import json
import os
import subprocess
import sys
import warnings

import ot
import ot.bsp
import numpy as np
import pytest
from ot.backend import torch


def test_bsp_ot_exact_identity():
    # test that bsp-ot is exact for similarity transform
    n = 50
    rng = np.random.RandomState(0)

    A = rng.randn(n, 2)
    # create B as a similarity transform of A, scale by 2 and translate x coord by 1
    B = 2 * A + np.array([1, 0])

    _, perm, _ = ot.bsp.compute_bspot_bijection(A, B, 1)

    # check that the permutation is the identity
    np.testing.assert_allclose(perm, np.arange(n))


def test_bsp_ot_identity_null_cost():
    # test that the cost returned by bsp-ot is zero for the identity transform
    n = 50
    rng = np.random.RandomState(0)

    A = rng.randn(n, 2)

    cost, _, _ = ot.bsp.compute_bspot_bijection(A, A, 1)

    # check that the cost is zero
    np.testing.assert_allclose(cost, 0)


def test_bsp_ot_bijective():
    # test that the output of bsp-ot is indeed a bijection
    n = 50
    rng = np.random.RandomState(0)

    A = rng.randn(n, 2)
    # create B as a similarity transform of A, scale by 2 and translate x coord by 1
    B = 2 * A + np.array([1, 0])

    _, perm, _ = ot.bsp.compute_bspot_bijection(A, B, 1)

    # check that the permutation is a bijection
    assert len(set(perm)) == n, "Permutation is not a bijection"

    # check that the permutation is a valid permutation
    assert set(perm) == set(range(n)), "Permutation is not valid"


def test_bsp_ot_plan_merge_decrease():
    # test that merging two plans gives a smaller cost than the plans separately
    n = 50
    rng = np.random.RandomState(0)

    A = rng.randn(n, 2)
    B = rng.randn(n, 2)

    cost, plan, plans = ot.bsp.compute_bspot_bijection(A, B, 2)

    # evaluate mean squared cost lambda
    def cost_lambda(A, B, T):
        return np.mean(np.sum((A - B[T]) ** 2, axis=1))

    cost_plan_1 = cost_lambda(A, B, plans[0])
    cost_plan_2 = cost_lambda(A, B, plans[1])

    # check that the cost of the merged plan is smaller than the cost of the separate plans
    assert (
        cost <= cost_plan_1
    ), "Cost of merged plan is not smaller than one of the input plans"
    assert (
        cost <= cost_plan_2
    ), "Cost of merged plan is not smaller than one of the input plans"


def test_bsp_ot_relative_error():
    # test that the cost returned by bsp-ot is close to the cost of the optimal transport plan
    n = 100
    rng = np.random.RandomState(0)

    A = rng.randn(n, 2)
    B = rng.randn(n, 2)

    cost, perm, _ = ot.bsp.compute_bspot_bijection(A, B, 1000)

    w = ot.utils.unif(n)

    # solve exact with ot solver
    M = ot.dist(A, B, metric="sqeuclidean")
    G = ot.emd(w, w, M)

    cost_ot = np.sum(G * M)

    # check that the relative error is small
    relative_error = np.abs(cost - cost_ot) / (cost_ot + 1e-8)
    assert relative_error < 0.1, "Relative error is too large: {}".format(
        relative_error
    )


@pytest.mark.skipif(not torch, reason="PyTorch not installed")
def test_bsp_ot_torch_backend():
    # test that bsp-ot works with torch tensors
    n = 50
    rng = np.random.RandomState(0)

    A = rng.randn(n, 2)
    B = rng.randn(n, 2)

    A_torch = torch.from_numpy(A)
    B_torch = torch.from_numpy(B)
    A_torch.requires_grad_()
    B_torch.requires_grad_()

    cost, perm, _ = ot.bsp.compute_bspot_bijection(A_torch, B_torch, 1)

    # compute gradients w.r.t. A and B
    cost.backward()

    # advected points (should match B)
    A_new = A - A_torch.grad.detach().numpy() * n / 2

    # compute new cost with advected points
    cost_new, _, _ = ot.bsp.compute_bspot_bijection(A_new, B, 1)

    # cost should be zero
    np.testing.assert_allclose(cost_new, 0, atol=1e-5)


def _two_clouds(n=100, data_seed=0):
    rng = np.random.RandomState(data_seed)
    return rng.randn(n, 2), rng.randn(n, 2)


def _bsp_fingerprint(A, B, n_plans, seed):
    cost, perm, perms = ot.bsp.compute_bspot_bijection(A, B, n_plans, seed=seed)
    return float(np.asarray(cost)), np.asarray(perm), np.asarray(perms)


def test_bsp_seed_repeated_calls_are_reproducible():
    # ``seed`` is documented as "Random seed for reproducibility", so two calls with
    # the same arguments must return the same matchings, not just a similar cost.
    A, B = _two_clouds()
    cost0, perm0, perms0 = _bsp_fingerprint(A, B, 32, 0)
    cost1, perm1, perms1 = _bsp_fingerprint(A, B, 32, 0)

    np.testing.assert_array_equal(perm0, perm1)
    np.testing.assert_array_equal(perms0, perms1)
    np.testing.assert_allclose(cost0, cost1, rtol=0, atol=0)


def test_bsp_seed_is_independent_of_call_history():
    # A preceding call with another seed must not disturb the seeded stream.
    A, B = _two_clouds()
    _bsp_fingerprint(A, B, 16, 7)
    cost0, perm0, perms0 = _bsp_fingerprint(A, B, 32, 0)
    _bsp_fingerprint(A, B, 16, 11)
    cost1, perm1, perms1 = _bsp_fingerprint(A, B, 32, 0)

    np.testing.assert_array_equal(perm0, perm1)
    np.testing.assert_array_equal(perms0, perms1)
    np.testing.assert_allclose(cost0, cost1, rtol=0, atol=0)


_FRESH_PROCESS_SNIPPET = """
import hashlib, json, sys
import numpy as np
import ot
from ot.bsp import bsp_wrap

seed, n_plans = int(sys.argv[1]), int(sys.argv[2])
rng = np.random.RandomState(0)
A, B = rng.randn(100, 2), rng.randn(100, 2)
out = []
for _ in range(2):
    cost, perm, perms = ot.bsp.compute_bspot_bijection(A, B, n_plans, seed=seed)
    perm, perms = np.asarray(perm), np.asarray(perms)
    out.append({
        "cost": float(np.asarray(cost)),
        "perm": hashlib.sha256(perm.tobytes()).hexdigest(),
        "perms": hashlib.sha256(perms.tobytes()).hexdigest(),
    })
print(json.dumps({"results": out, "ot_file": ot.__file__,
                  "ext_sha256": hashlib.sha256(open(bsp_wrap.__file__, "rb").read()).hexdigest()}))
"""


def _ext_sha256():
    from ot.bsp import bsp_wrap

    return hashlib.sha256(open(bsp_wrap.__file__, "rb").read()).hexdigest()


def _bsp_in_fresh_process(seed, n_plans, omp_threads):
    """Run the fingerprint in a fresh interpreter.

    Returns the reported results, or None together with the raw process output when the
    child did not get as far as printing them.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ)
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    if omp_threads is not None:
        env["OMP_NUM_THREADS"] = str(omp_threads)
    proc = subprocess.run(
        [sys.executable, "-c", _FRESH_PROCESS_SNIPPET, str(seed), str(n_plans)],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        return None, proc.returncode, proc.stderr
    report = json.loads(proc.stdout)
    # a child that imported a different POT build would prove nothing
    assert report["ext_sha256"] == _ext_sha256(), "child loaded another build"
    return report["results"], proc.returncode, proc.stderr


def _run_in_fresh_process(seed=0, n_plans=32, omp_threads=None, attempts=3):
    # A child interpreter can die while importing POT on a host that is out of memory;
    # that shows up as a plain exit code 1 with a Python traceback and is retried.
    # Anything else -- a native crash such as 0xC0000005 or a signal -- is reported at
    # once, since a real defect is deterministic and retrying it would only hide it.
    failure = None
    for _ in range(attempts):
        results, returncode, stderr = _bsp_in_fresh_process(seed, n_plans, omp_threads)
        if results is not None:
            return results
        failure = (returncode, stderr)
        if returncode != 1:
            break
    raise AssertionError(
        f"BSP subprocess failed with rc={failure[0]}\n{failure[1]}"
    )


def test_bsp_seed_reproducible_in_fresh_process():
    # Reproducibility must not depend on whether the solver ran earlier in the
    # session, so the reference is a first call in a brand new interpreter.
    ref = _run_in_fresh_process()
    assert ref[0] == ref[1], "two first calls in one fresh interpreter already disagree"

    A, B = _two_clouds()
    _bsp_fingerprint(A, B, 16, 7)  # make the current session non-pristine
    cost, perm, perms = _bsp_fingerprint(A, B, 32, 0)
    assert cost == ref[0]["cost"]
    assert hashlib.sha256(np.asarray(perm).tobytes()).hexdigest() == ref[0]["perm"]
    assert hashlib.sha256(np.asarray(perms).tobytes()).hexdigest() == ref[0]["perms"]


def test_bsp_result_independent_of_openmp_thread_count():
    # The candidate plans are generated by an OpenMP loop. How the loop happens to be
    # split over threads is not part of the solver, so the answer must not change with
    # the number of available threads (CI images and laptops differ here). Staying
    # parallel is a performance property, checked by the benchmarks, not by this test.
    ref = _run_in_fresh_process(omp_threads=1)
    for threads in (2, 4):
        got = _run_in_fresh_process(omp_threads=threads)
        assert got == ref, f"OMP_NUM_THREADS={threads} gives a different result"


def test_bsp_candidate_plans_are_determined_by_seed_and_index():
    # The property the seeding gives: candidate plan i depends on
    # (A, B, p, gaussian_slicing, seed, i) alone, so asking for 4 plans and asking for 37
    # must agree on the first four rows, whatever the OpenMP chunking made of them and
    # whatever was computed before.  Whether the plans are pairwise distinct is a property
    # of the instance and of the standard library's distributions, not of the seeding --
    # for a similarity transform every BSP candidate legitimately is the identity -- so it
    # is not asserted here.
    A, B = _two_clouds()
    _, _, few = _bsp_fingerprint(A, B, 4, 0)
    _bsp_fingerprint(A, B, 16, 5)  # unrelated call in between
    _, _, many = _bsp_fingerprint(A, B, 37, 0)
    np.testing.assert_array_equal(few, many[:4])
    _, _, few_again = _bsp_fingerprint(A, B, 4, 0)
    np.testing.assert_array_equal(few, few_again)


def test_bsp_permutation_structure_and_cost():
    # Structural invariants that must hold whatever random sequence is drawn.
    # compute_bspot_bijection warns when the cost it recomputes from ``perm`` disagrees
    # with the cost the solver reports for it, so warnings are turned into errors here:
    # that compares the solver's own bookkeeping against the permutation it returned.
    A, B = _two_clouds(n=80, data_seed=3)
    M = ot.dist(A, B, metric="sqeuclidean")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for seed in (0, 1, 2):
            cost, perm, perms = _bsp_fingerprint(A, B, 16, seed)
    perm = np.asarray(perm)
    np.testing.assert_array_equal(np.sort(perm), np.arange(80))
    np.testing.assert_allclose(cost, np.mean(M[np.arange(80), perm]), rtol=1e-9)
    for row in np.asarray(perms):
        np.testing.assert_array_equal(np.sort(row), np.arange(80))
