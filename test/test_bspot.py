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


def _mean_squared_cost(A, B, perm):
    # cost of ``perm`` computed here, independently of what the solver reports
    return float(np.mean(np.sum((A - B[np.asarray(perm)]) ** 2, axis=1)))


def test_bsp_merge_fewer_plans_than_samples():
    # perms is a stack of (n_plans, n_samples) permutations, so the merge has n_plans of
    # them to read. With one plan over 32 samples the old code read one plan per sample
    # and walked off the end of the array (issue #910).
    n = 32
    rng = np.random.RandomState(0)

    A = rng.randn(n, 2)
    B = rng.randn(n, 2)
    perms = np.arange(n, dtype=np.int32).reshape(1, n)

    with warnings.catch_warnings():
        # merge_bijections warns when its reported cost does not match the permutation
        warnings.simplefilter("error")
        cost, perm = ot.bsp.merge_bijections(A, B, perms, 2)

    perm = np.asarray(perm)
    np.testing.assert_array_equal(np.sort(perm), np.arange(n))
    np.testing.assert_allclose(cost, _mean_squared_cost(A, B, perm), rtol=1e-9)
    assert cost <= _mean_squared_cost(A, B, np.arange(n)) + 1e-12


def test_bsp_merge_uses_every_supplied_plan():
    # The plans past the n_samples-th one must reach the merge. Both clouds are the same,
    # so the identity is the only plan with zero cost and the only one that can produce
    # it: if it is dropped the merged cost stays on the mediocre plan.
    n = 8
    rng = np.random.RandomState(0)

    A = rng.randn(n, 2)
    B = A.copy()

    mediocre = np.roll(np.arange(n), 1).astype(np.int32)
    without = np.tile(mediocre, (n, 1))
    with_last = np.vstack([without, np.arange(n)]).astype(np.int32)

    cost_without, _ = ot.bsp.merge_bijections(A, B, without, 2)
    cost_with, perm_with = ot.bsp.merge_bijections(A, B, with_last, 2)

    assert cost_with < cost_without, "the plan past n_samples was ignored"
    np.testing.assert_array_equal(np.asarray(perm_with), np.arange(n))
    np.testing.assert_allclose(cost_with, 0.0, atol=1e-12)


def test_bsp_merge_rejects_plans_of_wrong_length():
    # A stack whose rows are not one entry per point cannot be merged; it must fail here
    # instead of being read as a number of plans by C++.
    n = 8
    rng = np.random.RandomState(0)

    A = rng.randn(n, 2)
    B = rng.randn(n, 2)

    with pytest.raises(ValueError, match="n_samples is the number of points"):
        ot.bsp.merge_bijections(A, B, np.tile(np.roll(np.arange(7), 1), (3, 1)), 2)

    with pytest.raises(ValueError, match="2D"):
        ot.bsp.merge_bijections(A, B, np.arange(n, dtype=np.int32), 2)

    with pytest.raises(ValueError, match="2D"):
        ot.bsp.merge_bijections(A, B, np.zeros((n, n, 1), dtype=np.int32), 2)

    # the same refusal at the native boundary, so no caller can reach the overrun
    from ot.bsp import bsp_wrap

    with pytest.raises(ValueError, match="one entry per point"):
        bsp_wrap.merge_bijections_c(
            np.ascontiguousarray(A, dtype=np.float64),
            np.ascontiguousarray(B, dtype=np.float64),
            np.ascontiguousarray(
                np.tile(np.roll(np.arange(7), 1), (3, 1)), dtype=np.int32
            ),
            2,
        )


def test_bsp_merge_on_a_standard_stack():
    # The ordinary case must keep its documented behaviour: a bijection no worse than any
    # plan it merged, and a cost that is the cost of that bijection.
    n = 50
    rng = np.random.RandomState(0)

    A = rng.randn(n, 2)
    B = rng.randn(n, 2)
    perms = np.stack([np.roll(np.arange(n), s) for s in (1, 7, 23, 41)]).astype(np.int32)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cost, perm = ot.bsp.merge_bijections(A, B, perms, 2)

    perm = np.asarray(perm)
    np.testing.assert_array_equal(np.sort(perm), np.arange(n))
    np.testing.assert_allclose(cost, _mean_squared_cost(A, B, perm), rtol=1e-9)
    for plan in perms:
        assert cost <= _mean_squared_cost(A, B, plan) + 1e-12


def test_bsp_merge_converts_dtypes_and_refuses_views():
    # np.asarray(perms, dtype=np.int32) already accepted these, and the fix must not change
    # that; a non-contiguous view was refused before the fix too, rather than read loosely.
    n = 8
    rng = np.random.RandomState(0)

    A = rng.randn(n, 2)
    B = rng.randn(n, 2)
    plans = np.tile(np.roll(np.arange(n), 1), (4, 1))

    ref_cost, ref_perm = ot.bsp.merge_bijections(A, B, plans.astype(np.int32), 2)
    for other in (plans.astype(np.int64), plans.astype(np.float64), plans.tolist()):
        cost, perm = ot.bsp.merge_bijections(A, B, other, 2)
        np.testing.assert_allclose(cost, ref_cost, rtol=0, atol=0)
        np.testing.assert_array_equal(np.asarray(perm), np.asarray(ref_perm))

    view = np.asfortranarray(plans.astype(np.int32))
    assert not view.flags["C_CONTIGUOUS"]
    with pytest.raises(ValueError, match="C-contiguous"):
        ot.bsp.merge_bijections(A, B, view, 2)


def test_bsp_merge_rejects_empty_plan_stack():
    # An empty stack leaves the merge helper with nothing to read, which crashes the
    # interpreter, so it has to be refused before the helper is reached.
    n = 8
    rng = np.random.RandomState(0)

    A = rng.randn(n, 2)
    B = rng.randn(n, 2)

    with pytest.raises(ValueError, match="at least one plan"):
        ot.bsp.merge_bijections(A, B, np.zeros((0, n), dtype=np.int32), 2)

    # the native entry point is reachable on its own, so it must refuse it too
    from ot.bsp import bsp_wrap

    with pytest.raises(ValueError, match="at least one plan"):
        bsp_wrap.merge_bijections_c(
            np.ascontiguousarray(A),
            np.ascontiguousarray(B),
            np.zeros((0, n), dtype=np.int32),
            2,
        )


def test_bsp_merge_rejects_rows_that_are_not_permutations():
    # A row describes a bijection only if it is a permutation of 0..n-1. Anything else
    # makes the merge index outside the point clouds, so it is refused rather than read.
    n = 8
    rng = np.random.RandomState(0)

    A = rng.randn(n, 2)
    B = rng.randn(n, 2)
    identity = np.arange(n, dtype=np.int32)

    invalid = [
        np.array([0, 0, 2, 3, 4, 5, 6, 7], dtype=np.int32),  # repeated entry
        np.array([-1, 1, 2, 3, 4, 5, 6, 7], dtype=np.int32),  # negative index
        np.array([0, 1, 2, 3, 4, 5, 6, 8], dtype=np.int32),  # past the last point
        np.array([0, 1, 2, 3, 4, 5, 6, 6], dtype=np.int32),  # missing entry
    ]

    from ot.bsp import bsp_wrap

    for row in invalid:
        stack = row.reshape(1, n)
        with pytest.raises(ValueError, match="permutation of 0..n_samples-1"):
            ot.bsp.merge_bijections(A, B, stack, 2)
        with pytest.raises(ValueError, match="permutation of 0..n_samples-1"):
            bsp_wrap.merge_bijections_c(
                np.ascontiguousarray(A),
                np.ascontiguousarray(B),
                np.ascontiguousarray(stack, dtype=np.int32),
                2,
            )

        # one bad row is enough to refuse the whole stack, wherever it sits in it
        mixed = np.stack([identity, row, identity])
        with pytest.raises(ValueError, match="permutation of 0..n_samples-1"):
            ot.bsp.merge_bijections(A, B, mixed, 2)
        with pytest.raises(ValueError, match="permutation of 0..n_samples-1"):
            bsp_wrap.merge_bijections_c(
                np.ascontiguousarray(A),
                np.ascontiguousarray(B),
                np.ascontiguousarray(mixed, dtype=np.int32),
                2,
            )


def test_bsp_merge_rejects_plans_that_only_look_valid_after_casting():
    # The entries are checked before the conversion to int32. That cast truncates
    # fractional values and wraps integers that do not fit, and either one would silently
    # turn an invalid row into a permutation if the check ran after it.
    n = 8
    rng = np.random.RandomState(0)

    A = rng.randn(n, 2)
    B = rng.randn(n, 2)

    fractional = np.arange(n, dtype=np.float64).reshape(1, n)
    fractional[0, 3] = 3.5
    with pytest.raises(ValueError, match="permutation of 0..n_samples-1"):
        ot.bsp.merge_bijections(A, B, fractional, 2)

    wrapped = np.arange(n, dtype=np.int64).reshape(1, n)
    wrapped[0, 3] = 2**32 + 3
    assert wrapped.astype(np.int32)[0, 3] == 3, "the cast keeps the low bits only"
    with pytest.raises(ValueError, match="permutation of 0..n_samples-1"):
        ot.bsp.merge_bijections(A, B, wrapped, 2)

    nan = np.arange(n, dtype=np.float64).reshape(1, n)
    nan[0, 5] = np.nan
    with pytest.raises(ValueError, match="permutation of 0..n_samples-1"):
        ot.bsp.merge_bijections(A, B, nan, 2)


def _two_clouds(n=100, data_seed=0):
    rng = np.random.RandomState(data_seed)
    return rng.randn(n, 2), rng.randn(n, 2)


def _bsp_fingerprint(A, B, n_plans, seed):
    cost, perm, perms = ot.bsp.compute_bspot_bijection(A, B, n_plans, seed=seed)
    return float(np.asarray(cost)), np.asarray(perm), np.asarray(perms)


def test_bsp_seed_repeated_calls_are_reproducible():
    # Two calls with the same arguments must return the same matchings, not
    # just a similar cost.
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


# Third-party imports write to stdout on import (e.g. the KeOps warning printed on
# machines without CUDA), so the child prefixes its record with a unique marker.
_RESULT_MARKER = "__POT_BSP_JSON_V1__="

_FRESH_PROCESS_SNIPPET = """
import hashlib, json, sys
import numpy as np
import ot
from ot.bsp import bsp_wrap

seed, n_plans, marker = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
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
print(marker + json.dumps({"results": out, "ot_file": ot.__file__,
                           "ext_sha256": hashlib.sha256(open(bsp_wrap.__file__, "rb").read()).hexdigest()}),
      flush=True)
"""


def _ext_sha256():
    from ot.bsp import bsp_wrap

    return hashlib.sha256(open(bsp_wrap.__file__, "rb").read()).hexdigest()


def _parse_fresh_process_report(stdout, stderr, returncode):
    """Return the JSON record printed by the child after ``_RESULT_MARKER``.

    The marker is searched for anywhere in stdout, since the child's record is
    neither first nor necessarily last.
    """
    context = (
        f"returncode={returncode}\n"
        f"--- child stdout ---\n{stdout!r}\n"
        f"--- child stderr ---\n{stderr!r}"
    )
    occurrences = stdout.count(_RESULT_MARKER)
    if occurrences == 0:
        raise AssertionError(
            f"child stdout contains no {_RESULT_MARKER!r} record\n{context}"
        )
    if occurrences > 1:
        raise AssertionError(
            f"child stdout contains {occurrences} {_RESULT_MARKER!r} records\n{context}"
        )
    start = stdout.index(_RESULT_MARKER) + len(_RESULT_MARKER)
    end = stdout.find("\n", start)
    payload = stdout[start:] if end == -1 else stdout[start:end]
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise AssertionError(
            f"invalid JSON in {_RESULT_MARKER!r} record: {payload!r}\n{context}"
        ) from exc


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
        [
            sys.executable,
            "-c",
            _FRESH_PROCESS_SNIPPET,
            str(seed),
            str(n_plans),
            _RESULT_MARKER,
        ],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        return None, proc.returncode, proc.stderr
    report = _parse_fresh_process_report(proc.stdout, proc.stderr, proc.returncode)
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


def test_fresh_process_report_parsing_ignores_foreign_stdout():
    report = {"results": [{"cost": 0.5}], "ot_file": "ot.py", "ext_sha256": "abc"}
    stdout = (
        "[KeOps] Warning : CUDA libraries not found or could not be loaded; "
        "Switching to CPU only."
        + _RESULT_MARKER
        + json.dumps(report)
        + "\ntrailing log line\n"
    )
    assert _parse_fresh_process_report(stdout, "", 0) == report


def test_fresh_process_report_parsing_diagnoses_protocol_errors():
    record = _RESULT_MARKER + '{"results": []}'

    with pytest.raises(AssertionError, match="contains no") as missing:
        _parse_fresh_process_report("only foreign output\n", "boom", 0)
    assert "boom" in str(missing.value)

    with pytest.raises(AssertionError, match="contains 2"):
        _parse_fresh_process_report(record + "\n" + record + "\n", "", 0)

    with pytest.raises(AssertionError, match="invalid JSON"):
        _parse_fresh_process_report(_RESULT_MARKER + "{not json}\n", "", 0)


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
    # How the candidate-plan loop happens to be split over threads is not part of the
    # solver, so the answer must not change with the number of available threads.
    # Staying parallel is a performance property, not asserted here.
    ref = _run_in_fresh_process(omp_threads=1)
    for threads in (2, 4):
        got = _run_in_fresh_process(omp_threads=threads)
        assert got == ref, f"OMP_NUM_THREADS={threads} gives a different result"


def test_bsp_candidate_plans_are_determined_by_seed_and_index():
    # The property the seeding gives: candidate plan i depends on
    # (A, B, p, gaussian_slicing, seed, i) alone, so asking for 4 plans and asking for 37
    # must agree on the first four rows. Whether the plans are pairwise distinct is a
    # property of the instance and of the standard library's distributions, not of the
    # seeding -- for a similarity transform every candidate legitimately is the identity
    # -- so it is not asserted here.
    A, B = _two_clouds()
    _, _, few = _bsp_fingerprint(A, B, 4, 0)
    _bsp_fingerprint(A, B, 16, 5)  # unrelated call in between
    _, _, many = _bsp_fingerprint(A, B, 37, 0)
    np.testing.assert_array_equal(few, many[:4])
    _, _, few_again = _bsp_fingerprint(A, B, 4, 0)
    np.testing.assert_array_equal(few, few_again)


def test_bsp_permutation_structure_and_cost():
    # Invariants that must hold whatever random sequence is drawn. The solver warns when
    # the cost it recomputes from ``perm`` disagrees with the cost it reports for it, so
    # warnings are turned into errors here.
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
