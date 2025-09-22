"""
=================================================
Solving Many Optimal Transport Problems in Parallel
=================================================

In some situations, one may want to solve many OT problems with the same
structure (same number of samples, same cost function, etc.) at the same time.

In that case using a for loop to solve the problems sequentially is inefficient.
This example shows how to use the batch solvers implemented in POT to solve
many problems in parallel on CPU or GPU (even more efficient on GPU).

"""

# Author: Paul Krzakala <paul.krzakala@gmail.com>
# License: MIT License

# sphinx_gallery_thumbnail_number = 1


#############################################################################
#
# Computing the Cost Matrices
# ---------------------------------------------
#
# We want to create a batch of optimal transport problems with
# :math:`n` samples in :math:`d` dimensions.
#
# To do this, we first need to compute the cost matrices for each problem.
#
# .. note::
#    A straightforward approach would be to use a Python loop and
#    :func:`ot.dist`.
#    However, this is inefficient when working with batches.
#
# Instead, you can directly use :func:`ot.batch.dist_batch`, which computes
# all cost matrices in parallel.

import ot
import numpy as np

n_problems = 4  # nb problems/batch size
n_samples = 8  # nb samples
dim = 2  # nb dimensions

np.random.seed(0)
samples_source = np.random.randn(n_problems, n_samples, dim)
samples_target = samples_source + 0.1 * np.random.randn(n_problems, n_samples, dim)

# Naive approach
M_list = []
for i in range(n_problems):
    M_list.append(
        ot.dist(samples_source[i], samples_target[i])
    )  # List of cost matrices n_samples x n_samples
# Batched approach
M_batch = ot.dist_batch(
    samples_source, samples_target
)  # Array of cost matrices n_problems x n_samples x n_samples

for i in range(n_problems):
    assert np.allclose(M_list[i], M_batch[i])

#############################################################################
#
# Solving the Problems
# ---------------------------------------------
#
# Once the cost matrices are computed, we can solve the corresponding
# optimal transport problems.
#
# .. note::
#    One option is to solve them sequentially with a Python loop using
#    :func:`ot.solve`.
#    This is simple but inefficient for large batches.
#
# Instead, you can use :func:`ot.batch.solve_batch`, which solves all
# problems in parallel.

reg = 1.0
max_iter = 100
tol = 1e-3

# Naive approach
results_values_list = []
for i in range(n_problems):
    res = ot.solve(M_list[i], reg=reg, max_iter=max_iter, tol=tol, reg_type="entropy")
    results_values_list.append(res.value_linear)

# Batched approach
results_batch = ot.solve_batch(
    M=M_batch, reg=reg, max_iter=max_iter, tol=tol, reg_type="entropy"
)
results_values_batch = results_batch.value_linear

assert np.allclose(np.array(results_values_list), results_values_batch, atol=tol * 10)

#############################################################################
#
# Comparing Computation Time
# ---------------------------------------------
#
# We now compare the runtime of the two approaches on larger problems.
#
# .. note::
#    The speedup obtained with :mod:`ot.batch` can be even more
#    significant when computations are performed on a GPU.


from time import perf_counter

n_problems = 128
n_samples = 8
dim = 2
reg = 10.0
max_iter = 1000
tol = 1e-3

samples_source = np.random.randn(n_problems, n_samples, dim)
samples_target = samples_source + 0.1 * np.random.randn(n_problems, n_samples, dim)


def benchmark_naive(samples_source, samples_target):
    start = perf_counter()
    for i in range(n_problems):
        M = ot.dist(samples_source[i], samples_target[i])
        res = ot.solve(M, reg=reg, max_iter=max_iter, tol=tol, reg_type="entropy")
    end = perf_counter()
    return end - start


def benchmark_batch(samples_source, samples_target):
    start = perf_counter()
    M_batch = ot.dist_batch(samples_source, samples_target)
    res_batch = ot.solve_batch(
        M=M_batch, reg=reg, max_iter=max_iter, tol=tol, reg_type="entropy"
    )
    end = perf_counter()
    return end - start


time_naive = benchmark_naive(samples_source, samples_target)
time_batch = benchmark_batch(samples_source, samples_target)

print(f"Naive approach time: {time_naive:.4f} seconds")
print(f"Batched approach time: {time_batch:.4f} seconds")

#############################################################################
#
# Gromov-Wasserstein
# ---------------------------------------------
#
# The :mod:`ot.batch` module also provides a batched Gromov-Wasserstein solver.
#
# .. note::
#    This solver is **not** equivalent to calling :func:`ot.solve_gromov`
#    repeatedly in a loop.
#
# Key differences:
#
# - :func:`ot.solve_gromov`
#   Uses the conditional gradient algorithm. Each inner iteration relies on
#   an exact EMD solver.
#
# - :func:`ot.batch.solve_gromov_batch`
#   Uses a proximal variant, where each inner iteration applies entropic
#   regularization.
#
# As a result:
#
# - :func:`ot.solve_gromov` is usually faster on CPU
# - :func:`ot.batch.solve_gromov_batch` is slower on CPU, but provides
#   better objective values.
#
# .. tip::
#    If your data is on a GPU, :func:`ot.batch.solve_gromov_batch`
#    is significantly faster AND provides better objective values.

from ot import solve_gromov, solve_gromov_batch


def benchmark_naive_gw(samples_source, samples_target):
    start = perf_counter()
    avg_value = 0
    for i in range(n_problems):
        C1 = ot.dist(samples_source[i], samples_source[i])
        C2 = ot.dist(samples_target[i], samples_target[i])
        res = solve_gromov(C1, C2, max_iter=1000, tol=tol)
        avg_value += res.value
    avg_value /= n_problems
    end = perf_counter()
    return end - start, avg_value


def benchmark_batch_gw(samples_source, samples_target):
    start = perf_counter()
    C1_batch = ot.dist_batch(samples_source, samples_source)
    C2_batch = ot.dist_batch(samples_target, samples_target)
    res_batch = solve_gromov_batch(
        C1_batch, C2_batch, reg=1, max_iter=100, max_iter_inner=50, tol=tol
    )
    avg_value = np.mean(res_batch.value)
    end = perf_counter()
    return end - start, avg_value


time_naive_gw, avg_value_naive_gw = benchmark_naive_gw(samples_source, samples_target)
time_batch_gw, avg_value_batch_gw = benchmark_batch_gw(samples_source, samples_target)

print(f"{'Method':<20}{'Time (s)':<15}{'Avg Value':<15}")
print(f"{'Naive GW':<20}{time_naive_gw:<15.4f}{avg_value_naive_gw:<15.4f}")
print(f"{'Batched GW':<20}{time_batch_gw:<15.4f}{avg_value_batch_gw:<15.4f}")

#############################################################################
#
# In summary: no more for loops!
# ---------------------------------------------

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(4, 4))
ax.text(0.5, 0.5, "For", fontsize=160, ha="center", va="center", zorder=0)
ax.axis("off")
ax.plot([0, 1], [0, 1], color="red", linewidth=10, zorder=1)
ax.plot([0, 1], [1, 0], color="red", linewidth=10, zorder=1)
plt.show()
