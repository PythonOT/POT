# -*- coding: utf-8 -*-
r"""
=================================================================================
1D Wasserstein barycenter: LP Barycenter vs DEMD
=================================================================================

Compares the performance of two methods for computing the 1D Wasserstein
barycenter:
1. Linear Programming (LP) method
2. Discrete Earth Mover's Distance (DEMD) method

The comparison is performed by generating random Gaussian distributions with
increasing numbers of bins and measuring the computation time of each method.
The results are then plotted for visualization.
"""

# Author: Ronak Mehta <ronakrm@cs.wisc.edu>
#         Xizheng Yu <xyu354@wisc.edu>
#
# License: MIT License

import numpy as np
import matplotlib.pyplot as pl
import ot

# %%
# Define 1d Barycenter Function and Compare Function
# --------------------------------------------------
# This section defines the functions `lp_1d_bary` and `compare_all`. The
# `lp_1d_bary` function computes the barycenter using the LP method. The
# `compare_all` function compares the LP method and DEMD method in terms of
# computation time and objective values.


def lp_1d_bary(data, M, n, d):
    A = np.vstack(data).T

    alpha = 1.0  # /d  # 0<=alpha<=1
    weights = np.array(d*[alpha])

    bary, bary_log = ot.lp.barycenter(
        A, M, weights, solver='interior-point', verbose=False, log=True)

    return bary_log['fun'], bary


def compare_all(data, M, n, d):
    print('IP LP Iterations:')
    ot.tic()
    lp_bary, lp_obj = lp_1d_bary(np.vstack(data), M, n, d)
    lp_time = ot.toc('')
    print('Obj\t: ', lp_bary)
    print('Time\t: ', lp_time)

    print('')
    print('D-EMD Algorithm:')
    ot.tic()
    demd_obj = ot.demd(np.vstack(data), n, d)
    demd_time = ot.toc('')
    print('Obj\t: ', demd_obj)
    print('Time\t: ', demd_time)
    return lp_time, demd_time

# %%
# 2 Random Dists with Increasing Bins
# -----------------------------------
# Generates two random Gaussian distributions with increasing bin
# sizes and compares the LP and DEMD methods


def random2d(n=4):
    print('*'*10)
    d = 2
    # Gaussian distributions
    a1 = ot.datasets.make_1D_gauss(n, m=20, s=5)  # m= mean, s= std
    a2 = ot.datasets.make_1D_gauss(n, m=60, s=8)
    print(a1)
    print(a2)
    x = np.arange(n, dtype=np.float64).reshape((n, 1))
    M = ot.utils.dist(x, metric='minkowski')
    lp_time, demd_time = compare_all([a1, a2], M, n, d)
    print('*'*10, '\n')
    return lp_time, demd_time


def increasing_bins():
    lp_times, demd_times = [], []
    ns = [5, 10, 20, 50, 100]
    for n in ns:
        lp_time, demd_time = random2d(n=n)
        lp_times.append(lp_time)
        demd_times.append(demd_time)
    return ns, lp_times, demd_times


ns, lp_times, demd_times = increasing_bins()


# %%
# Plot and Compare data
# ---------------------
# plots the computation times for the LP and DEMD methods for
# different bin sizes


pl.plot(ns, lp_times, 'o', linestyle="-", label="LP Barycenter")
pl.plot(ns, demd_times, 'o', linestyle="-", label="DEMD")
# pl.yscale('log')
pl.ylabel('Time Per Epoch (Seconds)')
pl.xlabel('Number of Distributions')
pl.legend()
