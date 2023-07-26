# -*- coding: utf-8 -*-
r"""
===============================================================================
Comparation of LP, dMMOT solvers under different Monge Matrics
===============================================================================

We also provided uniqueness test betweeen Entropic Regularization Barycenter,
LP Barycenter, and dMMOT Barycenter.
"""

# Author: Xizheng Yu <xyu354@wisc.edu>
#
# License: MIT License

# %%
# Generating distributions and functions setup
# -----
import numpy as np
import ot
import matplotlib.pyplot as pl

n = 100  # number of bins
d = 2


def monge_cost_matrix(matric):
    MM = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            MM[i, j] = ot.lp.dmmot.ground_cost([i, j], matric)
    return MM

labels = ['monge', 'monge_mean', "monge_square",
          'monge_sqrt', 'monge_log', 'monge_exp']
Ms = [monge_cost_matrix(label) for label in labels]

a1 = ot.datasets.make_1D_gauss(n, m=20, s=5)
a2 = ot.datasets.make_1D_gauss(n, m=60, s=8)

A = np.vstack((a1, a2)).T
x = np.arange(n, dtype=np.float64)
weights = np.ones(d) / d

l2_bary = A.dot(weights)

pl.figure(1, figsize=(6.4, 3))
pl.plot(x, a1, 'b', label='Source distribution')
pl.plot(x, a2, 'r', label='Target distribution')
pl.legend()


# %%
# Minimize using monge costs in LP solver and dmmot solver
# -----
lp_barys = []

for M, label in zip(Ms, labels):
    lp_bary_temp, _ = ot.lp.barycenter(
        A, M, weights, solver='interior-point', verbose=False, log=True)
    lp_barys.append(lp_bary_temp)

barys = ot.lp.dmmot_monge_1dgrid_optimize(
    A, niters=4000, lr_init=1e-5, lr_decay=0.997)
barys_mean = ot.lp.dmmot_monge_1dgrid_optimize(
    A, niters=3000, lr_init=1e-5, lr_decay=0.999, metric="monge_mean")
barys_square = ot.lp.dmmot_monge_1dgrid_optimize(
    A, niters=3000, lr_init=1e-6, lr_decay=0.999, metric="monge_square")
barys_sqrt = ot.lp.dmmot_monge_1dgrid_optimize(
    A, niters=3000, lr_init=1e-5, lr_decay=0.999, metric="monge_sqrt")
barys_log = ot.lp.dmmot_monge_1dgrid_optimize(
    A, niters=3000, lr_init=1e-5, lr_decay=0.9995, metric="monge_log")
barys_exp = ot.lp.dmmot_monge_1dgrid_optimize(
    A, niters=3000, lr_init=1e-3, lr_decay=0.999, metric="monge_exp")

dmmot_barys = [barys[0], barys_mean[0], barys_square[0], barys_sqrt[0],
               barys_log[0], barys_exp[0]]

# %%
# Compare Barycenters with different monge costs in LP Solver
# -----
fig, axes = pl.subplots(2, 3, figsize=(6.4, 3))
axes = axes.ravel()

for i in range(6):  # iterate over each subplot
    axes[i].plot(x, a1, 'b', label='Source distribution')
    axes[i].plot(x, a2, 'r', label='Target distribution')
    axes[i].plot(x, lp_barys[i], 'g-', label=labels[i])
    axes[i].set_title(labels[i])
    axes[i].set_xticklabels([])
    axes[i].set_yticklabels([])

fig.suptitle('LP Solver: Barycenters with Different Monge Costs')
pl.tight_layout()
pl.show()

# %%
# Compare Barycenters with different monge costs
# -----
fig, axes = pl.subplots(2, 3, figsize=(6.4, 3))
axes = axes.ravel()

for i in range(6):
    axes[i].plot(x, a1, 'b', label='Source distribution')
    axes[i].plot(x, a2, 'r', label='Target distribution')
    axes[i].plot(x, dmmot_barys[i], 'g-', label=labels[i])
    axes[i].plot(x, l2_bary, 'k', label='L2 Bary')
    axes[i].set_title(labels[i])
    axes[i].set_xticklabels([])
    axes[i].set_yticklabels([])

fig.suptitle('dmmot Solver: Barycenters with Different Monge Costs')
pl.tight_layout()
pl.show()


# %%
# Compare Barycenters with different monge costs
# -----
pl.figure(1, figsize=(6.4, 3))

pl.plot(x, barys[0], label='Monge')
pl.plot(x, barys_mean[0], label='Monge Mean')
pl.plot(x, barys_square[0], label='Monge Square')
pl.plot(x, barys_sqrt[0], label='Monge Sqrt')
pl.plot(x, barys_log[0], label='Monge Log')
pl.plot(x, barys_exp[0], label='Monge Exp')

pl.plot(x, l2_bary, 'k', label='L2 Bary')
# pl.plot(x, a1, 'b', label='Source')
# pl.plot(x, a2, 'r', label='Target')
pl.title('Barycenters of Different Monge Costs')
pl.legend()


# %%
# Uniqueness Test betweeen Entropic Regularization Barycenter, LP Barycenter,
# dMMOT Barycenter
# -----
def obj(A, M, bary):
    tmp = 0.0
    for x in A.T:
        _, log = ot.lp.emd(
            x, np.array(bary / np.sum(bary)), M, log=True)
        tmp += log['cost']
    return tmp


def entropy_reg(A, M):
    # Entropic Regularization Barycenter
    bary_wass, _ = ot.bregman.barycenter(A, M, 1e-2, weights, log=True)
    return bary_wass

print("\t\tReg\t\tLP\t\tdMMOT\t")
for M, dmmot_bary, lp_bary, label in zip(Ms, dmmot_barys, lp_barys, labels):
    M /= M.max()
    bary_wass = entropy_reg(A, M)
    print(f'{label}\t', f'{obj(A, M, bary_wass):.7f}\t',
          f'{obj(A, M, lp_bary):.7f}\t', f'{obj(A, M, dmmot_bary):.7f}')

# %%
