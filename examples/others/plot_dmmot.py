# -*- coding: utf-8 -*-
r"""
===============================================================================
Computing d-dimensional Barycenters via d-MMOT
===============================================================================

.. note::
    Example added in release: 0.9.1.

When the cost is discretized (Monge), the d-MMOT solver can more quickly
compute and minimize the distance between many distributions without the need
for intermediate barycenter computations. This example compares the time to
identify, and the quality of, solutions for the d-MMOT problem using a
primal/dual algorithm and classical LP barycenter approaches.
"""

# Author: Ronak Mehta <ronakrm@cs.wisc.edu>
#         Xizheng Yu <xyu354@wisc.edu>
#
# License: MIT License

# %%
# Generating 2 distributions
# -----
import numpy as np
import matplotlib.pyplot as pl
import ot

np.random.seed(0)

n = 100
d = 2
# Gaussian distributions
a1 = ot.datasets.make_1D_gauss(n, m=20, s=5)  # m=mean, s=std
a2 = ot.datasets.make_1D_gauss(n, m=60, s=8)
A = np.vstack((a1, a2)).T
x = np.arange(n, dtype=np.float64)
M = ot.utils.dist(x.reshape((n, 1)), metric="minkowski")

pl.figure(1, figsize=(6.4, 3))
pl.plot(x, a1, "b", label="Source distribution")
pl.plot(x, a2, "r", label="Target distribution")
pl.legend()

# %%
# Minimize the distances among distributions, identify the Barycenter
# -----
# The objective being minimized is different for both methods, so the objective
# values cannot be compared.

# L2 Iteration
weights = np.ones(d) / d
l2_bary = A.dot(weights)

print("LP Iterations:")
weights = np.ones(d) / d
lp_bary, lp_log = ot.lp.barycenter(
    A, M, weights, solver="interior-point", verbose=False, log=True
)
print("Time\t: ", ot.toc(""))
print("Obj\t: ", lp_log["fun"])

print("")
print("Discrete MMOT Algorithm:")
ot.tic()
barys, log = ot.lp.dmmot_monge_1dgrid_optimize(
    A, niters=4000, lr_init=1e-5, lr_decay=0.997, log=True
)
dmmot_obj = log["primal objective"]
print("Time\t: ", ot.toc(""))
print("Obj\t: ", dmmot_obj)

# %%
# Compare Barycenters in both methods
# -----
pl.figure(1, figsize=(6.4, 3))
for i in range(len(barys)):
    if i == 0:
        pl.plot(x, barys[i], "g-*", label="Discrete MMOT")
    else:
        continue
        # pl.plot(x, barys[i], 'g-*')
pl.plot(x, lp_bary, label="LP Barycenter")
pl.plot(x, l2_bary, label="L2 Barycenter")
pl.plot(x, a1, "b", label="Source distribution")
pl.plot(x, a2, "r", label="Target distribution")
pl.title("Monge Cost: Barycenters from LP Solver and dmmot solver")
pl.legend()


# %%
# More than 2 distributions
# --------------------------------------------------
# Generate 7 pseudorandom gaussian distributions with 50 bins.
n = 50  # nb bins
d = 7
vecsize = n * d

data = []
for i in range(d):
    m = n * (0.5 * np.random.rand(1)) * float(np.random.randint(2) + 1)
    a = ot.datasets.make_1D_gauss(n, m=m, s=5)
    data.append(a)

x = np.arange(n, dtype=np.float64)
M = ot.utils.dist(x.reshape((n, 1)), metric="minkowski")
A = np.vstack(data).T

pl.figure(1, figsize=(6.4, 3))
for i in range(len(data)):
    pl.plot(x, data[i])

pl.title("Distributions")
pl.legend()

# %%
# Minimizing Distances Among Many Distributions
# ---------------
# The objective being minimized is different for both methods, so the objective
# values cannot be compared.

# Perform gradient descent optimization using the d-MMOT method.
barys = ot.lp.dmmot_monge_1dgrid_optimize(A, niters=3000, lr_init=1e-4, lr_decay=0.997)

# after minimization, any distribution can be used as a estimate of barycenter.
bary = barys[0]

# Compute 1D Wasserstein barycenter using the L2/LP method
weights = ot.unif(d)
l2_bary = A.dot(weights)
lp_bary, bary_log = ot.lp.barycenter(
    A, M, weights, solver="interior-point", verbose=False, log=True
)

# %%
# Compare Barycenters in both methods
# ---------
pl.figure(1, figsize=(6.4, 3))
pl.plot(x, bary, "g-*", label="Discrete MMOT")
pl.plot(x, l2_bary, "k", label="L2 Barycenter")
pl.plot(x, lp_bary, "k-", label="LP Wasserstein")
pl.title("Barycenters")
pl.legend()

# %%
# Compare with original distributions
# ---------
pl.figure(1, figsize=(6.4, 3))
for i in range(len(data)):
    pl.plot(x, data[i])
for i in range(len(barys)):
    if i == 0:
        pl.plot(x, barys[i], "g-*", label="Discrete MMOT")
    else:
        continue
        # pl.plot(x, barys[i], 'g')
pl.plot(x, l2_bary, "k^", label="L2")
pl.plot(x, lp_bary, "o", color="grey", label="LP")
pl.title("Barycenters")
pl.legend()
pl.show()

# %%
