# -*- coding: utf-8 -*-
r"""
=================================================================================
d-MMOT vs LP Gradient Decent without Pytorch
=================================================================================

Compare the loss convergence between LP and DEMD. The comparison is performed using random
Gaussian or uniform distributions and calculating the loss for each method
during the optimization process.
"""

# Author: Ronak Mehta <ronakrm@cs.wisc.edu>
#         Xizheng Yu <xyu354@wisc.edu>
#
# License: MIT License

# %%
# 2 distributions
# -----
import numpy as np
import matplotlib.pyplot as pl
import ot

n = 100
d = 2
# Gaussian distributions
a1 = ot.datasets.make_1D_gauss(n, m=20, s=5)  # m=mean, s=std
a2 = ot.datasets.make_1D_gauss(n, m=60, s=8)
A = np.vstack((a1, a2)).T
x = np.arange(n, dtype=np.float64)
# M = ot.utils.dist0(n)
# M /= M.max()
M = ot.utils.dist(x.reshape((n, 1)), metric='minkowski')

pl.figure(1, figsize=(6.4, 3))
pl.plot(x, a1, 'b', label='Source distribution')
pl.plot(x, a2, 'r', label='Target distribution')
pl.legend()

# %%
# Run test
# -----

print('LP Iterations:')
ot.tic()
alpha = 1  # /d  # 0<=alpha<=1
weights = np.array(d * [alpha])
lp_bary, lp_log = ot.lp.barycenter(
    A, M, weights, solver='interior-point', verbose=False, log=True)
print('Time\t: ', ot.toc(''))
print('Obj\t: ', lp_log['fun'])

print('')
print('Discrete MMOT Algorithm:')
ot.tic()
# dmmot_obj, log = ot.lp.discrete_mmot(A.T, n, d)
barys, log = ot.lp.discrete_mmot_converge(A.T, niters=3000, lr=0.000002, log=True)
dmmot_obj = log['primal objective']
print('Time\t: ', ot.toc(''))
print('Obj\t: ', dmmot_obj)

# %%
# Compare Barycenters in both methods
# ---------
pl.figure(1, figsize=(6.4, 3))
for i in range(len(barys)):
    if i == 0:
        pl.plot(x, barys[i], 'g', label='Discrete MMOT')
    else:
        pl.plot(x, barys[i], 'g')
pl.plot(x, a1, 'b', label='Source distribution')
pl.plot(x, a2, 'r', label='Target distribution')
pl.title('Barycenters')
pl.legend()

# %%
# Compare d-MMOOT with original distributions
# ---------
pl.figure(1, figsize=(6.4, 3))
for i in range(len(barys)):
    if i == 0:
        pl.plot(x, barys[i], 'g', label='Discrete MMOT')
    else:
        pl.plot(x, barys[i], 'g')
# pl.plot(x, bary, 'g', label='Discrete MMOT')
pl.plot(x, lp_bary, 'b', label='LP Wasserstein')
pl.title('Barycenters')
pl.legend()

# %%
# Define parameters, generate and plot distributions
# --------------------------------------------------
# The following code generates random (n, d) data with in gauss
n = 50  # nb bins
d = 7
vecsize = n * d

data = []
for i in range(d):
    m = n * (0.5 * np.random.rand(1)) * float(np.random.randint(2) + 1)
    a = ot.datasets.make_1D_gauss(n, m=m, s=5)
    data.append(a)
    
x = np.arange(n, dtype=np.float64)
M = ot.utils.dist(x.reshape((n, 1)), metric='minkowski')
A = np.vstack(data).T

print(A.shape)

pl.figure(1, figsize=(6.4, 3))
for i in range(len(data)):
    pl.plot(x, data[i])

pl.title('Distributions')
pl.legend()

# %%
# Gradient Decent
# ---------------
# The following section performs gradient descent optimization using
# the DEMD method

barys = ot.lp.discrete_mmot_converge(A.T, niters=9000, lr=0.00001)

# after minimization, any distribution can be used as a estimate of barycenter
# bary = barys[0]


# %% lp barycenter
# ----------------
# The following section computes 1D Wasserstein barycenter using the LP method
weights = ot.unif(d)
lp_bary, bary_log = ot.lp.barycenter(A, M, weights, solver='interior-point', p
                                      verbose=True, log=True)

# %%
# Compare Barycenters in both methods
# ---------
pl.figure(1, figsize=(6.4, 3))
for i in range(len(barys)):
    if i == 0:
        pl.plot(x, barys[i], 'g', label='Discrete MMOT')
    else:
        pl.plot(x, barys[i], 'g')
# pl.plot(x, bary, 'g', label='Discrete MMOT')
pl.plot(x, lp_bary, 'b', label='LP Wasserstein')
pl.title('Barycenters')
pl.legend()

# %%
# Compare d-MMOOT with original distributions
# ---------
pl.figure(1, figsize=(6.4, 3))
for i in range(len(barys)):
    if i == 0:
        pl.plot(x, barys[i], 'g', label='Discrete MMOT')
    else:
        pl.plot(x, barys[i], 'g')
# pl.plot(x, bary, 'g', label='Discrete MMOT')
for i in range(len(data)):
    pl.plot(x, data[i])
pl.title('Barycenters')
pl.legend()


# %%
# Compare the loss between DEMD and LP Barycenter
# ---------
# The barycenter approach does not minize the distance between
# the distributions, while our DEMD does.
