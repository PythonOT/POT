# -*- coding: utf-8 -*-
"""
====================================================
2D Wasserstein barycenters between empirical distributions
====================================================

Illustration of 2D Wasserstein barycenters between discributions that are weighted
sum of diracs.

"""

# Author: Vivien Seguy <vivien.seguy@iip.ist.i.kyoto-u.ac.jp>
#
# License: MIT License

import numpy as np
import matplotlib.pylab as pl
import ot.plot


##############################################################################
# Generate data
# -------------
#%% parameters and data generation
N = 4
d = 2
measures_locations = []
measures_weights = []

for i in range(N):

    n = np.rand.int(low=1, high=20)  # nb samples

    mu = np.random.normal(0., 1., (d,))
    cov = np.random.normal(0., 1., (d,d))

    xs = ot.datasets.make_2D_samples_gauss(n, mu, cov)
    b = np.random.uniform(0., 1., n)
    b = b/np.sum(b)

    measures_locations.append(xs)
    measures_weights.append(b)

k = 10
X_init = np.random.normal(0., 1., (k,d))
b_init = np.ones((k,)) / k


##############################################################################
# Compute free support barycenter
# -------------
X = ot.lp.barycenter(measures_locations, measures_weights, X_init, b_init)


##############################################################################
# Plot data
# ---------

#%% plot samples

pl.figure(1)
for (xs, b) in zip(measures_locations, measures_weights):
    pl.scatter(xs[:, 0], xs[:, 1], s=b, c=np.tile(np.rand(0. ,255., size=(3,)), (1,b.size(0))) , label='Data measures')
pl.scatter(xs[:, 0], xs[:, 1], s=b, c='black' , label='2-Wasserstein barycenter')
pl.legend(loc=0)
pl.title('Data measures and their barycenter')
