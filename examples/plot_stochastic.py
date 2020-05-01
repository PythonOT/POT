"""
===================
Stochastic examples
===================

This example is designed to show how to use the stochatic optimization
algorithms for discrete and semi-continuous measures from the POT library.

[18] Genevay, A., Cuturi, M., Peyré, G. & Bach, F.
Stochastic Optimization for Large-scale Optimal Transport.
Advances in Neural Information Processing Systems (2016).

[19] Seguy, V., Bhushan Damodaran, B., Flamary, R., Courty, N., Rolet, A. &
Blondel, M. Large-scale Optimal Transport and Mapping Estimation.
International Conference on Learning Representation (2018)

"""

# Author: Kilian Fatras <kilian.fatras@gmail.com>
#
# License: MIT License

import matplotlib.pylab as pl
import numpy as np
import ot
import ot.plot


#############################################################################
# Compute the Transportation Matrix for the Semi-Dual Problem
# -----------------------------------------------------------
#
# Discrete case
# `````````````
#
# Sample two discrete measures for the discrete case and compute their cost
# matrix c.

n_source = 7
n_target = 4
reg = 1
numItermax = 1000

a = ot.utils.unif(n_source)
b = ot.utils.unif(n_target)

rng = np.random.RandomState(0)
X_source = rng.randn(n_source, 2)
Y_target = rng.randn(n_target, 2)
M = ot.dist(X_source, Y_target)

#############################################################################
# Call the "SAG" method to find the transportation matrix in the discrete case

method = "SAG"
sag_pi = ot.stochastic.solve_semi_dual_entropic(a, b, M, reg, method,
                                                numItermax)
print(sag_pi)

#############################################################################
# Semi-Continuous Case
# ````````````````````
#
# Sample one general measure a, one discrete measures b for the semicontinous
# case, the points where source and target measures are defined and compute the
# cost matrix.

n_source = 7
n_target = 4
reg = 1
numItermax = 1000
log = True

a = ot.utils.unif(n_source)
b = ot.utils.unif(n_target)

rng = np.random.RandomState(0)
X_source = rng.randn(n_source, 2)
Y_target = rng.randn(n_target, 2)
M = ot.dist(X_source, Y_target)

#############################################################################
# Call the "ASGD" method to find the transportation matrix in the semicontinous
# case.

method = "ASGD"
asgd_pi, log_asgd = ot.stochastic.solve_semi_dual_entropic(a, b, M, reg, method,
                                                           numItermax, log=log)
print(log_asgd['alpha'], log_asgd['beta'])
print(asgd_pi)

#############################################################################
# Compare the results with the Sinkhorn algorithm

sinkhorn_pi = ot.sinkhorn(a, b, M, reg)
print(sinkhorn_pi)


##############################################################################
# Plot Transportation Matrices
# ````````````````````````````
#
# For SAG

pl.figure(4, figsize=(5, 5))
ot.plot.plot1D_mat(a, b, sag_pi, 'semi-dual : OT matrix SAG')
pl.show()


##############################################################################
# For ASGD

pl.figure(4, figsize=(5, 5))
ot.plot.plot1D_mat(a, b, asgd_pi, 'semi-dual : OT matrix ASGD')
pl.show()


##############################################################################
# For Sinkhorn

pl.figure(4, figsize=(5, 5))
ot.plot.plot1D_mat(a, b, sinkhorn_pi, 'OT matrix Sinkhorn')
pl.show()


#############################################################################
# Compute the Transportation Matrix for the Dual Problem
# ------------------------------------------------------
#
# Semi-continuous case
# ````````````````````
#
# Sample one general measure a, one discrete measures b for the semi-continuous
# case and compute the cost matrix c.

n_source = 7
n_target = 4
reg = 1
numItermax = 100000
lr = 0.1
batch_size = 3
log = True

a = ot.utils.unif(n_source)
b = ot.utils.unif(n_target)

rng = np.random.RandomState(0)
X_source = rng.randn(n_source, 2)
Y_target = rng.randn(n_target, 2)
M = ot.dist(X_source, Y_target)

#############################################################################
#
# Call the "SGD" dual method to find the transportation matrix in the
# semi-continuous case

sgd_dual_pi, log_sgd = ot.stochastic.solve_dual_entropic(a, b, M, reg,
                                                         batch_size, numItermax,
                                                         lr, log=log)
print(log_sgd['alpha'], log_sgd['beta'])
print(sgd_dual_pi)

#############################################################################
#
# Compare the results with the Sinkhorn algorithm
# ```````````````````````````````````````````````
#
# Call the Sinkhorn algorithm from POT

sinkhorn_pi = ot.sinkhorn(a, b, M, reg)
print(sinkhorn_pi)

##############################################################################
# Plot Transportation Matrices
# ````````````````````````````
#
# For SGD

pl.figure(4, figsize=(5, 5))
ot.plot.plot1D_mat(a, b, sgd_dual_pi, 'dual : OT matrix SGD')
pl.show()


##############################################################################
# For Sinkhorn

pl.figure(4, figsize=(5, 5))
ot.plot.plot1D_mat(a, b, sinkhorn_pi, 'OT matrix Sinkhorn')
pl.show()
