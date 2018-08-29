"""
==========================
Stochastic examples
==========================

This example is designed to show how to use the stochatic optimization
algorithms for descrete and semicontinous measures from the POT library.

"""

# Author: Kilian Fatras <kilian.fatras@gmail.com>
#
# License: MIT License

import matplotlib.pylab as pl
import numpy as np
import ot
import ot.plot


#############################################################################
# COMPUTE TRANSPORTATION MATRIX FOR SEMI-DUAL PROBLEM
#############################################################################
print("------------SEMI-DUAL PROBLEM------------")
#############################################################################
# DISCRETE CASE
# Sample two discrete measures for the discrete case
# ---------------------------------------------
#
# Define 2 discrete measures a and b, the points where are defined the source
# and the target measures and finally the cost matrix c.

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
#
# Call the "SAG" method to find the transportation matrix in the discrete case
# ---------------------------------------------
#
# Define the method "SAG", call ot.solve_semi_dual_entropic and plot the
# results.

method = "SAG"
sag_pi = ot.stochastic.solve_semi_dual_entropic(a, b, M, reg, method,
                                                numItermax)
print(sag_pi)

#############################################################################
# SEMICONTINOUS CASE
# Sample one general measure a, one discrete measures b for the semicontinous
# case
# ---------------------------------------------
#
# Define one general measure a, one discrete measures b, the points where
# are defined the source and the target measures and finally the cost matrix c.

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
#
# Call the "ASGD" method to find the transportation matrix in the semicontinous
# case
# ---------------------------------------------
#
# Define the method "ASGD", call ot.solve_semi_dual_entropic and plot the
# results.

method = "ASGD"
asgd_pi, log_asgd = ot.stochastic.solve_semi_dual_entropic(a, b, M, reg, method,
                                                           numItermax, log=log)
print(log_asgd['alpha'], log_asgd['beta'])
print(asgd_pi)

#############################################################################
#
# Compare the results with the Sinkhorn algorithm
# ---------------------------------------------
#
# Call the Sinkhorn algorithm from POT

sinkhorn_pi = ot.sinkhorn(a, b, M, reg)
print(sinkhorn_pi)


##############################################################################
# PLOT TRANSPORTATION MATRIX
##############################################################################

##############################################################################
# Plot SAG results
# ----------------

pl.figure(4, figsize=(5, 5))
ot.plot.plot1D_mat(a, b, sag_pi, 'semi-dual : OT matrix SAG')
pl.show()


##############################################################################
# Plot ASGD results
# -----------------

pl.figure(4, figsize=(5, 5))
ot.plot.plot1D_mat(a, b, asgd_pi, 'semi-dual : OT matrix ASGD')
pl.show()


##############################################################################
# Plot Sinkhorn results
# ---------------------

pl.figure(4, figsize=(5, 5))
ot.plot.plot1D_mat(a, b, sinkhorn_pi, 'OT matrix Sinkhorn')
pl.show()


#############################################################################
# COMPUTE TRANSPORTATION MATRIX FOR DUAL PROBLEM
#############################################################################
print("------------DUAL PROBLEM------------")
#############################################################################
# SEMICONTINOUS CASE
# Sample one general measure a, one discrete measures b for the semicontinous
# case
# ---------------------------------------------
#
# Define one general measure a, one discrete measures b, the points where
# are defined the source and the target measures and finally the cost matrix c.

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
# semicontinous case
# ---------------------------------------------
#
# Call ot.solve_dual_entropic and plot the results.

sgd_dual_pi, log_sgd = ot.stochastic.solve_dual_entropic(a, b, M, reg,
                                                         batch_size, numItermax,
                                                         lr, log=log)
print(log_sgd['alpha'], log_sgd['beta'])
print(sgd_dual_pi)

#############################################################################
#
# Compare the results with the Sinkhorn algorithm
# ---------------------------------------------
#
# Call the Sinkhorn algorithm from POT

sinkhorn_pi = ot.sinkhorn(a, b, M, reg)
print(sinkhorn_pi)

##############################################################################
# Plot  SGD results
# -----------------

pl.figure(4, figsize=(5, 5))
ot.plot.plot1D_mat(a, b, sgd_dual_pi, 'dual : OT matrix SGD')
pl.show()


##############################################################################
# Plot Sinkhorn results
# ---------------------

pl.figure(4, figsize=(5, 5))
ot.plot.plot1D_mat(a, b, sinkhorn_pi, 'OT matrix Sinkhorn')
pl.show()
