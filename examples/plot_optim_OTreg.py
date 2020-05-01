# -*- coding: utf-8 -*-
"""
==================================
Regularized OT with generic solver
==================================

Illustrates the use of the generic solver for regularized OT with
user-designed regularization term. It uses Conditional gradient as in [6] and
generalized Conditional Gradient as proposed in [5,7].


[5] N. Courty; R. Flamary; D. Tuia; A. Rakotomamonjy, Optimal Transport for
Domain Adaptation, in IEEE Transactions on Pattern Analysis and Machine
Intelligence , vol.PP, no.99, pp.1-1.

[6] Ferradans, S., Papadakis, N., Peyré, G., & Aujol, J. F. (2014).
Regularized discrete optimal transport. SIAM Journal on Imaging
Sciences, 7(3), 1853-1882.

[7] Rakotomamonjy, A., Flamary, R., & Courty, N. (2015). Generalized
conditional gradient: analysis of convergence and applications.
arXiv preprint arXiv:1510.06567.



"""
# sphinx_gallery_thumbnail_number = 4

import numpy as np
import matplotlib.pylab as pl
import ot
import ot.plot

##############################################################################
# Generate data
# -------------

#%% parameters

n = 100  # nb bins

# bin positions
x = np.arange(n, dtype=np.float64)

# Gaussian distributions
a = ot.datasets.make_1D_gauss(n, m=20, s=5)  # m= mean, s= std
b = ot.datasets.make_1D_gauss(n, m=60, s=10)

# loss matrix
M = ot.dist(x.reshape((n, 1)), x.reshape((n, 1)))
M /= M.max()

##############################################################################
# Solve EMD
# ---------

#%% EMD

G0 = ot.emd(a, b, M)

pl.figure(3, figsize=(5, 5))
ot.plot.plot1D_mat(a, b, G0, 'OT matrix G0')

##############################################################################
# Solve EMD with Frobenius norm regularization
# --------------------------------------------

#%% Example with Frobenius norm regularization


def f(G):
    return 0.5 * np.sum(G**2)


def df(G):
    return G


reg = 1e-1

Gl2 = ot.optim.cg(a, b, M, reg, f, df, verbose=True)

pl.figure(3)
ot.plot.plot1D_mat(a, b, Gl2, 'OT matrix Frob. reg')

##############################################################################
# Solve EMD with entropic regularization
# --------------------------------------

#%% Example with entropic regularization


def f(G):
    return np.sum(G * np.log(G))


def df(G):
    return np.log(G) + 1.


reg = 1e-3

Ge = ot.optim.cg(a, b, M, reg, f, df, verbose=True)

pl.figure(4, figsize=(5, 5))
ot.plot.plot1D_mat(a, b, Ge, 'OT matrix Entrop. reg')

##############################################################################
# Solve EMD with Frobenius norm + entropic regularization
# -------------------------------------------------------

#%% Example with Frobenius norm + entropic regularization with gcg


def f(G):
    return 0.5 * np.sum(G**2)


def df(G):
    return G


reg1 = 1e-3
reg2 = 1e-1

Gel2 = ot.optim.gcg(a, b, M, reg1, reg2, f, df, verbose=True)

pl.figure(5, figsize=(5, 5))
ot.plot.plot1D_mat(a, b, Gel2, 'OT entropic + matrix Frob. reg')
pl.show()
