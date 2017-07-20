# -*- coding: utf-8 -*-
"""
==================================
Regularized OT with generic solver
==================================


"""

import numpy as np
import matplotlib.pylab as pl
import ot


#%% parameters

n = 100  # nb bins

# bin positions
x = np.arange(n, dtype=np.float64)

# Gaussian distributions
a = ot.datasets.get_1D_gauss(n, m=20, s=5)  # m= mean, s= std
b = ot.datasets.get_1D_gauss(n, m=60, s=10)

# loss matrix
M = ot.dist(x.reshape((n, 1)), x.reshape((n, 1)))
M /= M.max()

#%% EMD

G0 = ot.emd(a, b, M)

pl.figure(3, figsize=(5, 5))
ot.plot.plot1D_mat(a, b, G0, 'OT matrix G0')

#%% Example with Frobenius norm regularization


def f(G):
    return 0.5 * np.sum(G**2)


def df(G):
    return G


reg = 1e-1

Gl2 = ot.optim.cg(a, b, M, reg, f, df, verbose=True)

pl.figure(3)
ot.plot.plot1D_mat(a, b, Gl2, 'OT matrix Frob. reg')

#%% Example with entropic regularization


def f(G):
    return np.sum(G * np.log(G))


def df(G):
    return np.log(G) + 1.


reg = 1e-3

Ge = ot.optim.cg(a, b, M, reg, f, df, verbose=True)

pl.figure(4, figsize=(5, 5))
ot.plot.plot1D_mat(a, b, Ge, 'OT matrix Entrop. reg')

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
