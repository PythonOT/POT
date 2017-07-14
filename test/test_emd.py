#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import pylab as pl
import ot

from ot.datasets import get_1D_gauss as gauss
reload(ot.lp)

#%% parameters

n=5000 # nb bins
m=6000 # nb bins

mean1 = 1000
mean2 = 1100

tol = 1e-6

# bin positions
x=np.arange(n,dtype=np.float64)
y=np.arange(m,dtype=np.float64)

# Gaussian distributions
a=gauss(n,m=mean1,s=5) # m= mean, s= std

b=gauss(m,m=mean2,s=10)

# loss matrix
M=ot.dist(x.reshape((-1,1)), y.reshape((-1,1))) ** (1./2)
print M[0,:]
#M/=M.max()

#%%

print('Computing {} EMD '.format(1))

# emd loss 1 proc
ot.tic()
G = ot.emd(a,b,M)
ot.toc('1 proc : {} s')

cost1 = (G * M).sum()

ot.tic()
G = ot.emd(b, a, np.ascontiguousarray(M.T))
ot.toc('1 proc : {} s')

cost2 = (G * M.T).sum()

assert np.abs(cost1-cost2) < tol
assert np.abs(cost1-np.abs(mean1-mean2)) < tol
