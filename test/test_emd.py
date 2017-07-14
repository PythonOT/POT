#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import pylab as pl
import ot

from ot.datasets import get_1D_gauss as gauss
reload(ot.lp)

#%% parameters

n=5000 # nb bins

# bin positions
x=np.arange(n,dtype=np.float64)

# Gaussian distributions
a=gauss(n,m=20,s=5) # m= mean, s= std

b=gauss(n,m=30,s=10)

# loss matrix
M=ot.dist(x.reshape((n,1)),x.reshape((n,1)))
#M/=M.max()

#%%

print('Computing {} EMD '.format(1))

# emd loss 1 proc
ot.tic()
emd_loss4 = ot.emd(a,b,M)
ot.toc('1 proc : {} s')

