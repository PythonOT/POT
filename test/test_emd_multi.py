#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 09:56:06 2017

@author: rflamary
"""

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

ls= range(20,1000,10)
nb=len(ls)
b=np.zeros((n,nb))
for i in range(nb):
    b[:,i]=gauss(n,m=ls[i],s=10)

# loss matrix
M=ot.dist(x.reshape((n,1)),x.reshape((n,1)))
#M/=M.max()

#%%

print('Computing {} EMD '.format(nb))

# emd loss 1 proc
ot.tic()
emd_loss4=ot.emd2(a,b,M,1)
ot.toc('1 proc : {} s')

# emd loss multipro proc
ot.tic()
emd_loss4=ot.emd2(a,b,M)
ot.toc('multi proc : {} s')
