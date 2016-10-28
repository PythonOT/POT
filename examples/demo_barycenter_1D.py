# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 09:51:45 2016

@author: rflamary
"""

import numpy as np
import matplotlib.pylab as pl
import ot



#%% parameters

n=100 # nb bins

# bin positions
x=np.arange(n,dtype=np.float64)

# Gaussian distributions
a1=ot.datasets.get_1D_gauss(n,m=20,s=20) # m= mean, s= std
a2=ot.datasets.get_1D_gauss(n,m=60,s=60)

A=np.vstack((a1,a2)).T
nbd=A.shape[1]

# loss matrix
M=ot.utils.dist0(n)
M/=M.max()

#%% plot the distributions

pl.figure(1)
for i in range(nbd):
    pl.plot(x,A[:,i])
pl.title('Distributions')

#%% barucenter computation

# l2bary
bary_l2=A.mean(1)

# wasserstein
reg=1e-3
bary_wass=ot.bregman.barycenter(A,M,reg)

pl.figure(2)
pl.clf()
pl.subplot(2,1,1)
for i in range(nbd):
    pl.plot(x,A[:,i])
pl.title('Distributions')

pl.subplot(2,1,2)
pl.plot(x,bary_l2,'r',label='l2')
pl.plot(x,bary_wass,'g',label='Wasserstein')
pl.legend()
pl.title('Barycenters')
