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
a=ot.datasets.get_1D_gauss(n,m=20,s=20) # m= mean, s= std
b=ot.datasets.get_1D_gauss(n,m=60,s=60)

# loss matrix
M=ot.dist(x.reshape((n,1)),x.reshape((n,1)))
M/=M.max()

#%% plot the distributions

pl.figure(1)

pl.plot(x,a,'b',label='Source distribution')
pl.plot(x,b,'r',label='Target distribution')

pl.legend()

#%% plot distributions and loss matrix

pl.figure(2)
ot.plot.otplot1D(a,b,M,'Cost matrix M')


#%% EMD

G0=ot.emd(a,b,M)

pl.figure(3)
ot.plot.otplot1D(a,b,G0,'OT matrix G0')

#%% Sinkhorn
lambd=1e-3

Gs=ot.sinkhorn(a,b,M,lambd)

pl.figure(4)
ot.plot.otplot1D(a,b,Gs,'OT matrix Sinkhorn')
