# -*- coding: utf-8 -*-
"""
Demo for 1D optimal transport

@author: rflamary
"""

import numpy as np
import matplotlib.pylab as pl
import ot
from ot.datasets import get_1D_gauss as gauss


#%% parameters

n=100 # nb bins

# bin positions
x=np.arange(n,dtype=np.float64)

# Gaussian distributions
a=gauss(n,m=n*.2,s=5) # m= mean, s= std
b=gauss(n,m=n*.6,s=10)

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
ot.plot.plot1D_mat(a,b,M,'Cost matrix M')

#%% EMD

G0=ot.emd(a,b,M)

pl.figure(3)
ot.plot.plot1D_mat(a,b,G0,'OT matrix G0')

#%% Sinkhorn

lambd=1e-3
Gs=ot.sinkhorn(a,b,M,lambd,verbose=True)

pl.figure(4)
ot.plot.plot1D_mat(a,b,Gs,'OT matrix Sinkhorn')

#%% Sinkhorn

lambd=1e-4
Gss,log=ot.bregman.sinkhorn_stabilized(a,b,M,lambd,verbose=True,log=True)
Gss2,log2=ot.bregman.sinkhorn_stabilized(a,b,M,lambd,verbose=True,log=True,warmstart=log['warmstart'])

pl.figure(5)
ot.plot.plot1D_mat(a,b,Gss,'OT matrix Sinkhorn stabilized')

#%% Sinkhorn

lambd=1e-11
Gss=ot.bregman.sinkhorn_epsilon_scaling(a,b,M,lambd,verbose=True)

pl.figure(5)
ot.plot.plot1D_mat(a,b,Gss,'OT matrix Sinkhorn stabilized')
