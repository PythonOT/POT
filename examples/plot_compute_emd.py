# -*- coding: utf-8 -*-
"""
====================
1D optimal transport
====================

@author: rflamary
"""

import numpy as np
import matplotlib.pylab as pl
import ot
from ot.datasets import get_1D_gauss as gauss


#%% parameters

n=100 # nb bins
n_target=10 # nb target distributions


# bin positions
x=np.arange(n,dtype=np.float64)

lst_m=np.linspace(20,90,n_target)

# Gaussian distributions
a=gauss(n,m=20,s=5) # m= mean, s= std

B=np.zeros((n,n_target))

for i,m in enumerate(lst_m):
    B[:,i]=gauss(n,m=m,s=5)

# loss matrix
M=ot.dist(x.reshape((n,1)),x.reshape((n,1)),'euclidean')
M2=ot.dist(x.reshape((n,1)),x.reshape((n,1)),'sqeuclidean')

#%% plot the distributions

pl.figure(1)
pl.subplot(2,1,1)
pl.plot(x,a,'b',label='Source distribution')
pl.title('Source distribution')
pl.subplot(2,1,2)
pl.plot(x,B,label='Target distributions')
pl.title('Target distributions')

#%% plot distributions and loss matrix

emd=ot.emd2(a,B,M)
emd2=ot.emd2(a,B,M2)
pl.figure(2)
pl.plot(emd,label='Euclidean loss')
pl.plot(emd,label='Squared Euclidean loss')
pl.legend()

