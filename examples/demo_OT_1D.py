# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 09:51:45 2016

@author: rflamary
"""

import numpy as np
import matplotlib.pylab as pl
from matplotlib import gridspec
import ot



#%% parameters

n=100 # nb bins

ma=20 # mean of a
mb=60 # mean of b

sa=20 # std of a
sb=60 # std of b

# bin positions
x=np.arange(n,dtype=np.float64)

# Gaussian distributions
a=ot.datasets.get_1D_gauss(n,ma,sa)
b=ot.datasets.get_1D_gauss(n,mb,sb)

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
gs = gridspec.GridSpec(3, 3)

ax1=pl.subplot(gs[0,1:])
pl.plot(x,b,'r',label='Target distribution')
pl.yticks(())

#pl.axis('off')

ax2=pl.subplot(gs[1:,0])
pl.plot(a,x,'b',label='Source distribution')
pl.gca().invert_xaxis()
pl.gca().invert_yaxis()
pl.xticks(())
#pl.ylim((0,n))
#pl.axis('off')

pl.subplot(gs[1:,1:],sharex=ax1,sharey=ax2)
pl.imshow(M,interpolation='nearest')

pl.xlim((0,n))
#pl.ylim((0,n))
#pl.axis('off')

#%% EMD

G0=ot.emd(a,b,M)

#%% plot EMD optimal tranport matrix
pl.figure(3)
gs = gridspec.GridSpec(3, 3)

ax1=pl.subplot(gs[0,1:])
pl.plot(x,b,'r',label='Target distribution')
pl.yticks(())

#pl.axis('off')

ax2=pl.subplot(gs[1:,0])
pl.plot(a,x,'b',label='Source distribution')
pl.gca().invert_xaxis()
pl.gca().invert_yaxis()
pl.xticks(())
#pl.ylim((0,n))
#pl.axis('off')

pl.subplot(gs[1:,1:],sharex=ax1,sharey=ax2)
pl.imshow(G0,interpolation='nearest')

pl.xlim((0,n))
#pl.ylim((0,n))
#pl.axis('off')

#%% Sinkhorn
lambd=1e3

Gs=ot.sinkhorn(a,b,M,lambd)


#%% plot Sikhorn optimal tranport matrix
pl.figure(3)
gs = gridspec.GridSpec(3, 3)

ax1=pl.subplot(gs[0,1:])
pl.plot(x,b,'r',label='Target distribution')
pl.yticks(())

#pl.axis('off')

ax2=pl.subplot(gs[1:,0])
pl.plot(a,x,'b',label='Source distribution')
pl.gca().invert_xaxis()
pl.gca().invert_yaxis()
pl.xticks(())
#pl.ylim((0,n))
#pl.axis('off')

pl.subplot(gs[1:,1:],sharex=ax1,sharey=ax2)
pl.imshow(Gs,interpolation='nearest')

pl.xlim((0,n))