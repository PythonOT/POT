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

def plotmat(M,title=''):
    """ Plot a matrix woth the 1D distribution """
    gs = gridspec.GridSpec(3, 3)
    
    ax1=pl.subplot(gs[0,1:])
    pl.plot(x,b,'r',label='Target distribution')
    pl.yticks(())
    pl.title(title)
    
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

pl.figure(2)

plotmat(M,'Cost matrix M')


    
    
    
#pl.ylim((0,n))
#pl.axis('off')

#%% EMD

G0=ot.emd(a,b,M)

pl.figure(3)
plotmat(G0,'OT matrix G0')

#%% Sinkhorn
lambd=1e-3

Gs=ot.sinkhorn(a,b,M,lambd)

pl.figure(4)
plotmat(Gs,'OT matrix Sinkhorn')
