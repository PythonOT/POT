# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 09:51:45 2016

@author: rflamary
"""

import numpy as np
import matplotlib.pylab as pl
import ot



#%% parameters

n=20 # nb bins

mu_s=np.array([0,0])
cov_s=np.array([[1,0],[0,1]])

mu_t=np.array([4,4])
cov_t=np.array([[1,-.8],[-.8,1]])

xs=ot.datasets.get_2D_samples_gauss(n,mu_s,cov_s)
xt=ot.datasets.get_2D_samples_gauss(n,mu_t,cov_t)

a,b = ot.unif(n),ot.unif(n)

# loss matrix
M=ot.dist(xs,xt)
#M/=M.max()

#%% plot samples

pl.figure(1)
pl.plot(xs[:,0],xs[:,1],'+b',label='Source samples')
pl.plot(xt[:,0],xt[:,1],'xr',label='Target samples')
pl.legend(loc=0)
pl.title('Source and traget distributions')

pl.figure(2)
pl.imshow(M,interpolation='nearest')
pl.title('Cost matrix M')


#%% EMD

G0=ot.emd(a,b,M)


pl.figure(3)
pl.imshow(G0,interpolation='nearest')
pl.title('Cost matrix M')

pl.figure(4)
ot.plot.otplot2D_samples(xs,xt,G0,c=[.5,.5,1])
pl.plot(xs[:,0],xs[:,1],'+b',label='Source samples')
pl.plot(xt[:,0],xt[:,1],'xr',label='Target samples')
pl.legend(loc=0)
pl.title('OT matrix')


#%% sinkhorn

lambd=.8e-1
Gs=ot.sinkhorn(a,b,M,lambd)


pl.figure(5)
pl.imshow(Gs,interpolation='nearest')
pl.title('Cost matrix M')

pl.figure(6)
ot.plot.otplot2D_samples(xs,xt,Gs,color=[.5,.5,1])
pl.plot(xs[:,0],xs[:,1],'+b',label='Source samples')
pl.plot(xt[:,0],xt[:,1],'xr',label='Target samples')
pl.legend(loc=0)
pl.title('OT matrix Sinkhorn')

#
#pl.figure(3)
#ot.plot.otplot1D(a,b,G0,'OT matrix G0')
#
##%% Sinkhorn
#
#lambd=1e-3
#Gs=ot.sinkhorn(a,b,M,lambd)
#
#pl.figure(4)
#ot.plot.otplot1D(a,b,Gs,'OT matrix Sinkhorn')
