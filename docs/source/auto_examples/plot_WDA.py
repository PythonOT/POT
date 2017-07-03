# -*- coding: utf-8 -*-
"""
=================================
Wasserstein Discriminant Analysis
=================================

@author: rflamary
"""

import numpy as np
import matplotlib.pylab as pl
import ot
from ot.datasets import get_1D_gauss as gauss
from ot.dr import wda


#%% parameters

n=1000 # nb samples in source and target datasets
nz=0.2
xs,ys=ot.datasets.get_data_classif('3gauss',n,nz)
xt,yt=ot.datasets.get_data_classif('3gauss',n,nz)

nbnoise=8

xs=np.hstack((xs,np.random.randn(n,nbnoise)))
xt=np.hstack((xt,np.random.randn(n,nbnoise)))

#%% plot samples

pl.figure(1)


pl.scatter(xt[:,0],xt[:,1],c=ys,marker='+',label='Source samples')
pl.legend(loc=0)
pl.title('Discriminant dimensions')


#%% plot distributions and loss matrix
p=2
reg=1
k=10
maxiter=100

P,proj = wda(xs,ys,p,reg,k,maxiter=maxiter)

#%% plot samples

xsp=proj(xs)
xtp=proj(xt)

pl.figure(1,(10,5))

pl.subplot(1,2,1)
pl.scatter(xsp[:,0],xsp[:,1],c=ys,marker='+',label='Projected samples')
pl.legend(loc=0)
pl.title('Projected training samples')


pl.subplot(1,2,2)
pl.scatter(xtp[:,0],xtp[:,1],c=ys,marker='+',label='Projected samples')
pl.legend(loc=0)
pl.title('Projected test samples')
