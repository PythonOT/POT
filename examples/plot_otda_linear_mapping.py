#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 14:31:15 2018

@author: rflamary
"""

import numpy as np
import pylab as pl
import ot



#%%


n=1000
d=2
sigma=.1

angles=np.random.rand(n,1)*2*np.pi
xs=np.concatenate((np.sin(angles),np.cos(angles)),axis=1)+sigma*np.random.randn(n,2)

xs[:n//2,1]+=2

anglet=np.random.rand(n,1)*2*np.pi
xt=np.concatenate((np.sin(anglet),np.cos(anglet)),axis=1)+sigma*np.random.randn(n,2)
xt[:n//2,1]+=2


A=np.array([[1.5,.7],[.7,1.5]])
b=np.array([[4,2]])
xt=xt.dot(A)+b

#%%

pl.figure(1,(5,5))
pl.plot(xs[:,0],xs[:,1],'+')
pl.plot(xt[:,0],xt[:,1],'o')

#%%

Ae,be=ot.da.OT_mapping_linear(xs,xt)

xst=xs.dot(Ae)+be

##%%

pl.figure(1,(5,5))
pl.clf()
pl.plot(xs[:,0],xs[:,1],'+')
pl.plot(xt[:,0],xt[:,1],'o')
pl.plot(xst[:,0],xst[:,1],'+')