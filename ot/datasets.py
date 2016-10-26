"""
Simple example datasets for OT
"""


import numpy as np
import scipy as sp


def get_1D_gauss(n,m,s):
    "return a 1D histogram for a gaussian distribution (n bins, mean m and std s) "
    x=np.arange(n,dtype=np.float64)
    h=np.exp(-(x-m)**2/(2*s^2))
    return h/h.sum()
    
    
def get_2D_samples_gauss(n,m,sigma):
    "return samples from 2D gaussian (n samples, mean m and cov sigma) "
    if  np.isscalar(sigma):
        sigma=np.array([sigma,])
    if len(sigma)>1:
        P=sp.linalg.sqrtm(sigma)
        res= np.random.randn(n,2).dot(P)+m
    else:
        res= np.random.randn(n,2)*np.sqrt(sigma)+m
    return res

def get_data_classif(dataset,n,nz=.5,**kwargs):
    """
    dataset generation
    """
    if dataset.lower()=='3gauss':
        y=np.floor((np.arange(n)*1.0/n*3))+1
        x=np.zeros((n,2))
        # class 1
        x[y==1,0]=-1.; x[y==1,1]=-1.
        x[y==2,0]=-1.; x[y==2,1]=1.
        x[y==3,0]=1. ; x[y==3,1]=0
        
        x[y!=3,:]+=1.5*nz*np.random.randn(sum(y!=3),2)
        x[y==3,:]+=2*nz*np.random.randn(sum(y==3),2)
        
    elif dataset.lower()=='3gauss2':
        y=np.floor((np.arange(n)*1.0/n*3))+1
        x=np.zeros((n,2))
        y[y==4]=3
        # class 1
        x[y==1,0]=-2.; x[y==1,1]=-2.
        x[y==2,0]=-2.; x[y==2,1]=2.
        x[y==3,0]=2. ; x[y==3,1]=0
        
        x[y!=3,:]+=nz*np.random.randn(sum(y!=3),2)
        x[y==3,:]+=2*nz*np.random.randn(sum(y==3),2)   
#    elif dataset.lower()=='sinreg':
#        
#        x=np.random.rand(n,1)
#        y=4*x+np.sin(2*np.pi*x)+nz*np.random.randn(n,1) 
         
    else:
        x=0
        y=0
        print("unknown dataset")
    
    return x,y