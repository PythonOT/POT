"""
Simple example datasets for OT
"""


import numpy as np
import scipy as sp


def get_1D_gauss(n,m,s):
    """return a 1D histogram for a gaussian distribution (n bins, mean m and std s) 
    
    Parameters
    ----------

    n : int
        number of bins in the histogram
    m : float
        mean value of the gaussian distribution
    s : float
        standard deviaton of the gaussian distribution

  
    Returns
    -------
    h : np.array (n,)
          1D histogram for a gaussian distribution      
    
    """
    x=np.arange(n,dtype=np.float64)
    h=np.exp(-(x-m)**2/(2*s^2))
    return h/h.sum()
    
    
def get_2D_samples_gauss(n,m,sigma):
    """return n samples drawn from 2D gaussian N(m,sigma) 
    
    Parameters
    ----------

    n : int
        number of bins in the histogram
    m : np.array (2,)
        mean value of the gaussian distribution
    sigma : np.array (2,2)
        covariance matrix of the gaussian distribution

  
    Returns
    -------
    X : np.array (n,2)
          n samples drawn from  N(m,sigma)       
    
    """
    if  np.isscalar(sigma):
        sigma=np.array([sigma,])
    if len(sigma)>1:
        P=sp.linalg.sqrtm(sigma)
        res= np.random.randn(n,2).dot(P)+m
    else:
        res= np.random.randn(n,2)*np.sqrt(sigma)+m
    return res

def get_data_classif(dataset,n,nz=.5,**kwargs):
    """ dataset generation for classification problems
    
    Parameters
    ----------

    dataset : str
        type of classification problem (see code)
    n : int
        number of training samples
    nz : float
        noise level (>0)

  
    Returns
    -------
    X : np.array (n,d)
          n observation of size d      
    y : np.array (n,)
          labels of the samples         

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
    else:
        x=0
        y=0
        print("unknown dataset")
    
    return x,y.astype(int)