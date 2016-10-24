
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
    