
import numpy as np



def get_1D_gauss(n,m,s):
    "return a 1D histogram for a gaussian distribution (n bins, mean m and std s) "
    x=np.arange(n,dtype=np.float64)
    h=np.exp(-(x-m)**2/(2*s^2))
    return h/h.sum()