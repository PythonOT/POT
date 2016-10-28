"""
Functions for plotting OT matrices
"""


import numpy as np
import matplotlib.pylab as pl
from matplotlib import gridspec


def plot1D_mat(a,b,M,title=''):
    """ Plot matrix M  with the source and target 1D distribution 
    
    Creates a subplot with the source distribution a on the left and 
    target distribution b on the tot. The matrix M is shown in between.
    
    
    Parameters
    ----------

    a : np.array (na,)
        Source distribution
    b : np.array (nb,)
        Target distribution  
    M : np.array (na,nb)
        Matrix to plot
    
    
    
    """
    
    na=M.shape[0]
    nb=M.shape[1]
    
    gs = gridspec.GridSpec(3, 3)
    
    
    xa=np.arange(na)
    xb=np.arange(nb)
    
    
    ax1=pl.subplot(gs[0,1:])
    pl.plot(xb,b,'r',label='Target distribution')
    pl.yticks(())
    pl.title(title)
    
    #pl.axis('off')
    
    ax2=pl.subplot(gs[1:,0])
    pl.plot(a,xa,'b',label='Source distribution')
    pl.gca().invert_xaxis()
    pl.gca().invert_yaxis()
    pl.xticks(())
    #pl.ylim((0,n))
    #pl.axis('off')
    
    pl.subplot(gs[1:,1:],sharex=ax1,sharey=ax2)
    pl.imshow(M,interpolation='nearest')
    
    pl.xlim((0,nb))


def plot2D_samples_mat(xs,xt,G,thr=1e-8,**kwargs):
    """ Plot matrix M  in 2D with  lines using alpha values
    
    Plot lines between source and target 2D samples with a color 
    proportional to the value of the matrix G between samples.
    
    
    Parameters
    ----------

    xs : np.array (ns,2)
        Source samples positions
    b : np.array (nt,2)
        Target samples positions
    G : np.array (na,nb)
        OT matrix
    thr : float, optional
        threshold above which the line is drawn
    **kwargs : dict
        paameters given to the plot functions (default color is black if nothing given)
    
    """
    if ('color' not in kwargs) and ('c' not  in kwargs):
        kwargs['color']='k'
    mx=G.max()
    for i in range(xs.shape[0]):
        for j in range(xt.shape[0]):
            if G[i,j]/mx>thr:
                pl.plot([xs[i,0],xt[j,0]],[xs[i,1],xt[j,1]],alpha=G[i,j]/mx,**kwargs)
    