

.. _sphx_glr_auto_examples_plot_OT_conv.py:


==============================
1D Wasserstein barycenter demo
==============================


@author: rflamary




.. code-block:: pytb

    Traceback (most recent call last):
      File "/home/rflamary/.local/lib/python2.7/site-packages/sphinx_gallery/gen_rst.py", line 518, in execute_code_block
        exec(code_block, example_globals)
      File "<string>", line 86, in <module>
    TypeError: unsupported operand type(s) for *: 'float' and 'Mock'





.. code-block:: python


    import numpy as np
    import matplotlib.pylab as pl
    import ot
    from mpl_toolkits.mplot3d import Axes3D #necessary for 3d plot even if not used
    import scipy as sp
    import scipy.signal as sps
    #%% parameters

    n=10 # nb bins

    # bin positions
    x=np.arange(n,dtype=np.float64)

    xx,yy=np.meshgrid(x,x)


    xpos=np.hstack((xx.reshape(-1,1),yy.reshape(-1,1)))

    M=ot.dist(xpos)


    I0=((xx-5)**2+(yy-5)**2<3**2)*1.0
    I1=((xx-7)**2+(yy-7)**2<3**2)*1.0

    I0/=I0.sum()
    I1/=I1.sum()

    i0=I0.ravel()
    i1=I1.ravel()

    M=M[i0>0,:][:,i1>0].copy()
    i0=i0[i0>0]
    i1=i1[i1>0]
    Itot=np.concatenate((I0[:,:,np.newaxis],I1[:,:,np.newaxis]),2)


    #%% plot the distributions

    pl.figure(1)
    pl.subplot(2,2,1)
    pl.imshow(I0)
    pl.subplot(2,2,2)
    pl.imshow(I1)


    #%% barycenter computation

    alpha=0.5 # 0<=alpha<=1
    weights=np.array([1-alpha,alpha])


    def conv2(I,k):
        return sp.ndimage.convolve1d(sp.ndimage.convolve1d(I,k,axis=1),k,axis=0)

    def conv2n(I,k):
        res=np.zeros_like(I)
        for i in range(I.shape[2]):
            res[:,:,i]=conv2(I[:,:,i],k)
        return res


    def get_1Dkernel(reg,thr=1e-16,wmax=1024):
        w=max(min(wmax,2*int((-np.log(thr)*reg)**(.5))),3)
        x=np.arange(w,dtype=np.float64)
        return np.exp(-((x-w/2)**2)/reg)
    
    thr=1e-16
    reg=1e0

    k=get_1Dkernel(reg)
    pl.figure(2)
    pl.plot(k)

    I05=conv2(I0,k)

    pl.figure(1)
    pl.subplot(2,2,1)
    pl.imshow(I0)
    pl.subplot(2,2,2)
    pl.imshow(I05)

    #%%

    G=ot.emd(i0,i1,M)
    r0=np.sum(M*G)

    reg=1e-1
    Gs=ot.bregman.sinkhorn_knopp(i0,i1,M,reg=reg)
    rs=np.sum(M*Gs)

    #%%

    def mylog(u):
        tmp=np.log(u)
        tmp[np.isnan(tmp)]=0
        return tmp

    def sinkhorn_conv(a,b, reg, numItermax = 1000, stopThr=1e-9, verbose=False, log=False,**kwargs):


        a=np.asarray(a,dtype=np.float64)
        b=np.asarray(b,dtype=np.float64)
        
    
        if len(b.shape)>2:
            nbb=b.shape[2]
            a=a[:,:,np.newaxis]
        else:
            nbb=0
    

        if log:
            log={'err':[]}

        # we assume that no distances are null except those of the diagonal of distances
        if nbb:
            u = np.ones((a.shape[0],a.shape[1],nbb))/(np.prod(a.shape[:2]))
            v = np.ones((a.shape[0],a.shape[1],nbb))/(np.prod(b.shape[:2]))
            a0=1.0/(np.prod(b.shape[:2]))
        else:
            u = np.ones((a.shape[0],a.shape[1]))/(np.prod(a.shape[:2]))
            v = np.ones((a.shape[0],a.shape[1]))/(np.prod(b.shape[:2]))
            a0=1.0/(np.prod(b.shape[:2]))
        
        
        k=get_1Dkernel(reg)
    
        if nbb:
            K=lambda I: conv2n(I,k)
        else:
            K=lambda I: conv2(I,k)

        cpt = 0
        err=1
        while (err>stopThr and cpt<numItermax):
            uprev = u
            vprev = v
        
            v = np.divide(b, K(u))
            u = np.divide(a, K(v))

            if (np.any(np.isnan(u)) or np.any(np.isnan(v)) 
                or np.any(np.isinf(u)) or np.any(np.isinf(v))):
                # we have reached the machine precision
                # come back to previous solution and quit loop
                print('Warning: numerical errors at iteration', cpt)
                u = uprev
                v = vprev
                break
            if cpt%10==0:
                # we can speed up the process by checking for the error only all the 10th iterations

                err = np.sum((u-uprev)**2)/np.sum((u)**2)+np.sum((v-vprev)**2)/np.sum((v)**2)

                if log:
                    log['err'].append(err)

                if verbose:
                    if cpt%200 ==0:
                        print('{:5s}|{:12s}'.format('It.','Err')+'\n'+'-'*19)
                    print('{:5d}|{:8e}|'.format(cpt,err))
            cpt = cpt +1
        if log:
            log['u']=u
            log['v']=v
        
        if nbb: #return only loss 
            res=np.zeros((nbb))
            for i in range(nbb):
                res[i]=np.sum(u[:,i].reshape((-1,1))*K*v[:,i].reshape((1,-1))*M)
            if log:
                return res,log
            else:
                return res        
        
        else: # return OT matrix
            res=reg*a0*np.sum(a*mylog(u+(u==0))+b*mylog(v+(v==0)))
            if log:
            
                return res,log
            else:
                return res

    reg=1e0
    r,log=sinkhorn_conv(I0,I1,reg,verbose=True,log=True)
    a=I0
    b=I1
    u=log['u']
    v=log['v']
    #%% barycenter interpolation

**Total running time of the script:** ( 0 minutes  0.000 seconds)



.. container:: sphx-glr-footer


  .. container:: sphx-glr-download

     :download:`Download Python source code: plot_OT_conv.py <plot_OT_conv.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: plot_OT_conv.ipynb <plot_OT_conv.ipynb>`

.. rst-class:: sphx-glr-signature

    `Generated by Sphinx-Gallery <http://sphinx-gallery.readthedocs.io>`_
