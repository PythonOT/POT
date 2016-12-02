

.. _sphx_glr_auto_examples_plot_OTDA_mapping_color_images.py:


====================================================================================
OT for domain adaptation with image color adaptation [6] with mapping estimation [8]
====================================================================================

[6] Ferradans, S., Papadakis, N., Peyre, G., & Aujol, J. F. (2014). Regularized
    discrete optimal transport. SIAM Journal on Imaging Sciences, 7(3), 1853-1882.
[8] M. Perrot, N. Courty, R. Flamary, A. Habrard, "Mapping estimation for
    discrete optimal transport", Neural Information Processing Systems (NIPS), 2016.





.. rst-class:: sphx-glr-horizontal


    *

      .. image:: /auto_examples/images/sphx_glr_plot_OTDA_mapping_color_images_001.png
            :scale: 47

    *

      .. image:: /auto_examples/images/sphx_glr_plot_OTDA_mapping_color_images_002.png
            :scale: 47


.. rst-class:: sphx-glr-script-out

 Out::

    It.  |Loss        |Delta loss
    --------------------------------
        0|3.624802e+02|0.000000e+00
        1|3.547180e+02|-2.141395e-02
        2|3.545494e+02|-4.753955e-04
        3|3.544646e+02|-2.391784e-04
        4|3.544126e+02|-1.466280e-04
        5|3.543775e+02|-9.921805e-05
        6|3.543518e+02|-7.245828e-05
        7|3.543323e+02|-5.491924e-05
        8|3.543170e+02|-4.342401e-05
        9|3.543046e+02|-3.472174e-05
       10|3.542945e+02|-2.878681e-05
       11|3.542859e+02|-2.417065e-05
       12|3.542786e+02|-2.058131e-05
       13|3.542723e+02|-1.768262e-05
       14|3.542668e+02|-1.551616e-05
       15|3.542620e+02|-1.371909e-05
       16|3.542577e+02|-1.213326e-05
       17|3.542538e+02|-1.085481e-05
       18|3.542531e+02|-1.996006e-06
    It.  |Loss        |Delta loss
    --------------------------------
        0|3.555768e+02|0.000000e+00
        1|3.510071e+02|-1.285164e-02
        2|3.509110e+02|-2.736701e-04
        3|3.508748e+02|-1.031476e-04
        4|3.508506e+02|-6.910585e-05
        5|3.508330e+02|-5.014608e-05
        6|3.508195e+02|-3.839166e-05
        7|3.508090e+02|-3.004218e-05
        8|3.508005e+02|-2.417627e-05
        9|3.507935e+02|-2.004621e-05
       10|3.507876e+02|-1.681731e-05




|


.. code-block:: python


    import numpy as np
    import scipy.ndimage as spi
    import matplotlib.pylab as pl
    import ot


    #%% Loading images

    I1=spi.imread('../data/ocean_day.jpg').astype(np.float64)/256
    I2=spi.imread('../data/ocean_sunset.jpg').astype(np.float64)/256

    #%% Plot images

    pl.figure(1)

    pl.subplot(1,2,1)
    pl.imshow(I1)
    pl.title('Image 1')

    pl.subplot(1,2,2)
    pl.imshow(I2)
    pl.title('Image 2')

    pl.show()

    #%% Image conversion and dataset generation

    def im2mat(I):
        """Converts and image to matrix (one pixel per line)"""
        return I.reshape((I.shape[0]*I.shape[1],I.shape[2]))

    def mat2im(X,shape):
        """Converts back a matrix to an image"""
        return X.reshape(shape)

    X1=im2mat(I1)
    X2=im2mat(I2)

    # training samples
    nb=1000
    idx1=np.random.randint(X1.shape[0],size=(nb,))
    idx2=np.random.randint(X2.shape[0],size=(nb,))

    xs=X1[idx1,:]
    xt=X2[idx2,:]

    #%% Plot image distributions


    pl.figure(2,(10,5))

    pl.subplot(1,2,1)
    pl.scatter(xs[:,0],xs[:,2],c=xs)
    pl.axis([0,1,0,1])
    pl.xlabel('Red')
    pl.ylabel('Blue')
    pl.title('Image 1')

    pl.subplot(1,2,2)
    #pl.imshow(I2)
    pl.scatter(xt[:,0],xt[:,2],c=xt)
    pl.axis([0,1,0,1])
    pl.xlabel('Red')
    pl.ylabel('Blue')
    pl.title('Image 2')

    pl.show()



    #%% domain adaptation between images
    def minmax(I):
        return np.minimum(np.maximum(I,0),1)
    # LP problem
    da_emd=ot.da.OTDA()     # init class
    da_emd.fit(xs,xt)       # fit distributions

    X1t=da_emd.predict(X1)  # out of sample
    I1t=minmax(mat2im(X1t,I1.shape))

    # sinkhorn regularization
    lambd=1e-1
    da_entrop=ot.da.OTDA_sinkhorn()
    da_entrop.fit(xs,xt,reg=lambd)

    X1te=da_entrop.predict(X1)
    I1te=minmax(mat2im(X1te,I1.shape))

    # linear mapping estimation
    eta=1e-8   # quadratic regularization for regression
    mu=1e0     # weight of the OT linear term
    bias=True  # estimate a bias

    ot_mapping=ot.da.OTDA_mapping_linear()
    ot_mapping.fit(xs,xt,mu=mu,eta=eta,bias=bias,numItermax = 20,verbose=True)

    X1tl=ot_mapping.predict(X1) # use the estimated mapping
    I1tl=minmax(mat2im(X1tl,I1.shape))

    # nonlinear mapping estimation
    eta=1e-2   # quadratic regularization for regression
    mu=1e0     # weight of the OT linear term
    bias=False  # estimate a bias
    sigma=1    # sigma bandwidth fot gaussian kernel


    ot_mapping_kernel=ot.da.OTDA_mapping_kernel()
    ot_mapping_kernel.fit(xs,xt,mu=mu,eta=eta,sigma=sigma,bias=bias,numItermax = 10,verbose=True)

    X1tn=ot_mapping_kernel.predict(X1) # use the estimated mapping
    I1tn=minmax(mat2im(X1tn,I1.shape))
    #%% plot images


    pl.figure(2,(10,8))

    pl.subplot(2,3,1)

    pl.imshow(I1)
    pl.title('Im. 1')

    pl.subplot(2,3,2)

    pl.imshow(I2)
    pl.title('Im. 2')


    pl.subplot(2,3,3)
    pl.imshow(I1t)
    pl.title('Im. 1 Interp LP')

    pl.subplot(2,3,4)
    pl.imshow(I1te)
    pl.title('Im. 1 Interp Entrop')


    pl.subplot(2,3,5)
    pl.imshow(I1tl)
    pl.title('Im. 1 Linear mapping')

    pl.subplot(2,3,6)
    pl.imshow(I1tn)
    pl.title('Im. 1 nonlinear mapping')

    pl.show()

**Total running time of the script:** ( 1 minutes  59.537 seconds)



.. container:: sphx-glr-footer


  .. container:: sphx-glr-download

     :download:`Download Python source code: plot_OTDA_mapping_color_images.py <plot_OTDA_mapping_color_images.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: plot_OTDA_mapping_color_images.ipynb <plot_OTDA_mapping_color_images.ipynb>`

.. rst-class:: sphx-glr-signature

    `Generated by Sphinx-Gallery <http://sphinx-gallery.readthedocs.io>`_
