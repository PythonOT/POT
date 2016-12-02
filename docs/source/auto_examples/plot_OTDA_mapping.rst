

.. _sphx_glr_auto_examples_plot_OTDA_mapping.py:


===============================================
OT mapping estimation for domain adaptation [8]
===============================================

[8] M. Perrot, N. Courty, R. Flamary, A. Habrard, "Mapping estimation for
    discrete optimal transport", Neural Information Processing Systems (NIPS), 2016.




.. rst-class:: sphx-glr-horizontal


    *

      .. image:: /auto_examples/images/sphx_glr_plot_OTDA_mapping_001.png
            :scale: 47

    *

      .. image:: /auto_examples/images/sphx_glr_plot_OTDA_mapping_002.png
            :scale: 47


.. rst-class:: sphx-glr-script-out

 Out::

    It.  |Loss        |Delta loss
    --------------------------------
        0|4.009366e+03|0.000000e+00
        1|3.999933e+03|-2.352753e-03
        2|3.999520e+03|-1.031984e-04
        3|3.999362e+03|-3.936391e-05
        4|3.999281e+03|-2.032868e-05
        5|3.999238e+03|-1.083083e-05
        6|3.999229e+03|-2.125291e-06
    It.  |Loss        |Delta loss
    --------------------------------
        0|4.026841e+02|0.000000e+00
        1|3.990791e+02|-8.952439e-03
        2|3.987954e+02|-7.107124e-04
        3|3.986554e+02|-3.512453e-04
        4|3.985721e+02|-2.087997e-04
        5|3.985141e+02|-1.456184e-04
        6|3.984729e+02|-1.034624e-04
        7|3.984435e+02|-7.366943e-05
        8|3.984199e+02|-5.922497e-05
        9|3.984016e+02|-4.593063e-05
       10|3.983867e+02|-3.733061e-05




|


.. code-block:: python


    import numpy as np
    import matplotlib.pylab as pl
    import ot



    #%% dataset generation

    np.random.seed(0) # makes example reproducible

    n=100 # nb samples in source and target datasets
    theta=2*np.pi/20
    nz=0.1
    xs,ys=ot.datasets.get_data_classif('gaussrot',n,nz=nz)
    xt,yt=ot.datasets.get_data_classif('gaussrot',n,theta=theta,nz=nz)

    # one of the target mode changes its variance (no linear mapping)
    xt[yt==2]*=3
    xt=xt+4


    #%% plot samples

    pl.figure(1,(8,5))
    pl.clf()

    pl.scatter(xs[:,0],xs[:,1],c=ys,marker='+',label='Source samples')
    pl.scatter(xt[:,0],xt[:,1],c=yt,marker='o',label='Target samples')

    pl.legend(loc=0)
    pl.title('Source and target distributions')



    #%% OT linear mapping estimation

    eta=1e-8   # quadratic regularization for regression
    mu=1e0     # weight of the OT linear term
    bias=True  # estimate a bias

    ot_mapping=ot.da.OTDA_mapping_linear()
    ot_mapping.fit(xs,xt,mu=mu,eta=eta,bias=bias,numItermax = 20,verbose=True)

    xst=ot_mapping.predict(xs) # use the estimated mapping
    xst0=ot_mapping.interp()   # use barycentric mapping


    pl.figure(2,(10,7))
    pl.clf()
    pl.subplot(2,2,1)
    pl.scatter(xt[:,0],xt[:,1],c=yt,marker='o',label='Target samples',alpha=.3)
    pl.scatter(xst0[:,0],xst0[:,1],c=ys,marker='+',label='barycentric mapping')
    pl.title("barycentric mapping")

    pl.subplot(2,2,2)
    pl.scatter(xt[:,0],xt[:,1],c=yt,marker='o',label='Target samples',alpha=.3)
    pl.scatter(xst[:,0],xst[:,1],c=ys,marker='+',label='Learned mapping')
    pl.title("Learned mapping")



    #%% Kernel mapping estimation

    eta=1e-5   # quadratic regularization for regression
    mu=1e-1     # weight of the OT linear term
    bias=True  # estimate a bias
    sigma=1    # sigma bandwidth fot gaussian kernel


    ot_mapping_kernel=ot.da.OTDA_mapping_kernel()
    ot_mapping_kernel.fit(xs,xt,mu=mu,eta=eta,sigma=sigma,bias=bias,numItermax = 10,verbose=True)

    xst_kernel=ot_mapping_kernel.predict(xs) # use the estimated mapping
    xst0_kernel=ot_mapping_kernel.interp()   # use barycentric mapping


    #%% Plotting the mapped samples

    pl.figure(2,(10,7))
    pl.clf()
    pl.subplot(2,2,1)
    pl.scatter(xt[:,0],xt[:,1],c=yt,marker='o',label='Target samples',alpha=.2)
    pl.scatter(xst0[:,0],xst0[:,1],c=ys,marker='+',label='Mapped source samples')
    pl.title("Bary. mapping (linear)")
    pl.legend(loc=0)

    pl.subplot(2,2,2)
    pl.scatter(xt[:,0],xt[:,1],c=yt,marker='o',label='Target samples',alpha=.2)
    pl.scatter(xst[:,0],xst[:,1],c=ys,marker='+',label='Learned mapping')
    pl.title("Estim. mapping (linear)")

    pl.subplot(2,2,3)
    pl.scatter(xt[:,0],xt[:,1],c=yt,marker='o',label='Target samples',alpha=.2)
    pl.scatter(xst0_kernel[:,0],xst0_kernel[:,1],c=ys,marker='+',label='barycentric mapping')
    pl.title("Bary. mapping (kernel)")

    pl.subplot(2,2,4)
    pl.scatter(xt[:,0],xt[:,1],c=yt,marker='o',label='Target samples',alpha=.2)
    pl.scatter(xst_kernel[:,0],xst_kernel[:,1],c=ys,marker='+',label='Learned mapping')
    pl.title("Estim. mapping (kernel)")

**Total running time of the script:** ( 0 minutes  0.882 seconds)



.. container:: sphx-glr-footer


  .. container:: sphx-glr-download

     :download:`Download Python source code: plot_OTDA_mapping.py <plot_OTDA_mapping.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: plot_OTDA_mapping.ipynb <plot_OTDA_mapping.ipynb>`

.. rst-class:: sphx-glr-signature

    `Generated by Sphinx-Gallery <http://sphinx-gallery.readthedocs.io>`_
