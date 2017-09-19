

.. _sphx_glr_auto_examples_plot_otda_classes.py:


========================
OT for domain adaptation
========================

This example introduces a domain adaptation in a 2D setting and the 4 OTDA
approaches currently supported in POT.




.. code-block:: python


    # Authors: Remi Flamary <remi.flamary@unice.fr>
    #          Stanislas Chambon <stan.chambon@gmail.com>
    #
    # License: MIT License

    import matplotlib.pylab as pl
    import ot








Generate data
-------------



.. code-block:: python


    n_source_samples = 150
    n_target_samples = 150

    Xs, ys = ot.datasets.get_data_classif('3gauss', n_source_samples)
    Xt, yt = ot.datasets.get_data_classif('3gauss2', n_target_samples)








Instantiate the different transport algorithms and fit them
-----------------------------------------------------------



.. code-block:: python


    # EMD Transport
    ot_emd = ot.da.EMDTransport()
    ot_emd.fit(Xs=Xs, Xt=Xt)

    # Sinkhorn Transport
    ot_sinkhorn = ot.da.SinkhornTransport(reg_e=1e-1)
    ot_sinkhorn.fit(Xs=Xs, Xt=Xt)

    # Sinkhorn Transport with Group lasso regularization
    ot_lpl1 = ot.da.SinkhornLpl1Transport(reg_e=1e-1, reg_cl=1e0)
    ot_lpl1.fit(Xs=Xs, ys=ys, Xt=Xt)

    # Sinkhorn Transport with Group lasso regularization l1l2
    ot_l1l2 = ot.da.SinkhornL1l2Transport(reg_e=1e-1, reg_cl=2e0, max_iter=20,
                                          verbose=True)
    ot_l1l2.fit(Xs=Xs, ys=ys, Xt=Xt)

    # transport source samples onto target samples
    transp_Xs_emd = ot_emd.transform(Xs=Xs)
    transp_Xs_sinkhorn = ot_sinkhorn.transform(Xs=Xs)
    transp_Xs_lpl1 = ot_lpl1.transform(Xs=Xs)
    transp_Xs_l1l2 = ot_l1l2.transform(Xs=Xs)






.. rst-class:: sphx-glr-script-out

 Out::

    It.  |Loss        |Delta loss
    --------------------------------
        0|1.003747e+01|0.000000e+00
        1|1.953263e+00|-4.138821e+00
        2|1.744456e+00|-1.196969e-01
        3|1.689268e+00|-3.267022e-02
        4|1.666355e+00|-1.374998e-02
        5|1.656125e+00|-6.177356e-03
        6|1.651753e+00|-2.646960e-03
        7|1.647261e+00|-2.726957e-03
        8|1.642274e+00|-3.036672e-03
        9|1.639926e+00|-1.431818e-03
       10|1.638750e+00|-7.173837e-04
       11|1.637558e+00|-7.281753e-04
       12|1.636248e+00|-8.002067e-04
       13|1.634555e+00|-1.036074e-03
       14|1.633547e+00|-6.166646e-04
       15|1.633531e+00|-1.022614e-05
       16|1.632957e+00|-3.510986e-04
       17|1.632853e+00|-6.380944e-05
       18|1.632704e+00|-9.122988e-05
       19|1.632237e+00|-2.861276e-04
    It.  |Loss        |Delta loss
    --------------------------------
       20|1.632174e+00|-3.896483e-05


Fig 1 : plots source and target samples
---------------------------------------



.. code-block:: python


    pl.figure(1, figsize=(10, 5))
    pl.subplot(1, 2, 1)
    pl.scatter(Xs[:, 0], Xs[:, 1], c=ys, marker='+', label='Source samples')
    pl.xticks([])
    pl.yticks([])
    pl.legend(loc=0)
    pl.title('Source  samples')

    pl.subplot(1, 2, 2)
    pl.scatter(Xt[:, 0], Xt[:, 1], c=yt, marker='o', label='Target samples')
    pl.xticks([])
    pl.yticks([])
    pl.legend(loc=0)
    pl.title('Target samples')
    pl.tight_layout()





.. image:: /auto_examples/images/sphx_glr_plot_otda_classes_001.png
    :align: center




Fig 2 : plot optimal couplings and transported samples
------------------------------------------------------



.. code-block:: python


    param_img = {'interpolation': 'nearest', 'cmap': 'spectral'}

    pl.figure(2, figsize=(15, 8))
    pl.subplot(2, 4, 1)
    pl.imshow(ot_emd.coupling_, **param_img)
    pl.xticks([])
    pl.yticks([])
    pl.title('Optimal coupling\nEMDTransport')

    pl.subplot(2, 4, 2)
    pl.imshow(ot_sinkhorn.coupling_, **param_img)
    pl.xticks([])
    pl.yticks([])
    pl.title('Optimal coupling\nSinkhornTransport')

    pl.subplot(2, 4, 3)
    pl.imshow(ot_lpl1.coupling_, **param_img)
    pl.xticks([])
    pl.yticks([])
    pl.title('Optimal coupling\nSinkhornLpl1Transport')

    pl.subplot(2, 4, 4)
    pl.imshow(ot_l1l2.coupling_, **param_img)
    pl.xticks([])
    pl.yticks([])
    pl.title('Optimal coupling\nSinkhornL1l2Transport')

    pl.subplot(2, 4, 5)
    pl.scatter(Xt[:, 0], Xt[:, 1], c=yt, marker='o',
               label='Target samples', alpha=0.3)
    pl.scatter(transp_Xs_emd[:, 0], transp_Xs_emd[:, 1], c=ys,
               marker='+', label='Transp samples', s=30)
    pl.xticks([])
    pl.yticks([])
    pl.title('Transported samples\nEmdTransport')
    pl.legend(loc="lower left")

    pl.subplot(2, 4, 6)
    pl.scatter(Xt[:, 0], Xt[:, 1], c=yt, marker='o',
               label='Target samples', alpha=0.3)
    pl.scatter(transp_Xs_sinkhorn[:, 0], transp_Xs_sinkhorn[:, 1], c=ys,
               marker='+', label='Transp samples', s=30)
    pl.xticks([])
    pl.yticks([])
    pl.title('Transported samples\nSinkhornTransport')

    pl.subplot(2, 4, 7)
    pl.scatter(Xt[:, 0], Xt[:, 1], c=yt, marker='o',
               label='Target samples', alpha=0.3)
    pl.scatter(transp_Xs_lpl1[:, 0], transp_Xs_lpl1[:, 1], c=ys,
               marker='+', label='Transp samples', s=30)
    pl.xticks([])
    pl.yticks([])
    pl.title('Transported samples\nSinkhornLpl1Transport')

    pl.subplot(2, 4, 8)
    pl.scatter(Xt[:, 0], Xt[:, 1], c=yt, marker='o',
               label='Target samples', alpha=0.3)
    pl.scatter(transp_Xs_l1l2[:, 0], transp_Xs_l1l2[:, 1], c=ys,
               marker='+', label='Transp samples', s=30)
    pl.xticks([])
    pl.yticks([])
    pl.title('Transported samples\nSinkhornL1l2Transport')
    pl.tight_layout()

    pl.show()



.. image:: /auto_examples/images/sphx_glr_plot_otda_classes_003.png
    :align: center




**Total running time of the script:** ( 0 minutes  2.308 seconds)



.. container:: sphx-glr-footer


  .. container:: sphx-glr-download

     :download:`Download Python source code: plot_otda_classes.py <plot_otda_classes.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: plot_otda_classes.ipynb <plot_otda_classes.ipynb>`

.. rst-class:: sphx-glr-signature

    `Generated by Sphinx-Gallery <http://sphinx-gallery.readthedocs.io>`_
