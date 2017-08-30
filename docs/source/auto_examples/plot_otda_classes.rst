

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








generate data
#############################################################################



.. code-block:: python


    n_source_samples = 150
    n_target_samples = 150

    Xs, ys = ot.datasets.get_data_classif('3gauss', n_source_samples)
    Xt, yt = ot.datasets.get_data_classif('3gauss2', n_target_samples)








Instantiate the different transport algorithms and fit them
#############################################################################



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
        0|9.456043e+00|0.000000e+00
        1|2.059035e+00|-3.592463e+00
        2|1.839814e+00|-1.191540e-01
        3|1.787860e+00|-2.905942e-02
        4|1.766582e+00|-1.204485e-02
        5|1.760573e+00|-3.413038e-03
        6|1.755288e+00|-3.010556e-03
        7|1.749124e+00|-3.523968e-03
        8|1.744159e+00|-2.846760e-03
        9|1.741007e+00|-1.810862e-03
       10|1.739839e+00|-6.710130e-04
       11|1.737221e+00|-1.507260e-03
       12|1.736011e+00|-6.970742e-04
       13|1.734948e+00|-6.126425e-04
       14|1.733901e+00|-6.038775e-04
       15|1.733768e+00|-7.618542e-05
       16|1.732821e+00|-5.467723e-04
       17|1.732678e+00|-8.226843e-05
       18|1.731934e+00|-4.300066e-04
       19|1.731850e+00|-4.848002e-05
    It.  |Loss        |Delta loss
    --------------------------------
       20|1.731699e+00|-8.729590e-05


Fig 1 : plots source and target samples
#############################################################################



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
#############################################################################



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




**Total running time of the script:** ( 0 minutes  1.906 seconds)



.. container:: sphx-glr-footer


  .. container:: sphx-glr-download

     :download:`Download Python source code: plot_otda_classes.py <plot_otda_classes.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: plot_otda_classes.ipynb <plot_otda_classes.ipynb>`

.. rst-class:: sphx-glr-signature

    `Generated by Sphinx-Gallery <http://sphinx-gallery.readthedocs.io>`_
