

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
        0|9.984935e+00|0.000000e+00
        1|2.126803e+00|-3.694808e+00
        2|1.867272e+00|-1.389895e-01
        3|1.803858e+00|-3.515488e-02
        4|1.783036e+00|-1.167761e-02
        5|1.774823e+00|-4.627422e-03
        6|1.771947e+00|-1.623526e-03
        7|1.767564e+00|-2.479535e-03
        8|1.763484e+00|-2.313667e-03
        9|1.761138e+00|-1.331780e-03
       10|1.758879e+00|-1.284576e-03
       11|1.758034e+00|-4.806014e-04
       12|1.757595e+00|-2.497155e-04
       13|1.756749e+00|-4.818562e-04
       14|1.755316e+00|-8.161432e-04
       15|1.754988e+00|-1.866236e-04
       16|1.754964e+00|-1.382474e-05
       17|1.754032e+00|-5.315971e-04
       18|1.753595e+00|-2.492359e-04
       19|1.752900e+00|-3.961403e-04
    It.  |Loss        |Delta loss
    --------------------------------
       20|1.752850e+00|-2.869262e-05


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




**Total running time of the script:** ( 0 minutes  1.576 seconds)



.. container:: sphx-glr-footer


  .. container:: sphx-glr-download

     :download:`Download Python source code: plot_otda_classes.py <plot_otda_classes.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: plot_otda_classes.ipynb <plot_otda_classes.ipynb>`

.. rst-class:: sphx-glr-signature

    `Generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_
