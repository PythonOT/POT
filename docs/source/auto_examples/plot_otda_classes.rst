

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

    Xs, ys = ot.datasets.make_data_classif('3gauss', n_source_samples)
    Xt, yt = ot.datasets.make_data_classif('3gauss2', n_target_samples)








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
        0|9.566309e+00|0.000000e+00
        1|2.169680e+00|-3.409088e+00
        2|1.914989e+00|-1.329986e-01
        3|1.860251e+00|-2.942498e-02
        4|1.838073e+00|-1.206621e-02
        5|1.827064e+00|-6.025122e-03
        6|1.820899e+00|-3.386082e-03
        7|1.817290e+00|-1.985705e-03
        8|1.814644e+00|-1.458223e-03
        9|1.812661e+00|-1.093816e-03
       10|1.810239e+00|-1.338121e-03
       11|1.809100e+00|-6.296940e-04
       12|1.807939e+00|-6.420646e-04
       13|1.806965e+00|-5.389118e-04
       14|1.806822e+00|-7.889599e-05
       15|1.806193e+00|-3.482356e-04
       16|1.805735e+00|-2.536930e-04
       17|1.805321e+00|-2.292667e-04
       18|1.804389e+00|-5.170222e-04
       19|1.803908e+00|-2.661907e-04
    It.  |Loss        |Delta loss
    --------------------------------
       20|1.803696e+00|-1.178279e-04


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


    param_img = {'interpolation': 'nearest'}

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




**Total running time of the script:** ( 0 minutes  1.423 seconds)



.. only :: html

 .. container:: sphx-glr-footer


  .. container:: sphx-glr-download

     :download:`Download Python source code: plot_otda_classes.py <plot_otda_classes.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: plot_otda_classes.ipynb <plot_otda_classes.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_
