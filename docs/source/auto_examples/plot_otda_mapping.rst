

.. _sphx_glr_auto_examples_plot_otda_mapping.py:


===========================================
OT mapping estimation for domain adaptation
===========================================

This example presents how to use MappingTransport to estimate at the same
time both the coupling transport and approximate the transport map with either
a linear or a kernelized mapping as introduced in [8].

[8] M. Perrot, N. Courty, R. Flamary, A. Habrard,
    "Mapping estimation for discrete optimal transport",
    Neural Information Processing Systems (NIPS), 2016.



.. code-block:: python


    # Authors: Remi Flamary <remi.flamary@unice.fr>
    #          Stanislas Chambon <stan.chambon@gmail.com>
    #
    # License: MIT License

    import numpy as np
    import matplotlib.pylab as pl
    import ot








Generate data
-------------



.. code-block:: python


    n_source_samples = 100
    n_target_samples = 100
    theta = 2 * np.pi / 20
    noise_level = 0.1

    Xs, ys = ot.datasets.get_data_classif(
        'gaussrot', n_source_samples, nz=noise_level)
    Xs_new, _ = ot.datasets.get_data_classif(
        'gaussrot', n_source_samples, nz=noise_level)
    Xt, yt = ot.datasets.get_data_classif(
        'gaussrot', n_target_samples, theta=theta, nz=noise_level)

    # one of the target mode changes its variance (no linear mapping)
    Xt[yt == 2] *= 3
    Xt = Xt + 4







Plot data
---------



.. code-block:: python


    pl.figure(1, (10, 5))
    pl.clf()
    pl.scatter(Xs[:, 0], Xs[:, 1], c=ys, marker='+', label='Source samples')
    pl.scatter(Xt[:, 0], Xt[:, 1], c=yt, marker='o', label='Target samples')
    pl.legend(loc=0)
    pl.title('Source and target distributions')





.. image:: /auto_examples/images/sphx_glr_plot_otda_mapping_001.png
    :align: center




Instantiate the different transport algorithms and fit them
-----------------------------------------------------------



.. code-block:: python


    # MappingTransport with linear kernel
    ot_mapping_linear = ot.da.MappingTransport(
        kernel="linear", mu=1e0, eta=1e-8, bias=True,
        max_iter=20, verbose=True)

    ot_mapping_linear.fit(Xs=Xs, Xt=Xt)

    # for original source samples, transform applies barycentric mapping
    transp_Xs_linear = ot_mapping_linear.transform(Xs=Xs)

    # for out of source samples, transform applies the linear mapping
    transp_Xs_linear_new = ot_mapping_linear.transform(Xs=Xs_new)


    # MappingTransport with gaussian kernel
    ot_mapping_gaussian = ot.da.MappingTransport(
        kernel="gaussian", eta=1e-5, mu=1e-1, bias=True, sigma=1,
        max_iter=10, verbose=True)
    ot_mapping_gaussian.fit(Xs=Xs, Xt=Xt)

    # for original source samples, transform applies barycentric mapping
    transp_Xs_gaussian = ot_mapping_gaussian.transform(Xs=Xs)

    # for out of source samples, transform applies the gaussian mapping
    transp_Xs_gaussian_new = ot_mapping_gaussian.transform(Xs=Xs_new)






.. rst-class:: sphx-glr-script-out

 Out::

    It.  |Loss        |Delta loss
    --------------------------------
        0|4.231423e+03|0.000000e+00
        1|4.217955e+03|-3.182835e-03
        2|4.217580e+03|-8.885864e-05
        3|4.217451e+03|-3.043162e-05
        4|4.217368e+03|-1.978325e-05
        5|4.217312e+03|-1.338471e-05
        6|4.217307e+03|-1.000290e-06
    It.  |Loss        |Delta loss
    --------------------------------
        0|4.257004e+02|0.000000e+00
        1|4.208978e+02|-1.128168e-02
        2|4.205168e+02|-9.052112e-04
        3|4.203566e+02|-3.810681e-04
        4|4.202570e+02|-2.369884e-04
        5|4.201844e+02|-1.726132e-04
        6|4.201341e+02|-1.196461e-04
        7|4.200941e+02|-9.525441e-05
        8|4.200630e+02|-7.405552e-05
        9|4.200377e+02|-6.031884e-05
       10|4.200168e+02|-4.968324e-05


Plot transported samples
------------------------



.. code-block:: python


    pl.figure(2)
    pl.clf()
    pl.subplot(2, 2, 1)
    pl.scatter(Xt[:, 0], Xt[:, 1], c=yt, marker='o',
               label='Target samples', alpha=.2)
    pl.scatter(transp_Xs_linear[:, 0], transp_Xs_linear[:, 1], c=ys, marker='+',
               label='Mapped source samples')
    pl.title("Bary. mapping (linear)")
    pl.legend(loc=0)

    pl.subplot(2, 2, 2)
    pl.scatter(Xt[:, 0], Xt[:, 1], c=yt, marker='o',
               label='Target samples', alpha=.2)
    pl.scatter(transp_Xs_linear_new[:, 0], transp_Xs_linear_new[:, 1],
               c=ys, marker='+', label='Learned mapping')
    pl.title("Estim. mapping (linear)")

    pl.subplot(2, 2, 3)
    pl.scatter(Xt[:, 0], Xt[:, 1], c=yt, marker='o',
               label='Target samples', alpha=.2)
    pl.scatter(transp_Xs_gaussian[:, 0], transp_Xs_gaussian[:, 1], c=ys,
               marker='+', label='barycentric mapping')
    pl.title("Bary. mapping (kernel)")

    pl.subplot(2, 2, 4)
    pl.scatter(Xt[:, 0], Xt[:, 1], c=yt, marker='o',
               label='Target samples', alpha=.2)
    pl.scatter(transp_Xs_gaussian_new[:, 0], transp_Xs_gaussian_new[:, 1], c=ys,
               marker='+', label='Learned mapping')
    pl.title("Estim. mapping (kernel)")
    pl.tight_layout()

    pl.show()



.. image:: /auto_examples/images/sphx_glr_plot_otda_mapping_003.png
    :align: center




**Total running time of the script:** ( 0 minutes  0.970 seconds)



.. container:: sphx-glr-footer


  .. container:: sphx-glr-download

     :download:`Download Python source code: plot_otda_mapping.py <plot_otda_mapping.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: plot_otda_mapping.ipynb <plot_otda_mapping.ipynb>`

.. rst-class:: sphx-glr-signature

    `Generated by Sphinx-Gallery <http://sphinx-gallery.readthedocs.io>`_
