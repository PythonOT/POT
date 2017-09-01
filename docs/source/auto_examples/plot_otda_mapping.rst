

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
#############################################################################



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
#############################################################################



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
#############################################################################



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
        0|4.481482e+03|0.000000e+00
        1|4.469389e+03|-2.698549e-03
        2|4.468825e+03|-1.261217e-04
        3|4.468580e+03|-5.486064e-05
        4|4.468438e+03|-3.161220e-05
        5|4.468352e+03|-1.930800e-05
        6|4.468309e+03|-9.570658e-06
    It.  |Loss        |Delta loss
    --------------------------------
        0|4.504654e+02|0.000000e+00
        1|4.461571e+02|-9.564116e-03
        2|4.459105e+02|-5.528043e-04
        3|4.457895e+02|-2.712398e-04
        4|4.457041e+02|-1.914829e-04
        5|4.456431e+02|-1.369704e-04
        6|4.456032e+02|-8.944784e-05
        7|4.455700e+02|-7.447824e-05
        8|4.455447e+02|-5.688965e-05
        9|4.455229e+02|-4.890051e-05
       10|4.455084e+02|-3.262490e-05


Plot transported samples
#############################################################################



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




**Total running time of the script:** ( 0 minutes  0.869 seconds)



.. container:: sphx-glr-footer


  .. container:: sphx-glr-download

     :download:`Download Python source code: plot_otda_mapping.py <plot_otda_mapping.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: plot_otda_mapping.ipynb <plot_otda_mapping.ipynb>`

.. rst-class:: sphx-glr-signature

    `Generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_
