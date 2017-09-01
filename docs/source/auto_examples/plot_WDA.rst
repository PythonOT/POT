

.. _sphx_glr_auto_examples_plot_WDA.py:


=================================
Wasserstein Discriminant Analysis
=================================

This example illustrate the use of WDA as proposed in [11].


[11] Flamary, R., Cuturi, M., Courty, N., & Rakotomamonjy, A. (2016). 
Wasserstein Discriminant Analysis.




.. code-block:: python


    # Author: Remi Flamary <remi.flamary@unice.fr>
    #
    # License: MIT License

    import numpy as np
    import matplotlib.pylab as pl

    from ot.dr import wda, fda








Generate data
#############################################################################



.. code-block:: python


    #%% parameters

    n = 1000  # nb samples in source and target datasets
    nz = 0.2

    # generate circle dataset
    t = np.random.rand(n) * 2 * np.pi
    ys = np.floor((np.arange(n) * 1.0 / n * 3)) + 1
    xs = np.concatenate(
        (np.cos(t).reshape((-1, 1)), np.sin(t).reshape((-1, 1))), 1)
    xs = xs * ys.reshape(-1, 1) + nz * np.random.randn(n, 2)

    t = np.random.rand(n) * 2 * np.pi
    yt = np.floor((np.arange(n) * 1.0 / n * 3)) + 1
    xt = np.concatenate(
        (np.cos(t).reshape((-1, 1)), np.sin(t).reshape((-1, 1))), 1)
    xt = xt * yt.reshape(-1, 1) + nz * np.random.randn(n, 2)

    nbnoise = 8

    xs = np.hstack((xs, np.random.randn(n, nbnoise)))
    xt = np.hstack((xt, np.random.randn(n, nbnoise)))







Plot data
#############################################################################



.. code-block:: python


    #%% plot samples
    pl.figure(1, figsize=(6.4, 3.5))

    pl.subplot(1, 2, 1)
    pl.scatter(xt[:, 0], xt[:, 1], c=ys, marker='+', label='Source samples')
    pl.legend(loc=0)
    pl.title('Discriminant dimensions')

    pl.subplot(1, 2, 2)
    pl.scatter(xt[:, 2], xt[:, 3], c=ys, marker='+', label='Source samples')
    pl.legend(loc=0)
    pl.title('Other dimensions')
    pl.tight_layout()




.. image:: /auto_examples/images/sphx_glr_plot_WDA_001.png
    :align: center




Compute Fisher Discriminant Analysis
#############################################################################



.. code-block:: python


    #%% Compute FDA
    p = 2

    Pfda, projfda = fda(xs, ys, p)







Compute Wasserstein Discriminant Analysis
#############################################################################



.. code-block:: python


    #%% Compute WDA
    p = 2
    reg = 1e0
    k = 10
    maxiter = 100

    Pwda, projwda = wda(xs, ys, p, reg, k, maxiter=maxiter)






.. rst-class:: sphx-glr-script-out

 Out::

    Compiling cost function...
    Computing gradient of cost function...
     iter              cost val         grad. norm
        1   +7.7038877420882157e-01 6.30647522e-01
        2   +3.3969600919721271e-01 2.83791849e-01
        3   +3.0014000762425608e-01 2.56139137e-01
        4   +2.3397191702411621e-01 6.41134216e-02
        5   +2.3107227220070231e-01 2.24837190e-02
        6   +2.3072327156158298e-01 1.71334761e-03
        7   +2.3072143589220098e-01 6.30059431e-04
        8   +2.3072133109125159e-01 4.88673790e-04
        9   +2.3072119579341774e-01 1.74129117e-04
       10   +2.3072118662364521e-01 1.27046386e-04
       11   +2.3072118228917746e-01 9.70877451e-05
       12   +2.3072117734120351e-01 4.17292699e-05
       13   +2.3072117623493599e-01 4.46062100e-06
       14   +2.3072117622383431e-01 1.59801454e-06
       15   +2.3072117622300498e-01 1.12117391e-06
       16   +2.3072117622220378e-01 4.14581994e-08
    Terminated - min grad norm reached after 16 iterations, 7.77 seconds.


Plot 2D projections
#############################################################################



.. code-block:: python


    #%% plot samples

    xsp = projfda(xs)
    xtp = projfda(xt)

    xspw = projwda(xs)
    xtpw = projwda(xt)

    pl.figure(2)

    pl.subplot(2, 2, 1)
    pl.scatter(xsp[:, 0], xsp[:, 1], c=ys, marker='+', label='Projected samples')
    pl.legend(loc=0)
    pl.title('Projected training samples FDA')

    pl.subplot(2, 2, 2)
    pl.scatter(xtp[:, 0], xtp[:, 1], c=ys, marker='+', label='Projected samples')
    pl.legend(loc=0)
    pl.title('Projected test samples FDA')

    pl.subplot(2, 2, 3)
    pl.scatter(xspw[:, 0], xspw[:, 1], c=ys, marker='+', label='Projected samples')
    pl.legend(loc=0)
    pl.title('Projected training samples WDA')

    pl.subplot(2, 2, 4)
    pl.scatter(xtpw[:, 0], xtpw[:, 1], c=ys, marker='+', label='Projected samples')
    pl.legend(loc=0)
    pl.title('Projected test samples WDA')
    pl.tight_layout()

    pl.show()



.. image:: /auto_examples/images/sphx_glr_plot_WDA_003.png
    :align: center




**Total running time of the script:** ( 0 minutes  8.568 seconds)



.. container:: sphx-glr-footer


  .. container:: sphx-glr-download

     :download:`Download Python source code: plot_WDA.py <plot_WDA.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: plot_WDA.ipynb <plot_WDA.ipynb>`

.. rst-class:: sphx-glr-signature

    `Generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_
