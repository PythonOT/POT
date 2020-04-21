.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download_auto_examples_plot_OT_L1_vs_L2.py>`     to download the full example code
    .. rst-class:: sphx-glr-example-title

    .. _sphx_glr_auto_examples_plot_OT_L1_vs_L2.py:


==========================================
2D Optimal transport for different metrics
==========================================

2D OT on empirical distributio  with different gound metric.

Stole the figure idea from Fig. 1 and 2 in
https://arxiv.org/pdf/1706.07650.pdf




.. code-block:: default


    # Author: Remi Flamary <remi.flamary@unice.fr>
    #
    # License: MIT License

    import numpy as np
    import matplotlib.pylab as pl
    import ot
    import ot.plot








Dataset 1 : uniform sampling
----------------------------


.. code-block:: default


    n = 20  # nb samples
    xs = np.zeros((n, 2))
    xs[:, 0] = np.arange(n) + 1
    xs[:, 1] = (np.arange(n) + 1) * -0.001  # to make it strictly convex...

    xt = np.zeros((n, 2))
    xt[:, 1] = np.arange(n) + 1

    a, b = ot.unif(n), ot.unif(n)  # uniform distribution on samples

    # loss matrix
    M1 = ot.dist(xs, xt, metric='euclidean')
    M1 /= M1.max()

    # loss matrix
    M2 = ot.dist(xs, xt, metric='sqeuclidean')
    M2 /= M2.max()

    # loss matrix
    Mp = np.sqrt(ot.dist(xs, xt, metric='euclidean'))
    Mp /= Mp.max()

    # Data
    pl.figure(1, figsize=(7, 3))
    pl.clf()
    pl.plot(xs[:, 0], xs[:, 1], '+b', label='Source samples')
    pl.plot(xt[:, 0], xt[:, 1], 'xr', label='Target samples')
    pl.axis('equal')
    pl.title('Source and target distributions')


    # Cost matrices
    pl.figure(2, figsize=(7, 3))

    pl.subplot(1, 3, 1)
    pl.imshow(M1, interpolation='nearest')
    pl.title('Euclidean cost')

    pl.subplot(1, 3, 2)
    pl.imshow(M2, interpolation='nearest')
    pl.title('Squared Euclidean cost')

    pl.subplot(1, 3, 3)
    pl.imshow(Mp, interpolation='nearest')
    pl.title('Sqrt Euclidean cost')
    pl.tight_layout()




.. rst-class:: sphx-glr-horizontal


    *

      .. image:: /auto_examples/images/sphx_glr_plot_OT_L1_vs_L2_001.png
            :class: sphx-glr-multi-img

    *

      .. image:: /auto_examples/images/sphx_glr_plot_OT_L1_vs_L2_002.png
            :class: sphx-glr-multi-img





Dataset 1 : Plot OT Matrices
----------------------------


.. code-block:: default

    G1 = ot.emd(a, b, M1)
    G2 = ot.emd(a, b, M2)
    Gp = ot.emd(a, b, Mp)

    # OT matrices
    pl.figure(3, figsize=(7, 3))

    pl.subplot(1, 3, 1)
    ot.plot.plot2D_samples_mat(xs, xt, G1, c=[.5, .5, 1])
    pl.plot(xs[:, 0], xs[:, 1], '+b', label='Source samples')
    pl.plot(xt[:, 0], xt[:, 1], 'xr', label='Target samples')
    pl.axis('equal')
    # pl.legend(loc=0)
    pl.title('OT Euclidean')

    pl.subplot(1, 3, 2)
    ot.plot.plot2D_samples_mat(xs, xt, G2, c=[.5, .5, 1])
    pl.plot(xs[:, 0], xs[:, 1], '+b', label='Source samples')
    pl.plot(xt[:, 0], xt[:, 1], 'xr', label='Target samples')
    pl.axis('equal')
    # pl.legend(loc=0)
    pl.title('OT squared Euclidean')

    pl.subplot(1, 3, 3)
    ot.plot.plot2D_samples_mat(xs, xt, Gp, c=[.5, .5, 1])
    pl.plot(xs[:, 0], xs[:, 1], '+b', label='Source samples')
    pl.plot(xt[:, 0], xt[:, 1], 'xr', label='Target samples')
    pl.axis('equal')
    # pl.legend(loc=0)
    pl.title('OT sqrt Euclidean')
    pl.tight_layout()

    pl.show()





.. image:: /auto_examples/images/sphx_glr_plot_OT_L1_vs_L2_003.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    /home/rflamary/PYTHON/POT/examples/plot_OT_L1_vs_L2.py:113: UserWarning: Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure.
      pl.show()




Dataset 2 : Partial circle
--------------------------


.. code-block:: default


    n = 50  # nb samples
    xtot = np.zeros((n + 1, 2))
    xtot[:, 0] = np.cos(
        (np.arange(n + 1) + 1.0) * 0.9 / (n + 2) * 2 * np.pi)
    xtot[:, 1] = np.sin(
        (np.arange(n + 1) + 1.0) * 0.9 / (n + 2) * 2 * np.pi)

    xs = xtot[:n, :]
    xt = xtot[1:, :]

    a, b = ot.unif(n), ot.unif(n)  # uniform distribution on samples

    # loss matrix
    M1 = ot.dist(xs, xt, metric='euclidean')
    M1 /= M1.max()

    # loss matrix
    M2 = ot.dist(xs, xt, metric='sqeuclidean')
    M2 /= M2.max()

    # loss matrix
    Mp = np.sqrt(ot.dist(xs, xt, metric='euclidean'))
    Mp /= Mp.max()


    # Data
    pl.figure(4, figsize=(7, 3))
    pl.clf()
    pl.plot(xs[:, 0], xs[:, 1], '+b', label='Source samples')
    pl.plot(xt[:, 0], xt[:, 1], 'xr', label='Target samples')
    pl.axis('equal')
    pl.title('Source and traget distributions')


    # Cost matrices
    pl.figure(5, figsize=(7, 3))

    pl.subplot(1, 3, 1)
    pl.imshow(M1, interpolation='nearest')
    pl.title('Euclidean cost')

    pl.subplot(1, 3, 2)
    pl.imshow(M2, interpolation='nearest')
    pl.title('Squared Euclidean cost')

    pl.subplot(1, 3, 3)
    pl.imshow(Mp, interpolation='nearest')
    pl.title('Sqrt Euclidean cost')
    pl.tight_layout()




.. rst-class:: sphx-glr-horizontal


    *

      .. image:: /auto_examples/images/sphx_glr_plot_OT_L1_vs_L2_004.png
            :class: sphx-glr-multi-img

    *

      .. image:: /auto_examples/images/sphx_glr_plot_OT_L1_vs_L2_005.png
            :class: sphx-glr-multi-img





Dataset 2 : Plot  OT Matrices
-----------------------------


.. code-block:: default

    G1 = ot.emd(a, b, M1)
    G2 = ot.emd(a, b, M2)
    Gp = ot.emd(a, b, Mp)

    # OT matrices
    pl.figure(6, figsize=(7, 3))

    pl.subplot(1, 3, 1)
    ot.plot.plot2D_samples_mat(xs, xt, G1, c=[.5, .5, 1])
    pl.plot(xs[:, 0], xs[:, 1], '+b', label='Source samples')
    pl.plot(xt[:, 0], xt[:, 1], 'xr', label='Target samples')
    pl.axis('equal')
    # pl.legend(loc=0)
    pl.title('OT Euclidean')

    pl.subplot(1, 3, 2)
    ot.plot.plot2D_samples_mat(xs, xt, G2, c=[.5, .5, 1])
    pl.plot(xs[:, 0], xs[:, 1], '+b', label='Source samples')
    pl.plot(xt[:, 0], xt[:, 1], 'xr', label='Target samples')
    pl.axis('equal')
    # pl.legend(loc=0)
    pl.title('OT squared Euclidean')

    pl.subplot(1, 3, 3)
    ot.plot.plot2D_samples_mat(xs, xt, Gp, c=[.5, .5, 1])
    pl.plot(xs[:, 0], xs[:, 1], '+b', label='Source samples')
    pl.plot(xt[:, 0], xt[:, 1], 'xr', label='Target samples')
    pl.axis('equal')
    # pl.legend(loc=0)
    pl.title('OT sqrt Euclidean')
    pl.tight_layout()

    pl.show()



.. image:: /auto_examples/images/sphx_glr_plot_OT_L1_vs_L2_006.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    /home/rflamary/PYTHON/POT/examples/plot_OT_L1_vs_L2.py:208: UserWarning: Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure.
      pl.show()





.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  1.002 seconds)


.. _sphx_glr_download_auto_examples_plot_OT_L1_vs_L2.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: plot_OT_L1_vs_L2.py <plot_OT_L1_vs_L2.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: plot_OT_L1_vs_L2.ipynb <plot_OT_L1_vs_L2.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
