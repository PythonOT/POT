.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download_auto_examples_plot_gromov.py>`     to download the full example code
    .. rst-class:: sphx-glr-example-title

    .. _sphx_glr_auto_examples_plot_gromov.py:


==========================
Gromov-Wasserstein example
==========================

This example is designed to show how to use the Gromov-Wassertsein distance
computation in POT.


.. code-block:: default


    # Author: Erwan Vautier <erwan.vautier@gmail.com>
    #         Nicolas Courty <ncourty@irisa.fr>
    #
    # License: MIT License

    import scipy as sp
    import numpy as np
    import matplotlib.pylab as pl
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    import ot








Sample two Gaussian distributions (2D and 3D)
---------------------------------------------

The Gromov-Wasserstein distance allows to compute distances with samples that
do not belong to the same metric space. For demonstration purpose, we sample
two Gaussian distributions in 2- and 3-dimensional spaces.


.. code-block:: default



    n_samples = 30  # nb samples

    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    mu_t = np.array([4, 4, 4])
    cov_t = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])


    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s)
    P = sp.linalg.sqrtm(cov_t)
    xt = np.random.randn(n_samples, 3).dot(P) + mu_t








Plotting the distributions
--------------------------


.. code-block:: default



    fig = pl.figure()
    ax1 = fig.add_subplot(121)
    ax1.plot(xs[:, 0], xs[:, 1], '+b', label='Source samples')
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(xt[:, 0], xt[:, 1], xt[:, 2], color='r')
    pl.show()




.. image:: /auto_examples/images/sphx_glr_plot_gromov_001.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    /home/rflamary/PYTHON/POT/examples/plot_gromov.py:56: UserWarning: Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure.
      pl.show()




Compute distance kernels, normalize them and then display
---------------------------------------------------------


.. code-block:: default



    C1 = sp.spatial.distance.cdist(xs, xs)
    C2 = sp.spatial.distance.cdist(xt, xt)

    C1 /= C1.max()
    C2 /= C2.max()

    pl.figure()
    pl.subplot(121)
    pl.imshow(C1)
    pl.subplot(122)
    pl.imshow(C2)
    pl.show()




.. image:: /auto_examples/images/sphx_glr_plot_gromov_002.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    /home/rflamary/PYTHON/POT/examples/plot_gromov.py:75: UserWarning: Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure.
      pl.show()




Compute Gromov-Wasserstein plans and distance
---------------------------------------------


.. code-block:: default


    p = ot.unif(n_samples)
    q = ot.unif(n_samples)

    gw0, log0 = ot.gromov.gromov_wasserstein(
        C1, C2, p, q, 'square_loss', verbose=True, log=True)

    gw, log = ot.gromov.entropic_gromov_wasserstein(
        C1, C2, p, q, 'square_loss', epsilon=5e-4, log=True, verbose=True)


    print('Gromov-Wasserstein distances: ' + str(log0['gw_dist']))
    print('Entropic Gromov-Wasserstein distances: ' + str(log['gw_dist']))


    pl.figure(1, (10, 5))

    pl.subplot(1, 2, 1)
    pl.imshow(gw0, cmap='jet')
    pl.title('Gromov Wasserstein')

    pl.subplot(1, 2, 2)
    pl.imshow(gw, cmap='jet')
    pl.title('Entropic Gromov Wasserstein')

    pl.show()



.. image:: /auto_examples/images/sphx_glr_plot_gromov_003.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    It.  |Loss        |Relative loss|Absolute loss
    ------------------------------------------------
        0|8.019265e-02|0.000000e+00|0.000000e+00
        1|3.734805e-02|1.147171e+00|4.284460e-02
        2|2.923853e-02|2.773572e-01|8.109516e-03
        3|2.478957e-02|1.794691e-01|4.448961e-03
        4|2.444720e-02|1.400444e-02|3.423693e-04
        5|2.444720e-02|0.000000e+00|0.000000e+00
    It.  |Err         
    -------------------
        0|8.259147e-02|
       10|6.113732e-04|
       20|1.650651e-08|
       30|5.671192e-12|
    Gromov-Wasserstein distances: 0.024447198979060496
    Entropic Gromov-Wasserstein distances: 0.02488439679981518
    /home/rflamary/PYTHON/POT/examples/plot_gromov.py:106: UserWarning: Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure.
      pl.show()





.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  0.999 seconds)


.. _sphx_glr_download_auto_examples_plot_gromov.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: plot_gromov.py <plot_gromov.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: plot_gromov.ipynb <plot_gromov.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
