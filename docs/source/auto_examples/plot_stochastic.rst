.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download_auto_examples_plot_stochastic.py>`     to download the full example code
    .. rst-class:: sphx-glr-example-title

    .. _sphx_glr_auto_examples_plot_stochastic.py:


==========================
Stochastic examples
==========================

This example is designed to show how to use the stochatic optimization
algorithms for descrete and semicontinous measures from the POT library.



.. code-block:: default


    # Author: Kilian Fatras <kilian.fatras@gmail.com>
    #
    # License: MIT License

    import matplotlib.pylab as pl
    import numpy as np
    import ot
    import ot.plot









COMPUTE TRANSPORTATION MATRIX FOR SEMI-DUAL PROBLEM
############################################################################
############################################################################
 DISCRETE CASE:

 Sample two discrete measures for the discrete case
 ---------------------------------------------

 Define 2 discrete measures a and b, the points where are defined the source
 and the target measures and finally the cost matrix c.


.. code-block:: default


    n_source = 7
    n_target = 4
    reg = 1
    numItermax = 1000

    a = ot.utils.unif(n_source)
    b = ot.utils.unif(n_target)

    rng = np.random.RandomState(0)
    X_source = rng.randn(n_source, 2)
    Y_target = rng.randn(n_target, 2)
    M = ot.dist(X_source, Y_target)








Call the "SAG" method to find the transportation matrix in the discrete case
---------------------------------------------

Define the method "SAG", call ot.solve_semi_dual_entropic and plot the
results.


.. code-block:: default


    method = "SAG"
    sag_pi = ot.stochastic.solve_semi_dual_entropic(a, b, M, reg, method,
                                                    numItermax)
    print(sag_pi)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [[2.55553509e-02 9.96395660e-02 1.76579142e-02 4.31178196e-06]
     [1.21640234e-01 1.25357448e-02 1.30225078e-03 7.37891338e-03]
     [3.56123975e-03 7.61451746e-02 6.31505947e-02 1.33831456e-07]
     [2.61515202e-02 3.34246014e-02 8.28734709e-02 4.07550428e-04]
     [9.85500870e-03 7.52288517e-04 1.08262628e-02 1.21423583e-01]
     [2.16904253e-02 9.03825797e-04 1.87178503e-03 1.18391107e-01]
     [4.15462212e-02 2.65987989e-02 7.23177216e-02 2.39440107e-03]]




SEMICONTINOUS CASE:

Sample one general measure a, one discrete measures b for the semicontinous
case
---------------------------------------------

Define one general measure a, one discrete measures b, the points where
are defined the source and the target measures and finally the cost matrix c.


.. code-block:: default


    n_source = 7
    n_target = 4
    reg = 1
    numItermax = 1000
    log = True

    a = ot.utils.unif(n_source)
    b = ot.utils.unif(n_target)

    rng = np.random.RandomState(0)
    X_source = rng.randn(n_source, 2)
    Y_target = rng.randn(n_target, 2)
    M = ot.dist(X_source, Y_target)








Call the "ASGD" method to find the transportation matrix in the semicontinous
case
---------------------------------------------

Define the method "ASGD", call ot.solve_semi_dual_entropic and plot the
results.


.. code-block:: default


    method = "ASGD"
    asgd_pi, log_asgd = ot.stochastic.solve_semi_dual_entropic(a, b, M, reg, method,
                                                               numItermax, log=log)
    print(log_asgd['alpha'], log_asgd['beta'])
    print(asgd_pi)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [3.89943264 7.64823414 3.9284189  2.67501041 1.42825446 3.26039819
     2.79237712] [-2.50786905 -2.42684838 -0.93647774  5.87119517]
    [[2.50229922e-02 1.00367920e-01 1.74615056e-02 4.72486104e-06]
     [1.20583329e-01 1.27839737e-02 1.30373565e-03 8.18610462e-03]
     [3.49243139e-03 7.68200813e-02 6.25444833e-02 1.46879008e-07]
     [2.58205995e-02 3.39501207e-02 8.26360982e-02 4.50324517e-04]
     [8.94164918e-03 7.02183713e-04 9.92028326e-03 1.23293027e-01]
     [1.97360234e-02 8.46022708e-04 1.72001583e-03 1.20555081e-01]
     [4.10386980e-02 2.70289873e-02 7.21425804e-02 2.64687723e-03]]




Compare the results with the Sinkhorn algorithm
---------------------------------------------

Call the Sinkhorn algorithm from POT


.. code-block:: default


    sinkhorn_pi = ot.sinkhorn(a, b, M, reg)
    print(sinkhorn_pi)






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [[2.55553508e-02 9.96395661e-02 1.76579142e-02 4.31178193e-06]
     [1.21640234e-01 1.25357448e-02 1.30225079e-03 7.37891333e-03]
     [3.56123974e-03 7.61451746e-02 6.31505947e-02 1.33831455e-07]
     [2.61515201e-02 3.34246014e-02 8.28734709e-02 4.07550425e-04]
     [9.85500876e-03 7.52288523e-04 1.08262629e-02 1.21423583e-01]
     [2.16904255e-02 9.03825804e-04 1.87178504e-03 1.18391107e-01]
     [4.15462212e-02 2.65987989e-02 7.23177217e-02 2.39440105e-03]]




PLOT TRANSPORTATION MATRIX
#############################################################################

Plot SAG results
----------------


.. code-block:: default


    pl.figure(4, figsize=(5, 5))
    ot.plot.plot1D_mat(a, b, sag_pi, 'semi-dual : OT matrix SAG')
    pl.show()





.. image:: /auto_examples/images/sphx_glr_plot_stochastic_001.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    /home/rflamary/PYTHON/POT/examples/plot_stochastic.py:119: UserWarning: Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure.
      pl.show()




Plot ASGD results
-----------------


.. code-block:: default


    pl.figure(4, figsize=(5, 5))
    ot.plot.plot1D_mat(a, b, asgd_pi, 'semi-dual : OT matrix ASGD')
    pl.show()





.. image:: /auto_examples/images/sphx_glr_plot_stochastic_002.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    /home/rflamary/PYTHON/POT/examples/plot_stochastic.py:128: UserWarning: Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure.
      pl.show()




Plot Sinkhorn results
---------------------


.. code-block:: default


    pl.figure(4, figsize=(5, 5))
    ot.plot.plot1D_mat(a, b, sinkhorn_pi, 'OT matrix Sinkhorn')
    pl.show()





.. image:: /auto_examples/images/sphx_glr_plot_stochastic_003.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    /home/rflamary/PYTHON/POT/examples/plot_stochastic.py:137: UserWarning: Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure.
      pl.show()




COMPUTE TRANSPORTATION MATRIX FOR DUAL PROBLEM
############################################################################
############################################################################
 SEMICONTINOUS CASE:

 Sample one general measure a, one discrete measures b for the semicontinous
 case
 ---------------------------------------------

 Define one general measure a, one discrete measures b, the points where
 are defined the source and the target measures and finally the cost matrix c.


.. code-block:: default


    n_source = 7
    n_target = 4
    reg = 1
    numItermax = 100000
    lr = 0.1
    batch_size = 3
    log = True

    a = ot.utils.unif(n_source)
    b = ot.utils.unif(n_target)

    rng = np.random.RandomState(0)
    X_source = rng.randn(n_source, 2)
    Y_target = rng.randn(n_target, 2)
    M = ot.dist(X_source, Y_target)








Call the "SGD" dual method to find the transportation matrix in the
semicontinous case
---------------------------------------------

Call ot.solve_dual_entropic and plot the results.


.. code-block:: default


    sgd_dual_pi, log_sgd = ot.stochastic.solve_dual_entropic(a, b, M, reg,
                                                             batch_size, numItermax,
                                                             lr, log=log)
    print(log_sgd['alpha'], log_sgd['beta'])
    print(sgd_dual_pi)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [0.91421006 2.78075506 1.06828701 0.01979397 0.60914807 1.81887037
     0.1152939 ] [0.33964624 0.47604281 1.57223631 4.93843308]
    [[2.18038772e-02 9.24355133e-02 1.08426805e-02 9.39355366e-08]
     [1.59966167e-02 1.79248770e-03 1.23251128e-04 2.47779034e-05]
     [3.44864558e-03 8.01760930e-02 4.40119061e-02 3.30922887e-09]
     [3.12954103e-02 4.34915712e-02 7.13747533e-02 1.24533534e-05]
     [6.79742497e-02 5.64192090e-03 5.37416946e-02 2.13851205e-02]
     [8.05141568e-02 3.64790957e-03 5.00040902e-03 1.12213345e-02]
     [4.86643900e-02 3.38763749e-02 6.09634969e-02 7.16139950e-05]]




Compare the results with the Sinkhorn algorithm
---------------------------------------------

Call the Sinkhorn algorithm from POT


.. code-block:: default


    sinkhorn_pi = ot.sinkhorn(a, b, M, reg)
    print(sinkhorn_pi)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [[2.55553508e-02 9.96395661e-02 1.76579142e-02 4.31178193e-06]
     [1.21640234e-01 1.25357448e-02 1.30225079e-03 7.37891333e-03]
     [3.56123974e-03 7.61451746e-02 6.31505947e-02 1.33831455e-07]
     [2.61515201e-02 3.34246014e-02 8.28734709e-02 4.07550425e-04]
     [9.85500876e-03 7.52288523e-04 1.08262629e-02 1.21423583e-01]
     [2.16904255e-02 9.03825804e-04 1.87178504e-03 1.18391107e-01]
     [4.15462212e-02 2.65987989e-02 7.23177217e-02 2.39440105e-03]]




Plot  SGD results
-----------------


.. code-block:: default


    pl.figure(4, figsize=(5, 5))
    ot.plot.plot1D_mat(a, b, sgd_dual_pi, 'dual : OT matrix SGD')
    pl.show()





.. image:: /auto_examples/images/sphx_glr_plot_stochastic_004.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    /home/rflamary/PYTHON/POT/examples/plot_stochastic.py:199: UserWarning: Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure.
      pl.show()




Plot Sinkhorn results
---------------------


.. code-block:: default


    pl.figure(4, figsize=(5, 5))
    ot.plot.plot1D_mat(a, b, sinkhorn_pi, 'OT matrix Sinkhorn')
    pl.show()



.. image:: /auto_examples/images/sphx_glr_plot_stochastic_005.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    /home/rflamary/PYTHON/POT/examples/plot_stochastic.py:208: UserWarning: Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure.
      pl.show()





.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  8.885 seconds)


.. _sphx_glr_download_auto_examples_plot_stochastic.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: plot_stochastic.py <plot_stochastic.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: plot_stochastic.ipynb <plot_stochastic.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
