

.. _sphx_glr_auto_examples_plot_stochastic.py:


==========================
Stochastic examples
==========================

This example is designed to show how to use the stochatic optimization
algorithms for descrete and semicontinous measures from the POT library.




.. code-block:: python


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



.. code-block:: python


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



.. code-block:: python


    method = "SAG"
    sag_pi = ot.stochastic.solve_semi_dual_entropic(a, b, M, reg, method,
                                                    numItermax)
    print(sag_pi)





.. rst-class:: sphx-glr-script-out

 Out::

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



.. code-block:: python


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



.. code-block:: python


    method = "ASGD"
    asgd_pi, log_asgd = ot.stochastic.solve_semi_dual_entropic(a, b, M, reg, method,
                                                               numItermax, log=log)
    print(log_asgd['alpha'], log_asgd['beta'])
    print(asgd_pi)





.. rst-class:: sphx-glr-script-out

 Out::

    [3.98220325 7.76235856 3.97645524 2.72051681 1.23219313 3.07696856
     2.84476972] [-2.65544161 -2.50838395 -0.9397765   6.10360206]
    [[2.34528761e-02 1.00491956e-01 1.89058354e-02 6.47543413e-06]
     [1.16616747e-01 1.32074516e-02 1.45653361e-03 1.15764107e-02]
     [3.16154850e-03 7.42892944e-02 6.54061055e-02 1.94426150e-07]
     [2.33152216e-02 3.27486992e-02 8.61986263e-02 5.94595747e-04]
     [6.34131496e-03 5.31975896e-04 8.12724003e-03 1.27856612e-01]
     [1.41744829e-02 6.49096245e-04 1.42704389e-03 1.26606520e-01]
     [3.73127657e-02 2.62526499e-02 7.57727161e-02 3.51901117e-03]]


Compare the results with the Sinkhorn algorithm
---------------------------------------------

Call the Sinkhorn algorithm from POT



.. code-block:: python


    sinkhorn_pi = ot.sinkhorn(a, b, M, reg)
    print(sinkhorn_pi)






.. rst-class:: sphx-glr-script-out

 Out::

    [[2.55535622e-02 9.96413843e-02 1.76578860e-02 4.31043335e-06]
     [1.21640742e-01 1.25369034e-02 1.30234529e-03 7.37715259e-03]
     [3.56096458e-03 7.61460101e-02 6.31500344e-02 1.33788624e-07]
     [2.61499607e-02 3.34255577e-02 8.28741973e-02 4.07427179e-04]
     [9.85698720e-03 7.52505948e-04 1.08291770e-02 1.21418473e-01]
     [2.16947591e-02 9.04086158e-04 1.87228707e-03 1.18386011e-01]
     [4.15442692e-02 2.65998963e-02 7.23192701e-02 2.39370724e-03]]


PLOT TRANSPORTATION MATRIX
#############################################################################


Plot SAG results
----------------



.. code-block:: python


    pl.figure(4, figsize=(5, 5))
    ot.plot.plot1D_mat(a, b, sag_pi, 'semi-dual : OT matrix SAG')
    pl.show()





.. image:: /auto_examples/images/sphx_glr_plot_stochastic_004.png
    :align: center




Plot ASGD results
-----------------



.. code-block:: python


    pl.figure(4, figsize=(5, 5))
    ot.plot.plot1D_mat(a, b, asgd_pi, 'semi-dual : OT matrix ASGD')
    pl.show()





.. image:: /auto_examples/images/sphx_glr_plot_stochastic_005.png
    :align: center




Plot Sinkhorn results
---------------------



.. code-block:: python


    pl.figure(4, figsize=(5, 5))
    ot.plot.plot1D_mat(a, b, sinkhorn_pi, 'OT matrix Sinkhorn')
    pl.show()





.. image:: /auto_examples/images/sphx_glr_plot_stochastic_006.png
    :align: center




COMPUTE TRANSPORTATION MATRIX FOR DUAL PROBLEM
############################################################################
############################################################################
 SEMICONTINOUS CASE:

 Sample one general measure a, one discrete measures b for the semicontinous
 case
 ---------------------------------------------

 Define one general measure a, one discrete measures b, the points where
 are defined the source and the target measures and finally the cost matrix c.



.. code-block:: python


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



.. code-block:: python


    sgd_dual_pi, log_sgd = ot.stochastic.solve_dual_entropic(a, b, M, reg,
                                                             batch_size, numItermax,
                                                             lr, log=log)
    print(log_sgd['alpha'], log_sgd['beta'])
    print(sgd_dual_pi)





.. rst-class:: sphx-glr-script-out

 Out::

    [0.92449986 2.75486107 1.07923806 0.02741145 0.61355413 1.81961594
     0.12072562] [0.33831611 0.46806842 1.5640451  4.96947652]
    [[2.20001105e-02 9.26497883e-02 1.08654588e-02 9.78995555e-08]
     [1.55669974e-02 1.73279561e-03 1.19120878e-04 2.49058251e-05]
     [3.48198483e-03 8.04151063e-02 4.41335396e-02 3.45115752e-09]
     [3.14927954e-02 4.34760520e-02 7.13338154e-02 1.29442395e-05]
     [6.81836550e-02 5.62182457e-03 5.35386584e-02 2.21568095e-02]
     [8.04671052e-02 3.62163462e-03 4.96331605e-03 1.15837801e-02]
     [4.88644009e-02 3.37903481e-02 6.07955004e-02 7.42743505e-05]]


Compare the results with the Sinkhorn algorithm
---------------------------------------------

Call the Sinkhorn algorithm from POT



.. code-block:: python


    sinkhorn_pi = ot.sinkhorn(a, b, M, reg)
    print(sinkhorn_pi)





.. rst-class:: sphx-glr-script-out

 Out::

    [[2.55535622e-02 9.96413843e-02 1.76578860e-02 4.31043335e-06]
     [1.21640742e-01 1.25369034e-02 1.30234529e-03 7.37715259e-03]
     [3.56096458e-03 7.61460101e-02 6.31500344e-02 1.33788624e-07]
     [2.61499607e-02 3.34255577e-02 8.28741973e-02 4.07427179e-04]
     [9.85698720e-03 7.52505948e-04 1.08291770e-02 1.21418473e-01]
     [2.16947591e-02 9.04086158e-04 1.87228707e-03 1.18386011e-01]
     [4.15442692e-02 2.65998963e-02 7.23192701e-02 2.39370724e-03]]


Plot  SGD results
-----------------



.. code-block:: python


    pl.figure(4, figsize=(5, 5))
    ot.plot.plot1D_mat(a, b, sgd_dual_pi, 'dual : OT matrix SGD')
    pl.show()





.. image:: /auto_examples/images/sphx_glr_plot_stochastic_007.png
    :align: center




Plot Sinkhorn results
---------------------



.. code-block:: python


    pl.figure(4, figsize=(5, 5))
    ot.plot.plot1D_mat(a, b, sinkhorn_pi, 'OT matrix Sinkhorn')
    pl.show()



.. image:: /auto_examples/images/sphx_glr_plot_stochastic_008.png
    :align: center




**Total running time of the script:** ( 0 minutes  20.889 seconds)



.. only :: html

 .. container:: sphx-glr-footer


  .. container:: sphx-glr-download

     :download:`Download Python source code: plot_stochastic.py <plot_stochastic.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: plot_stochastic.ipynb <plot_stochastic.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_
