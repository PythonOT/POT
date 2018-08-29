

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



.. code-block:: python

    print("------------SEMI-DUAL PROBLEM------------")




.. rst-class:: sphx-glr-script-out

 Out::

    ------------SEMI-DUAL PROBLEM------------


DISCRETE CASE
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


SEMICONTINOUS CASE
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

    [3.9018759  7.63059124 3.93260224 2.67274989 1.43888443 3.26904884
     2.78748299] [-2.48511647 -2.43621119 -0.93585194  5.8571796 ]
    [[2.56614773e-02 9.96758169e-02 1.75151781e-02 4.67049862e-06]
     [1.21201047e-01 1.24433535e-02 1.28173754e-03 7.93100436e-03]
     [3.58778167e-03 7.64232233e-02 6.28459924e-02 1.45441936e-07]
     [2.63551754e-02 3.35577920e-02 8.25011211e-02 4.43054320e-04]
     [9.24518246e-03 7.03074064e-04 1.00325744e-02 1.22876312e-01]
     [2.03656325e-02 8.45420425e-04 1.73604569e-03 1.19910044e-01]
     [4.17781688e-02 2.66463708e-02 7.18353075e-02 2.59729583e-03]]


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



.. code-block:: python

    print("------------DUAL PROBLEM------------")




.. rst-class:: sphx-glr-script-out

 Out::

    ------------DUAL PROBLEM------------


SEMICONTINOUS CASE
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

    [ 1.29325617  5.0435082   1.30996326  0.05538236 -1.08113283  0.73711558
      0.18086364] [0.08840343 0.17710082 1.68604226 8.37377551]
    [[2.47763879e-02 1.00144623e-01 1.77492330e-02 4.25988443e-06]
     [1.19568278e-01 1.27740478e-02 1.32714202e-03 7.39121816e-03]
     [3.41581121e-03 7.57137404e-02 6.27992039e-02 1.30808430e-07]
     [2.52245323e-02 3.34219732e-02 8.28754229e-02 4.00582912e-04]
     [9.75329554e-03 7.71824343e-04 1.11085400e-02 1.22456628e-01]
     [2.12304276e-02 9.17096580e-04 1.89946234e-03 1.18084973e-01]
     [4.04179693e-02 2.68253041e-02 7.29410047e-02 2.37369404e-03]]


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




**Total running time of the script:** ( 0 minutes  22.857 seconds)



.. only :: html

 .. container:: sphx-glr-footer


  .. container:: sphx-glr-download

     :download:`Download Python source code: plot_stochastic.py <plot_stochastic.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: plot_stochastic.ipynb <plot_stochastic.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_
