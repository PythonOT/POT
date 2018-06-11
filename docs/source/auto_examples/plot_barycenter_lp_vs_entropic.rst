

.. _sphx_glr_auto_examples_plot_barycenter_lp_vs_entropic.py:


=================================================================================
1D Wasserstein barycenter comparison between exact LP and entropic regularization
=================================================================================

This example illustrates the computation of regularized Wasserstein Barycenter
as proposed in [3] and exact LP barycenters using standard LP solver.

It reproduces approximately Figure 3.1 and 3.2 from the following paper:
Cuturi, M., & Peyré, G. (2016). A smoothed dual approach for variational
Wasserstein problems. SIAM Journal on Imaging Sciences, 9(1), 320-343.

[3] Benamou, J. D., Carlier, G., Cuturi, M., Nenna, L., & Peyré, G. (2015).
Iterative Bregman projections for regularized transportation problems
SIAM Journal on Scientific Computing, 37(2), A1111-A1138.




.. code-block:: python


    # Author: Remi Flamary <remi.flamary@unice.fr>
    #
    # License: MIT License

    import numpy as np
    import matplotlib.pylab as pl
    import ot
    # necessary for 3d plot even if not used
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    from matplotlib.collections import PolyCollection  # noqa

    #import ot.lp.cvx as cvx







Gaussian Data
-------------



.. code-block:: python


    #%% parameters

    problems = []

    n = 100  # nb bins

    # bin positions
    x = np.arange(n, dtype=np.float64)

    # Gaussian distributions
    # Gaussian distributions
    a1 = ot.datasets.make_1D_gauss(n, m=20, s=5)  # m= mean, s= std
    a2 = ot.datasets.make_1D_gauss(n, m=60, s=8)

    # creating matrix A containing all distributions
    A = np.vstack((a1, a2)).T
    n_distributions = A.shape[1]

    # loss matrix + normalization
    M = ot.utils.dist0(n)
    M /= M.max()


    #%% plot the distributions

    pl.figure(1, figsize=(6.4, 3))
    for i in range(n_distributions):
        pl.plot(x, A[:, i])
    pl.title('Distributions')
    pl.tight_layout()

    #%% barycenter computation

    alpha = 0.5  # 0<=alpha<=1
    weights = np.array([1 - alpha, alpha])

    # l2bary
    bary_l2 = A.dot(weights)

    # wasserstein
    reg = 1e-3
    ot.tic()
    bary_wass = ot.bregman.barycenter(A, M, reg, weights)
    ot.toc()


    ot.tic()
    bary_wass2 = ot.lp.barycenter(A, M, weights, solver='interior-point', verbose=True)
    ot.toc()

    pl.figure(2)
    pl.clf()
    pl.subplot(2, 1, 1)
    for i in range(n_distributions):
        pl.plot(x, A[:, i])
    pl.title('Distributions')

    pl.subplot(2, 1, 2)
    pl.plot(x, bary_l2, 'r', label='l2')
    pl.plot(x, bary_wass, 'g', label='Reg Wasserstein')
    pl.plot(x, bary_wass2, 'b', label='LP Wasserstein')
    pl.legend()
    pl.title('Barycenters')
    pl.tight_layout()

    problems.append([A, [bary_l2, bary_wass, bary_wass2]])




.. rst-class:: sphx-glr-horizontal


    *

      .. image:: /auto_examples/images/sphx_glr_plot_barycenter_lp_vs_entropic_001.png
            :scale: 47

    *

      .. image:: /auto_examples/images/sphx_glr_plot_barycenter_lp_vs_entropic_002.png
            :scale: 47


.. rst-class:: sphx-glr-script-out

 Out::

    Elapsed time : 0.010712385177612305 s
    Primal Feasibility  Dual Feasibility    Duality Gap         Step             Path Parameter      Objective          
    1.0                 1.0                 1.0                 -                1.0                 1700.336700337      
    0.006776453137632   0.006776453137633   0.006776453137633   0.9932238647293  0.006776453137633   125.6700527543      
    0.004018712867874   0.004018712867874   0.004018712867874   0.4301142633     0.004018712867874   12.26594150093      
    0.001172775061627   0.001172775061627   0.001172775061627   0.7599932455029  0.001172775061627   0.3378536968897     
    0.0004375137005385  0.0004375137005385  0.0004375137005385  0.6422331807989  0.0004375137005385  0.1468420566358     
    0.000232669046734   0.0002326690467341  0.000232669046734   0.5016999460893  0.000232669046734   0.09381703231432    
    7.430121674303e-05  7.430121674303e-05  7.430121674303e-05  0.7035962305812  7.430121674303e-05  0.0577787025717     
    5.321227838876e-05  5.321227838875e-05  5.321227838876e-05  0.308784186441   5.321227838876e-05  0.05266249477203    
    1.990900379199e-05  1.990900379196e-05  1.990900379199e-05  0.6520472013244  1.990900379199e-05  0.04526054405519    
    6.305442046799e-06  6.30544204682e-06   6.3054420468e-06    0.7073953304075  6.305442046798e-06  0.04237597591383    
    2.290148391577e-06  2.290148391582e-06  2.290148391578e-06  0.6941812711492  2.29014839159e-06   0.041522849321      
    1.182864875387e-06  1.182864875406e-06  1.182864875427e-06  0.508455204675   1.182864875445e-06  0.04129461872827    
    3.626786381529e-07  3.626786382468e-07  3.626786382923e-07  0.7101651572101  3.62678638267e-07   0.04113032448923    
    1.539754244902e-07  1.539754249276e-07  1.539754249356e-07  0.6279322066282  1.539754253892e-07  0.04108867636379    
    5.193221323143e-08  5.193221463044e-08  5.193221462729e-08  0.6843453436759  5.193221708199e-08  0.04106859618414    
    1.888205054507e-08  1.888204779723e-08  1.88820477688e-08   0.6673444085651  1.888205650952e-08  0.041062141752      
    5.676855206925e-09  5.676854518888e-09  5.676854517651e-09  0.7281705804232  5.676885442702e-09  0.04105958648713    
    3.501157668218e-09  3.501150243546e-09  3.501150216347e-09  0.414020345194   3.501164437194e-09  0.04105916265261    
    1.110594251499e-09  1.110590786827e-09  1.11059083379e-09   0.6998954759911  1.110636623476e-09  0.04105870073485    
    5.770971626386e-10  5.772456113791e-10  5.772456200156e-10  0.4999769658132  5.77013379477e-10   0.04105859769135    
    1.535218204536e-10  1.536993317032e-10  1.536992771966e-10  0.7516471627141  1.536205005991e-10  0.04105851679958    
    6.724209350756e-11  6.739211232927e-11  6.739210470901e-11  0.5944802416166  6.735465384341e-11  0.04105850033766    
    1.743382199199e-11  1.736445896691e-11  1.736448490761e-11  0.7573407808104  1.734254328931e-11  0.04105849088824    
    Optimization terminated successfully.
    Elapsed time : 2.883899211883545 s


Dirac Data
----------



.. code-block:: python


    #%% parameters

    a1 = 1.0 * (x > 10) * (x < 50)
    a2 = 1.0 * (x > 60) * (x < 80)

    a1 /= a1.sum()
    a2 /= a2.sum()

    # creating matrix A containing all distributions
    A = np.vstack((a1, a2)).T
    n_distributions = A.shape[1]

    # loss matrix + normalization
    M = ot.utils.dist0(n)
    M /= M.max()


    #%% plot the distributions

    pl.figure(1, figsize=(6.4, 3))
    for i in range(n_distributions):
        pl.plot(x, A[:, i])
    pl.title('Distributions')
    pl.tight_layout()


    #%% barycenter computation

    alpha = 0.5  # 0<=alpha<=1
    weights = np.array([1 - alpha, alpha])

    # l2bary
    bary_l2 = A.dot(weights)

    # wasserstein
    reg = 1e-3
    ot.tic()
    bary_wass = ot.bregman.barycenter(A, M, reg, weights)
    ot.toc()


    ot.tic()
    bary_wass2 = ot.lp.barycenter(A, M, weights, solver='interior-point', verbose=True)
    ot.toc()


    problems.append([A, [bary_l2, bary_wass, bary_wass2]])

    pl.figure(2)
    pl.clf()
    pl.subplot(2, 1, 1)
    for i in range(n_distributions):
        pl.plot(x, A[:, i])
    pl.title('Distributions')

    pl.subplot(2, 1, 2)
    pl.plot(x, bary_l2, 'r', label='l2')
    pl.plot(x, bary_wass, 'g', label='Reg Wasserstein')
    pl.plot(x, bary_wass2, 'b', label='LP Wasserstein')
    pl.legend()
    pl.title('Barycenters')
    pl.tight_layout()

    #%% parameters

    a1 = np.zeros(n)
    a2 = np.zeros(n)

    a1[10] = .25
    a1[20] = .5
    a1[30] = .25
    a2[80] = 1


    a1 /= a1.sum()
    a2 /= a2.sum()

    # creating matrix A containing all distributions
    A = np.vstack((a1, a2)).T
    n_distributions = A.shape[1]

    # loss matrix + normalization
    M = ot.utils.dist0(n)
    M /= M.max()


    #%% plot the distributions

    pl.figure(1, figsize=(6.4, 3))
    for i in range(n_distributions):
        pl.plot(x, A[:, i])
    pl.title('Distributions')
    pl.tight_layout()


    #%% barycenter computation

    alpha = 0.5  # 0<=alpha<=1
    weights = np.array([1 - alpha, alpha])

    # l2bary
    bary_l2 = A.dot(weights)

    # wasserstein
    reg = 1e-3
    ot.tic()
    bary_wass = ot.bregman.barycenter(A, M, reg, weights)
    ot.toc()


    ot.tic()
    bary_wass2 = ot.lp.barycenter(A, M, weights, solver='interior-point', verbose=True)
    ot.toc()


    problems.append([A, [bary_l2, bary_wass, bary_wass2]])

    pl.figure(2)
    pl.clf()
    pl.subplot(2, 1, 1)
    for i in range(n_distributions):
        pl.plot(x, A[:, i])
    pl.title('Distributions')

    pl.subplot(2, 1, 2)
    pl.plot(x, bary_l2, 'r', label='l2')
    pl.plot(x, bary_wass, 'g', label='Reg Wasserstein')
    pl.plot(x, bary_wass2, 'b', label='LP Wasserstein')
    pl.legend()
    pl.title('Barycenters')
    pl.tight_layout()





.. rst-class:: sphx-glr-horizontal


    *

      .. image:: /auto_examples/images/sphx_glr_plot_barycenter_lp_vs_entropic_003.png
            :scale: 47

    *

      .. image:: /auto_examples/images/sphx_glr_plot_barycenter_lp_vs_entropic_004.png
            :scale: 47


.. rst-class:: sphx-glr-script-out

 Out::

    Elapsed time : 0.014938592910766602 s
    Primal Feasibility  Dual Feasibility    Duality Gap         Step             Path Parameter      Objective          
    1.0                 1.0                 1.0                 -                1.0                 1700.336700337      
    0.006776466288966   0.006776466288966   0.006776466288966   0.9932238515788  0.006776466288966   125.6649255808      
    0.004036918865495   0.004036918865495   0.004036918865495   0.4272973099316  0.004036918865495   12.3471617011       
    0.00121923268707    0.00121923268707    0.00121923268707    0.749698685599   0.00121923268707    0.3243835647408     
    0.0003837422984432  0.0003837422984432  0.0003837422984432  0.6926882608284  0.0003837422984432  0.1361719397493     
    0.0001070128410183  0.0001070128410183  0.0001070128410183  0.7643889137854  0.0001070128410183  0.07581952832518    
    0.0001001275033711  0.0001001275033711  0.0001001275033711  0.07058704837812 0.0001001275033712  0.0734739493635     
    4.550897507844e-05  4.550897507841e-05  4.550897507844e-05  0.5761172484828  4.550897507845e-05  0.05555077655047    
    8.557124125522e-06  8.5571241255e-06    8.557124125522e-06  0.8535925441152  8.557124125522e-06  0.04439814660221    
    3.611995628407e-06  3.61199562841e-06   3.611995628414e-06  0.6002277331554  3.611995628415e-06  0.04283007762152    
    7.590393750365e-07  7.590393750491e-07  7.590393750378e-07  0.8221486533416  7.590393750381e-07  0.04192322976248    
    8.299929287441e-08  8.299929286079e-08  8.299929287532e-08  0.9017467938799  8.29992928758e-08   0.04170825633295    
    3.117560203449e-10  3.117560130137e-10  3.11756019954e-10   0.997039969226   3.11756019952e-10   0.04168179329766    
    1.559749653711e-14  1.558073160926e-14  1.559756940692e-14  0.9999499686183  1.559750643989e-14  0.04168169240444    
    Optimization terminated successfully.
    Elapsed time : 2.642659902572632 s
    Elapsed time : 0.002908945083618164 s
    Primal Feasibility  Dual Feasibility    Duality Gap         Step             Path Parameter      Objective          
    1.0                 1.0                 1.0                 -                1.0                 1700.336700337      
    0.006774675520727   0.006774675520727   0.006774675520727   0.9932256422636  0.006774675520727   125.6956034743      
    0.002048208707562   0.002048208707562   0.002048208707562   0.7343095368143  0.002048208707562   5.213991622123      
    0.000269736547478   0.0002697365474781  0.0002697365474781  0.8839403501193  0.000269736547478   0.505938390389      
    6.832109993943e-05  6.832109993944e-05  6.832109993944e-05  0.7601171075965  6.832109993943e-05  0.2339657807272     
    2.437682932219e-05  2.43768293222e-05   2.437682932219e-05  0.6663448297475  2.437682932219e-05  0.1471256246325     
    1.13498321631e-05   1.134983216308e-05  1.13498321631e-05   0.5553643816404  1.13498321631e-05   0.1181584941171     
    3.342312725885e-06  3.342312725884e-06  3.342312725885e-06  0.7238133571615  3.342312725885e-06  0.1006387519747     
    7.078561231603e-07  7.078561231509e-07  7.078561231604e-07  0.8033142552512  7.078561231603e-07  0.09474734646269    
    1.966870956916e-07  1.966870954537e-07  1.966870954468e-07  0.752547917788   1.966870954633e-07  0.09354342735766    
    4.19989524849e-10   4.199895164852e-10  4.199895238758e-10  0.9984019849375  4.19989523951e-10   0.09310367785861    
    2.101015938666e-14  2.100625691113e-14  2.101023853438e-14  0.999949974425   2.101023691864e-14  0.09310274466458    
    Optimization terminated successfully.
    Elapsed time : 2.690450668334961 s


Final figure
------------




.. code-block:: python


    #%% plot

    nbm = len(problems)
    nbm2 = (nbm // 2)


    pl.figure(2, (20, 6))
    pl.clf()

    for i in range(nbm):

        A = problems[i][0]
        bary_l2 = problems[i][1][0]
        bary_wass = problems[i][1][1]
        bary_wass2 = problems[i][1][2]

        pl.subplot(2, nbm, 1 + i)
        for j in range(n_distributions):
            pl.plot(x, A[:, j])
        if i == nbm2:
            pl.title('Distributions')
        pl.xticks(())
        pl.yticks(())

        pl.subplot(2, nbm, 1 + i + nbm)

        pl.plot(x, bary_l2, 'r', label='L2 (Euclidean)')
        pl.plot(x, bary_wass, 'g', label='Reg Wasserstein')
        pl.plot(x, bary_wass2, 'b', label='LP Wasserstein')
        if i == nbm - 1:
            pl.legend()
        if i == nbm2:
            pl.title('Barycenters')

        pl.xticks(())
        pl.yticks(())



.. image:: /auto_examples/images/sphx_glr_plot_barycenter_lp_vs_entropic_006.png
    :align: center




**Total running time of the script:** ( 0 minutes  8.892 seconds)



.. only :: html

 .. container:: sphx-glr-footer


  .. container:: sphx-glr-download

     :download:`Download Python source code: plot_barycenter_lp_vs_entropic.py <plot_barycenter_lp_vs_entropic.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: plot_barycenter_lp_vs_entropic.ipynb <plot_barycenter_lp_vs_entropic.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_
