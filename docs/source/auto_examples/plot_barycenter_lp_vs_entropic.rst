.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download_auto_examples_plot_barycenter_lp_vs_entropic.py>`     to download the full example code
    .. rst-class:: sphx-glr-example-title

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



.. code-block:: default


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


.. code-block:: default


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










.. code-block:: default


    pl.figure(1, figsize=(6.4, 3))
    for i in range(n_distributions):
        pl.plot(x, A[:, i])
    pl.title('Distributions')
    pl.tight_layout()




.. image:: /auto_examples/images/sphx_glr_plot_barycenter_lp_vs_entropic_001.png
    :class: sphx-glr-single-img






.. code-block:: default


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




.. image:: /auto_examples/images/sphx_glr_plot_barycenter_lp_vs_entropic_002.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Elapsed time : 0.0049059391021728516 s
    Primal Feasibility  Dual Feasibility    Duality Gap         Step             Path Parameter      Objective          
    1.0                 1.0                 1.0                 -                1.0                 1700.336700337      
    0.006776453137632   0.006776453137632   0.006776453137632   0.9932238647293  0.006776453137632   125.6700527543      
    0.004018712867873   0.004018712867873   0.004018712867873   0.4301142633001  0.004018712867873   12.26594150092      
    0.001172775061627   0.001172775061627   0.001172775061627   0.7599932455027  0.001172775061627   0.3378536968898     
    0.0004375137005386  0.0004375137005386  0.0004375137005386  0.6422331807989  0.0004375137005386  0.1468420566359     
    0.0002326690467339  0.0002326690467339  0.0002326690467339  0.5016999460898  0.0002326690467339  0.09381703231428    
    7.430121674299e-05  7.4301216743e-05    7.430121674299e-05  0.7035962305811  7.430121674299e-05  0.05777870257169    
    5.321227838943e-05  5.321227838945e-05  5.321227838944e-05  0.3087841864307  5.321227838944e-05  0.05266249477219    
    1.990900379216e-05  1.99090037922e-05   1.990900379216e-05  0.6520472013271  1.990900379216e-05  0.04526054405523    
    6.305442046834e-06  6.305442046856e-06  6.305442046837e-06  0.7073953304085  6.305442046837e-06  0.04237597591384    
    2.290148391591e-06  2.290148391631e-06  2.290148391602e-06  0.6941812711476  2.29014839161e-06   0.04152284932101    
    1.182864875578e-06  1.182864875548e-06  1.182864875555e-06  0.5084552046229  1.182864875567e-06  0.04129461872829    
    3.626786386894e-07  3.626786386985e-07  3.626786386845e-07  0.7101651569095  3.626786385995e-07  0.0411303244893     
    1.539754244475e-07  1.539754247164e-07  1.539754247197e-07  0.6279322077522  1.539754251915e-07  0.04108867636377    
    5.193221608537e-08  5.19322169648e-08   5.193221696942e-08  0.6843453280956  5.193221892276e-08  0.04106859618454    
    1.888205219929e-08  1.88820500654e-08   1.888205006369e-08  0.6673443828803  1.888205852187e-08  0.04106214175236    
    5.676837529301e-09  5.676842740457e-09  5.676842761502e-09  0.7281712198286  5.676877044229e-09  0.04105958648535    
    3.501170987746e-09  3.501167688027e-09  3.501167721804e-09  0.4140142115019  3.501183058995e-09  0.04105916265728    
    1.110582426269e-09  1.110580273241e-09  1.110580239523e-09  0.6999003212726  1.110624310022e-09  0.04105870073273    
    5.768753963318e-10  5.769422203363e-10  5.769421938248e-10  0.5002521235315  5.767522037401e-10  0.04105859764872    
    1.534102102874e-10  1.535920569433e-10  1.535921107494e-10  0.7516900610544  1.535251083958e-10  0.04105851678411    
    6.717475002202e-11  6.735435784522e-11  6.735430717133e-11  0.5944268235824  6.732253215483e-11  0.04105850033323    
    1.751321118837e-11  1.74043080851e-11   1.740429001123e-11  0.7566075167358  1.736956306927e-11  0.0410584908946     
    Optimization terminated successfully.
             Current function value: 0.041058    
             Iterations: 22
    Elapsed time : 2.149055242538452 s




Stair Data
----------


.. code-block:: default


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










.. code-block:: default


    pl.figure(1, figsize=(6.4, 3))
    for i in range(n_distributions):
        pl.plot(x, A[:, i])
    pl.title('Distributions')
    pl.tight_layout()





.. image:: /auto_examples/images/sphx_glr_plot_barycenter_lp_vs_entropic_003.png
    :class: sphx-glr-single-img






.. code-block:: default


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





.. image:: /auto_examples/images/sphx_glr_plot_barycenter_lp_vs_entropic_004.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Elapsed time : 0.008316993713378906 s
    Primal Feasibility  Dual Feasibility    Duality Gap         Step             Path Parameter      Objective          
    1.0                 1.0                 1.0                 -                1.0                 1700.336700337      
    0.006776466288938   0.006776466288938   0.006776466288938   0.9932238515788  0.006776466288938   125.66492558        
    0.004036918865472   0.004036918865472   0.004036918865472   0.4272973099325  0.004036918865472   12.347161701        
    0.001219232687076   0.001219232687076   0.001219232687076   0.7496986855957  0.001219232687076   0.3243835647418     
    0.0003837422984467  0.0003837422984467  0.0003837422984467  0.6926882608271  0.0003837422984467  0.1361719397498     
    0.0001070128410194  0.0001070128410194  0.0001070128410194  0.7643889137854  0.0001070128410194  0.07581952832542    
    0.0001001275033713  0.0001001275033714  0.0001001275033713  0.07058704838615 0.0001001275033713  0.07347394936346    
    4.550897507807e-05  4.550897507807e-05  4.550897507807e-05  0.576117248486   4.550897507807e-05  0.05555077655034    
    8.557124125834e-06  8.557124125853e-06  8.557124125835e-06  0.853592544106   8.557124125835e-06  0.0443981466023     
    3.611995628666e-06  3.611995628643e-06  3.611995628672e-06  0.6002277331398  3.611995628673e-06  0.0428300776216     
    7.590393750111e-07  7.590393750273e-07  7.590393750129e-07  0.8221486533655  7.590393750133e-07  0.04192322976247    
    8.299929287077e-08  8.299929283415e-08  8.299929287126e-08  0.901746793884   8.299929287181e-08  0.04170825633295    
    3.117560207452e-10  3.117560192413e-10  3.117560199213e-10  0.9970399692253  3.117560200234e-10  0.04168179329766    
    1.559774508975e-14  1.559825507727e-14  1.559755309294e-14  0.9999499686993  1.559748033629e-14  0.04168169240444    
    Optimization terminated successfully.
             Current function value: 0.041682    
             Iterations: 13
    Elapsed time : 2.0333712100982666 s




Dirac Data
----------


.. code-block:: default


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










.. code-block:: default


    pl.figure(1, figsize=(6.4, 3))
    for i in range(n_distributions):
        pl.plot(x, A[:, i])
    pl.title('Distributions')
    pl.tight_layout()





.. image:: /auto_examples/images/sphx_glr_plot_barycenter_lp_vs_entropic_005.png
    :class: sphx-glr-single-img






.. code-block:: default


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





.. image:: /auto_examples/images/sphx_glr_plot_barycenter_lp_vs_entropic_006.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Elapsed time : 0.001787424087524414 s
    Primal Feasibility  Dual Feasibility    Duality Gap         Step             Path Parameter      Objective          
    1.0                 1.0                 1.0                 -                1.0                 1700.336700337      
    0.00677467552072    0.006774675520719   0.006774675520719   0.9932256422636  0.006774675520719   125.6956034741      
    0.002048208707556   0.002048208707555   0.002048208707555   0.734309536815   0.002048208707555   5.213991622102      
    0.0002697365474791  0.0002697365474791  0.0002697365474791  0.8839403501183  0.0002697365474791  0.5059383903908     
    6.832109993919e-05  6.832109993918e-05  6.832109993918e-05  0.7601171075982  6.832109993918e-05  0.2339657807271     
    2.437682932221e-05  2.437682932221e-05  2.437682932221e-05  0.6663448297463  2.437682932221e-05  0.1471256246325     
    1.134983216308e-05  1.134983216308e-05  1.134983216308e-05  0.5553643816417  1.134983216308e-05  0.1181584941171     
    3.342312725863e-06  3.34231272585e-06   3.342312725863e-06  0.7238133571629  3.342312725863e-06  0.1006387519746     
    7.078561231536e-07  7.078561231537e-07  7.078561231535e-07  0.803314255252   7.078561231535e-07  0.09474734646268    
    1.966870949422e-07  1.966870952674e-07  1.966870952717e-07  0.7525479180433  1.966870953014e-07  0.09354342735758    
    4.199895266495e-10  4.199895367352e-10  4.19989526535e-10   0.9984019849265  4.199895265747e-10  0.09310367785861    
    2.101053559204e-14  2.100331212975e-14  2.101054034304e-14  0.9999499736903  2.101053604307e-14  0.09310274466458    
    Optimization terminated successfully.
             Current function value: 0.093103    
             Iterations: 11
    Elapsed time : 2.1853578090667725 s




Final figure
------------



.. code-block:: default


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



.. image:: /auto_examples/images/sphx_glr_plot_barycenter_lp_vs_entropic_007.png
    :class: sphx-glr-single-img






.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  7.697 seconds)


.. _sphx_glr_download_auto_examples_plot_barycenter_lp_vs_entropic.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: plot_barycenter_lp_vs_entropic.py <plot_barycenter_lp_vs_entropic.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: plot_barycenter_lp_vs_entropic.ipynb <plot_barycenter_lp_vs_entropic.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
