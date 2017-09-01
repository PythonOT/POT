

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
        1   +8.6305817354868675e-01 4.10110152e-01
        2   +4.6939060757969131e-01 2.98553763e-01
        3   +4.2106314200107775e-01 1.48552668e-01
        4   +4.1376389458568069e-01 1.12319011e-01
        5   +4.0984854988792835e-01 1.01126129e-01
        6   +4.0415292614140025e-01 3.90875165e-02
        7   +4.0297967887432584e-01 2.73716014e-02
        8   +4.0252319029045258e-01 3.76498956e-02
        9   +4.0158635935184972e-01 1.31986577e-02
       10   +4.0118906894272482e-01 3.40307273e-02
       11   +4.0052579694802176e-01 7.79567347e-03
       12   +4.0049330810825384e-01 9.77921941e-03
       13   +4.0042500151972926e-01 4.63602913e-03
       14   +4.0031705300038767e-01 1.69742018e-02
       15   +4.0013705338124350e-01 7.40310798e-03
       16   +4.0006224569843946e-01 1.08829949e-02
       17   +3.9998280287782945e-01 1.25733450e-02
       18   +3.9986405111843215e-01 1.05626807e-02
       19   +3.9974905002724365e-01 9.93566406e-03
       20   +3.9971323753531823e-01 2.21199533e-02
       21   +3.9958582328238779e-01 1.73335808e-02
       22   +3.9937139582811110e-01 1.09182412e-02
       23   +3.9923748818499571e-01 1.77304913e-02
       24   +3.9900530515251881e-01 1.15381586e-02
       25   +3.9883316307006128e-01 1.80225446e-02
       26   +3.9860317631835845e-01 1.65011032e-02
       27   +3.9852130309759393e-01 2.81245689e-02
       28   +3.9824281033694675e-01 2.01114810e-02
       29   +3.9799657608114836e-01 2.66040929e-02
       30   +3.9746233677210713e-01 1.45779937e-02
       31   +3.9671794378467928e-01 4.27487207e-02
       32   +3.9573357685391913e-01 2.20071520e-02
       33   +3.9536725156297214e-01 2.00817458e-02
       34   +3.9515994339814914e-01 3.81472315e-02
       35   +3.9448966390371887e-01 2.52129049e-02
       36   +3.9351423238681266e-01 5.60677866e-02
       37   +3.9082703288308568e-01 4.26859586e-02
       38   +3.7139409489868136e-01 1.26067835e-01
       39   +2.8085932518253526e-01 1.70133509e-01
       40   +2.7330384726281814e-01 1.95523507e-01
       41   +2.4806985554269162e-01 1.31192016e-01
       42   +2.3748356968454920e-01 8.71616829e-02
       43   +2.3501927152342389e-01 7.02789537e-02
       44   +2.3183578112546338e-01 2.62025296e-02
       45   +2.3154208568082749e-01 1.67845346e-02
       46   +2.3139316710346300e-01 8.27285074e-03
       47   +2.3136034106523354e-01 4.64818210e-03
       48   +2.3134548827742521e-01 4.53144806e-04
       49   +2.3134540503271503e-01 2.91010390e-04
       50   +2.3134535764073319e-01 1.25662481e-04
       51   +2.3134534692621381e-01 1.24751216e-05
       52   +2.3134534685831357e-01 7.44008265e-06
       53   +2.3134534684658337e-01 6.16933546e-06
       54   +2.3134534682129679e-01 5.12152219e-07
    Terminated - min grad norm reached after 54 iterations, 24.53 seconds.


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




**Total running time of the script:** ( 0 minutes  25.326 seconds)



.. container:: sphx-glr-footer


  .. container:: sphx-glr-download

     :download:`Download Python source code: plot_WDA.py <plot_WDA.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: plot_WDA.ipynb <plot_WDA.ipynb>`

.. rst-class:: sphx-glr-signature

    `Generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_
