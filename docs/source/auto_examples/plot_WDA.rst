

.. _sphx_glr_auto_examples_plot_WDA.py:


=================================
Wasserstein Discriminant Analysis
=================================





.. rst-class:: sphx-glr-horizontal


    *

      .. image:: /auto_examples/images/sphx_glr_plot_WDA_001.png
            :scale: 47

    *

      .. image:: /auto_examples/images/sphx_glr_plot_WDA_002.png
            :scale: 47


.. rst-class:: sphx-glr-script-out

 Out::

    Compiling cost function...
    Computing gradient of cost function...
     iter              cost val         grad. norm
        1   +8.9741888001949222e-01 3.71269078e-01
        2   +4.9103998133976140e-01 3.46687543e-01
        3   +4.2142651893148553e-01 1.04789602e-01
        4   +4.1573609749588841e-01 5.21726648e-02
        5   +4.1486046805261961e-01 5.35335513e-02
        6   +4.1315953904635105e-01 2.17803599e-02
        7   +4.1313030162717523e-01 6.06901182e-02
        8   +4.1301511591963386e-01 5.88598758e-02
        9   +4.1258349404769817e-01 5.14307874e-02
       10   +4.1139242901051226e-01 2.03198793e-02
       11   +4.1113798965164017e-01 1.18944721e-02
       12   +4.1103446820878486e-01 2.21783648e-02
       13   +4.1076586830791861e-01 9.51495863e-03
       14   +4.1036935287519144e-01 3.74973214e-02
       15   +4.0958729714575060e-01 1.23810902e-02
       16   +4.0898266309095005e-01 4.01999918e-02
       17   +4.0816076944357715e-01 2.27240277e-02
       18   +4.0788116701894767e-01 4.42815945e-02
       19   +4.0695443744952403e-01 3.28464304e-02
       20   +4.0293834480911150e-01 7.76000681e-02
       21   +3.8488003705202750e-01 1.49378022e-01
       22   +3.0767344927282614e-01 2.15432117e-01
       23   +2.3849425361868334e-01 1.07942382e-01
       24   +2.3845125762548214e-01 1.08953278e-01
       25   +2.3828007730494005e-01 1.07934830e-01
       26   +2.3760839060570119e-01 1.03822134e-01
       27   +2.3514215179705886e-01 8.67263481e-02
       28   +2.2978886197588613e-01 9.26609306e-03
       29   +2.2972671019495342e-01 2.59476089e-03
       30   +2.2972355865247496e-01 1.57205146e-03
       31   +2.2972296662351968e-01 1.29300760e-03
       32   +2.2972181557051569e-01 8.82375756e-05
       33   +2.2972181277025336e-01 6.20536544e-05
       34   +2.2972181023486152e-01 7.01884014e-06
       35   +2.2972181020400181e-01 1.60415765e-06
       36   +2.2972181020236590e-01 2.44290966e-07
    Terminated - min grad norm reached after 36 iterations, 13.41 seconds.




|


.. code-block:: python


    # Author: Remi Flamary <remi.flamary@unice.fr>
    #
    # License: MIT License

    import numpy as np
    import matplotlib.pylab as pl

    from ot.dr import wda, fda


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

    #%% Compute FDA
    p = 2

    Pfda, projfda = fda(xs, ys, p)

    #%% Compute WDA
    p = 2
    reg = 1e0
    k = 10
    maxiter = 100

    Pwda, projwda = wda(xs, ys, p, reg, k, maxiter=maxiter)

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

**Total running time of the script:** ( 0 minutes  19.853 seconds)



.. container:: sphx-glr-footer


  .. container:: sphx-glr-download

     :download:`Download Python source code: plot_WDA.py <plot_WDA.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: plot_WDA.ipynb <plot_WDA.ipynb>`

.. rst-class:: sphx-glr-signature

    `Generated by Sphinx-Gallery <http://sphinx-gallery.readthedocs.io>`_
