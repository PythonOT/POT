
How to ?
========

In the following we provide some pointers about which functions and classes 
to use for different problems related to optimal transport (OTs).

1. **How to solve a discrete optimal transport problem ?**

    The solver for discrete  is the function :py:mod:`ot.emd` that returns
    the OT transport matrix. If you want to solve a regularized OT you can 
    use :py:mod:`ot.sinkhorn`.

    More detailed examples can be seen on this :ref:`auto_examples/plot_OT_2D_samples`

    Here is a simple use case:

   .. code:: python

       # a,b are 1D histograms (sum to 1 and positive)
       # M is the ground cost matrix
       T=ot.emd(a,b,M) # exact linear program
       T_reg=ot.sinkhorn(a,b,M,reg) # entropic regularized OT


