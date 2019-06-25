
Quick start
===========



In the following we provide some pointers about which functions and classes 
to use for different problems related to optimal transport (OT).


Optimal transport and Wasserstein distance
------------------------------------------


Solving optimal transport
^^^^^^^^^^^^^^^^^^^^^^^^^

The optimal transport problem between discrete distributions is often expressed
as
    .. math::
        \gamma^* = arg\min_\gamma \sum_{i,j}\gamma_{i,j}M_{i,j}

        s.t. \gamma 1 = a; \gamma^T 1= b; \gamma\geq 0

where :

- :math:`M\in\mathbb{R}_+^{m\times n}` is the metric cost matrix defining the cost to move mass from bin :math:`a_i` to bin :math:`b_j`.
- :math:`a` and :math:`b` are histograms (positive, sum to 1) that represent the weights of each samples in the source an target distributions.

Solving the linear program above can be done using the function :any:`ot.emd`
that will return the optimal transport matrix :math:`\gamma^*`:

.. code:: python

    # a,b are 1D histograms (sum to 1 and positive)
    # M is the ground cost matrix
    T=ot.emd(a,b,M) # exact linear program

.. hint::
    Examples of use for :any:`ot.emd` are available in the following examples:

    - :any:`auto_examples/plot_OT_2D_samples`
    - :any:`auto_examples/plot_OT_1D` 
    - :any:`auto_examples/plot_OT_L1_vs_L2` 

Computing Wasserstein distance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The value of the OT solution is often more of interest that the OT matrix :

    .. math::
        W(a,b)=\min_\gamma \sum_{i,j}\gamma_{i,j}M_{i,j}

        s.t. \gamma 1 = a; \gamma^T 1= b; \gamma\geq 0


where :math:`W(a,b)` is the  `Wasserstein distance
<https://en.wikipedia.org/wiki/Wasserstein_metric>`_ between distributions a and b
It is a metrix that has nice statistical
properties. It can computed from an already estimated OT matrix with
:code:`np.sum(T*M)` or directly with the function :any:`ot.emd2`.

.. code:: python

    # a,b are 1D histograms (sum to 1 and positive)
    # M is the ground cost matrix
    W=ot.emd2(a,b,M) # Wasserstein distance / EMD value


.. hint::
    Examples of use for :any:`ot.emd2` are available in the following examples:

    - :any:`auto_examples/plot_compute_emd`
 

.. note::
    In POT, most functions that solve OT or regularized OT problems have two
    versions that return the OT matrix or the value of the optimal solution. Fir
    instance :any:`ot.emd` return the OT matrix and :any:`ot.emd2` return the
    Wassertsein distance.


Regularized Optimal Transport
-----------------------------

Wasserstein Barycenters
-----------------------

Monge mapping and Domain adaptation with Optimal transport
----------------------------------------


Other applications
------------------


GPU acceleration
----------------



How to?
-------



1. **How to solve a discrete optimal transport problem ?**

    The solver for discrete  is the function :py:mod:`ot.emd` that returns
    the OT transport matrix. If you want to solve a regularized OT you can 
    use :py:mod:`ot.sinkhorn`.

    

    Here is a simple use case:

    .. code:: python

       # a,b are 1D histograms (sum to 1 and positive)
       # M is the ground cost matrix
       T=ot.emd(a,b,M) # exact linear program
       T_reg=ot.sinkhorn(a,b,M,reg) # entropic regularized OT

    More detailed examples can be seen on this
    :doc:`auto_examples/plot_OT_2D_samples`
    

2. **Compute a Wasserstein distance**




