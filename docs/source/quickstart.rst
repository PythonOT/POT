
Quick start guide
=================

In the following we provide some pointers about which functions and classes 
to use for different problems related to optimal transport (OT).

This document is not a tutorial on numerical optimal transport. For this we strongly
recommend to read the very nice book [15]_ . 


Optimal transport and Wasserstein distance
------------------------------------------

.. note::
    In POT, most functions that solve OT or regularized OT problems have two
    versions that return the OT matrix or the value of the optimal solution. For
    instance :any:`ot.emd` return the OT matrix and :any:`ot.emd2` return the
    Wassertsein distance.

Solving optimal transport
^^^^^^^^^^^^^^^^^^^^^^^^^

The optimal transport problem between discrete distributions is often expressed
as

.. math::
    \gamma^* = arg\min_\gamma \quad \sum_{i,j}\gamma_{i,j}M_{i,j}

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

The method used for solving the OT problem is the network simplex, it is
implemented in C from  [1]_. It has a complexity of :math:`O(n^3)` but the
solver is quite efficient and uses sparsity of the solution.

.. hint::
    Examples of use for :any:`ot.emd` are available in the following examples:

    - :any:`auto_examples/plot_OT_2D_samples`
    - :any:`auto_examples/plot_OT_1D` 
    - :any:`auto_examples/plot_OT_L1_vs_L2` 

Computing Wasserstein distance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The value of the OT solution is often more of interest that the OT matrix :

    .. math::
        OT(a,b)=\min_\gamma \quad \sum_{i,j}\gamma_{i,j}M_{i,j}

        s.t. \gamma 1 = a; \gamma^T 1= b; \gamma\geq 0


It can computed from an already estimated OT matrix with
:code:`np.sum(T*M)` or directly with the function :any:`ot.emd2`.

.. code:: python

    # a,b are 1D histograms (sum to 1 and positive)
    # M is the ground cost matrix
    W=ot.emd2(a,b,M) # Wasserstein distance / EMD value

Note that the well known  `Wasserstein distance
<https://en.wikipedia.org/wiki/Wasserstein_metric>`_ between distributions a and
b is defined as


    .. math::

        W_p(a,b)=(\min_\gamma \sum_{i,j}\gamma_{i,j}\|x_i-y_j\|_p)^\frac{1}{p}

        s.t. \gamma 1 = a; \gamma^T 1= b; \gamma\geq 0

This means that if you want to compute the :math:`W_2` you need to compute the
square root of :any:`ot.emd2` when providing
:code:`M=ot.dist(xs,xt)` that use the squared euclidean distance by default. Computing
the :math:`W_1` wasserstein distance can be done directly with  :any:`ot.emd2`
when providing :code:`M=ot.dist(xs,xt, metric='euclidean')` to use the euclidean
distance.

 

.. hint::
    Examples of use for :any:`ot.emd2` are available in the following examples:

    - :any:`auto_examples/plot_compute_emd`
 

Special cases
^^^^^^^^^^^^^

Note that the OT problem and the corresponding Wasserstein distance can in some
special cases be computed very efficiently. 

For instance when the samples are in 1D, then the OT problem can be solved in
:math:`O(n\log(n))` by using a simple sorting. In this case we provide the
function :any:`ot.emd_1d` and   :any:`ot.emd2_1d` to return respectively the OT
matrix and value. Note that since the solution is very sparse the :code:`sparse`
parameter of :any:`ot.emd_1d` allows for solving and returning the solution for
very large problems. Note that in order to computed directly the :math:`W_p`
Wasserstein distance in 1D we provide the function :any:`ot.wasserstein_1d` that
takes :code:`p` as a parameter. 

Another specials for estimating OT and Monge mapping is between Gaussian
distributions. In this case there exists a close form solution given in Remark
2.29 in [15]_ and the Monge mapping is an affine function and can be
also computed from the covariances and means of the source and target
distributions. In this case when the finite sample dataset is supposed gaussian, we provide 
:any:`ot.da.OT_mapping_linear` that returns the parameters for the Monge
mapping.


Regularized Optimal Transport
-----------------------------

Recent developments have shown the interest of regularized OT both in terms of
computational and statistical properties.

We address in this section the regularized OT problem that can be expressed as

.. math::
    \gamma^* = arg\min_\gamma \quad \sum_{i,j}\gamma_{i,j}M_{i,j} + \lambda\Omega(\gamma)

        s.t. \gamma 1 = a; \gamma^T 1= b; \gamma\geq 0


where :

- :math:`M\in\mathbb{R}_+^{m\times n}` is the metric cost matrix defining the cost to move mass from bin :math:`a_i` to bin :math:`b_j`.
- :math:`a` and :math:`b` are histograms (positive, sum to 1) that represent the weights of each samples in the source an target distributions.
- :math:`\Omega` is the regularization term.

We discuss in the following specific algorithms that can be used depending on
the regularization term.



Entropic regularized OT
^^^^^^^^^^^^^^^^^^^^^^^

This is the most common regularization used for optimal transport. It has been
proposed in the ML community by Marco Cuturi in his seminal paper [2]_. This
regularization has the following expression

.. math::
    \Omega(\gamma)=\sum_{i,j}\gamma_{i,j}\log(\gamma_{i,j})


The use of the regularization term above in the optimization problem has a very
strong impact. First it makes the problem smooth which leads to new optimization
procedures such as L-BFGS (see :any:`ot.smooth` ). Next it makes the problem
strictly convex meaning that there will be a unique solution. Finally the
solution of the resulting optimization problem can be expressed as:

.. math::

    \gamma_\lambda^*=\text{diag}(u)K\text{diag}(v)

where :math:`u` and :math:`v` are vectors and :math:`K=\exp(-M/\lambda)` where
the :math:`\exp` is taken component-wise. In order to solve the optimization
problem, on can use an alternative projection algorithm that can be very
efficient for large values if regularization. 

The main function is POT are  :any:`ot.sinkhorn` and
:any:`ot.sinkhorn2` that return respectively the OT matrix and the value of the
linear term. Note that the regularization parameter :math:`\lambda` in the
equation above is given to those function with the parameter :code:`reg`.

    >>> import ot
    >>> a=[.5,.5]
    >>> b=[.5,.5]
    >>> M=[[0.,1.],[1.,0.]]
    >>> ot.sinkhorn(a,b,M,1)
    array([[ 0.36552929,  0.13447071],
        [ 0.13447071,  0.36552929]])



More details about the algorithm used is given in the following note.


.. note::
    The main function to solve entropic regularized OT is :any:`ot.sinkhorn`.
    This function is a wrapper and the parameter :code:`method` help you select
    the actual algorithm used to solve the problem:

    + :code:`method='sinkhorn'` calls :any:`ot.bregman.sinkhorn_knopp`  the
      classic algorithm [2]_.
    + :code:`method='sinkhorn_stabilized'` calls :any:`ot.bregman.sinkhorn_stabilized`  the
      log stabilized version of the algorithm [9]_.    
    + :code:`method='sinkhorn_epsilon_scaling'` calls
      :any:`ot.bregman.sinkhorn_epsilon_scaling`  the epsilon scaling version
      of the algorithm [9]_.   
    + :code:`method='greenkhorn'` calls :any:`ot.bregman.greenkhorn`  the
      greedy sinkhorn verison of the algorithm [22]_.   

    In addition to all those variants of sinkhorn, we have another
    implementation solving the problem in the smooth dual or semi-dual in
    :any:`ot.smooth`. This solver uses the :any:`scipy.optimize.minimize`
    function to solve the smooth problem with :code:`L-BFGS` algorithm. Tu use
    this solver, use functions :any:`ot.smooth.smooth_ot_dual` or
    :any:`ot.smooth.smooth_ot_semi_dual` with parameter :code:`reg_type='kl'` to
    choose entropic/Kullbach Leibler regularization.




Recently [23]_ introduced the sinkhorn divergence that build from entropic
regularization to compute fast and differentiable geometric diveregnce between
empirical distributions.  



Finally note that we also provide in :any:`ot.stochastic` several implementation
of stochastic solvers for entropic regularized OT [18]_ [19]_.  

.. hint::
    Examples of use for :any:`ot.sinkhorn` are available in the following examples:

    - :any:`auto_examples/plot_OT_2D_samples`
    - :any:`auto_examples/plot_OT_1D` 
    - :any:`auto_examples/plot_OT_1D_smooth`
    - :any:`auto_examples/plot_stochastic`


Other regularization
^^^^^^^^^^^^^^^^^^^^

While entropic OT is the most common and favored in practice, there exist other
kind of regularization. We provide in POT two specific solvers for other
regularization terms: namely quadratic regularization and group lasso
regularization. But we also provide in :any:`ot.optim`  two generic solvers that allows solving any
smooth regularization in practice. 

Quadratic regularization
""""""""""""""""""""""""

The first general regularization term we can solve is the quadratic
regularization of the form 

.. math::
    \Omega(\gamma)=\sum_{i,j} \gamma_{i,j}^2

this regularization term has a similar effect to entropic regularization in
densifying the OT matrix but it keeps some sort of sparsity that is lost with
entropic regularization as soon as :math:`\lambda>0` [17]_. This problem cen be
solved with POT using solvers from :any:`ot.smooth`, more specifically
functions :any:`ot.smooth.smooth_ot_dual` or
:any:`ot.smooth.smooth_ot_semi_dual` with parameter :code:`reg_type='l2'` to 
choose the quadratic regularization.

.. hint::
    Examples of quadratic regularization are available in the following examples:

    - :any:`auto_examples/plot_OT_1D_smooth`
    - :any:`auto_examples/plot_optim_OTreg`



Group Lasso regularization
""""""""""""""""""""""""""

Another regularization that has been used in recent years is the group lasso
regularization

.. math::
    \Omega(\gamma)=\sum_{j,G\in\mathcal{G}} \|\gamma_{G,j}\|_q^p

where :math:`\mathcal{G}` contains non overlapping groups of lines in the OT
matrix. This regularization proposed in [5]_ will promote sparsity at the group level and for
instance will force target samples to get mass from a small number of groups.
Note that the exact OT solution is already sparse so this regularization does
not make sens if it is not combined with others such as entropic. Depending on
the choice of :code:`p` and :code:`q`, the problem can be solved with different
approaches.  When :code:`q=1` and :code:`p<1` the problem is non convex but can
be solved using an efficient majoration minimization approach  with
:any:`ot.sinkhorn_lpl1_mm`. When :code:`q=2` and :code:`p=1` we recover the
convex gourp lasso and we provide a solver using generalized conditional
gradient algorithm [7]_ in function
:any:`ot.da.sinkhorn_l1l2_gl`.

.. hint::
    Examples of group Lasso regularization are available in the following examples:

    - :any:`auto_examples/plot_otda_classes` 
    - :any:`auto_examples/plot_otda_d2`


Generic solvers
"""""""""""""""

Finally we propose in POT generic solvers that can be used to solve any
regularization as long as you can provide a function computing the
regularization and a function computing its gradient.

In order to solve 

.. math::
    \gamma^* = arg\min_\gamma \quad \sum_{i,j}\gamma_{i,j}M_{i,j} + \lambda\Omega(\gamma)

        s.t. \gamma 1 = a; \gamma^T 1= b; \gamma\geq 0

you can use function :any:`ot.optim.cg` that will use a conditional gradient as
proposed in [6]_ . you need to provide the regularization function as parameter
``f`` and its gradient as parameter  ``df``. Note that the conditional gradient relies on
iterative solving of a linearization of the problem using the exact
:any:`ot.emd` so it can be  slow in practice. Still it always returns a
transport matrix that does not violates the marginals.

Another solver is proposed to solve the problem

.. math::
    \gamma^* = arg\min_\gamma \quad \sum_{i,j}\gamma_{i,j}M_{i,j}+ \lambda_e\Omega_e(\gamma) + \lambda\Omega(\gamma)

        s.t. \gamma 1 = a; \gamma^T 1= b; \gamma\geq 0

where :math:`\Omega_e` is the entropic regularization. In this case we use a
generalized conditional gradient [7]_ implemented in :any:`ot.opim.gcg`  that does not linearize the entropic term and
relies on :any:`ot.sinkhorn` for its iterations. 

.. hint::
    Example of generic solvers are available in the following example:

    - :any:`auto_examples/plot_optim_OTreg` 


Wasserstein Barycenters
-----------------------

A Wasserstein barycenter is a distribution that minimize its Wasserstein
distance with respect to other distributions [16]_. It corresponds to minimizing the
following problem by seaching a distribution :math:`\mu` 

.. math::
    \min_\mu \quad \sum_{k} w_kW(\mu,\mu_k)


In practice we model a distribution with a finite number of support position:

.. math::
    \mu=\sum_{i=1}^n a_i\delta_{x_i}

where :math:`a` is an histogram on the simplex and the :math:`\{x_i\}` are the
position of the support. We can clearly see here that optimizing :math:`\mu` can
be done by searching for optimal weights :math:`a` or optimal support
:math:`\{x_i\}` (optimizing both is also an option).
We provide in POT solvers to estimate a discrete
Wasserstein barycenter in both cases.

Barycenters with fixed support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When optimizing a barycenter with a fixed support, the optimization problem can
be expressed as


.. math::
    \min_a \quad \sum_{k} w_k W(a,b_k)

where :math:`b_k` are also weights in the simplex. In the non-regularized case,
the problem above is a classical linear program. In this case we propose a
solver :any:`ot.lp.barycenter` that rely on generic LP solvers. By default the
function uses :any:`scipy.optimize.linprog`, but more efficient LP solvers from
cvxopt can be also used by changing parameter :code:`solver`. Note that these
solver require to solve a very large linear program and can be very slow in
practice. 

Similarly to the OT problem, OT barycenters can be computed in the regularized
case. When using entropic regularization the problem can be solved with a
generalization of the sinkhorn algorithm based on bregman projections [3]_. This
algorithm is provided in function :any:`ot.bregman.barycenter` also available as
:any:`ot.barycenter`. In this case, the algorithm scales better to large
distributions and rely only on matrix multiplications that can be performed in
parallel.

In addition to teh speedup brought by regularization, one can also greatly
accelerate the estimation of Wasserstein barycenter when the support has a
separable structure [21]_. In teh case of 2D images for instance one can replace
the matrix vector production in teh bregman projections by convolution
operators. We provide an implementation of this algorithm in function
:any:`ot.bregman.convolutional_barycenter2d`.

.. hint::
    Example of Wasserstein (:any:`ot.lp.barycenter`) and regularized wassrestein
    barycenter (:any:`ot.bregman.barycenter`) computation are available in the following examples:

    - :any:`auto_examples/plot_barycenter_1D` 
    - :any:`auto_examples/plot_barycenter_lp_vs_entropic` 

    Example of convolutional barycenter (:any:`ot.bregman.convolutional_barycenter2d`) computation for 2D images is available
    in the following example:

    - :any:`auto_examples/plot_convolutional_barycenter`



Barycenters with free support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^




Monge mapping and Domain adaptation
-----------------------------------


Other applications
------------------

Wasserstein Discriminant Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Gromov-Wasserstein
^^^^^^^^^^^^^^^^^^


GPU acceleration
----------------

We provide several implementation of our OT solvers in :any:`ot.gpu`. Those
implementation use the :code:`cupy` toolbox.   



FAQ
---



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
    

2. **pip install POT fails with error : ImportError: No module named Cython.Build**

    As discussed shortly in the README file. POT requires to have :code:`numpy`
    and :code:`cython` installed to build. This corner case is not yet handled
    by :code:`pip` and for now you need to install both library prior to
    installing POT.

    Note that this problem do not occur when using conda-forge since the packages
    there are pre-compiled. 

    See `Issue #59 <https://github.com/rflamary/POT/issues/59>`__ for more
    details.

3. **Why is Sinkhorn slower than EMD ?**

    This might come from the choice of the regularization term. The speed of
    convergence of sinkhorn depends directly on this term [22]_ and when the
    regularization gets very small the problem try and approximate the exact OT
    which leads to slow convergence in addition to numerical problems. In other
    words, for large regularization sinkhorn will be very fast to converge, for
    small regularization (when you need an OT matrix close to the true OT), it
    might be quicker to use the EMD solver.

    Also note that the numpy implementation of the sinkhorn can use parallel
    computation depending on the configuration of your system but very important
    speedup can be obtained by using a GPU implementation since all operations
    are matrix/vector products.

4. **Using GPU fails with error:  module 'ot' has no attribute 'gpu'**

    In order to limit import time and hard dependencies in POT. we do not import
    some sub-modules automatically with :code:`import ot`. In order to use the
    acceleration in :any:`ot.gpu` you need first to import is with
    :code:`import ot.gpu`.  

    See `Issue #85 <https://github.com/rflamary/POT/issues/85>`__ and :any:`ot.gpu`
    for more details.


References
----------

.. [1] Bonneel, N., Van De Panne, M., Paris, S., & Heidrich, W. (2011,
    December). `Displacement  nterpolation using Lagrangian mass transport
    <https://people.csail.mit.edu/sparis/publi/2011/sigasia/Bonneel_11_Displacement_Interpolation.pdf>`__.
    In ACM Transactions on Graphics (TOG) (Vol. 30, No. 6, p. 158). ACM. 

.. [2] Cuturi, M. (2013). `Sinkhorn distances: Lightspeed computation of
    optimal transport <https://arxiv.org/pdf/1306.0895.pdf>`__. In Advances
    in Neural Information Processing Systems (pp. 2292-2300).

.. [3] Benamou, J. D., Carlier, G., Cuturi, M., Nenna, L., & Peyré, G.
    (2015). `Iterative Bregman projections for regularized transportation
    problems <https://arxiv.org/pdf/1412.5154.pdf>`__. SIAM Journal on
    Scientific Computing, 37(2), A1111-A1138.

.. [4] S. Nakhostin, N. Courty, R. Flamary, D. Tuia, T. Corpetti,
    `Supervised planetary unmixing with optimal
    transport <https://hal.archives-ouvertes.fr/hal-01377236/document>`__,
    Whorkshop on Hyperspectral Image and Signal Processing : Evolution in
    Remote Sensing (WHISPERS), 2016.

.. [5] N. Courty; R. Flamary; D. Tuia; A. Rakotomamonjy, `Optimal Transport
    for Domain Adaptation <https://arxiv.org/pdf/1507.00504.pdf>`__, in IEEE
    Transactions on Pattern Analysis and Machine Intelligence , vol.PP,
    no.99, pp.1-1

.. [6] Ferradans, S., Papadakis, N., Peyré, G., & Aujol, J. F. (2014).
    `Regularized discrete optimal
    transport <https://arxiv.org/pdf/1307.5551.pdf>`__. SIAM Journal on
    Imaging Sciences, 7(3), 1853-1882.

.. [7] Rakotomamonjy, A., Flamary, R., & Courty, N. (2015). `Generalized
    conditional gradient: analysis of convergence and
    applications <https://arxiv.org/pdf/1510.06567.pdf>`__. arXiv preprint
    arXiv:1510.06567.

.. [8] M. Perrot, N. Courty, R. Flamary, A. Habrard (2016), `Mapping
    estimation for discrete optimal
    transport <http://remi.flamary.com/biblio/perrot2016mapping.pdf>`__,
    Neural Information Processing Systems (NIPS).

.. [9] Schmitzer, B. (2016). `Stabilized Sparse Scaling Algorithms for
    Entropy Regularized Transport
    Problems <https://arxiv.org/pdf/1610.06519.pdf>`__. arXiv preprint
    arXiv:1610.06519.

.. [10] Chizat, L., Peyré, G., Schmitzer, B., & Vialard, F. X. (2016).
    `Scaling algorithms for unbalanced transport
    problems <https://arxiv.org/pdf/1607.05816.pdf>`__. arXiv preprint
    arXiv:1607.05816.

.. [11] Flamary, R., Cuturi, M., Courty, N., & Rakotomamonjy, A. (2016).
    `Wasserstein Discriminant
    Analysis <https://arxiv.org/pdf/1608.08063.pdf>`__. arXiv preprint
    arXiv:1608.08063.

.. [12] Gabriel Peyré, Marco Cuturi, and Justin Solomon (2016),
    `Gromov-Wasserstein averaging of kernel and distance
    matrices <http://proceedings.mlr.press/v48/peyre16.html>`__
    International Conference on Machine Learning (ICML).

.. [13] Mémoli, Facundo (2011). `Gromov–Wasserstein distances and the
    metric approach to object
    matching <https://media.adelaide.edu.au/acvt/Publications/2011/2011-Gromov%E2%80%93Wasserstein%20Distances%20and%20the%20Metric%20Approach%20to%20Object%20Matching.pdf>`__.
    Foundations of computational mathematics 11.4 : 417-487.

.. [14] Knott, M. and Smith, C. S. (1984).`On the optimal mapping of
    distributions <https://link.springer.com/article/10.1007/BF00934745>`__,
    Journal of Optimization Theory and Applications Vol 43.

.. [15] Peyré, G., & Cuturi, M. (2018). `Computational Optimal
    Transport <https://arxiv.org/pdf/1803.00567.pdf>`__ .

.. [16] Agueh, M., & Carlier, G. (2011). `Barycenters in the Wasserstein
    space <https://hal.archives-ouvertes.fr/hal-00637399/document>`__. SIAM
    Journal on Mathematical Analysis, 43(2), 904-924.

.. [17] Blondel, M., Seguy, V., & Rolet, A. (2018). `Smooth and Sparse
    Optimal Transport <https://arxiv.org/abs/1710.06276>`__. Proceedings of
    the Twenty-First International Conference on Artificial Intelligence and
    Statistics (AISTATS).

.. [18] Genevay, A., Cuturi, M., Peyré, G. & Bach, F. (2016) `Stochastic
    Optimization for Large-scale Optimal
    Transport <https://arxiv.org/abs/1605.08527>`__. Advances in Neural
    Information Processing Systems (2016).

.. [19] Seguy, V., Bhushan Damodaran, B., Flamary, R., Courty, N., Rolet,
    A.& Blondel, M. `Large-scale Optimal Transport and Mapping
    Estimation <https://arxiv.org/pdf/1711.02283.pdf>`__. International
    Conference on Learning Representation (2018)

.. [20] Cuturi, M. and Doucet, A. (2014) `Fast Computation of Wasserstein
    Barycenters <http://proceedings.mlr.press/v32/cuturi14.html>`__.
    International Conference in Machine Learning

.. [21] Solomon, J., De Goes, F., Peyré, G., Cuturi, M., Butscher, A.,
    Nguyen, A. & Guibas, L. (2015). `Convolutional wasserstein distances:
    Efficient optimal transportation on geometric
    domains <https://dl.acm.org/citation.cfm?id=2766963>`__. ACM
    Transactions on Graphics (TOG), 34(4), 66.

.. [22] J. Altschuler, J.Weed, P. Rigollet, (2017) `Near-linear time
    approximation algorithms for optimal transport via Sinkhorn
    iteration <https://papers.nips.cc/paper/6792-near-linear-time-approximation-algorithms-for-optimal-transport-via-sinkhorn-iteration.pdf>`__,
    Advances in Neural Information Processing Systems (NIPS) 31

.. [23] Aude, G., Peyré, G., Cuturi, M., `Learning Generative Models with
    Sinkhorn Divergences <https://arxiv.org/abs/1706.00292>`__, Proceedings
    of the Twenty-First International Conference on Artficial Intelligence
    and Statistics, (AISTATS) 21, 2018

.. [24] Vayer, T., Chapel, L., Flamary, R., Tavenard, R. and Courty, N.
    (2019). `Optimal Transport for structured data with application on
    graphs <http://proceedings.mlr.press/v97/titouan19a.html>`__ Proceedings
    of the 36th International Conference on Machine Learning (ICML).