
Quick start guide
=================

In the following we provide some pointers about which functions and classes
to use for different problems related to optimal transport (OT) and machine
learning. We refer when we can to concrete examples in the documentation that
are also available as notebooks on the POT Github.

.. note::

    For a  good introduction to numerical optimal transport we refer the reader
    to `the book <https://arxiv.org/pdf/1803.00567.pdf>`_ by Peyré and Cuturi
    [15]_. For more detailed introduction to OT and how it can be used
    in ML applications we refer the reader to the following `OTML tutorial
    <https://remi.flamary.com/cours/tuto_otml.html>`_.
    
.. note::

    Since version 0.8, POT provides a backend to automatically solve some OT
    problems independently from the toolbox used by the user (numpy/torch/jax).
    We provide a discussion about which functions are compatible in section
    `Backend section <#solving-ot-with-multiple-backends>`_ .


Why Optimal Transport ?
-----------------------


When to use OT
^^^^^^^^^^^^^^

Optimal Transport (OT) is a mathematical  problem introduced by Gaspard Monge in
1781 that aim at finding the most efficient way to move mass between
distributions. The cost of moving a unit of mass between two positions is called
the ground cost and the objective is to minimize the overall cost of moving one
mass distribution onto another one. The optimization problem can be expressed
for two distributions :math:`\mu_s` and :math:`\mu_t` as

.. math:: 
    \min_{m, m \# \mu_s = \mu_t} \int c(x,m(x))d\mu_s(x) ,

where :math:`c(\cdot,\cdot)` is the ground cost and the constraint
:math:`m \# \mu_s = \mu_t`  ensures that  :math:`\mu_s` is completely transported to :math:`\mu_t`.
This problem is particularly difficult to solve because of this constraint and
has been replaced in practice (on discrete distributions) by a
linear program easier to solve. It corresponds to the Kantorovitch formulation
where the Monge mapping :math:`m` is replaced by a joint distribution
(OT matrix expressed in the next section) (see :ref:`kantorovitch_solve`). 

From the optimization problem above we can see that there are two main aspects
to the OT solution that can be used in practical applications:

- The optimal value (Wasserstein distance): Measures similarity between distributions.
- The optimal mapping (Monge mapping, OT matrix): Finds correspondences between distributions.


In the first case, OT can be used to measure similarity between distributions
(or datasets), in this case the Wasserstein distance (the optimal value of the
problem) is used. In the second case one can be interested in the way the mass
is moved between the distributions (the mapping). This mapping can then be used
to transfer knowledge between distributions.


Wasserstein distance between distributions
""""""""""""""""""""""""""""""""""""""""""

OT is often used to measure similarity between distributions, especially
when they do not share the same support.  When the support between the
distributions is disjoint OT-based Wasserstein distances compare  favorably to
popular f-divergences including the popular Kullback-Leibler, Jensen-Shannon
divergences, and the Total Variation distance. What is particularly interesting
for data science applications is that one can compute meaningful sub-gradients
of the Wasserstein distance. For these reasons it became a very efficient tool
for machine learning applications that need to measure and optimize similarity
between empirical distributions.


Numerous contributions make use of this an approach is the machine learning (ML)
literature. For example OT was used for training `Generative
Adversarial Networks (GANs) <https://arxiv.org/pdf/1701.07875.pdf>`_
in order to overcome the vanishing gradient problem. It has also
been used to find `discriminant <https://arxiv.org/pdf/1608.08063.pdf>`_ or
`robust <https://arxiv.org/pdf/1901.08949.pdf>`_ subspaces for a dataset. The
Wasserstein distance has also been used to measure `similarity between word
embeddings of documents <http://proceedings.mlr.press/v37/kusnerb15.pdf>`_ or
between `signals
<https://www.math.ucdavis.edu/~saito/data/acha.read.s19/kolouri-etal_optimal-mass-transport.pdf>`_
or `spectra <https://arxiv.org/pdf/1609.09799.pdf>`_. 



OT for mapping estimation
"""""""""""""""""""""""""

A very interesting aspect of OT problem is the OT mapping in itself. When
computing optimal transport between discrete distributions one output is the OT
matrix that will provide you with correspondences between the samples in each
distributions.


This correspondence is estimated with respect to the OT criterion and is found
in a non-supervised way, which makes it very interesting on problems of transfer
between datasets. It has been used to perform
`color transfer between images <https://arxiv.org/pdf/1307.5551.pdf>`_ or in
the context of `domain adaptation <https://arxiv.org/pdf/1507.00504.pdf>`_.
More recent applications include the use of extension of OT (Gromov-Wasserstein)
to find correspondences between languages in `word embeddings
<https://arxiv.org/pdf/1809.00013.pdf>`_.


When to use POT
^^^^^^^^^^^^^^^


The main objective of POT is to provide OT solvers for the rapidly growing area
of OT in the context of machine learning. To this end we implement a number of
solvers that have been proposed in research papers. Doing so we aim to promote
reproducible research and foster novel developments.


One very important aspect of POT is its ability to be easily extended. For
instance we provide a very generic OT solver :any:`ot.optim.cg` that can solve
OT problems with any smooth/continuous regularization term making it
particularly practical for research purpose. Note that this generic solver has
been used to solve both graph Laplacian regularization OT and Gromov
Wasserstein [30]_.


.. note::

    POT is originally designed to solve OT problems with Numpy interface and
    is not yet compatible with Pytorch API. We are currently working on a torch
    submodule that will provide OT solvers and losses for the most common deep
    learning configurations.


When not to use POT
"""""""""""""""""""

While POT has to the best of our knowledge one of the most efficient exact OT
solvers, it has not been designed to handle large scale OT problems. For
instance the memory cost for an OT problem is always :math:`\mathcal{O}(n^2)` in
memory because the cost matrix has to be computed. The exact solver in of time
complexity :math:`\mathcal{O}(n^3\log(n))` and the Sinkhorn solver has been
proven to be nearly :math:`\mathcal{O}(n^2)` which is still too complex for very
large scale solvers.


If you need to solve OT with large number of samples, we recommend to use
entropic regularization and memory efficient implementation of Sinkhorn as
proposed in `GeomLoss <https://www.kernel-operations.io/geomloss/>`_. This
implementation is compatible with Pytorch and can handle large number of
samples. Another approach to estimate the Wasserstein distance for very large
number of sample is to use the trick from `Wasserstein GAN
<https://arxiv.org/pdf/1701.07875.pdf>`_ that solves the problem
in the dual with a neural network estimating the dual variable. Note that in this
case you are only solving an approximation of the Wasserstein distance because
the 1-Lipschitz constraint on the dual cannot be enforced exactly (approximated
through filter thresholding or regularization). Finally note that in order to
avoid solving large scale OT problems, a number of recent approached minimized
the expected Wasserstein distance on minibtaches that is different from the
Wasserstein but has better computational and
`statistical properties <https://arxiv.org/pdf/1910.04091.pdf>`_.


Optimal transport and Wasserstein distance
------------------------------------------

.. note::

    In POT, most functions that solve OT or regularized OT problems have two
    versions that return the OT matrix or the value of the optimal solution. For
    instance :any:`ot.emd` returns the OT matrix and :any:`ot.emd2` returns the
    Wassertsein distance. This approach has been implemented in practice for all
    solvers that return an OT matrix (even Gromov-Wasserstsein).

.. _kantorovitch_solve:

Solving optimal transport
^^^^^^^^^^^^^^^^^^^^^^^^^

The optimal transport problem between discrete distributions is often expressed
as

.. math::
    \gamma^* = arg\min_{\gamma \in \mathbb{R}_+^{m\times n}} \quad \sum_{i,j}\gamma_{i,j}M_{i,j}

    s.t. \gamma 1 = a; \gamma^T 1= b; \gamma\geq 0

where:

  - :math:`M\in\mathbb{R}_+^{m\times n}` is the metric cost matrix defining the cost to move mass from bin :math:`a_i` to bin :math:`b_j`.

  - :math:`a` and :math:`b` are histograms on the simplex (positive, sum to 1) that represent the weights of each samples in the source an target distributions.

Solving the linear program above can be done using the function :any:`ot.emd`
that will return the optimal transport matrix :math:`\gamma^*`:

.. code:: python

    # a and b are 1D histograms (sum to 1 and positive)
    # M is the ground cost matrix
    T = ot.emd(a, b, M)  # exact linear program

The method implemented for solving the OT problem is the network simplex. It is
implemented in C from [1]_. It has a complexity of :math:`O(n^3)` but the
solver is quite efficient and uses sparsity of the solution.



.. minigallery:: ot.emd
    :add-heading: Examples of use for :any:`ot.emd`
    :heading-level: "



Computing Wasserstein distance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The value of the OT solution is often more interesting than the OT matrix:

.. math::

    OT(a,b) = \min_{\gamma \in \mathbb{R}_+^{m\times n}} \quad \sum_{i,j}\gamma_{i,j}M_{i,j}

    s.t. \gamma 1 = a; \gamma^T 1= b; \gamma\geq 0


It can computed from an already estimated OT matrix with
:code:`np.sum(T*M)` or directly with the function :any:`ot.emd2`.

.. code:: python

    # a and b are 1D histograms (sum to 1 and positive)
    # M is the ground cost matrix
    W = ot.emd2(a, b, M)  # Wasserstein distance / EMD value

Note that the well known  `Wasserstein distance
<https://en.wikipedia.org/wiki/Wasserstein_metric>`_ between distributions a and
b is defined as


    .. math::

        W_p(a,b)=(\min_{\gamma \in \mathbb{R}_+^{m\times n}} \sum_{i,j}\gamma_{i,j}\|x_i-y_j\|_p)^\frac{1}{p}

        s.t. \gamma 1 = a; \gamma^T 1= b; \gamma\geq 0

This means that if you want to compute the :math:`W_2` you need to compute the
square root of :any:`ot.emd2` when providing
:code:`M = ot.dist(xs, xt)`, that uses the squared euclidean distance by default. Computing
the :math:`W_1` Wasserstein distance can be done directly with :any:`ot.emd2`
when providing :code:`M = ot.dist(xs, xt, metric='euclidean')` to use the Euclidean
distance.

.. minigallery:: ot.emd2
    :add-heading: Examples of use for :any:`ot.emd2`
    :heading-level: "


Special cases
^^^^^^^^^^^^^

Note that the OT problem and the corresponding Wasserstein distance can in some
special cases be computed very efficiently.

For instance when the samples are in 1D, then the OT problem can be solved in
:math:`O(n\log(n))` by using a simple sorting. In this case we provide the
function :any:`ot.emd_1d` and   :any:`ot.emd2_1d` to return respectively the OT
matrix and value. Note that since the solution is very sparse the :code:`sparse`
parameter of :any:`ot.emd_1d` allows for solving and returning the solution for
very large problems. Note that in order to compute directly the :math:`W_p`
Wasserstein distance in 1D we provide the function :any:`ot.wasserstein_1d` that
takes :code:`p` as a parameter.

Another special case for estimating OT and Monge mapping is between Gaussian
distributions. In this case there exists a close form solution given in Remark
2.29 in [15]_ and the Monge mapping is an affine function and can be
also computed from the covariances and means of the source and target
distributions. In the case when the finite sample dataset is supposed Gaussian,
we provide :any:`ot.da.OT_mapping_linear` that returns the parameters for the
Monge mapping.


Regularized Optimal Transport
-----------------------------

Recent developments have shown the interest of regularized OT both in terms of
computational and statistical properties.
We address in this section the regularized OT problems that can be expressed as

.. math::
    \gamma^* = arg\min_{\gamma \in \mathbb{R}_+^{m\times n}} \quad \sum_{i,j}\gamma_{i,j}M_{i,j} + \lambda\Omega(\gamma)

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
procedures such as the well known Sinkhorn algorithm [2]_ or L-BFGS (see
:any:`ot.smooth` ). Next it makes the problem
strictly convex meaning that there will be a unique solution. Finally the
solution of the resulting optimization problem can be expressed as:

.. math::

    \gamma_\lambda^*=\text{diag}(u)K\text{diag}(v)

where :math:`u` and :math:`v` are vectors and :math:`K=\exp(-M/\lambda)` where
the :math:`\exp` is taken component-wise. In order to solve the optimization
problem, one can use an alternative projection algorithm called Sinkhorn-Knopp
that can be very efficient for large values of regularization.

The Sinkhorn-Knopp algorithm is implemented in :any:`ot.sinkhorn` and
:any:`ot.sinkhorn2` that return respectively the OT matrix and the value of the
linear term. Note that the regularization parameter :math:`\lambda` in the
equation above is given to those functions with the parameter :code:`reg`.

    >>> import ot
    >>> a = [.5, .5]
    >>> b = [.5, .5]
    >>> M = [[0., 1.], [1., 0.]]
    >>> ot.sinkhorn(a, b, M, 1)
    array([[ 0.36552929,  0.13447071],
        [ 0.13447071,  0.36552929]])

More details about the algorithms used are given in the following note.

.. note::
    The main function to solve entropic regularized OT is :any:`ot.sinkhorn`.
    This function is a wrapper and the parameter :code:`method` allows you to select
    the actual algorithm used to solve the problem:

    + :code:`method='sinkhorn'` calls :any:`ot.bregman.sinkhorn_knopp`  the
      classic algorithm [2]_.
    + :code:`method='sinkhorn_log'` calls :any:`ot.bregman.sinkhorn_log`  the
      sinkhorn algorithm in log space [2]_ that is more stable but can be
      slower in numpy since `logsumexp` is not implmemented in parallel. 
      It is the recommended solver for applications that requires
      differentiability with a  small number of iterations.
    + :code:`method='sinkhorn_stabilized'` calls :any:`ot.bregman.sinkhorn_stabilized`  the
      log stabilized version of the algorithm [9]_.
    + :code:`method='sinkhorn_epsilon_scaling'` calls
      :any:`ot.bregman.sinkhorn_epsilon_scaling`  the epsilon scaling version
      of the algorithm [9]_.
    + :code:`method='greenkhorn'` calls :any:`ot.bregman.greenkhorn`  the
      greedy Sinkhorn version of the algorithm [22]_.
    + :code:`method='screenkhorn'` calls :any:`ot.bregman.screenkhorn`  the
      screening sinkhorn version of the algorithm [26]_.

    In addition to all those variants of Sinkhorn, we have another
    implementation solving the problem in the smooth dual or semi-dual in
    :any:`ot.smooth`. This solver uses the :any:`scipy.optimize.minimize`
    function to solve the smooth problem with :code:`L-BFGS-B` algorithm. Tu use
    this solver, use functions :any:`ot.smooth.smooth_ot_dual` or
    :any:`ot.smooth.smooth_ot_semi_dual` with parameter :code:`reg_type='kl'` to
    choose entropic/Kullbach Leibler regularization.

    **Choosing a Sinkhorn solver**

    By default and when using a regularization parameter that is not too small
    the default Sinkhorn solver should be enough. If you need to use a small
    regularization to get sharper OT matrices, you should use the
    :any:`ot.bregman.sinkhorn_stabilized` solver that will avoid numerical
    errors. This last solver can be very slow in practice and might not even
    converge to a reasonable OT matrix in a finite time. This is why
    :any:`ot.bregman.sinkhorn_epsilon_scaling` that relie on iterating the value
    of the regularization (and using warm start) sometimes leads to better
    solutions. Note that the greedy version of the Sinkhorn
    :any:`ot.bregman.greenkhorn` can also lead to a speedup and the screening
    version of the Sinkhorn :any:`ot.bregman.screenkhorn` aim a providing a
    fast approximation of the Sinkhorn problem. For use of GPU and gradient
    computation with small number of iterations we strongly recommend the 
    :any:`ot.bregman.sinkhorn_log` solver that will no need to check for 
    numerical problems.



Recently Genevay et al. [23]_ introduced the Sinkhorn divergence that build from entropic
regularization to compute fast and differentiable geometric divergence between
empirical distributions.  Note that we provide a function that computes directly
(with no need to precompute the :code:`M` matrix)
the Sinkhorn divergence for empirical distributions in
:any:`ot.bregman.empirical_sinkhorn_divergence`. Similarly one can compute the
OT matrix and loss for empirical distributions with respectively
:any:`ot.bregman.empirical_sinkhorn` and :any:`ot.bregman.empirical_sinkhorn2`.


Finally note that we also provide in :any:`ot.stochastic` several implementation
of stochastic solvers for entropic regularized OT [18]_ [19]_.  Those pure Python
implementations are not optimized for speed but provide a robust implementation
of algorithms in [18]_ [19]_.


.. minigallery:: ot.sinkhorn
    :add-heading: Examples of use for :any:`ot.sinkhorn`
    :heading-level: "

.. minigallery:: ot.sinkhorn2
    :add-heading: Examples of use for :any:`ot.sinkhorn2`
    :heading-level: "


Other regularizations
^^^^^^^^^^^^^^^^^^^^^

While entropic OT is the most common and favored in practice, there exists other
kinds of regularizations. We provide in POT two specific solvers for other
regularization terms, namely quadratic regularization and group Lasso
regularization. But we also provide in :any:`ot.optim`  two generic solvers
that allows solving any smooth regularization in practice.

Quadratic regularization
""""""""""""""""""""""""

The first general regularization term we can solve is the quadratic
regularization of the form

.. math::
    \Omega(\gamma)=\sum_{i,j} \gamma_{i,j}^2

This regularization term has an effect similar to entropic regularization by
densifying the OT matrix, yet it keeps some sort of sparsity that is lost with
entropic regularization as soon as :math:`\lambda>0` [17]_. This problem can be
solved with POT using solvers from :any:`ot.smooth`, more specifically
functions :any:`ot.smooth.smooth_ot_dual` or
:any:`ot.smooth.smooth_ot_semi_dual` with parameter :code:`reg_type='l2'` to
choose the quadratic regularization.

.. minigallery:: ot.smooth.smooth_ot_dual ot.smooth.smooth_ot_semi_dual ot.optim.cg
    :add-heading: Examples of use of quadratic regularization
    :heading-level: "


Group Lasso regularization
""""""""""""""""""""""""""

Another regularization that has been used in recent years [5]_ is the group Lasso
regularization

.. math::
    \Omega(\gamma)=\sum_{j,G\in\mathcal{G}} \|\gamma_{G,j}\|_q^p

where :math:`\mathcal{G}` contains non-overlapping groups of lines in the OT
matrix. This regularization proposed in [5]_ promotes sparsity at the group level and for
instance will force target samples to get mass from a small number of groups.
Note that the exact OT solution is already sparse so this regularization does
not make sense if it is not combined with entropic regularization. Depending on
the choice of :code:`p` and :code:`q`, the problem can be solved with different
approaches.  When :code:`q=1` and :code:`p<1` the problem is non-convex but can
be solved using an efficient majoration minimization approach with
:any:`ot.sinkhorn_lpl1_mm`. When :code:`q=2` and :code:`p=1` we recover the
convex group lasso and we provide a solver using generalized conditional
gradient algorithm [7]_ in function :any:`ot.da.sinkhorn_l1l2_gl`.

.. minigallery::  ot.da.SinkhornLpl1Transport  ot.da.SinkhornL1l2Transport ot.da.sinkhorn_l1l2_gl ot.da.sinkhorn_lpl1_mm
    :add-heading: Examples of group Lasso regularization
    :heading-level: "


Generic solvers
"""""""""""""""

Finally we propose in POT generic solvers that can be used to solve any
regularization as long as you can provide a function computing the
regularization and a function computing its gradient (or sub-gradient).

In order to solve

.. math::
    \gamma^* = arg\min_\gamma \quad \sum_{i,j}\gamma_{i,j}M_{i,j} + \lambda\Omega(\gamma)

        s.t. \gamma 1 = a; \gamma^T 1= b; \gamma\geq 0

you can use function :any:`ot.optim.cg` that will use a conditional gradient as
proposed in [6]_ . You need to provide the regularization function as parameter
``f`` and its gradient as parameter  ``df``. Note that the conditional gradient relies on
iterative solving of a linearization of the problem using the exact
:any:`ot.emd` so it can be quite slow in practice. However, being an interior point
algorithm, it always returns a transport matrix that does not violates the marginals.

Another generic solver is proposed to solve the problem:

.. math::
    \gamma^* = arg\min_\gamma \quad \sum_{i,j}\gamma_{i,j}M_{i,j}+ \lambda_e\Omega_e(\gamma) + \lambda\Omega(\gamma)

        s.t. \gamma 1 = a; \gamma^T 1= b; \gamma\geq 0

where :math:`\Omega_e` is the entropic regularization. In this case we use a
generalized conditional gradient [7]_ implemented in :any:`ot.optim.gcg`  that
does not linearize the entropic term but
relies on :any:`ot.sinkhorn` for its iterations.

.. minigallery:: ot.optim.cg ot.optim.gcg
    :add-heading: Examples of the generic solvers
    :heading-level: "


Wasserstein Barycenters
-----------------------

A Wasserstein barycenter is a distribution that minimizes its Wasserstein
distance with respect to other distributions [16]_. It corresponds to minimizing the
following problem by searching a distribution :math:`\mu` such that

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
solver :meth:`ot.lp.barycenter` that relies on generic LP solvers. By default the
function uses :any:`scipy.optimize.linprog`, but more efficient LP solvers from
cvxopt can be also used by changing parameter :code:`solver`. Note that this problem
requires to solve a very large linear program and can be very slow in
practice.

Similarly to the OT problem, OT barycenters can be computed in the regularized
case. When entropic regularization is used, the problem can be solved with a
generalization of the Sinkhorn algorithm based on Bregman projections [3]_. This
algorithm is provided in function :any:`ot.bregman.barycenter` also available as
:any:`ot.barycenter`. In this case, the algorithm scales better to large
distributions and relies only on matrix multiplications that can be performed in
parallel.

In addition to the speedup brought by regularization, one can also greatly
accelerate the estimation of Wasserstein barycenter when the support has a
separable structure [21]_. In the case of 2D images for instance one can replace
the matrix vector production in the Bregman projections by convolution
operators. We provide an implementation of this algorithm in function
:any:`ot.bregman.convolutional_barycenter2d`.



.. minigallery:: ot.lp.barycenter ot.bregman.barycenter ot.barycenter
    :add-heading: Examples of Wasserstein and regularized Wasserstein barycenters
    :heading-level: "

.. minigallery:: ot.bregman.convolutional_barycenter2d
    :add-heading:  An example of convolutional barycenter (:any:`ot.bregman.convolutional_barycenter2d`) computation
    :heading-level: "



Barycenters with free support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Estimating the Wasserstein barycenter with free support but fixed weights
corresponds to solving the following optimization problem:

.. math::
    \min_{\{x_i\}} \quad \sum_{k} w_kW(\mu,\mu_k)

    s.t. \quad \mu=\sum_{i=1}^n a_i\delta_{x_i}

We provide a solver based on [20]_ in
:any:`ot.lp.free_support_barycenter`. This function minimize the problem and
return a locally optimal support :math:`\{x_i\}` for uniform or given weights
:math:`a`.

.. minigallery:: ot.lp.free_support_barycenter
    :add-heading: Examples of free support barycenter estimation
    :heading-level: "



Monge mapping and Domain adaptation
-----------------------------------

The original transport problem investigated by Gaspard Monge was seeking for a
mapping function that maps (or transports) between a source and target
distribution but that minimizes the transport loss. The existence and uniqueness of this
optimal mapping is still an open problem in the general case but has been proven
for smooth distributions by Brenier in his eponym `theorem
<https://who.rocq.inria.fr/Jean-David.Benamou/demiheure.pdf>`__. We provide in
:any:`ot.da` several solvers for smooth Monge mapping estimation and domain
adaptation from discrete distributions.

Monge Mapping estimation
^^^^^^^^^^^^^^^^^^^^^^^^

We now discuss several approaches that are implemented in POT to estimate or
approximate a Monge mapping from finite distributions.

First note that when the source and target distributions are supposed to be Gaussian
distributions, there exists a close form solution for the mapping and its an
affine function [14]_ of the form :math:`T(x)=Ax+b` . In this case we provide the function
:any:`ot.da.OT_mapping_linear` that returns the operator :math:`A` and vector
:math:`b`. Note that if the number of samples is too small there is a parameter
:code:`reg` that provides a regularization for the covariance matrix estimation.

For a more general mapping estimation we also provide the barycentric mapping
proposed in [6]_. It is implemented in the class :any:`ot.da.EMDTransport` and
other transport-based classes in :any:`ot.da` . Those classes are discussed more
in the following but follow an interface similar to scikit-learn classes. Finally a
method proposed in [8]_ that estimates a continuous mapping approximating the
barycentric mapping is provided in :any:`ot.da.joint_OT_mapping_linear` for
linear mapping and :any:`ot.da.joint_OT_mapping_kernel` for non-linear mapping.

.. minigallery:: ot.da.joint_OT_mapping_linear ot.da.joint_OT_mapping_linear ot.da.OT_mapping_linear
    :add-heading: Examples of Monge mapping estimation
    :heading-level: "


Domain adaptation classes
^^^^^^^^^^^^^^^^^^^^^^^^^

The use of OT for domain adaptation (OTDA) has been first proposed in [5]_ that also
introduced the group Lasso regularization. The main idea of OTDA is to estimate
a mapping of the samples between source and target distributions which allows to
transport labeled source samples onto the target distribution with no labels.

We provide several classes based on :any:`ot.da.BaseTransport` that provide
several OT and mapping estimations. The interface of those classes is similar to
classifiers in scikit-learn. At initialization, several parameters such as
regularization parameter value can be set. Then one needs to estimate the
mapping with function :any:`ot.da.BaseTransport.fit`. Finally one can map the
samples from source to target with  :any:`ot.da.BaseTransport.transform` and
from target to source with :any:`ot.da.BaseTransport.inverse_transform`.

Here is an example for class :any:`ot.da.EMDTransport`:

.. code::

    ot_emd = ot.da.EMDTransport()
    ot_emd.fit(Xs=Xs, Xt=Xt)
    Xs_mapped = ot_emd.transform(Xs=Xs)

A list of the provided implementation is given in the following note.

.. note::

    Here is a list of the OT mapping classes inheriting from
    :any:`ot.da.BaseTransport`

    * :any:`ot.da.EMDTransport`: Barycentric mapping with EMD transport
    * :any:`ot.da.SinkhornTransport`: Barycentric mapping with Sinkhorn transport
    * :any:`ot.da.SinkhornL1l2Transport`: Barycentric mapping with Sinkhorn +
      group Lasso regularization [5]_
    * :any:`ot.da.SinkhornLpl1Transport`: Barycentric mapping with Sinkhorn +
      non convex group Lasso regularization [5]_
    * :any:`ot.da.LinearTransport`: Linear mapping estimation  between Gaussians
      [14]_
    * :any:`ot.da.MappingTransport`: Nonlinear mapping estimation [8]_


.. minigallery:: ot.da.SinkhornTransport ot.da.LinearTransport 
    :add-heading: Examples of the use of OTDA classes
    :heading-level: "


Other applications
------------------

We discuss in the following several OT related problems and tools that has been
proposed in the OT and machine learning community.

Wasserstein Discriminant Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Wasserstein Discriminant Analysis [11]_ is a generalization of `Fisher Linear Discriminant
Analysis <https://en.wikipedia.org/wiki/Linear_discriminant_analysis>`__ that
allows discrimination between classes that are not linearly separable. It
consists in finding a linear projector optimizing the following criterion

.. math::
    P = \text{arg}\min_P \frac{\sum_i OT_e(\mu_i\#P,\mu_i\#P)}{\sum_{i,j\neq i}
    OT_e(\mu_i\#P,\mu_j\#P)}

where :math:`\#` is the push-forward operator, :math:`OT_e` is the entropic OT
loss and :math:`\mu_i` is the
distribution of samples from class :math:`i`.  :math:`P` is also constrained to
be in the Stiefel manifold. WDA can be solved in POT using function
:any:`ot.dr.wda`. It requires to have installed :code:`pymanopt` and
:code:`autograd` for manifold optimization and automatic differentiation
respectively. Note that we also provide the Fisher discriminant estimator in
:any:`ot.dr.fda` for easy comparison.

.. warning::

    Note that due to the hard dependency on  :code:`pymanopt` and
    :code:`autograd`, :any:`ot.dr` is not imported by default. If you want to
    use it you have to specifically import it with :code:`import ot.dr` .

.. minigallery:: ot.dr.wda
    :add-heading: Examples of the use of WDA
    :heading-level: "



Unbalanced optimal transport
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Unbalanced OT is a relaxation of the entropy regularized OT problem where the violation of
the constraint on the marginals is added to the objective of the optimization
problem. The unbalanced OT metric between two unbalanced histograms a and b is defined as [25]_ [10]_:

.. math::
    W_u(a, b) = \min_\gamma \quad \sum_{i,j}\gamma_{i,j}M_{i,j} + reg\cdot\Omega(\gamma) + reg_m KL(\gamma 1, a) + reg_m KL(\gamma^T 1, b)

    s.t. \quad  \gamma\geq 0


where KL is the Kullback-Leibler divergence. This formulation allows for
computing approximate mapping between distributions that do not have the same
amount of mass. Interestingly the problem can be solved with a generalization of
the Bregman projections algorithm [10]_. We provide a solver for unbalanced OT
in :any:`ot.unbalanced`. Computing the optimal transport
plan or the transport cost is similar to the balanced case. The Sinkhorn-Knopp
algorithm is implemented in :any:`ot.sinkhorn_unbalanced` and :any:`ot.sinkhorn_unbalanced2`
that return respectively the OT matrix and the value of the
linear term.

.. note::
    The main function to solve entropic regularized UOT is :any:`ot.sinkhorn_unbalanced`.
    This function is a wrapper and the parameter :code:`method` helps you select
    the actual algorithm used to solve the problem:

    + :code:`method='sinkhorn'` calls :any:`ot.unbalanced.sinkhorn_knopp_unbalanced`
      the generalized Sinkhorn algorithm [25]_ [10]_.
    + :code:`method='sinkhorn_stabilized'` calls :any:`ot.unbalanced.sinkhorn_stabilized_unbalanced`
      the log stabilized version of the algorithm [10]_.


.. minigallery:: ot.sinkhorn_unbalanced ot.sinkhorn_unbalanced2  ot.unbalanced.sinkhorn_unbalanced
    :add-heading: Examples of Unbalanced OT
    :heading-level: "


Unbalanced Barycenters
^^^^^^^^^^^^^^^^^^^^^^

As with balanced distributions, we can define a barycenter of a set of
histograms with different masses as a Fréchet Mean:

    .. math::
        \min_{\mu} \quad \sum_{k} w_kW_u(\mu,\mu_k)

where :math:`W_u` is the unbalanced Wasserstein metric defined above. This problem
can also be solved using generalized version of Sinkhorn's algorithm and it is
implemented the main function :any:`ot.barycenter_unbalanced`.


.. note::
    The main function to compute UOT barycenters is :any:`ot.barycenter_unbalanced`.
    This function is a wrapper and the parameter :code:`method` helps you select
    the actual algorithm used to solve the problem:

    + :code:`method='sinkhorn'` calls :meth:`ot.unbalanced.barycenter_unbalanced_sinkhorn_unbalanced`
      the generalized Sinkhorn algorithm [10]_.
    + :code:`method='sinkhorn_stabilized'` calls :any:`ot.unbalanced.barycenter_unbalanced_stabilized`
      the log stabilized version of the algorithm [10]_.


.. minigallery:: ot.barycenter_unbalanced  ot.unbalanced.barycenter_unbalanced
    :add-heading: Examples of Unbalanced OT barycenters
    :heading-level: "



Partial optimal transport
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Partial OT is a variant of the optimal transport problem when only a fixed amount of mass m
is to be transported. The partial OT metric between two histograms a and b is defined as [28]_:

.. math::
    \gamma = \arg\min_\gamma <\gamma,M>_F

    s.t.
        \gamma\geq 0 \\
        \gamma 1 \leq a\\
        \gamma^T 1 \leq b\\
        1^T \gamma^T 1 = m \leq \min\{\|a\|_1, \|b\|_1\}
             

Interestingly the problem can be casted into a regular OT problem by adding reservoir points
in which the surplus mass is sent [29]_. We provide a solver for partial OT
in :any:`ot.partial`. The exact resolution of the problem is computed in :any:`ot.partial.partial_wasserstein`
and :any:`ot.partial.partial_wasserstein2` that return respectively the OT matrix and the value of the
linear term. The entropic solution of the problem is computed in :any:`ot.partial.entropic_partial_wasserstein` 
(see [3]_).

The partial Gromov-Wasserstein formulation of the problem 

.. math::
    GW = \min_\gamma \sum_{i,j,k,l} L(C1_{i,k},C2_{j,l})*\gamma_{i,j}*\gamma_{k,l}

    s.t.
        \gamma\geq 0 \\
        \gamma 1 \leq a\\
        \gamma^T 1 \leq b\\
        1^T \gamma^T 1 = m \leq \min\{\|a\|_1, \|b\|_1\}

is computed in :any:`ot.partial.partial_gromov_wasserstein` and in 
:any:`ot.partial.entropic_partial_gromov_wasserstein` when considering the entropic 
regularization of the problem.


.. minigallery:: ot.partial.partial_wasserstein ot.partial.partial_gromov_wasserstein
    :add-heading: Examples of Partial OT
    :heading-level: "




Gromov-Wasserstein
^^^^^^^^^^^^^^^^^^

Gromov Wasserstein (GW) is a generalization of OT to distributions that do not lie in
the same space [13]_. In this case one cannot compute distance between samples
from the two distributions. [13]_ proposed instead to realign the metric spaces
by computing a transport between distance matrices. The Gromow Wasserstein
alignment between two distributions can be expressed as the one minimizing:

.. math::
    GW = \min_\gamma \sum_{i,j,k,l} L(C1_{i,k},C2_{j,l})*\gamma_{i,j}*\gamma_{k,l}

    s.t. \gamma 1 = a; \gamma^T 1= b; \gamma\geq 0

where ::math:`C1` is the distance matrix between samples in the source
distribution and :math:`C2` the one between samples in the target,
:math:`L(C1_{i,k},C2_{j,l})` is a measure of similarity between
:math:`C1_{i,k}` and :math:`C2_{j,l}` often chosen as
:math:`L(C1_{i,k},C2_{j,l})=\|C1_{i,k}-C2_{j,l}\|^2`. The optimization problem
above is a non-convex quadratic program but we provide a solver that finds a
local minimum using conditional gradient in :any:`ot.gromov.gromov_wasserstein`.
There also exists an entropic regularized variant of GW that has been proposed in
[12]_ and we provide an implementation of their algorithm in
:any:`ot.gromov.entropic_gromov_wasserstein`.


.. minigallery:: ot.gromov.gromov_wasserstein ot.gromov.entropic_gromov_wasserstein  ot.gromov.fused_gromov_wasserstein ot.gromov.gromov_wasserstein2    
    :add-heading: Examples of computation of GW, regularized G and FGW
    :heading-level: "


Note that similarly to Wasserstein distance GW allows for the definition of GW
barycenters that can be expressed as

.. math::
    \min_{C\geq 0} \quad \sum_{k} w_k GW(C,Ck)

where :math:`Ck` is the distance matrix between samples in distribution
:math:`k`. Note that interestingly the barycenter is defined as a symmetric
positive matrix. We provide a block coordinate optimization procedure in
:any:`ot.gromov.gromov_barycenters` and
:any:`ot.gromov.entropic_gromov_barycenters` for non-regularized and regularized
barycenters respectively.

Finally note that recently a fusion between Wasserstein and GW, coined Fused
Gromov-Wasserstein (FGW) has been proposed [24]_.
It allows to compute a similarity between objects that are only partly in
the same space. As such it can be used to measure similarity between labeled
graphs for instance and also provide computable barycenters.
The implementations of FGW and FGW barycenter is provided in functions
:any:`ot.gromov.fused_gromov_wasserstein` and :any:`ot.gromov.fgw_barycenters`.


.. minigallery:: ot.gromov.gromov_barycenters ot.gromov.fgw_barycenters   
    :add-heading: Examples of GW, regularized G and FGW barycenters
    :heading-level: "



Solving OT with Multiple backends on CPU/GPU
--------------------------------------------

.. _backends_section:

Since version 0.8, POT provides a backend that allows to code solvers
independently from the type of the input arrays. The idea is to provide the user
with a package that works seamlessly and returns a solution for instance as a
Pytorch tensors when the function has Pytorch tensors as input. 


How it works
^^^^^^^^^^^^

The aim of the backend is to use the same function independently of the type of
the input arrays.

For instance when executing the following code

.. code:: python

    # a and b are 1D histograms (sum to 1 and positive)
    # M is the ground cost matrix
    T = ot.emd(a, b, M)  # exact linear program
    w = ot.emd2(a, b, M)  # Wasserstein computation

the functions  :any:`ot.emd` and :any:`ot.emd2` can take inputs of the type
:any:`numpy.array`, :any:`torch.tensor` or  :any:`jax.numpy.array`. The output of
the function will be the same type as the inputs and on the same device. When
possible all computations are done on the same device and also when possible the
output will be differentiable with respect to the input of the function.

GPU acceleration
^^^^^^^^^^^^^^^^

The backends provide automatic computations/compatibility on GPU for most of the
POT functions.
Note that all solvers relying on the exact OT solver en C++ will need to solve the
problem on CPU which can incur some memory copy overhead and be far from optimal
when all other computations are done on GPU. They will still work on array on
GPU since the copy is done automatically.

Some of the functions that rely on the exact C++ solver are:

- :any:`ot.emd`, :any:`ot.emd2`
- :any:`ot.gromov_wasserstein`, :any:`ot.gromov_wasserstein2`
- :any:`ot.optim.cg`

List of compatible Backends
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- `Numpy <https://numpy.org/>`_ (all functions and solvers)
- `Pytorch <https://pytorch.org/>`_ (all outputs differentiable w.r.t. inputs)
- `Jax <https://github.com/google/jax>`_ (Some functions are differentiable some require a wrapper)
- `Tensorflow <https://www.tensorflow.org/>`_ (all outputs differentiable w.r.t. inputs)
- `Cupy <https://cupy.dev/>`_ (no differentiation, GPU only)


List of compatible modules
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This list will get longer for new releases and will hopefully disappear when POT
become fully implemented with the backend.

- :any:`ot.bregman`
- :any:`ot.gromov` (some functions use CPU only solvers with copy overhead)
- :any:`ot.optim` (some functions use CPU only solvers with copy overhead)
- :any:`ot.sliced`
- :any:`ot.utils` (partial)


FAQ
---

1. **How to solve a discrete optimal transport problem ?**

    The solver for discrete OT is the function :py:mod:`ot.emd` that returns
    the OT transport matrix. If you want to solve a regularized OT you can
    use :py:mod:`ot.sinkhorn`.


    Here is a simple use case:

    .. code:: python

       # a and b are 1D histograms (sum to 1 and positive)
       # M is the ground cost matrix
       T = ot.emd(a, b, M)  # exact linear program
       T_reg = ot.sinkhorn(a, b, M, reg)  # entropic regularized OT

    More detailed examples can be seen on this example:
    :doc:`auto_examples/plot_OT_2D_samples`


2. **pip install POT fails with error : ImportError: No module named Cython.Build**

    As discussed shortly in the README file. POT<0.8 requires to have :code:`numpy`
    and :code:`cython` installed to build. This corner case is not yet handled
    by :code:`pip` and for now you need to install both library prior to
    installing POT.

    Note that this problem do not occur when using conda-forge since the packages
    there are pre-compiled.

    See `Issue #59 <https://github.com/rflamary/POT/issues/59>`__ for more
    details.

3. **Why is Sinkhorn slower than EMD ?**

    This might come from the choice of the regularization term. The speed of
    convergence of Sinkhorn depends directly on this term [22]_. When the
    regularization gets very small the problem tries to approximate the exact OT
    which leads to slow convergence in addition to numerical problems. In other
    words, for large regularization Sinkhorn will be very fast to converge, for
    small regularization (when you need an OT matrix close to the true OT), it
    might be quicker to use the EMD solver.

    Also note that the numpy implementation of Sinkhorn can use parallel
    computation depending on the configuration of your system, yet very important
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

.. [14] Knott, M. and Smith, C. S. (1984). `On the optimal mapping of
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

.. [23] Genevay, A., Peyré, G., Cuturi, M., `Learning Generative Models with
    Sinkhorn Divergences <https://arxiv.org/abs/1706.00292>`__, Proceedings
    of the Twenty-First International Conference on Artficial Intelligence
    and Statistics, (AISTATS) 21, 2018

.. [24] Vayer, T., Chapel, L., Flamary, R., Tavenard, R. and Courty, N.
    (2019). `Optimal Transport for structured data with application on
    graphs <http://proceedings.mlr.press/v97/titouan19a.html>`__ Proceedings
    of the 36th International Conference on Machine Learning (ICML).

.. [25] Frogner C., Zhang C., Mobahi H., Araya-Polo M., Poggio T. :
    Learning with a Wasserstein Loss,  Advances in Neural Information
    Processing Systems (NIPS) 2015
    
.. [26] Alaya M. Z., Bérar M., Gasso G., Rakotomamonjy A. (2019). Screening Sinkhorn 
	Algorithm for Regularized Optimal Transport <https://papers.nips.cc/paper/9386-screening-sinkhorn-algorithm-for-regularized-optimal-transport>, 
	Advances in Neural Information Processing Systems 33 (NeurIPS).
	
.. [28] Caffarelli, L. A., McCann, R. J. (2020). Free boundaries in optimal transport and 
	Monge-Ampere obstacle problems <http://www.math.toronto.edu/~mccann/papers/annals2010.pdf>, 
	Annals of mathematics, 673-730.

.. [29] Chapel, L., Alaya, M., Gasso, G. (2019). Partial Gromov-Wasserstein with 
	Applications on Positive-Unlabeled Learning <https://arxiv.org/abs/2002.08276>, 
	arXiv preprint arXiv:2002.08276.

.. [30] Flamary, Rémi, et al. "Optimal transport with Laplacian regularization:
    Applications to domain adaptation and shape matching." NIPS Workshop on Optimal
    Transport and Machine Learning OTML. 2014.
