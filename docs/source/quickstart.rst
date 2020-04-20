
Quick start guide
=================

In the following we provide some pointers about which functions and classes
to use for different problems related to optimal transport (OT) and machine
learning. We refer when we can to concrete examples in the documentation that
are also available as notebooks on the POT Github.

This document is not a tutorial on numerical optimal transport. For this we strongly
recommend to read the very nice book [15]_ .


Optimal transport and Wasserstein distance
------------------------------------------

.. note::
    In POT, most functions that solve OT or regularized OT problems have two
    versions that return the OT matrix or the value of the optimal solution. For
    instance :any:`ot.emd` return the OT matrix and :any:`ot.emd2` return the
    Wassertsein distance. This approach has been implemented in practice for all
    solvers that return an OT matrix (even Gromov-Wasserstsein)

Solving optimal transport
^^^^^^^^^^^^^^^^^^^^^^^^^

The optimal transport problem between discrete distributions is often expressed
as

.. math::
    \gamma^* = arg\min_\gamma \quad \sum_{i,j}\gamma_{i,j}M_{i,j}

    s.t. \gamma 1 = a; \gamma^T 1= b; \gamma\geq 0

where :

- :math:`M\in\mathbb{R}_+^{m\times n}` is the metric cost matrix defining the cost to move mass from bin :math:`a_i` to bin :math:`b_j`.
- :math:`a` and :math:`b` are histograms on the simplex (positive, sum to 1) that represent the
weights of each samples in the source an target distributions.

Solving the linear program above can be done using the function :any:`ot.emd`
that will return the optimal transport matrix :math:`\gamma^*`:

.. code:: python

    # a,b are 1D histograms (sum to 1 and positive)
    # M is the ground cost matrix
    T=ot.emd(a,b,M) # exact linear program

The method implemented for solving the OT problem is the network simplex, it is
implemented in C from  [1]_. It has a complexity of :math:`O(n^3)` but the
solver is quite efficient and uses sparsity of the solution.

.. hint::
    Examples of use for :any:`ot.emd` are available in :

    - :any:`auto_examples/plot_OT_2D_samples`
    - :any:`auto_examples/plot_OT_1D`
    - :any:`auto_examples/plot_OT_L1_vs_L2`


Computing Wasserstein distance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The value of the OT solution is often more of interest than the OT matrix :

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
    An example of use for :any:`ot.emd2` is available in :

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
very large problems. Note that in order to compute directly the :math:`W_p`
Wasserstein distance in 1D we provide the function :any:`ot.wasserstein_1d` that
takes :code:`p` as a parameter.

Another special case for estimating OT and Monge mapping is between Gaussian
distributions. In this case there exists a close form solution given in Remark
2.29 in [15]_ and the Monge mapping is an affine function and can be
also computed from the covariances and means of the source and target
distributions. In the case when the finite sample dataset is supposed gaussian, we provide
:any:`ot.da.OT_mapping_linear` that returns the parameters for the Monge
mapping.


Regularized Optimal Transport
-----------------------------

Recent developments have shown the interest of regularized OT both in terms of
computational and statistical properties.
We address in this section the regularized OT problems that can be expressed as

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
procedures such as the well known Sinkhorn algorithm [2]_ or L-BFGS (see
:any:`ot.smooth` ). Next it makes the problem
strictly convex meaning that there will be a unique solution. Finally the
solution of the resulting optimization problem can be expressed as:

.. math::

    \gamma_\lambda^*=\text{diag}(u)K\text{diag}(v)

where :math:`u` and :math:`v` are vectors and :math:`K=\exp(-M/\lambda)` where
the :math:`\exp` is taken component-wise. In order to solve the optimization
problem, on can use an alternative projection algorithm called Sinkhorn-Knopp that can be very
efficient for large values if regularization.

The Sinkhorn-Knopp algorithm is implemented in :any:`ot.sinkhorn` and
:any:`ot.sinkhorn2` that return respectively the OT matrix and the value of the
linear term. Note that the regularization parameter :math:`\lambda` in the
equation above is given to those functions with the parameter :code:`reg`.

    >>> import ot
    >>> a=[.5,.5]
    >>> b=[.5,.5]
    >>> M=[[0.,1.],[1.,0.]]
    >>> ot.sinkhorn(a,b,M,1)
    array([[ 0.36552929,  0.13447071],
        [ 0.13447071,  0.36552929]])

More details about the algorithms used are given in the following note.

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
    function to solve the smooth problem with :code:`L-BFGS-B` algorithm. Tu use
    this solver, use functions :any:`ot.smooth.smooth_ot_dual` or
    :any:`ot.smooth.smooth_ot_semi_dual` with parameter :code:`reg_type='kl'` to
    choose entropic/Kullbach Leibler regularization.


Recently [23]_ introduced the sinkhorn divergence that build from entropic
regularization to compute fast and differentiable geometric divergence between
empirical distributions.  Note that we provide a function that compute directly
(with no need to pre compute the :code:`M` matrix)
the sinkhorn divergence for empirical distributions in
:any:`ot.bregman.empirical_sinkhorn_divergence`. Similarly one can compute the
OT matrix and loss for empirical distributions with respectively
:any:`ot.bregman.empirical_sinkhorn` and :any:`ot.bregman.empirical_sinkhorn2`.


Finally note that we also provide in :any:`ot.stochastic` several implementation
of stochastic solvers for entropic regularized OT [18]_ [19]_.  Those pure Python
implementations are not optimized for speed but provide a roust implementation
of algorithms in [18]_ [19]_.

.. hint::
    Examples of use for :any:`ot.sinkhorn` are available in :

    - :any:`auto_examples/plot_OT_2D_samples`
    - :any:`auto_examples/plot_OT_1D`
    - :any:`auto_examples/plot_OT_1D_smooth`
    - :any:`auto_examples/plot_stochastic`


Other regularization
^^^^^^^^^^^^^^^^^^^^

While entropic OT is the most common and favored in practice, there exist other
kind of regularization. We provide in POT two specific solvers for other
regularization terms, namely quadratic regularization and group lasso
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
entropic regularization as soon as :math:`\lambda>0` [17]_. This problem can be
solved with POT using solvers from :any:`ot.smooth`, more specifically
functions :any:`ot.smooth.smooth_ot_dual` or
:any:`ot.smooth.smooth_ot_semi_dual` with parameter :code:`reg_type='l2'` to
choose the quadratic regularization.

.. hint::
    Examples of quadratic regularization are available in :

    - :any:`auto_examples/plot_OT_1D_smooth`
    - :any:`auto_examples/plot_optim_OTreg`



Group Lasso regularization
""""""""""""""""""""""""""

Another regularization that has been used in recent years [5]_  is the group lasso
regularization

.. math::
    \Omega(\gamma)=\sum_{j,G\in\mathcal{G}} \|\gamma_{G,j}\|_q^p

where :math:`\mathcal{G}` contains non overlapping groups of lines in the OT
matrix. This regularization proposed in [5]_ will promote sparsity at the group level and for
instance will force target samples to get mass from a small number of groups.
Note that the exact OT solution is already sparse so this regularization does
not make sens if it is not combined with entropic regularization. Depending on
the choice of :code:`p` and :code:`q`, the problem can be solved with different
approaches.  When :code:`q=1` and :code:`p<1` the problem is non convex but can
be solved using an efficient majoration minimization approach with
:any:`ot.sinkhorn_lpl1_mm`. When :code:`q=2` and :code:`p=1` we recover the
convex group lasso and we provide a solver using generalized conditional
gradient algorithm [7]_ in function
:any:`ot.da.sinkhorn_l1l2_gl`.

.. hint::
    Examples of group Lasso regularization are available in :

    - :any:`auto_examples/plot_otda_classes`
    - :any:`auto_examples/plot_otda_d2`


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
:any:`ot.emd` so it can be  slow in practice. But, being an interior point
algorithm,  it always returns a
transport matrix that does not violates the marginals.

Another generic solver is proposed to solve the problem

.. math::
    \gamma^* = arg\min_\gamma \quad \sum_{i,j}\gamma_{i,j}M_{i,j}+ \lambda_e\Omega_e(\gamma) + \lambda\Omega(\gamma)

        s.t. \gamma 1 = a; \gamma^T 1= b; \gamma\geq 0

where :math:`\Omega_e` is the entropic regularization. In this case we use a
generalized conditional gradient [7]_ implemented in :any:`ot.optim.gcg`  that
does not linearize the entropic term but
relies on :any:`ot.sinkhorn` for its iterations.

.. hint::
    An example of generic solvers are available in :

    - :any:`auto_examples/plot_optim_OTreg`


Wasserstein Barycenters
-----------------------

A Wasserstein barycenter is a distribution that minimize its Wasserstein
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
solver :any:`ot.lp.barycenter` that rely on generic LP solvers. By default the
function uses :any:`scipy.optimize.linprog`, but more efficient LP solvers from
cvxopt can be also used by changing parameter :code:`solver`. Note that this problem
requires to solve a very large linear program and can be very slow in
practice.

Similarly to the OT problem, OT barycenters can be computed in the regularized
case. When using entropic regularization is used, the problem can be solved with a
generalization of the sinkhorn algorithm based on bregman projections [3]_. This
algorithm is provided in function :any:`ot.bregman.barycenter` also available as
:any:`ot.barycenter`. In this case, the algorithm scales better to large
distributions and rely only on matrix multiplications that can be performed in
parallel.

In addition to the speedup brought by regularization, one can also greatly
accelerate the estimation of Wasserstein barycenter when the support has a
separable structure [21]_. In the case of 2D images for instance one can replace
the matrix vector production in the Bregman projections by convolution
operators. We provide an implementation of this algorithm in function
:any:`ot.bregman.convolutional_barycenter2d`.

.. hint::
    Examples of Wasserstein (:any:`ot.lp.barycenter`) and regularized Wasserstein
    barycenter (:any:`ot.bregman.barycenter`) computation are available in :

    - :any:`auto_examples/plot_barycenter_1D`
    - :any:`auto_examples/plot_barycenter_lp_vs_entropic`

    An example of convolutional barycenter
    (:any:`ot.bregman.convolutional_barycenter2d`) computation
    for 2D images is available
    in :

    - :any:`auto_examples/plot_convolutional_barycenter`



Barycenters with free support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Estimating the Wasserstein barycenter with free support but fixed weights
corresponds to  solving the following optimization problem:

.. math::
    \min_{\{x_i\}} \quad \sum_{k} w_kW(\mu,\mu_k)

    s.t. \quad \mu=\sum_{i=1}^n a_i\delta_{x_i}

We provide a solver based on [20]_ in
:any:`ot.lp.free_support_barycenter`. This function minimize the problem and
return a locally optimal support :math:`\{x_i\}` for uniform or given weights
:math:`a`.

 .. hint::

    An example of the free support barycenter estimation is available
    in :

    - :any:`auto_examples/plot_free_support_barycenter`




Monge mapping and Domain adaptation
-----------------------------------

The original transport problem investigated by Gaspard Monge  was seeking for a
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
:any:`ot.da.OT_mapping_linear` that return the operator :math:`A` and vector
:math:`b`. Note that if the number of samples is too small there is a parameter
:code:`reg` that provide a regularization for the covariance matrix estimation.

For a more general mapping estimation we also provide the barycentric mapping
proposed in [6]_ . It is implemented in the class :any:`ot.da.EMDTransport` and
other transport based classes in :any:`ot.da` . Those classes are discussed more
in the following but follow an interface similar to sklearn classes. Finally a
method proposed in [8]_ that estimates a continuous mapping approximating the
barycentric mapping is provided in :any:`ot.da.joint_OT_mapping_linear` for
linear mapping and :any:`ot.da.joint_OT_mapping_kernel` for non linear mapping.

 .. hint::

    An example of the linear Monge mapping estimation is available
    in :

    - :any:`auto_examples/plot_otda_linear_mapping`

Domain adaptation classes
^^^^^^^^^^^^^^^^^^^^^^^^^

The use of OT for domain adaptation (OTDA) has been first proposed in [5]_ that also
introduced the group Lasso regularization. The main idea of OTDA is to estimate
a mapping of the samples between source and target distributions which allows to
transport labeled source samples onto the target distribution with no labels.

We provide several classes based on :any:`ot.da.BaseTransport` that provide
several OT and mapping estimations. The interface of those classes is similar to
classifiers in sklearn toolbox. At initialization, several parameters such as
 regularization parameter value can be set. Then one needs to estimate the
mapping with function :any:`ot.da.BaseTransport.fit`. Finally one can map the
samples from source to target with  :any:`ot.da.BaseTransport.transform` and
from target to source with :any:`ot.da.BaseTransport.inverse_transform`.

Here is
an example for class :any:`ot.da.EMDTransport` :

.. code::

    ot_emd = ot.da.EMDTransport()
    ot_emd.fit(Xs=Xs, Xt=Xt)

    Mapped_Xs= ot_emd.transform(Xs=Xs)

A list of the provided implementation is given in the following note.

.. note::

    Here is a list of the OT mapping classes inheriting from
    :any:`ot.da.BaseTransport`

    * :any:`ot.da.EMDTransport` : Barycentric mapping with EMD transport
    * :any:`ot.da.SinkhornTransport` : Barycentric mapping with Sinkhorn transport
    * :any:`ot.da.SinkhornL1l2Transport` : Barycentric mapping with Sinkhorn +
      group Lasso regularization [5]_
    * :any:`ot.da.SinkhornLpl1Transport` : Barycentric mapping with Sinkhorn +
      non convex group Lasso regularization [5]_
    * :any:`ot.da.LinearTransport` : Linear mapping estimation  between Gaussians
      [14]_
    * :any:`ot.da.MappingTransport` : Nonlinear mapping estimation [8]_

.. hint::

    Example of the use of OTDA classes are available in :

    - :any:`auto_examples/plot_otda_color_images`
    - :any:`auto_examples/plot_otda_mapping`
    - :any:`auto_examples/plot_otda_mapping_colors_images`
    - :any:`auto_examples/plot_otda_semi_supervised`

Other applications
------------------

We discuss in the following several OT related problems and tools that has been
proposed in the OT and machine learning community.

Wasserstein Discriminant Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Wasserstein Discriminant Analysis [11]_ is a generalization of `Fisher Linear Discriminant
Analysis <https://en.wikipedia.org/wiki/Linear_discriminant_analysis>`__ that
allows discrimination between classes that are not linearly separable. It
consist in finding a linear projector optimizing the following criterion

.. math::
    P = \text{arg}\min_P \frac{\sum_i OT_e(\mu_i\#P,\mu_i\#P)}{\sum_{i,j\neq i}
    OT_e(\mu_i\#P,\mu_j\#P)}

where :math:`\#` is the push-forward operator, :math:`OT_e` is the entropic OT
loss  and :math:`\mu_i` is the
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

.. hint::

    An example of the use of WDA is available in :

    - :any:`auto_examples/plot_WDA`


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


.. hint::

    Examples of the use of :any:`ot.sinkhorn_unbalanced` are available in :

    - :any:`auto_examples/plot_UOT_1D`


Unbalanced Barycenters
^^^^^^^^^^^^^^^^^^^^^^

As with balanced distributions, we can define a barycenter of a set of
histograms with different masses as a Fréchet Mean:

    .. math::
        \min_{\mu} \quad \sum_{k} w_kW_u(\mu,\mu_k)

Where :math:`W_u` is the unbalanced Wasserstein metric defined above. This problem
can also be solved using generalized version of Sinkhorn's algorithm and it is
implemented the main function :any:`ot.barycenter_unbalanced`.


.. note::
    The main function to compute UOT barycenters is :any:`ot.barycenter_unbalanced`.
    This function is a wrapper and the parameter :code:`method` help you select
    the actual algorithm used to solve the problem:

    + :code:`method='sinkhorn'` calls :any:`ot.unbalanced.barycenter_unbalanced_sinkhorn_unbalanced`
      the generalized Sinkhorn algorithm [10]_.
    + :code:`method='sinkhorn_stabilized'` calls :any:`ot.unbalanced.barycenter_unbalanced_stabilized`
      the log stabilized version of the algorithm [10]_.


.. hint::

      Examples of the use of :any:`ot.barycenter_unbalanced` are available in :

      - :any:`auto_examples/plot_UOT_barycenter_1D`


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


.. hint::

    Examples of the use of :any:`ot.partial` are available in :

    - :any:`auto_examples/plot_partial`



Gromov-Wasserstein
^^^^^^^^^^^^^^^^^^

Gromov Wasserstein (GW) is a generalization of OT to distributions that do not lie in
the same space [13]_. In this case one cannot compute distance between samples
from the two distributions. [13]_ proposed instead to realign the metric spaces
by computing a transport between distance matrices. The Gromow Wasserstein
alignement between two distributions can be expressed as the one minimizing:

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
Gromov-Wasserstein (FGW) has been proposed
in [24]_. It allows to compute a similarity between objects that are only partly in
the same space. As such it can be used to measure similarity between labeled
graphs for instance and also provide computable barycenters.
The implementations of FGW and FGW barycenter is provided in functions
:any:`ot.gromov.fused_gromov_wasserstein` and :any:`ot.gromov.fgw_barycenters`.

.. hint::

    Examples of computation of GW, regularized G and FGW are available in :

    - :any:`auto_examples/plot_gromov`
    - :any:`auto_examples/plot_fgw`

    Examples of GW, regularized GW and FGW barycenters are available in :

    - :any:`auto_examples/plot_gromov_barycenter`
    - :any:`auto_examples/plot_barycenter_fgw`


GPU acceleration
^^^^^^^^^^^^^^^^

We provide several implementation of our OT solvers in :any:`ot.gpu`. Those
implementations use the :code:`cupy` toolbox that obviously need to be installed.


.. note::

    Several implementations of POT functions (mainly those relying on linear
    algebra) have been implemented in :any:`ot.gpu`. Here is a short list on the
    main entries:

    -  :any:`ot.gpu.dist` : computation of distance matrix
    -  :any:`ot.gpu.sinkhorn` : computation of sinkhorn
    -  :any:`ot.gpu.sinkhorn_lpl1_mm` : computation of sinkhorn + group lasso

Note that while the :any:`ot.gpu` module has been designed to be compatible with
POT,  calling its function with :any:`numpy`  arrays will incur a large overhead due to
the memory copy of the array on GPU prior to computation and conversion of the
array after computation. To avoid this overhead, we provide functions
:any:`ot.gpu.to_gpu` and :any:`ot.gpu.to_np` that perform the conversion
explicitly.


.. warning::
    Note that due to the hard dependency on  :code:`cupy`, :any:`ot.gpu` is not
    imported by default. If you want to
    use it you have to specifically import it with :code:`import ot.gpu` .


FAQ
---



1. **How to solve a discrete optimal transport problem ?**

    The solver for discrete OT is the function :py:mod:`ot.emd` that returns
    the OT transport matrix. If you want to solve a regularized OT you can
    use :py:mod:`ot.sinkhorn`.


    Here is a simple use case:

    .. code:: python

       # a,b are 1D histograms (sum to 1 and positive)
       # M is the ground cost matrix
       T=ot.emd(a,b,M) # exact linear program
       T_reg=ot.sinkhorn(a,b,M,reg) # entropic regularized OT

    More detailed examples can be seen on this example:
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

.. [23] Aude, G., Peyré, G., Cuturi, M., `Learning Generative Models with
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

.. [27] Redko I., Courty N., Flamary R., Tuia D. (2019). Optimal Transport for Multi-source 
	Domain Adaptation under Target Shift <http://proceedings.mlr.press/v89/redko19a.html>, 
	Proceedings of the Twenty-Second International Conference on Artificial Intelligence 
	and Statistics (AISTATS) 22, 2019.
	
.. [28] Caffarelli, L. A., McCann, R. J. (2020). Free boundaries in optimal transport and 
	Monge-Ampere obstacle problems <http://www.math.toronto.edu/~mccann/papers/annals2010.pdf>, 
	Annals of mathematics, 673-730.

.. [29] Chapel, L., Alaya, M., Gasso, G. (2019). Partial Gromov-Wasserstein with 
	Applications on Positive-Unlabeled Learning <https://arxiv.org/abs/2002.08276>, 
	arXiv preprint arXiv:2002.08276.
