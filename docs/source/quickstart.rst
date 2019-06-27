
Quick start guide
=================

In the following we provide some pointers about which functions and classes 
to use for different problems related to optimal transport (OT).


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
 

Regularized Optimal Transport
-----------------------------

Recent developments have shown the interest of regularized OT both in terms of
computational and statistical properties.

We address in this section the regularized OT problem that can be expressed as

.. math::
    \gamma^* = arg\min_\gamma <\gamma,M>_F + reg*\Omega(\gamma)

    s.t. \gamma 1 = a

            \gamma^T 1= b

            \gamma\geq 0
where :

- :math:`M\in\mathbb{R}_+^{m\times n}` is the metric cost matrix defining the cost to move mass from bin :math:`a_i` to bin :math:`b_j`.
- :math:`a` and :math:`b` are histograms (positive, sum to 1) that represent the weights of each samples in the source an target distributions.
- :math:`\Omega` is the regularization term.

We disvuss in the following specific algorithms 



Entropic regularized OT
^^^^^^^^^^^^^^^^^^^^^^^


Other regularization
^^^^^^^^^^^^^^^^^^^^

Stochastic gradient decsent
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Wasserstein Barycenters
-----------------------

Monge mapping and Domain adaptation with Optimal transport
----------------------------------------


Other applications
------------------


GPU acceleration
----------------



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
    

2. **Compute a Wasserstein distance**


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