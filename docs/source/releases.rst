Releases
========

0.6
---

*July 2019*

This is the first official stable release of POT and this means a jump
to 0.6! The library has been used in the wild for a while now and we
have reached a state where a lot of fundamental OT solvers are available
and tested. It has been quite stable in the last months but kept the
beta flag in its Pypi classifiers until now.

Note that this release will be the last one supporting officially Python
2.7 (See https://python3statement.org/ for more reasons). For next
release we will keep the travis tests for Python 2 but will make them
non necessary for merge in 2020.

The features are never complete in a toolbox designed for solving
mathematical problems and research but with the new contributions we now
implement algorithms and solvers from 24 scientific papers (listed in
the README.md file). New features include a direct implementation of the
`empirical Sinkhorn
divergence <all.html#ot.bregman.empirical_sinkhorn_divergence>`__
, a new efficient (Cython implementation) solver for `EMD in
1D <all.html#ot.lp.emd_1d>`__ and
corresponding `Wasserstein
1D <all.html#ot.lp.wasserstein_1d>`__.
We now also have implementations for `Unbalanced
OT <auto_examples/plot_UOT_1D.html>`__
and a solver for `Unbalanced OT
barycenters <auto_examples/plot_UOT_barycenter_1D.html>`__.
A new variant of Gromov-Wasserstein divergence called `Fused
Gromov-Wasserstein <all.html?highlight=fused_#ot.gromov.fused_gromov_wasserstein>`__
has been also contributed with exemples of use on `structured
data <auto_examples/plot_fgw.html>`__
and computing `barycenters of labeld
graphs <auto_examples/plot_barycenter_fgw.html>`__.

A lot of work has been done on the documentation with several new
examples corresponding to the new features and a lot of corrections for
the docstrings. But the most visible change is a new `quick start
guide <quickstart.html>`__ for POT
that gives several pointers about which function or classes allow to
solve which specific OT problem. When possible a link is provided to
relevant examples.

We will also provide with this release some pre-compiled Python wheels
for Linux 64bit on github and pip. This will simplify the install
process that before required a C compiler and numpy/cython already
installed.

Finally we would like to acknowledge and thank the numerous contributors
of POT that has helped in the past build the foundation and are still
contributing to bring new features and solvers to the library.

Features
^^^^^^^^

-  Add compiled manylinux 64bits wheels to pip releases (PR #91)
-  Add quick start guide (PR #88)
-  Make doctest work on travis (PR #90)
-  Update documentation (PR #79, PR #84)
-  Solver for EMD in 1D (PR #89)
-  Solvers for regularized unbalanced OT (PR #87, PR#99)
-  Solver for Fused Gromov-Wasserstein (PR #86)
-  Add empirical Sinkhorn and empirical Sinkhorn divergences (PR #80)

Closed issues
^^^^^^^^^^^^^

-  Issue #59 fail when using "pip install POT" (new details in doc+
   hopefully wheels)
-  Issue #85 Cannot run gpu modules
-  Issue #75 Greenkhorn do not return log (solved in PR #76)
-  Issue #82 Gromov-Wasserstein fails when the cost matrices are
   slightly different
-  Issue #72 Macosx build problem

0.5.0
-----

*Sep 2018*

POT is 2 years old! This release brings numerous new features to the
toolbox as listed below but also several bug correction.

| Among the new features, we can highlight a `non-regularized
  Gromov-Wasserstein
  solver <auto_examples/plot_gromov.html>`__,
  a new `greedy variant of
  sinkhorn <all.html#ot.bregman.greenkhorn>`__,
| `non-regularized <all.html#ot.lp.barycenter>`__,
  `convolutional
  (2D) <auto_examples/plot_convolutional_barycenter.html>`__
  and `free
  support <auto_examples/plot_free_support_barycenter.html>`__
  Wasserstein barycenters and
  `smooth <https://github.com/rflamary/POT/blob/prV0.5/notebooks/plot_OT_1D_smooth.html>`__
  and
  `stochastic <all.html#ot.stochastic.sgd_entropic_regularization>`__
  implementation of entropic OT.

POT 0.5 also comes with a rewriting of ot.gpu using the cupy framework
instead of the unmaintained cudamat. Note that while we tried to keed
changes to the minimum, the OTDA classes were deprecated. If you are
happy with the cudamat implementation, we recommend you stay with stable
release 0.4 for now.

The code quality has also improved with 92% code coverage in tests that
is now printed to the log in the Travis builds. The documentation has
also been greatly improved with new modules and examples/notebooks.

This new release is so full of new stuff and corrections thanks to the
old and new POT contributors (you can see the list in the
`readme <https://github.com/rflamary/POT/blob/master/README.md>`__).

Features
^^^^^^^^

-  Add non regularized Gromov-Wasserstein solver (PR #41)
-  Linear OT mapping between empirical distributions and 90% test
   coverage (PR #42)
-  Add log parameter in class EMDTransport and SinkhornLpL1Transport (PR
   #44)
-  Add Markdown format for Pipy (PR #45)
-  Test for Python 3.5 and 3.6 on Travis (PR #46)
-  Non regularized Wasserstein barycenter with scipy linear solver
   and/or cvxopt (PR #47)
-  Rename dataset functions to be more sklearn compliant (PR #49)
-  Smooth and sparse Optimal transport implementation with entropic and
   quadratic regularization (PR #50)
-  Stochastic OT in the dual and semi-dual (PR #52 and PR #62)
-  Free support barycenters (PR #56)
-  Speed-up Sinkhorn function (PR #57 and PR #58)
-  Add convolutional Wassersein barycenters for 2D images (PR #64)
-  Add Greedy Sinkhorn variant (Greenkhorn) (PR #66)
-  Big ot.gpu update with cupy implementation (instead of un-maintained
   cudamat) (PR #67)

Deprecation
^^^^^^^^^^^

Deprecated OTDA Classes were removed from ot.da and ot.gpu for version
0.5 (PR #48 and PR #67). The deprecation message has been for a year
here since 0.4 and it is time to pull the plug.

Closed issues
^^^^^^^^^^^^^

-  Issue #35 : remove import plot from ot/\ **init**.py (See PR #41)
-  Issue #43 : Unusable parameter log for EMDTransport (See PR #44)
-  Issue #55 : UnicodeDecodeError: 'ascii' while installing with pip

0.4
---

*15 Sep 2017*

This release contains a lot of contribution from new contributors.

Features
^^^^^^^^

-  Automatic notebooks and doc update (PR #27)
-  Add gromov Wasserstein solver and Gromov Barycenters (PR #23)
-  emd and emd2 can now return dual variables and have max\_iter (PR #29
   and PR #25)
-  New domain adaptation classes compatible with scikit-learn (PR #22)
-  Proper tests with pytest on travis (PR #19)
-  PEP 8 tests (PR #13)

Closed issues
^^^^^^^^^^^^^

-  emd convergence problem du to fixed max iterations (#24)
-  Semi supervised DA error (#26)

0.3.1
-----

*11 Jul 2017*

-  Correct bug in emd on windows

0.3
---

*7 Jul 2017*

-  emd\* and sinkhorn\* are now performed in parallel for multiple
   target distributions
-  emd and sinkhorn are for OT matrix computation
-  emd2 and sinkhorn2 are for OT loss computation
-  new notebooks for emd computation and Wasserstein Discriminant
   Analysis
-  relocate notebooks
-  update documentation
-  clean\_zeros(a,b,M) for removimg zeros in sparse distributions
-  GPU implementations for sinkhorn and group lasso regularization

V0.2
----

*7 Apr 2017*

-  New dimensionality reduction method (WDA)
-  Efficient method emd2 returns only tarnsport (in paralell if several
   histograms given)

0.1.11
------

*5 Jan 2017*

-  Add sphinx gallery for better documentation
-  Small efficiency tweak in sinkhorn
-  Add simple tic() toc() functions for timing

0.1.10
------

*7 Nov 2016* \* numerical stabilization for sinkhorn (log domain and
epsilon scaling)

0.1.9
-----

*4 Nov 2016*

-  Update classes and examples for domain adaptation
-  Joint OT matrix and mapping estimation

0.1.7
-----

*31 Oct 2016*

-  Original Domain adaptation classes

0.1.3
-----

-  pipy works

First pre-release
-----------------

*28 Oct 2016*

It provides the following solvers: \* OT solver for the linear program/
Earth Movers Distance. \* Entropic regularization OT solver with
Sinkhorn Knopp Algorithm. \* Bregman projections for Wasserstein
barycenter [3] and unmixing. \* Optimal transport for domain adaptation
with group lasso regularization \* Conditional gradient and Generalized
conditional gradient for regularized OT.

Some demonstrations (both in Python and Jupyter Notebook format) are
available in the examples folder.
