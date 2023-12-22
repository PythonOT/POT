# POT: Python Optimal Transport

[![PyPI version](https://badge.fury.io/py/POT.svg)](https://badge.fury.io/py/POT)
[![Anaconda Cloud](https://anaconda.org/conda-forge/pot/badges/version.svg)](https://anaconda.org/conda-forge/pot)
[![Build Status](https://github.com/PythonOT/POT/actions/workflows/build_tests.yml/badge.svg)](https://github.com/PythonOT/POT/actions)
[![Codecov Status](https://codecov.io/gh/PythonOT/POT/branch/master/graph/badge.svg)](https://codecov.io/gh/PythonOT/POT)
[![Downloads](https://static.pepy.tech/badge/pot)](https://pepy.tech/project/pot)
[![Anaconda downloads](https://anaconda.org/conda-forge/pot/badges/downloads.svg)](https://anaconda.org/conda-forge/pot)
[![License](https://anaconda.org/conda-forge/pot/badges/license.svg)](https://github.com/PythonOT/POT/blob/master/LICENSE)

This open source Python library provide several solvers for optimization
problems related to Optimal Transport for signal, image processing and machine
learning.

Website and documentation: [https://PythonOT.github.io/](https://PythonOT.github.io/)

Source Code (MIT): [https://github.com/PythonOT/POT](https://github.com/PythonOT/POT)

POT provides the following generic OT solvers (links to examples):

* [OT Network Simplex solver](https://pythonot.github.io/auto_examples/plot_OT_1D.html) for the linear program/ Earth Movers Distance [1] .
* [Conditional gradient](https://pythonot.github.io/auto_examples/plot_optim_OTreg.html) [6] and [Generalized conditional gradient](https://pythonot.github.io/auto_examples/plot_optim_OTreg.html) for regularized OT [7].
* Entropic regularization OT solver with [Sinkhorn Knopp
  Algorithm](https://pythonot.github.io/auto_examples/plot_OT_1D.html) [2] ,
  stabilized version [9] [10] [34], lazy CPU/GPU solver from geomloss [60] [61], greedy Sinkhorn [22] and [Screening
  Sinkhorn [26]
  ](https://pythonot.github.io/auto_examples/plot_screenkhorn_1D.html).
* Bregman projections for [Wasserstein barycenter](https://pythonot.github.io/auto_examples/barycenters/plot_barycenter_lp_vs_entropic.html) [3], [convolutional barycenter](https://pythonot.github.io/auto_examples/barycenters/plot_convolutional_barycenter.html) [21]  and unmixing [4].
* Sinkhorn divergence [23] and entropic regularization OT from empirical data.
* Debiased Sinkhorn barycenters [Sinkhorn divergence barycenter](https://pythonot.github.io/auto_examples/barycenters/plot_debiased_barycenter.html) [37]
* [Smooth optimal transport solvers](https://pythonot.github.io/auto_examples/plot_OT_1D_smooth.html) (dual and semi-dual) for KL and squared L2 regularizations [17].
* Weak OT solver between empirical distributions [39]
* Non regularized [Wasserstein barycenters [16] ](https://pythonot.github.io/auto_examples/barycenters/plot_barycenter_lp_vs_entropic.html) with LP solver (only small scale).
* [Gromov-Wasserstein distances](https://pythonot.github.io/auto_examples/gromov/plot_gromov.html) and [GW barycenters](https://pythonot.github.io/auto_examples/gromov/plot_gromov_barycenter.html)  (exact [13] and regularized [12,51]), differentiable using gradients from Graph Dictionary Learning [38]
 * [Fused-Gromov-Wasserstein distances solver](https://pythonot.github.io/auto_examples/gromov/plot_fgw.html#sphx-glr-auto-examples-plot-fgw-py) and [FGW barycenters](https://pythonot.github.io/auto_examples/gromov/plot_barycenter_fgw.html) (exact [24] and regularized [12,51]).
* [Stochastic
  solver](https://pythonot.github.io/auto_examples/others/plot_stochastic.html) and
  [differentiable losses](https://pythonot.github.io/auto_examples/backends/plot_stoch_continuous_ot_pytorch.html) for
  Large-scale Optimal Transport (semi-dual problem [18] and dual problem [19])
* [Sampled solver of Gromov Wasserstein](https://pythonot.github.io/auto_examples/gromov/plot_gromov.html) for large-scale problem with any loss functions [33]
* Non regularized [free support Wasserstein barycenters](https://pythonot.github.io/auto_examples/barycenters/plot_free_support_barycenter.html) [20].
* [One dimensional Unbalanced OT](https://pythonot.github.io/auto_examples/unbalanced-partial/plot_UOT_1D.html) with KL relaxation and [barycenter](https://pythonot.github.io/auto_examples/unbalanced-partial/plot_UOT_barycenter_1D.html) [10, 25]. Also [exact unbalanced OT](https://pythonot.github.io/auto_examples/unbalanced-partial/plot_unbalanced_ot.html) with KL and quadratic regularization and the [regularization path of UOT](https://pythonot.github.io/auto_examples/unbalanced-partial/plot_regpath.html) [41]
* [Partial Wasserstein and Gromov-Wasserstein](https://pythonot.github.io/auto_examples/unbalanced-partial/plot_partial_wass_and_gromov.html) (exact [29] and entropic [3]
  formulations).
* [Sliced Wasserstein](https://pythonot.github.io/auto_examples/sliced-wasserstein/plot_variance.html) [31, 32] and Max-sliced Wasserstein [35] that can be used for gradient flows [36].
* [Wasserstein distance on the circle](https://pythonot.github.io/auto_examples/plot_compute_wasserstein_circle.html) [44, 45]
* [Spherical Sliced Wasserstein](https://pythonot.github.io/auto_examples/sliced-wasserstein/plot_variance_ssw.html) [46]
* [Graph Dictionary Learning solvers](https://pythonot.github.io/auto_examples/gromov/plot_gromov_wasserstein_dictionary_learning.html) [38].
* [Semi-relaxed (Fused) Gromov-Wasserstein divergences](https://pythonot.github.io/auto_examples/gromov/plot_semirelaxed_fgw.html) (exact and regularized [48]).
* [Efficient Discrete Multi Marginal Optimal Transport Regularization](https://pythonot.github.io/auto_examples/others/plot_demd_gradient_minimize.html) [50].
* [Several backends](https://pythonot.github.io/quickstart.html#solving-ot-with-multiple-backends) for easy use of POT with  [Pytorch](https://pytorch.org/)/[jax](https://github.com/google/jax)/[Numpy](https://numpy.org/)/[Cupy](https://cupy.dev/)/[Tensorflow](https://www.tensorflow.org/) arrays.
* Smooth Strongly Convex Nearest Brenier Potentials [58], with an extension to bounding potentials using [59].

POT provides the following Machine Learning related solvers:

* [Optimal transport for domain
  adaptation](https://pythonot.github.io/auto_examples/domain-adaptation/plot_otda_classes.html)
  with [group lasso regularization](https://pythonot.github.io/auto_examples/domain-adaptation/plot_otda_classes.html),   [Laplacian regularization](https://pythonot.github.io/auto_examples/domain-adaptation/plot_otda_laplacian.html) [5] [30] and [semi
  supervised setting](https://pythonot.github.io/auto_examples/domain-adaptation/plot_otda_semi_supervised.html).
* [Linear OT mapping](https://pythonot.github.io/auto_examples/domain-adaptation/plot_otda_linear_mapping.html) [14] and [Joint OT mapping estimation](https://pythonot.github.io/auto_examples/domain-adaptation/plot_otda_mapping.html) [8].
* [Wasserstein Discriminant Analysis](https://pythonot.github.io/auto_examples/others/plot_WDA.html) [11] (requires autograd + pymanopt).
* [JCPOT algorithm for multi-source domain adaptation with target shift](https://pythonot.github.io/auto_examples/domain-adaptation/plot_otda_jcpot.html) [27].
* [Graph Neural Network OT layers TFGW](https://pythonot.github.io/auto_examples/gromov/plot_gnn_TFGW.html) [52] and TW (OT-GNN) [53] 

Some other examples are available in the  [documentation](https://pythonot.github.io/auto_examples/index.html).

#### Using and citing the toolbox

If you use this toolbox in your research and find it useful, please cite POT
using the following reference from our [JMLR paper](https://jmlr.org/papers/v22/20-451.html):

    Rémi Flamary, Nicolas Courty, Alexandre Gramfort, Mokhtar Z. Alaya, Aurélie Boisbunon, Stanislas Chambon, Laetitia Chapel, Adrien Corenflos, Kilian Fatras, Nemo Fournier, Léo Gautheron, Nathalie T.H. Gayraud, Hicham Janati, Alain Rakotomamonjy, Ievgen Redko, Antoine Rolet, Antony Schutz, Vivien Seguy, Danica J. Sutherland, Romain Tavenard, Alexander Tong, Titouan Vayer,
    POT Python Optimal Transport library,
    Journal of Machine Learning Research, 22(78):1−8, 2021.
    Website: https://pythonot.github.io/

In Bibtex format:

```bibtex
@article{flamary2021pot,
  author  = {R{\'e}mi Flamary and Nicolas Courty and Alexandre Gramfort and Mokhtar Z. Alaya and Aur{\'e}lie Boisbunon and Stanislas Chambon and Laetitia Chapel and Adrien Corenflos and Kilian Fatras and Nemo Fournier and L{\'e}o Gautheron and Nathalie T.H. Gayraud and Hicham Janati and Alain Rakotomamonjy and Ievgen Redko and Antoine Rolet and Antony Schutz and Vivien Seguy and Danica J. Sutherland and Romain Tavenard and Alexander Tong and Titouan Vayer},
  title   = {POT: Python Optimal Transport},
  journal = {Journal of Machine Learning Research},
  year    = {2021},
  volume  = {22},
  number  = {78},
  pages   = {1-8},
  url     = {http://jmlr.org/papers/v22/20-451.html}
}
```

## Installation

The library has been tested on Linux, MacOSX and Windows. It requires a C++ compiler for building/installing the EMD solver and relies on the following Python modules:

- Numpy (>=1.16)
- Scipy (>=1.0)
- Cython (>=0.23) (build only, not necessary when installing from pip or conda)

#### Pip installation


You can install the toolbox through PyPI with:

```console
pip install POT
```

or get the very latest version by running:

```console
pip install -U https://github.com/PythonOT/POT/archive/master.zip # with --user for user install (no root)
```

#### Anaconda installation with conda-forge

If you use the Anaconda python distribution, POT is available in [conda-forge](https://conda-forge.org). To install it and the required dependencies:

```console
conda install -c conda-forge pot
```

#### Post installation check
After a correct installation, you should be able to import the module without errors:

```python
import ot
```

Note that for easier access the module is named `ot` instead of `pot`.


### Dependencies

Some sub-modules require additional dependencies which are discussed below

* **ot.dr** (Wasserstein dimensionality reduction) depends on autograd and pymanopt that can be installed with:

```shell
pip install pymanopt autograd
```


## Examples

### Short examples

* Import the toolbox

```python
import ot
```

* Compute Wasserstein distances

```python
# a,b are 1D histograms (sum to 1 and positive)
# M is the ground cost matrix
Wd = ot.emd2(a, b, M) # exact linear program
Wd_reg = ot.sinkhorn2(a, b, M, reg) # entropic regularized OT
# if b is a matrix compute all distances to a and return a vector
```

* Compute OT matrix

```python
# a,b are 1D histograms (sum to 1 and positive)
# M is the ground cost matrix
T = ot.emd(a, b, M) # exact linear program
T_reg = ot.sinkhorn(a, b, M, reg) # entropic regularized OT
```

* Compute Wasserstein barycenter

```python
# A is a n*d matrix containing d  1D histograms
# M is the ground cost matrix
ba = ot.barycenter(A, M, reg) # reg is regularization parameter
```

### Examples and Notebooks

The examples folder contain several examples and use case for the library. The full documentation with examples and output is available on [https://PythonOT.github.io/](https://PythonOT.github.io/).


## Acknowledgements

This toolbox has been created by

* [Rémi Flamary](https://remi.flamary.com/)
* [Nicolas Courty](http://people.irisa.fr/Nicolas.Courty/)

It is currently maintained by 

* [Rémi Flamary](https://remi.flamary.com/)
* [Cédric Vincent-Cuaz](https://cedricvincentcuaz.github.io/)

The numerous contributors to this library are listed [here](CONTRIBUTORS.md).

POT has benefited from the financing or manpower from the following partners:

<img src="https://pythonot.github.io/master/_static/images/logo_anr.jpg" alt="ANR" style="height:60px;"/><img src="https://pythonot.github.io/master/_static/images/logo_cnrs.jpg" alt="CNRS" style="height:60px;"/><img src="https://pythonot.github.io/master/_static/images/logo_3ia.jpg" alt="3IA" style="height:60px;"/><img src="https://pythonot.github.io/master/_static/images/logo_hiparis.png" alt="Hi!PARIS" style="height:60px;"/>



## Contributions and code of conduct

Every contribution is welcome and should respect the [contribution guidelines](https://pythonot.github.io/master/contributing.html). Each member of the project is expected to follow the [code of conduct](https://pythonot.github.io/master/code_of_conduct.html).

## Support

You can ask questions and join the development discussion:

* On the POT [slack channel](https://pot-toolbox.slack.com)
* On the POT [gitter channel](https://gitter.im/PythonOT/community)
* On the POT [mailing list](https://mail.python.org/mm3/mailman3/lists/pot.python.org/)

You can also post bug reports and feature requests in Github issues. Make sure to read our [guidelines](.github/CONTRIBUTING.md) first.

## References

[1] Bonneel, N., Van De Panne, M., Paris, S., & Heidrich, W. (2011, December). [Displacement interpolation using Lagrangian mass transport](https://people.csail.mit.edu/sparis/publi/2011/sigasia/Bonneel_11_Displacement_Interpolation.pdf). In ACM Transactions on Graphics (TOG) (Vol. 30, No. 6, p. 158). ACM.

[2] Cuturi, M. (2013). [Sinkhorn distances: Lightspeed computation of optimal transport](https://arxiv.org/pdf/1306.0895.pdf). In Advances in Neural Information Processing Systems (pp. 2292-2300).

[3] Benamou, J. D., Carlier, G., Cuturi, M., Nenna, L., & Peyré, G. (2015). [Iterative Bregman projections for regularized transportation problems](https://arxiv.org/pdf/1412.5154.pdf). SIAM Journal on Scientific Computing, 37(2), A1111-A1138.

[4] S. Nakhostin, N. Courty, R. Flamary, D. Tuia, T. Corpetti, [Supervised planetary unmixing with optimal transport](https://hal.archives-ouvertes.fr/hal-01377236/document), Workshop on Hyperspectral Image and Signal Processing : Evolution in Remote Sensing (WHISPERS), 2016.

[5] N. Courty; R. Flamary; D. Tuia; A. Rakotomamonjy, [Optimal Transport for Domain Adaptation](https://arxiv.org/pdf/1507.00504.pdf), in IEEE Transactions on Pattern Analysis and Machine Intelligence , vol.PP, no.99, pp.1-1

[6] Ferradans, S., Papadakis, N., Peyré, G., & Aujol, J. F. (2014). [Regularized discrete optimal transport](https://arxiv.org/pdf/1307.5551.pdf). SIAM Journal on Imaging Sciences, 7(3), 1853-1882.

[7] Rakotomamonjy, A., Flamary, R., & Courty, N. (2015). [Generalized conditional gradient: analysis of convergence and applications](https://arxiv.org/pdf/1510.06567.pdf). arXiv preprint arXiv:1510.06567.

[8] M. Perrot, N. Courty, R. Flamary, A. Habrard (2016), [Mapping estimation for discrete optimal transport](http://remi.flamary.com/biblio/perrot2016mapping.pdf), Neural Information Processing Systems (NIPS).

[9] Schmitzer, B. (2016). [Stabilized Sparse Scaling Algorithms for Entropy Regularized Transport Problems](https://arxiv.org/pdf/1610.06519.pdf). arXiv preprint arXiv:1610.06519.

[10] Chizat, L., Peyré, G., Schmitzer, B., & Vialard, F. X. (2016). [Scaling algorithms for unbalanced transport problems](https://arxiv.org/pdf/1607.05816.pdf). arXiv preprint arXiv:1607.05816.

[11] Flamary, R., Cuturi, M., Courty, N., & Rakotomamonjy, A. (2016). [Wasserstein Discriminant Analysis](https://arxiv.org/pdf/1608.08063.pdf). arXiv preprint arXiv:1608.08063.

[12] Gabriel Peyré, Marco Cuturi, and Justin Solomon (2016), [Gromov-Wasserstein averaging of kernel and distance matrices](http://proceedings.mlr.press/v48/peyre16.html)  International Conference on Machine Learning (ICML).

[13] Mémoli, Facundo (2011). [Gromov–Wasserstein distances and the metric approach to object matching](https://media.adelaide.edu.au/acvt/Publications/2011/2011-Gromov%E2%80%93Wasserstein%20Distances%20and%20the%20Metric%20Approach%20to%20Object%20Matching.pdf). Foundations of computational mathematics 11.4 : 417-487.

[14] Knott, M. and Smith, C. S. (1984).[On the optimal mapping of distributions](https://link.springer.com/article/10.1007/BF00934745), Journal of Optimization Theory and Applications Vol 43.

[15] Peyré, G., & Cuturi, M. (2018). [Computational Optimal Transport](https://arxiv.org/pdf/1803.00567.pdf) .

[16] Agueh, M., & Carlier, G. (2011). [Barycenters in the Wasserstein space](https://hal.archives-ouvertes.fr/hal-00637399/document). SIAM Journal on Mathematical Analysis, 43(2), 904-924.

[17] Blondel, M., Seguy, V., & Rolet, A. (2018). [Smooth and Sparse Optimal Transport](https://arxiv.org/abs/1710.06276). Proceedings of the Twenty-First International Conference on Artificial Intelligence and Statistics (AISTATS).

[18] Genevay, A., Cuturi, M., Peyré, G. & Bach, F. (2016) [Stochastic Optimization for Large-scale Optimal Transport](https://arxiv.org/abs/1605.08527). Advances in Neural Information Processing Systems (2016).

[19] Seguy, V., Bhushan Damodaran, B., Flamary, R., Courty, N., Rolet, A.& Blondel, M. [Large-scale Optimal Transport and Mapping Estimation](https://arxiv.org/pdf/1711.02283.pdf). International Conference on Learning Representation (2018)

[20] Cuturi, M. and Doucet, A. (2014) [Fast Computation of Wasserstein Barycenters](http://proceedings.mlr.press/v32/cuturi14.html). International Conference in Machine Learning

[21] Solomon, J., De Goes, F., Peyré, G., Cuturi, M., Butscher, A., Nguyen, A. & Guibas, L. (2015). [Convolutional wasserstein distances: Efficient optimal transportation on geometric domains](https://dl.acm.org/citation.cfm?id=2766963). ACM Transactions on Graphics (TOG), 34(4), 66.

[22] J. Altschuler, J.Weed, P. Rigollet, (2017) [Near-linear time approximation algorithms for optimal transport via Sinkhorn iteration](https://papers.nips.cc/paper/6792-near-linear-time-approximation-algorithms-for-optimal-transport-via-sinkhorn-iteration.pdf), Advances in Neural Information Processing Systems (NIPS) 31

[23] Aude, G., Peyré, G., Cuturi, M., [Learning Generative Models with Sinkhorn Divergences](https://arxiv.org/abs/1706.00292), Proceedings of the Twenty-First International Conference on Artificial Intelligence and Statistics, (AISTATS) 21, 2018

[24] Vayer, T., Chapel, L., Flamary, R., Tavenard, R. and Courty, N. (2019). [Optimal Transport for structured data with application on graphs](http://proceedings.mlr.press/v97/titouan19a.html) Proceedings of the 36th International Conference on Machine Learning (ICML).

[25] Frogner C., Zhang C., Mobahi H., Araya-Polo M., Poggio T. (2015). [Learning with a Wasserstein Loss](http://cbcl.mit.edu/wasserstein/)  Advances in Neural Information Processing Systems (NIPS).

[26] Alaya M. Z., Bérar M., Gasso G., Rakotomamonjy A. (2019). [Screening Sinkhorn Algorithm for Regularized Optimal Transport](https://papers.nips.cc/paper/9386-screening-sinkhorn-algorithm-for-regularized-optimal-transport), Advances in Neural Information Processing Systems 33 (NeurIPS).

[27] Redko I., Courty N., Flamary R., Tuia D. (2019). [Optimal Transport for Multi-source Domain Adaptation under Target Shift](http://proceedings.mlr.press/v89/redko19a.html), Proceedings of the Twenty-Second International Conference on Artificial Intelligence and Statistics (AISTATS) 22, 2019.

[28] Caffarelli, L. A., McCann, R. J. (2010). [Free boundaries in optimal transport and Monge-Ampere obstacle problems](http://www.math.toronto.edu/~mccann/papers/annals2010.pdf), Annals of mathematics, 673-730.

[29] Chapel, L., Alaya, M., Gasso, G. (2020). [Partial Optimal Transport with Applications on Positive-Unlabeled Learning](https://arxiv.org/abs/2002.08276), Advances in Neural Information Processing Systems (NeurIPS), 2020.

[30] Flamary R., Courty N., Tuia D., Rakotomamonjy A. (2014). [Optimal transport with Laplacian regularization: Applications to domain adaptation and shape matching](https://remi.flamary.com/biblio/flamary2014optlaplace.pdf), NIPS Workshop on Optimal Transport and Machine Learning OTML, 2014.

[31] Bonneel, Nicolas, et al. [Sliced and radon wasserstein barycenters of measures](https://perso.liris.cnrs.fr/nicolas.bonneel/WassersteinSliced-JMIV.pdf), Journal of Mathematical Imaging and Vision 51.1 (2015): 22-45

[32] Huang, M., Ma S., Lai, L. (2021). [A Riemannian Block Coordinate Descent Method for Computing the Projection Robust Wasserstein Distance](http://proceedings.mlr.press/v139/huang21e.html), Proceedings of the 38th International Conference on Machine Learning (ICML).

[33] Kerdoncuff T., Emonet R., Marc S. [Sampled Gromov Wasserstein](https://hal.archives-ouvertes.fr/hal-03232509/document), Machine Learning Journal (MJL), 2021

[34] Feydy, J., Séjourné, T., Vialard, F. X., Amari, S. I., Trouvé, A., & Peyré, G. (2019, April). [Interpolating between optimal transport and MMD using Sinkhorn divergences](http://proceedings.mlr.press/v89/feydy19a/feydy19a.pdf). In The 22nd International Conference on Artificial Intelligence and Statistics (pp. 2681-2690). PMLR.

[35] Deshpande, I., Hu, Y. T., Sun, R., Pyrros, A., Siddiqui, N., Koyejo, S., ... & Schwing, A. G. (2019). [Max-sliced wasserstein distance and its use for gans](https://openaccess.thecvf.com/content_CVPR_2019/papers/Deshpande_Max-Sliced_Wasserstein_Distance_and_Its_Use_for_GANs_CVPR_2019_paper.pdf). In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 10648-10656).

[36] Liutkus, A., Simsekli, U., Majewski, S., Durmus, A., & Stöter, F. R.
(2019, May). [Sliced-Wasserstein flows: Nonparametric generative modeling
via optimal transport and diffusions](http://proceedings.mlr.press/v97/liutkus19a/liutkus19a.pdf). In International Conference on
Machine Learning (pp. 4104-4113). PMLR.

[37] Janati, H., Cuturi, M., Gramfort, A. [Debiased sinkhorn barycenters](http://proceedings.mlr.press/v119/janati20a/janati20a.pdf) Proceedings of the 37th International
Conference on Machine Learning, PMLR 119:4692-4701, 2020

[38] C. Vincent-Cuaz, T. Vayer, R. Flamary, M. Corneli, N. Courty, [Online Graph
Dictionary Learning](https://arxiv.org/pdf/2102.06555.pdf), International Conference on Machine Learning (ICML), 2021.

[39] Gozlan, N., Roberto, C., Samson, P. M., & Tetali, P. (2017). [Kantorovich duality for general transport costs and applications](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.712.1825&rep=rep1&type=pdf). Journal of Functional Analysis, 273(11), 3327-3405.

[40] Forrow, A., Hütter, J. C., Nitzan, M., Rigollet, P., Schiebinger, G., & Weed, J. (2019, April). [Statistical optimal transport via factored couplings](http://proceedings.mlr.press/v89/forrow19a/forrow19a.pdf). In The 22nd International Conference on Artificial Intelligence and Statistics (pp. 2454-2465). PMLR.

[41] Chapel*, L., Flamary*, R., Wu, H., Févotte, C., Gasso, G. (2021). [Unbalanced Optimal Transport through Non-negative Penalized Linear Regression](https://proceedings.neurips.cc/paper/2021/file/c3c617a9b80b3ae1ebd868b0017cc349-Paper.pdf) Advances in Neural Information Processing Systems (NeurIPS), 2020. (Two first co-authors)

[42] Delon, J., Gozlan, N., and Saint-Dizier, A. [Generalized Wasserstein barycenters between probability measures living on different subspaces](https://arxiv.org/pdf/2105.09755). arXiv preprint arXiv:2105.09755, 2021.

[43]  Álvarez-Esteban, Pedro C., et al. [A fixed-point approach to barycenters in Wasserstein space.](https://arxiv.org/pdf/1511.05355.pdf) Journal of Mathematical Analysis and Applications 441.2 (2016): 744-762.

[44] Delon, Julie, Julien Salomon, and Andrei Sobolevski. [Fast transport optimization for Monge costs on the circle.](https://arxiv.org/abs/0902.3527) SIAM Journal on Applied Mathematics 70.7 (2010): 2239-2258.

[45] Hundrieser, Shayan, Marcel Klatt, and Axel Munk. [The statistics of circular optimal transport.](https://arxiv.org/abs/2103.15426) Directional Statistics for Innovative Applications: A Bicentennial Tribute to Florence Nightingale. Singapore: Springer Nature Singapore, 2022. 57-82.

[46] Bonet, C., Berg, P., Courty, N., Septier, F., Drumetz, L., & Pham, M. T. (2023). [Spherical Sliced-Wasserstein](https://openreview.net/forum?id=jXQ0ipgMdU). International Conference on Learning Representations.

[47] Chowdhury, S., & Mémoli, F. (2019). [The gromov–wasserstein distance between networks and stable network invariants](https://academic.oup.com/imaiai/article/8/4/757/5627736). Information and Inference: A Journal of the IMA, 8(4), 757-787.

[48] Cédric Vincent-Cuaz, Rémi Flamary, Marco Corneli, Titouan Vayer, Nicolas Courty (2022). [Semi-relaxed Gromov-Wasserstein divergence and applications on graphs](https://openreview.net/pdf?id=RShaMexjc-x). International Conference on Learning Representations (ICLR), 2022.

[49] Redko, I., Vayer, T., Flamary, R., and Courty, N. (2020). [CO-Optimal Transport](https://proceedings.neurips.cc/paper/2020/file/cc384c68ad503482fb24e6d1e3b512ae-Paper.pdf). Advances in Neural Information Processing Systems, 33.

[50] Liu, T., Puigcerver, J., & Blondel, M. (2023). [Sparsity-constrained optimal transport](https://openreview.net/forum?id=yHY9NbQJ5BP). Proceedings of the Eleventh International Conference on Learning Representations (ICLR).

[51] Xu, H., Luo, D., Zha, H., & Duke, L. C. (2019). [Gromov-wasserstein learning for graph matching and node embedding](http://proceedings.mlr.press/v97/xu19b.html). In International Conference on Machine Learning (ICML), 2019.

[52] Collas, A., Vayer, T., Flamary, F., & Breloy, A. (2023). [Entropic Wasserstein Component Analysis](https://arxiv.org/abs/2303.05119). ArXiv.

[53] C. Vincent-Cuaz, R. Flamary, M. Corneli, T. Vayer, N. Courty (2022). [Template based graph neural network with optimal transport distances](https://papers.nips.cc/paper_files/paper/2022/file/4d3525bc60ba1adc72336c0392d3d902-Paper-Conference.pdf). Advances in Neural Information Processing Systems, 35.

[54] Bécigneul, G., Ganea, O. E., Chen, B., Barzilay, R., & Jaakkola, T. S. (2020). [Optimal transport graph neural networks](https://arxiv.org/pdf/2006.04804).

[55] Ronak Mehta, Jeffery Kline, Vishnu Suresh Lokhande, Glenn Fung, & Vikas Singh (2023). [Efficient Discrete Multi Marginal Optimal Transport Regularization](https://openreview.net/forum?id=R98ZfMt-jE). In The Eleventh International Conference on Learning Representations (ICLR).

[56] Jeffery Kline. [Properties of the d-dimensional earth mover’s problem](https://www.sciencedirect.com/science/article/pii/S0166218X19301441). Discrete Applied Mathematics, 265: 128–141, 2019.

[57] Delon, J., Desolneux, A., & Salmona, A. (2022). [Gromov–Wasserstein
distances between Gaussian distributions](https://hal.science/hal-03197398v2/file/main.pdf). Journal of Applied Probability, 59(4),
1178-1198.

[58] Paty F-P., d’Aspremont 1., & Cuturi M. (2020). [Regularity as regularization:Smooth and strongly convex brenier potentials in optimal transport.](http://proceedings.mlr.press/v108/paty20a/paty20a.pdf) In International Conference on Artificial Intelligence and Statistics, pages 1222–1232. PMLR, 2020.

[59] Taylor A. B. (2017). [Convex interpolation and performance estimation of first-order methods for convex optimization.](https://dial.uclouvain.be/pr/boreal/object/boreal%3A182881/datastream/PDF_01/view) PhD thesis, Catholic University of Louvain, Louvain-la-Neuve, Belgium, 2017.

[60] Feydy, J., Roussillon, P., Trouvé, A., & Gori, P. (2019). [Fast and scalable optimal transport for brain tractograms](https://arxiv.org/pdf/2107.02010.pdf). In Medical Image Computing and Computer Assisted Intervention–MICCAI 2019: 22nd International Conference, Shenzhen, China, October 13–17, 2019, Proceedings, Part III 22 (pp. 636-644). Springer International Publishing.

[61] Charlier, B., Feydy, J., Glaunes, J. A., Collin, F. D., & Durif, G. (2021). [Kernel operations on the gpu, with autodiff, without memory overflows](https://www.jmlr.org/papers/volume22/20-275/20-275.pdf). The Journal of Machine Learning Research, 22(1), 3457-3462.

[62] H. Van Assel, C. Vincent-Cuaz, T. Vayer, R. Flamary, N. Courty (2023). [Interpolating between Clustering and Dimensionality Reduction with Gromov-Wasserstein](https://arxiv.org/pdf/2310.03398.pdf). NeurIPS 2023 Workshop Optimal Transport and Machine Learning.

[63] Li, J., Tang, J., Kong, L., Liu, H., Li, J., So, A. M. C., & Blanchet, J. (2022). [A Convergent Single-Loop Algorithm for Relaxation of Gromov-Wasserstein in Graph Data](https://openreview.net/pdf?id=0jxPyVWmiiF). In The Eleventh International Conference on Learning Representations.

[64] Ma, X., Chu, X., Wang, Y., Lin, Y., Zhao, J., Ma, L., & Zhu, W. (2023). [Fused Gromov-Wasserstein Graph Mixup for Graph-level Classifications](https://openreview.net/pdf?id=uqkUguNu40). In Thirty-seventh Conference on Neural Information Processing Systems.

[65] Scetbon, M., Cuturi, M., & Peyré, G. (2021). [Low-Rank Sinkhorn Factorization](https://arxiv.org/pdf/2103.04737.pdf).