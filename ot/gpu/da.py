# -*- coding: utf-8 -*-
"""
Domain adaptation with optimal transport with GPU implementation
"""

# Author: Remi Flamary <remi.flamary@unice.fr>
#         Nicolas Courty <ncourty@irisa.fr>
#         Michael Perrot <michael.perrot@univ-st-etienne.fr>
#         Leo Gautheron <https://github.com/aje>
#
# License: MIT License


import numpy as np
import cupy as cp

from ..utils import check_params
from ..da import BaseTransport, distribution_estimation_uniform
from .bregman import sinkhorn_knopp


def pairwiseEuclideanGPU(a, b, returnAsGPU=False, squared=False):
    """
    Compute the pairwise euclidean distance between matrices a and b.


    Parameters
    ----------
    a : np.ndarray (n, f)
        first matrix
    b : np.ndarray (m, f)
        second matrix
    returnAsGPU : boolean, optional (default False)
        if True, returns cupy matrix still on GPU, else return np.ndarray
    squared : boolean, optional (default False)
        if True, return squared euclidean distance matrix


    Returns
    -------
    c : (n x m) np.ndarray or cupy.ndarray
        pairwise euclidean distance distance matrix
    """
    # a is shape (n, f) and b shape (m, f). Return matrix c of shape (n, m).
    # First compute in c_GPU the squared euclidean distance. And return its
    # square root. At each cell [i,j] of c, we want to have
    # sum{k in range(f)} ( (a[i,k] - b[j,k])^2 ). We know that
    # (a-b)^2 = a^2 -2ab +b^2. Thus we want to have in each cell of c:
    # sum{k in range(f)} ( a[i,k]^2 -2a[i,k]b[j,k] +b[j,k]^2).
    a_GPU = cp.asarray(a)
    b_GPU = cp.asarray(b)

    # Multiply a by b transpose to obtain in each cell [i,j] of c the
    # value sum{k in range(f)} ( a[i,k]b[j,k] )
    c_GPU = a_GPU.dot(b_GPU.T)
    # multiply by -2 to have sum{k in range(f)} ( -2a[i,k]b[j,k] )
    cp.multiply(c_GPU, -2, out=c_GPU)

    # Compute the vectors of the sum of squared elements.
    a_GPU = cp.power(a_GPU, 2).sum(axis=1)
    b_GPU = cp.power(b_GPU, 2).sum(axis=1)

    # Add the vectors in each columns (respectivly rows) of c.
    # sum{k in range(f)} ( a[i,k]^2 -2a[i,k]b[j,k] )
    c_GPU += a_GPU.reshape(-1, 1)
    # sum{k in range(f)} ( a[i,k]^2 -2a[i,k]b[j,k] +b[j,k]^2)
    c_GPU += b_GPU

    if not squared:
        cp.sqrt(c_GPU, out=c_GPU)

    if returnAsGPU:
        return c_GPU
    else:
        return cp.asnumpy(c_GPU)


def sinkhorn_lpl1_mm(a, labels_a, b, M_GPU, reg, eta=0.1, numItermax=10,
                     numInnerItermax=200, stopInnerThr=1e-9,
                     verbose=False, log=False):
    """
    Solve the entropic regularization optimal transport problem with nonconvex
    group lasso regularization.

    The function solves the following optimization problem:

    .. math::
        \gamma = arg\min_\gamma <\gamma,M>_F + reg\cdot\Omega_e(\gamma) +
                 \eta \Omega_g(\gamma)

        s.t. \gamma 1 = a

             \gamma^T 1= b

             \gamma\geq 0
    where :

    - M is the (ns,nt) metric cost matrix
    - :math:`\Omega_e` is the entropic regularization term
      :math:`\Omega_e(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - :math:`\Omega_g` is the group lasso  regulaization term
      :math:`\Omega_g(\gamma)=\sum_{i,c} \|\gamma_{i,\mathcal{I}_c}\|^{1/2}_1`
      where  :math:`\mathcal{I}_c` are the index of samples from class c in the
      source domain.
    - a and b are source and target weights (sum to 1)

    The algorithm used for solving the problem is the generalised conditional
    gradient as proposed in  [5]_ [7]_


    Parameters
    ----------
    a : np.ndarray (ns,)
        samples weights in the source domain
    labels_a : np.ndarray (ns,)
        labels of samples in the source domain
    b : np.ndarray (nt,)
        samples weights in the target domain
    M_GPU : cupy.ndarray (ns,nt)
        loss matrix
    reg : float
        Regularization term for entropic regularization >0
    eta : float, optional
        Regularization term  for group lasso regularization >0
    numItermax : int, optional
        Max number of iterations
    numInnerItermax : int, optional
        Max number of iterations (inner sinkhorn solver)
    stopInnerThr : float, optional
        Stop threshold on error (inner sinkhorn solver) (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True


    Returns
    -------
    gamma : (ns x nt) np.ndarray
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters


    References
    ----------

    .. [5] N. Courty; R. Flamary; D. Tuia; A. Rakotomamonjy, "Optimal Transport
           for Domain Adaptation," in IEEE Transactions on Pattern Analysis and
           Machine Intelligence , vol.PP, no.99, pp.1-1
    .. [7] Rakotomamonjy, A., Flamary, R., & Courty, N. (2015). Generalized
           conditional gradient: analysis of convergence and applications.
           arXiv preprint arXiv:1510.06567.

    See Also
    --------
    ot.lp.emd : Unregularized OT
    ot.bregman.sinkhorn : Entropic regularized OT
    ot.optim.cg : General regularized OT

    """
    p = 0.5
    epsilon = 1e-3

    indices_labels = []
    classes = np.unique(labels_a)
    for c in classes:
        idxc, = np.where(labels_a == c)
        indices_labels.append(cp.asarray(idxc.reshape(1, -1)))

    W_GPU = cp.zeros(M_GPU.shape)
    Mreg_GPU = cp.empty(M_GPU.shape)
    for cpt in range(numItermax):
        cp.multiply(W_GPU, eta, out=Mreg_GPU)
        Mreg_GPU += M_GPU
        transp_GPU = sinkhorn_knopp(a, b, Mreg_GPU, reg,
                                    numItermax=numInnerItermax,
                                    stopThr=stopInnerThr, returnAsGPU=True)
        # the transport has been computed. Check if classes are really
        # separated
        W_GPU.fill(1)
        for (i, c) in enumerate(classes):
            majs_GPU = cp.sum(transp_GPU[indices_labels[i]][0], axis=0)
            majs_GPU = p * ((majs_GPU + epsilon)**(p - 1))
            W_GPU[indices_labels[i]] = majs_GPU

    return cp.asnumpy(transp_GPU)


def cost_normalization(C, norm=None):
    """ Apply normalization to the loss matrix


    Parameters
    ----------
    C : cupy.array (n1, n2)
        The cost matrix to normalize.
    norm : str
        type of normalization from 'median','max','log','loglog'. Any other
        value do not normalize.


    Returns
    -------

    C : cupy.array (n1, n2)
        The input cost matrix normalized according to given norm.

    """

    if norm == "median":
        C = cp.divide(np.median(cp.asnumpy(C)))
    elif norm == "max":
        C = cp.divide(cp.max(C))
    elif norm == "log":
        C = cp.log(1 + C)
    elif norm == "loglog":
        C = cp.log(1 + cp.log(1 + C))

    return C


class SinkhornTransport(BaseTransport):
    """Domain Adapatation OT method based on Sinkhorn Algorithm

    Parameters
    ----------
    reg_e : float, optional (default=1)
        Entropic regularization parameter
    max_iter : int, float, optional (default=1000)
        The minimum number of iteration before stopping the optimization
        algorithm if no it has not converged
    tol : float, optional (default=10e-9)
        The precision required to stop the optimization algorithm.
    mapping : string, optional (default="barycentric")
        The kind of mapping to apply to transport samples from a domain into
        another one.
        if "barycentric" only the samples used to estimate the coupling can
        be transported from a domain to another one.
    metric : string, optional (default="sqeuclidean")
        The ground metric for the Wasserstein problem
    norm : string, optional (default=None)
        If given, normalize the ground metric to avoid numerical errors that
        can occur with large metric values.
    distribution : string, optional (default="uniform")
        The kind of distribution estimation to employ
    verbose : int, optional (default=0)
        Controls the verbosity of the optimization algorithm
    log : int, optional (default=0)
        Controls the logs of the optimization algorithm
    limit_max: float, optional (defaul=np.infty)
        Controls the semi supervised mode. Transport between labeled source
        and target samples of different classes will exhibit an infinite cost

    Attributes
    ----------
    coupling_ : array-like, shape (n_source_samples, n_target_samples)
        The optimal coupling
    log_ : dictionary
        The dictionary of log, empty dic if parameter log is not True

    References
    ----------
    .. [1] N. Courty; R. Flamary; D. Tuia; A. Rakotomamonjy,
           "Optimal Transport for Domain Adaptation," in IEEE Transactions
           on Pattern Analysis and Machine Intelligence , vol.PP, no.99, pp.1-1
    .. [2] M. Cuturi, Sinkhorn Distances : Lightspeed Computation of Optimal
           Transport, Advances in Neural Information Processing Systems (NIPS)
           26, 2013
    """

    def __init__(self, reg_e=1., max_iter=1000,
                 tol=10e-9, verbose=False, log=False,
                 metric="sqeuclidean", norm=None,
                 distribution_estimation=distribution_estimation_uniform,
                 out_of_sample_map='ferradans', limit_max=np.infty):

        self.reg_e = reg_e
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.log = log
        self.metric = metric
        self.norm = norm
        self.limit_max = limit_max
        self.distribution_estimation = distribution_estimation
        self.out_of_sample_map = out_of_sample_map

    def fit(self, Xs=None, ys=None, Xt=None, yt=None):
        """Build a coupling matrix from source and target sets of samples
        (Xs, ys) and (Xt, yt)

        Parameters
        ----------
        Xs : array-like, shape (n_source_samples, n_features)
            The training input samples.
        ys : array-like, shape (n_source_samples,)
            The class labels
        Xt : array-like, shape (n_target_samples, n_features)
            The training input samples.
        yt : array-like, shape (n_target_samples,)
            The class labels.

        Returns
        -------
        self : object
            Returns self.
        """

        # Fit
        if not check_params(Xs=Xs, Xt=Xt):
            return
        # pairwise distance
        self.cost_ = pairwiseEuclideanGPU(Xs, Xt, returnAsGPU=True,
                                          squared=True)
        self.cost_ = cost_normalization(self.cost_, self.norm)

        # distribution estimation
        self.mu_s = self.distribution_estimation(Xs)
        self.mu_t = self.distribution_estimation(Xt)

        # store arrays of samples
        self.xs_ = Xs
        self.xt_ = Xt

        # coupling estimation
        returned_ = sinkhorn_knopp(
            a=self.mu_s, b=self.mu_t, M_GPU=self.cost_, reg=self.reg_e,
            numItermax=self.max_iter, stopThr=self.tol,
            verbose=self.verbose, log=self.log)

        # deal with the value of log
        if self.log:
            self.coupling_, self.log_ = returned_
        else:
            self.coupling_ = returned_
            self.log_ = dict()

        return self


class SinkhornLpl1Transport(BaseTransport):
    """Domain Adapatation OT method based on sinkhorn algorithm +
    LpL1 class regularization.

    Parameters
    ----------
    reg_e : float, optional (default=1)
        Entropic regularization parameter
    reg_cl : float, optional (default=0.1)
        Class regularization parameter
    mapping : string, optional (default="barycentric")
        The kind of mapping to apply to transport samples from a domain into
        another one.
        if "barycentric" only the samples used to estimate the coupling can
        be transported from a domain to another one.
    metric : string, optional (default="sqeuclidean")
        The ground metric for the Wasserstein problem
    norm : string, optional (default=None)
        If given, normalize the ground metric to avoid numerical errors that
        can occur with large metric values.
    distribution : string, optional (default="uniform")
        The kind of distribution estimation to employ
    max_iter : int, float, optional (default=10)
        The minimum number of iteration before stopping the optimization
        algorithm if no it has not converged
    max_inner_iter : int, float, optional (default=200)
        The number of iteration in the inner loop
    verbose : int, optional (default=0)
        Controls the verbosity of the optimization algorithm
    limit_max: float, optional (defaul=np.infty)
        Controls the semi supervised mode. Transport between labeled source
        and target samples of different classes will exhibit an infinite cost

    Attributes
    ----------
    coupling_ : array-like, shape (n_source_samples, n_target_samples)
        The optimal coupling

    References
    ----------

    .. [1] N. Courty; R. Flamary; D. Tuia; A. Rakotomamonjy,
       "Optimal Transport for Domain Adaptation," in IEEE
       Transactions on Pattern Analysis and Machine Intelligence ,
       vol.PP, no.99, pp.1-1
    .. [2] Rakotomamonjy, A., Flamary, R., & Courty, N. (2015).
       Generalized conditional gradient: analysis of convergence
       and applications. arXiv preprint arXiv:1510.06567.

    """

    def __init__(self, reg_e=1., reg_cl=0.1,
                 max_iter=10, max_inner_iter=200,
                 tol=10e-9, verbose=False,
                 metric="sqeuclidean", norm=None,
                 distribution_estimation=distribution_estimation_uniform,
                 out_of_sample_map='ferradans', limit_max=np.infty):

        self.reg_e = reg_e
        self.reg_cl = reg_cl
        self.max_iter = max_iter
        self.max_inner_iter = max_inner_iter
        self.tol = tol
        self.verbose = verbose
        self.metric = metric
        self.norm = norm
        self.distribution_estimation = distribution_estimation
        self.out_of_sample_map = out_of_sample_map
        self.limit_max = limit_max

    def fit(self, Xs, ys=None, Xt=None, yt=None):
        """Build a coupling matrix from source and target sets of samples
        (Xs, ys) and (Xt, yt)

        Parameters
        ----------
        Xs : array-like, shape (n_source_samples, n_features)
            The training input samples.
        ys : array-like, shape (n_source_samples,)
            The class labels
        Xt : array-like, shape (n_target_samples, n_features)
            The training input samples.
        yt : array-like, shape (n_target_samples,)
            The class labels. If some target samples are unlabeled, fill the
            yt's elements with -1.

            Warning: Note that, due to this convention -1 cannot be used as a
            class label

        Returns
        -------
        self : object
            Returns self.
        """

        # check the necessary inputs parameters are here
        if not check_params(Xs=Xs, Xt=Xt, ys=ys):
            return self

        # pairwise distance
        self.cost_ = pairwiseEuclideanGPU(Xs, Xt, returnAsGPU=True,
                                          squared=True)
        self.cost_ = cost_normalization(self.cost_, self.norm)

        # distribution estimation
        self.mu_s = self.distribution_estimation(Xs)
        self.mu_t = self.distribution_estimation(Xt)

        # store arrays of samples
        self.xs_ = Xs
        self.xt_ = Xt

        self.coupling_ = sinkhorn_lpl1_mm(
            a=self.mu_s, labels_a=ys, b=self.mu_t, M_GPU=self.cost_,
            reg=self.reg_e, eta=self.reg_cl, numItermax=self.max_iter,
            numInnerItermax=self.max_inner_iter, stopInnerThr=self.tol,
            verbose=self.verbose)

        return self
