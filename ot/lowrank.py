#################################################################################################################
############################################## WORK IN PROGRESS #################################################
#################################################################################################################

## Implementation of the LR-Dykstra algorithm and low rank sinkhorn algorithms 

import warnings
from .utils import unif, list_to_array, dist
from .backend import get_backend



################################## LR-DYSKTRA ALGORITHM ##########################################

def LR_Dysktra(eps1, eps2, eps3, p1, p2, alpha, dykstra_p): 
    """
    Implementation of the Dykstra algorithm for low rank sinkhorn
    """

    # get dykstra parameters 
    q3_1, q3_2, v1_, v2_, q1, q2 = dykstra_p

    # POT backend
    eps1, eps2, eps3, p1, p2 = list_to_array(eps1, eps2, eps3, p1, p2)
    q3_1, q3_2, v1_, v2_, q1, q2 = list_to_array(q3_1, q3_2, v1_, v2_, q1, q2)
    
    nx = get_backend(eps1, eps2, eps3, p1, p2, q3_1, q3_2, v1_, v2_, q1, q2)
    
    # ------- Dykstra algorithm ------
    g_ = eps3

    u1 = p1 / nx.dot(eps1, v1_)
    u2 = p2 / nx.dot(eps2, v2_)

    g = nx.maximum(alpha, g_ * q3_1)
    q3_1 = (g_ * q3_1) / g
    g_ = g 

    prod1 = ((v1_ * q1) * nx.dot(eps1.T, u1))
    prod2 = ((v2_ * q2) * nx.dot(eps2.T, u2))
    g = (g_ * q3_2 * prod1 * prod2)**(1/3)

    v1 = g / nx.dot(eps1.T,u1)
    v2 = g / nx.dot(eps2.T,u2)

    q1 = (v1_ * q1) / v1
    q2 = (v2_ * q2) / v2
    q3_2 = (g_ * q3_2) / g
    
    v1_, v2_ = v1, v2
    g_ = g

    # Compute error
    err1 = nx.sum(nx.abs(u1 * (eps1 @ v1) - p1))
    err2 = nx.sum(nx.abs(u2 * (eps2 @ v2) - p2))
    err = err1 + err2

    # Compute low rank matrices Q, R
    Q = u1[:,None] * eps1 * v1[None,:]
    R = u2[:,None] * eps2 * v2[None,:]

    dykstra_p = [q3_1, q3_2, v1_, v2_, q1, q2]

    return Q, R, g, err, dykstra_p
    


#################################### LOW RANK SINKHORN ALGORITHM #########################################


def lowrank_sinkhorn(X_s, X_t, a=None, b=None, reg=0, rank=2, metric='sqeuclidean', alpha="auto", 
                     numItermax=10000, stopThr=1e-9, warn=True, verbose=False):
    r'''
    Solve the entropic regularization optimal transport problem under low-nonnegative rank constraints.
    
    This function returns the two low-rank matrix decomposition of the OT plan (Q,R), as well as the weight vector g.
    
    Parameters
    ----------
    X_s : array-like, shape (n_samples_a, dim)
        samples in the source domain
    X_t : array-like, shape (n_samples_b, dim)
        samples in the target domain
    a : array-like, shape (n_samples_a,)
        samples weights in the source domain
    b : array-like, shape (n_samples_b,)
        samples weights in the target domain
    reg : float, optional
        Regularization term >0
    rank: int, optional 
        Nonnegative rank of the OT plan
    alpha: int, optional 
        Lower bound for the weight vector g (>0 and <1/r)
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshold on error (>0)

        
    Returns
    -------
    Q : array-like, shape (n_samples_a, r)
        First low-rank matrix decomposition of the OT plan 
    R: array-like, shape (n_samples_b, r)
        Second low-rank matrix decomposition of the OT plan
    g : array-like, shape (r, )
        Weight vector for the low-rank decomposition of the OT plan
    
        
    References
    ----------

    .. Scetbon, M., Cuturi, M., & Peyré, G (2021).
        Low-Rank Sinkhorn Factorization. arXiv preprint arXiv:2103.04737.

    '''

    X_s, X_t = list_to_array(X_s, X_t)
    nx = get_backend(X_s, X_t)

    ns, nt = X_s.shape[0], X_t.shape[0]
    if a is None:
        a = nx.from_numpy(unif(ns), type_as=X_s)
    if b is None:
        b = nx.from_numpy(unif(nt), type_as=X_s)
    
    # Compute cost matrix
    M = dist(X_s,X_t, metric=metric) 

    # Compute rank
    rank = min(ns, nt, rank)
    r = rank
    
    if alpha == 'auto':
        alpha = 1.0 / (r + 1)
        
    if (1/r < alpha) or (alpha < 0):
        warnings.warn("The provided alpha value might lead to instabilities.")

    
    # Compute gamma
    L = nx.sqrt((2/(alpha**4))*(nx.norm(M)**2) + (reg + (2/(alpha**3))*(nx.norm(M))**2))
    gamma = 1/(2*L)
    
    # Initialisation 
    Q, R, g = nx.ones((ns,r)), nx.ones((nt,r)), nx.ones(r) 
    q3_1, q3_2 = nx.ones(r), nx.ones(r)
    v1_, v2_ = nx.ones(r), nx.ones(r)
    q1, q2 = nx.ones(r), nx.ones(r)
    dykstra_p = [q3_1, q3_2, v1_, v2_, q1, q2]
    err = 1

    for ii in range(numItermax):
        CR = nx.dot(M,R)
        C_t_Q = nx.dot(M.T,Q)
        diag_g = (1/g)[:,None]

        eps1 = nx.exp(-gamma*(nx.dot(CR,diag_g)) - ((gamma*reg)-1)*nx.log(Q))
        eps2 = nx.exp(-gamma*(nx.dot(C_t_Q,diag_g)) - ((gamma*reg)-1)*nx.log(R))
        omega = nx.diag(nx.dot(Q.T, CR))
        eps3 = nx.exp(gamma*omega/(g**2) - (gamma*reg - 1)*nx.log(g))

        Q, R, g, err, dykstra_p = LR_Dysktra(eps1, eps2, eps3, a, b, alpha, dykstra_p)
        
        if err < stopThr:
            break

        if verbose:
                if ii % 200 == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(ii, err))

    else: 
        if warn:
            warnings.warn("Sinkhorn did not converge. You might want to "
                          "increase the number of iterations `numItermax` "
                          "or the regularization parameter `reg`.")
    
    return Q, R, g





############################################################################
## Test with X_s, X_t from ot.datasets
#############################################################################

# import numpy as np
# import ot 

# Xs, _ = ot.datasets.make_data_classif('3gauss', n=1000)
# Xt, _ = ot.datasets.make_data_classif('3gauss2', n=1500)

# ns = Xs.shape[0]
# nt = Xt.shape[0]

# a = unif(ns)
# b = unif(nt)

# Q, R, g = lowrank_sinkhorn(Xs, Xt, reg=0.1, metric='euclidean', verbose=True, numItermax=100)
# M = ot.dist(Xs,Xt)
# P = np.dot(Q,np.dot(np.diag(1/g),R.T))

# print(np.sum(P))




