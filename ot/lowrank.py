#################################################################################################################
############################################## WORK IN PROGRESS #################################################
#################################################################################################################


from ot.utils import unif, list_to_array
from ot.backend import get_backend
from ot.datasets import make_1D_gauss as gauss



################################## LR-DYSKTRA ALGORITHM ##########################################

def LR_Dysktra(eps1, eps2, eps3, p1, p2, alpha, dykstra_w): 
    """
    Implementation of the Dykstra algorithm for low rank Sinkhorn
    """

    # get dykstra parameters 
    q3_1, q3_2, v1_, v2_, q1, q2 = dykstra_w

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

    dykstra_w = [q3_1, q3_2, v1_, v2_, q1, q2]

    return Q, R, g, err, dykstra_w
    


#################################### LOW RANK SINKHORN ALGORITHM #########################################


def lowrank_sinkhorn(X_s, X_t, reg=0, a=None, b=None, r=4, metric='sqeuclidean', alpha=1e-10, numIterMax=10000, stopThr=1e-20):
    r'''
    Solve the entropic regularization optimal transport problem under low-nonnegative low rank constraints
    
    Parameters
    ----------
    X_s : array-like, shape (n_samples_a, dim)
        samples in the source domain
    X_t : array-like, shape (n_samples_b, dim)
        samples in the target domain
    reg : float
        Regularization term >0
    a : array-like, shape (n_samples_a,)
        samples weights in the source domain
    b : array-like, shape (n_samples_b,)
        samples weights in the target domain
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
        ...

    '''

    X_s, X_t = list_to_array(X_s, X_t)
    nx = get_backend(X_s, X_t)

    ns, nt = X_s.shape[0], X_t.shape[0]
    if a is None:
        a = nx.from_numpy(unif(ns), type_as=X_s)
    if b is None:
        b = nx.from_numpy(unif(nt), type_as=X_s)
    
    M = ot.dist(X_s,X_t, metric=metric) 
    
    # Compute rank
    r = min(ns, nt, r)
    
    # Compute gamma
    L = nx.sqrt((2/(alpha**4))*nx.norm(M)**2 + (reg + (2/(alpha**3))*nx.norm(M))**2)
    gamma = 1/(2*L)
    
    # Initialisation 
    Q, R, g = nx.ones((ns,r)), nx.ones((nt,r)), nx.ones(r) 
    q3_1, q3_2 = nx.ones(r), nx.ones(r)
    v1_, v2_ = nx.ones(r), nx.ones(r)
    q1, q2 = nx.ones(r), nx.ones(r)
    dykstra_w = [q3_1, q3_2, v1_, v2_, q1, q2]
    n_iter = 0
    err = 1

    while n_iter < numIterMax:
        if err > stopThr:
            n_iter = n_iter + 1
            
            CR = nx.dot(M,R)
            C_t_Q = nx.dot(M.T,Q)
            diag_g = (1/g)[:,None]

            eps1 = nx.exp(-gamma*(nx.dot(CR,diag_g)) - ((gamma*reg)-1)*nx.log(Q))
            eps2 = nx.exp(-gamma*(nx.dot(C_t_Q,diag_g)) - ((gamma*reg)-1)*nx.log(R))
            omega = nx.diag(nx.dot(Q.T, CR))
            eps3 = nx.exp(gamma*omega/(g**2) - (gamma*reg - 1)*nx.log(g))

            Q, R, g, err, dykstra_w = LR_Dysktra(eps1, eps2, eps3, a, b, alpha, dykstra_w)
        else:
            break
    
    return Q, R, g





############################################################################
## Test with X_s, X_t from ot.datasets
#############################################################################

import numpy as np
import ot 

Xs, _ = ot.datasets.make_data_classif('3gauss', n=1000)
Xt, _ = ot.datasets.make_data_classif('3gauss2', n=1500)


Q, R, g = lowrank_sinkhorn(Xs,Xt,reg=0.1)
M = ot.dist(Xs,Xt)
P = np.dot(Q,np.dot(np.diag(1/g),R.T))

print(np.sum(P))




