"""
Low rank OT solvers
"""

# Author: Laurène David <laurene.david@ip-paris.fr>
#
# License: MIT License



#################################################################################################################
############################################## WORK IN PROGRESS #################################################
#################################################################################################################


import warnings
from ot.utils import unif
from ot.backend import get_backend



################################## LR-DYSKTRA ALGORITHM ##########################################

def LR_Dysktra(eps1, eps2, eps3, p1, p2, alpha, dykstra_p, stopThr, nx=None): 
    """
    Implementation of the Dykstra algorithm for the Low rank sinkhorn solver
    
    """
    # Get dykstra parameters 
    g, q3_1, q3_2, v1_, v2_, q1, q2, u1, u2, v1, v2 = dykstra_p
    g_ = eps3.copy()
    err = 1

    # POT backend if needed 
    if nx is None:
        nx = get_backend(eps1, eps2, eps3, p1, p2, 
                         g, q3_1, q3_2, v1_, v2_, q1, q2, u1, u2)


    # ------- Dykstra algorithm ------    
    while err > stopThr :
        u1 = p1 / nx.dot(eps1, v1_)
        u2 = p2 / nx.dot(eps2, v2_)

        g = nx.maximum(alpha, g_ * q3_1)
        q3_1 = (g_ * q3_1) / g
        g_ = g.copy()

        prod1 = ((v1_ * q1) * nx.dot(eps1.T, u1))
        prod2 = ((v2_ * q2) * nx.dot(eps2.T, u2))
        g = (g_ * q3_2 * prod1 * prod2)**(1/3)

        v1 = g / nx.dot(eps1.T,u1)
        v2 = g / nx.dot(eps2.T,u2)
        q1 = (v1_ * q1) / v1

        q2 = (v2_ * q2) / v2
        q3_2 = (g_ * q3_2) / g
        
        v1_, v2_ = v1.copy(), v2.copy()
        g_ = g.copy()

        # Compute error
        err1 = nx.sum(nx.abs(u1 * (eps1 @ v1) - p1))
        err2 = nx.sum(nx.abs(u2 * (eps2 @ v2) - p2))
        err = err1 + err2

    # Compute low rank matrices Q, R
    Q = u1[:,None] * eps1 * v1[None,:]
    R = u2[:,None] * eps2 * v2[None,:]

    dykstra_p = [g, q3_1, q3_2, v1_, v2_, q1, q2, u1, u2, v1, v2]

    return Q, R, dykstra_p
    
    


#################################### LOW RANK SINKHORN ALGORITHM #########################################


def lowrank_sinkhorn(X_s, X_t, a=None, b=None, reg=0, rank=2, alpha="auto", 
                     numItermax=1000, stopThr=1e-9, warn=True, verbose=False): #stopThr = 1e-9
    
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
    warn: 

    verbose:

        
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
    # POT backend
    nx = get_backend(X_s, X_t)
    ns, nt = X_s.shape[0], X_t.shape[0]
    if a is None:
        a = unif(ns, type_as=X_s)
    if b is None:
        b = unif(nt, type_as=X_t)
    
    d = X_s.shape[1]
    
    # First low rank decomposition of the cost matrix (A)
    M1 = nx.zeros((ns,(d+2)))
    M1[:,0] = [nx.norm(X_s[i,:])**2 for i in range(ns)]
    M1[:,1] = nx.ones(ns)
    M1[:,2:] = -2*X_s

    # Second low rank decomposition of the cost matrix (B)
    M2 = nx.zeros((nt,(d+2)))
    M2[:,0] = nx.ones(nt)
    M2[:,1] = [nx.norm(X_t[i,:])**2 for i in range(nt)]
    M2[:,2:] = X_t

    # Compute rank
    rank = min(ns, nt, rank)
    r = rank
    
    # Alpha: lower bound for 1/rank
    if alpha == 'auto':
        alpha = 1e-3 # no convergence with alpha = 1 / (r+1)
        
    if (1/r < alpha) or (alpha < 0):
        warnings.warn("The provided alpha value might lead to instabilities.") 

    # Compute gamma
    L = nx.sqrt(3*(2/(alpha**4))*((nx.norm(M1)*nx.norm(M2))**2) + (reg + (2/(alpha**3))*(nx.norm(M1)*nx.norm(M2)))**2)
    gamma = 1/(2*L)
    
    # Initialize the low rank matrices Q, R, g 
    Q, R, g = nx.ones((ns,r)), nx.ones((nt,r)), nx.ones(r) 
    
    # Initialize parameters for Dykstra algorithm
    q3_1, q3_2 = nx.ones(r), nx.ones(r)
    u1, u2 = nx.ones(ns), nx.ones(nt)
    v1, v2 = nx.ones(r), nx.ones(r)
    v1_, v2_ = nx.ones(r), nx.ones(r)
    q1, q2 = nx.ones(r), nx.ones(r)
    dykstra_p = [g, q3_1, q3_2, v1_, v2_, q1, q2, u1, u2, v1, v2]


    for ii in range(numItermax): 
        CR_ = nx.dot(M2.T, R)
        CR = nx.dot(M1, CR_) 
        
        CQ_ = nx.dot(M1.T, Q)
        CQ = nx.dot(M2, CQ_)
        
        diag_g = (1/g)[:,None]

        eps1 = nx.exp(-gamma*(nx.dot(CR,diag_g)) - ((gamma*reg)-1)*nx.log(Q))
        eps2 = nx.exp(-gamma*(nx.dot(CQ,diag_g)) - ((gamma*reg)-1)*nx.log(R))
        omega = nx.diag(nx.dot(Q.T, CR))
        eps3 = nx.exp(gamma*omega/(g**2) - (gamma*reg - 1)*nx.log(g))

        Q, R, dykstra_p = LR_Dysktra(eps1, eps2, eps3, a, b, alpha, dykstra_p, stopThr, nx)
        g = dykstra_p[0]

    #     if verbose:
    #             if ii % 200 == 0:
    #                 print(
    #                     '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
    #             print('{:5d}|{:8e}|'.format(ii, err))

    # else: 
    #     if warn:
    #         warnings.warn("Sinkhorn did not converge. You might want to "
    #                       "increase the number of iterations `numItermax` "
    #                       "or the regularization parameter `reg`.")


    # Compute OT value using trace formula for scalar product 
    v1 = nx.dot(Q.T,M1)
    v2 = nx.dot(R,nx.dot(diag_g.T,v1))
    value_linear = nx.sum(nx.diag(nx.dot(v2,M2.T))) # compute Trace

    #value = value_linear + reg * nx.sum(plan * nx.log(plan + 1e-16))

    #value
    
    return value_linear, Q, R, g





############################################################################
## Test with X_s, X_t from ot.datasets
#############################################################################

import numpy as np
import ot 

Xs, _ = ot.datasets.make_data_classif('3gauss', n=1000)
Xt, _ = ot.datasets.make_data_classif('3gauss2', n=1500)

ns = Xs.shape[0]
nt = Xt.shape[0]

a = unif(ns)
b = unif(nt)

Q, R, g = lowrank_sinkhorn(Xs, Xt, reg=0.1, verbose=True, numItermax=20)
M = ot.dist(Xs,Xt)
P = np.dot(Q,np.dot(np.diag(1/g),R.T))

print(np.sum(P))



