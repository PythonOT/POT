"""
Low rank OT solvers
"""

# Author: Laurène David <laurene.david@ip-paris.fr>
#
# License: MIT License


import warnings
from .utils import unif, LazyTensor
from .backend import get_backend


def compute_lr_cost_matrix(X_s, X_t, nx=None):
    """
    Compute low rank decomposition of a sqeuclidean cost matrix. 
    This function won't work for other metrics. 

    See "Section 3.5, proposition 1" of the paper

    References
    ----------
    .. Scetbon, M., Cuturi, M., & Peyré, G (2021).
        Low-Rank Sinkhorn Factorization. arXiv preprint arXiv:2103.04737. 
    """

    if nx is None:
        nx = get_backend(X_s,X_t)
        
    ns = X_s.shape[0]
    nt = X_t.shape[0]
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

    return M1, M2



def LR_Dysktra(eps1, eps2, eps3, p1, p2, alpha, stopThr, numItermax, warn, nx=None): 
    """
    Implementation of the Dykstra algorithm for the Low Rank sinkhorn OT solver.

    References
    ----------
    .. Scetbon, M., Cuturi, M., & Peyré, G (2021).
        Low-Rank Sinkhorn Factorization. arXiv preprint arXiv:2103.04737.

    """

    # POT backend if None
    if nx is None:
        nx = get_backend(eps1, eps2, eps3, p1, p2)


    # ----------------- Initialisation of Dykstra algorithm -----------------
    r = len(eps3) # rank
    g_ = eps3.copy() # \tilde{g}
    q3_1, q3_2 = nx.ones(r), nx.ones(r) # q^{(3)}_1, q^{(3)}_2
    v1_, v2_ = nx.ones(r), nx.ones(r) # \tilde{v}^{(1)}, \tilde{v}^{(2)}
    q1, q2 = nx.ones(r), nx.ones(r) # q^{(1)}, q^{(2)} 
    err = 1 # initial error


    # --------------------- Dykstra algorithm -------------------------
    
    # See Section 3.3 - "Algorithm 2 LR-Dykstra" in paper
    
    for ii in range(numItermax):    
        if err > stopThr:

            # Compute u^{(1)} and u^{(2)}
            u1 = p1 / nx.dot(eps1, v1_)
            u2 = p2 / nx.dot(eps2, v2_)

            # Compute g, g^{(3)}_1 and update \tilde{g}
            g = nx.maximum(alpha, g_ * q3_1)
            q3_1 = (g_ * q3_1) / g
            g_ = g.copy()

            # Compute new value of g with \prod
            prod1 = ((v1_ * q1) * nx.dot(eps1.T, u1))
            prod2 = ((v2_ * q2) * nx.dot(eps2.T, u2))
            g = (g_ * q3_2 * prod1 * prod2)**(1/3)

            # Compute v^{(1)} and v^{(2)}
            v1 = g / nx.dot(eps1.T,u1)
            v2 = g / nx.dot(eps2.T,u2)

            # Compute q^{(1)}, q^{(2)} and q^{(3)}_2
            q1 = (v1_ * q1) / v1
            q2 = (v2_ * q2) / v2
            q3_2 = (g_ * q3_2) / g
            
            # Update values of \tilde{v}^{(1)}, \tilde{v}^{(2)} and \tilde{g}
            v1_, v2_ = v1.copy(), v2.copy()
            g_ = g.copy()

            # Compute error
            err1 = nx.sum(nx.abs(u1 * (eps1 @ v1) - p1))
            err2 = nx.sum(nx.abs(u2 * (eps2 @ v2) - p2))
            err = err1 + err2

        else:
            break
    
    else: 
        if warn:
            warnings.warn("Sinkhorn did not converge. You might want to "
                          "increase the number of iterations `numItermax` ")

    # Compute low rank matrices Q, R
    Q = u1[:,None] * eps1 * v1[None,:]
    R = u2[:,None] * eps2 * v2[None,:]

    return Q, R, g
    
    


#################################### LOW RANK SINKHORN ALGORITHM #########################################


def lowrank_sinkhorn(X_s, X_t, a=None, b=None, reg=0, rank="auto", alpha="auto", 
                     numItermax=10000, stopThr=1e-9, warn=True, shape_plan="auto"): 
    
    r'''
    Solve the entropic regularization optimal transport problem under low-nonnegative rank constraints.

    The function solves the following optimization problem:

    .. math::
        \mathop{\inf_{(Q,R,g) \in \mathcal{C(a,b,r)}}} \langle C, Q\mathrm{diag}(1/g)R^T \rangle - 
            \mathrm{reg} \cdot H((Q,R,g))
        
    where :
    - :math:`C` is the (`dim_a`, `dim_b`) metric cost matrix
    - :math:`H((Q,R,g))` is the values of the three respective entropies evaluated for each term.
    - :math: `Q` and `R` are the low-rank matrix decomposition of the OT plan
    - :math: `g` is the weight vector for the low-rank decomposition of the OT plan
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are source and target weights (histograms, both sum to 1)
    - :math: `r` is the rank of the OT plan
    - :math: `\mathcal{C(a,b,r)}` are the low-rank couplings of the OT problem 
        \mathcal{C(a,b,r)} = \mathcal{C_1(a,b,r)} \cap \mathcal{C_2(r)} with 
            \mathcal{C_1(a,b,r)} = \{ (Q,R,g) s.t Q\mathbb{1}_r = a, R^T \mathbb{1}_m = b \}
            \mathcal{C_2(r)} = \{ (Q,R,g) s.t Q\mathbb{1}_n = R^T \mathbb{1}_m = g \}
    

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
    rank: int, default "auto"
        Nonnegative rank of the OT plan
    alpha: int, default "auto"
        Lower bound for the weight vector g (>0 and <1/r)
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshold on error (>0)
    warn : bool, optional
        if True, raises a warning if the algorithm doesn't convergence.
    shape_plan : tuple 
        Shape of the lazy_plan 

        
    Returns
    -------
    lazy_plan : LazyTensor()
        OT plan in a LazyTensor object of shape (shape_plan) 
        See :any:`LazyTensor` for more information.
    value : float
        Optimal value of the optimization problem,
    value_linear : float
        Linear OT loss with the optimal OT 
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

    # Initialize weights a, b
    if a is None:
        a = unif(ns, type_as=X_s)
    if b is None:
        b = unif(nt, type_as=X_t)

    # Compute rank (see Section 3.1, def 1)
    r = rank
    if rank == "auto":
        r = min(ns, nt)

    # Check values of alpha, the lower bound for 1/rank 
    # (see "Section 3.2: The Low-rank OT Problem (LOT)" in the paper)
    if alpha == 'auto':
        alpha = 1e-10 

    if (1/r < alpha) or (alpha < 0):
        warnings.warn("The provided alpha value might lead to instabilities.")

    # Default value for shape tensor parameter in LazyTensor 
    if shape_plan == "auto":
        shape_plan = (ns,nt) 

    # Low rank decomposition of the sqeuclidean cost matrix (A, B)
    M1, M2 = compute_lr_cost_matrix(X_s, X_t, nx=None)

    # Compute gamma (see "Section 3.4, proposition 4" in the paper)
    L = nx.sqrt(3*(2/(alpha**4))*((nx.norm(M1)*nx.norm(M2))**2) + (reg + (2/(alpha**3))*(nx.norm(M1)*nx.norm(M2)))**2)
    gamma = 1/(2*L)
    
    # Initialize the low rank matrices Q, R, g 
    Q, R, g = nx.ones((ns,r)), nx.ones((nt,r)), nx.ones(r) 
    k = 100 # not specified in paper ?



    # -------------------------- Low rank algorithm ------------------------------
    # see "Section 3.3, Algorithm 3 LOT" in the paper

    for ii in range(k): 
        # Compute the C*R dot matrix using the lr decomposition of C
        CR_ = nx.dot(M2.T, R)
        CR = nx.dot(M1, CR_) 
        
        # Compute the C.t * Q dot matrix using the lr decomposition of C
        CQ_ = nx.dot(M1.T, Q)
        CQ = nx.dot(M2, CQ_)
        
        diag_g = nx.diag(1/g)
        
        eps1 = nx.exp(-gamma*(nx.dot(CR,diag_g)) - ((gamma*reg)-1)*nx.log(Q))
        eps2 = nx.exp(-gamma*(nx.dot(CQ,diag_g)) - ((gamma*reg)-1)*nx.log(R))
        omega = nx.diag(nx.dot(Q.T, CR))
        eps3 = nx.exp(gamma*omega/(g**2) - (gamma*reg - 1)*nx.log(g))

        Q, R, g = LR_Dysktra(eps1, eps2, eps3, a, b, alpha, stopThr, numItermax, warn, nx)



    # ----------------- Compute lazy_plan, value and value_linear  ------------------
    # see "Section 3.2: The Low-rank OT Problem" in the paper

    # Compute lazy plan (using LazyTensor class)
    plan1 = Q 
    plan2 = nx.dot(nx.diag(1/g),R.T) # low memory cost since shape (r*m)
    compute_plan = lambda i,j,P1,P2: nx.dot(P1[i,:], P2[:,j]) # function for LazyTensor
    lazy_plan = LazyTensor(shape_plan, compute_plan, P1=plan1, P2=plan2) 
    
    # Compute value_linear (using trace formula)
    v1 = nx.dot(Q.T,M1)
    v2 = nx.dot(R,nx.dot(diag_g.T,v1))
    value_linear = nx.sum(nx.diag(nx.dot(M2.T, v2)))

    # Compute value with entropy reg (entropy of Q, R, g must be computed separatly, see "Section 3.2" in the paper)
    reg_Q = nx.sum(Q * nx.log(Q + 1e-16)) # entropy for Q
    reg_g = nx.sum(g * nx.log(g + 1e-16)) # entropy for g
    reg_R = nx.sum(R * nx.log(R + 1e-16)) # entropy for R
    value = value_linear + reg * (reg_Q + reg_g + reg_R)

    return value, value_linear, lazy_plan, Q, R, g




