#####################################################################################################
####################################### WORK IN PROGRESS ############################################
#####################################################################################################


""" Test for low rank sinkhorn solvers """

import ot
import numpy as np
import pytest
from itertools import product


def test_LR_Dykstra():
    # test for LR_Dykstra algorithm ? catch nan values ?
    pass


@pytest.mark.parametrize("verbose, warn", product([True, False], [True, False]))
def test_lowrank_sinkhorn(verbose, warn):
    # test low rank sinkhorn
    n = 100
    a = ot.unif(n)
    b = ot.unif(n)

    X_s = np.reshape(1.0 * np.arange(n), (n, 1))
    X_t = np.reshape(1.0 * np.arange(0, n), (n, 1))

    Q_sqe, R_sqe, g_sqe = ot.lowrank.lowrank_sinkhorn(X_s, X_t, a, b, 0.1)
    P_sqe = np.dot(Q_sqe,np.dot(np.diag(1/g_sqe),R_sqe.T))

    Q_m, R_m, g_m = ot.lowrank.lowrank_sinkhorn(X_s, X_t, a, b, 0.1, metric='euclidean')
    P_m = np.dot(Q_m,np.dot(np.diag(1/g_m),R_m.T))

    # check constraints
    np.testing.assert_allclose(
        a, P_sqe.sum(1), atol=1e-05) # metric sqeuclidian
    np.testing.assert_allclose(
        b, P_sqe.sum(0), atol=1e-05) # metric sqeuclidian
    np.testing.assert_allclose(
        a, P_m.sum(1), atol=1e-05) # metric euclidian
    np.testing.assert_allclose(
        b, P_m.sum(0), atol=1e-05) # metric euclidian
    
    with pytest.warns(UserWarning):
        ot.lowrank.lowrank_sinkhorn(X_s, X_t, 0.1, a, b, stopThr=0, numItermax=1)



@pytest.mark.parametrize(("alpha, rank"),((0.8,2),(0.5,3),(0.2,4))) 
def test_lowrank_sinkhorn_alpha_warning(alpha,rank):
    # test warning for value of alpha 
    n = 100
    a = ot.unif(n)
    b = ot.unif(n)

    X_s = np.reshape(1.0 * np.arange(n), (n, 1))
    X_t = np.reshape(1.0 * np.arange(0, n), (n, 1))
    
    with pytest.warns(UserWarning):
        ot.lowrank.lowrank_sinkhorn(X_s, X_t, 0.1, a, b, r=rank, alpha=alpha, warn=False)
    


def test_lowrank_sinkhorn_backends(nx):
    # test low rank sinkhorn for different backends
    n = 100
    a = ot.unif(n)
    b = ot.unif(n)

    X_s = np.reshape(1.0 * np.arange(n), (n, 1))
    X_t = np.reshape(1.0 * np.arange(0, n), (n, 1))

    ab, bb, X_sb, X_tb = nx.from_numpy(a, b, X_s, X_t)

    Q, R, g = nx.to_numpy(ot.lowrank.lowrank_sinkhorn(X_sb, X_tb, ab, bb, 0.1))
    P = np.dot(Q,np.dot(np.diag(1/g),R.T))

    np.testing.assert_allclose(a, P.sum(1), atol=1e-05) 
    np.testing.assert_allclose(b, P.sum(0), atol=1e-05)




