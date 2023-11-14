#####################################################################################################
####################################### WORK IN PROGRESS ############################################
#####################################################################################################


""" Test for low rank sinkhorn solvers """

import ot
import numpy as np
import pytest
from itertools import product



################################################## WORK IN PROGRESS #######################################################

# Add test functions for each function in lowrank.py file ?

def test_lowrank_sinkhorn():
    # test low rank sinkhorn
    n = 100
    a = ot.unif(n)
    b = ot.unif(n)

    X_s = np.reshape(1.0 * np.arange(n), (n, 1))
    X_t = np.reshape(1.0 * np.arange(n), (n, 1))

    # what to test for value, value_linear, Q, R and g ?
    value, value_linear, lazy_plan, Q, R, g = ot.lowrank.lowrank_sinkhorn(X_s, X_t, a, b, 0.1)
    P = lazy_plan[:] # default shape for lazy_plan in lowrank_sinkhorn is (ns, nt)

    # check constraints for P
    np.testing.assert_allclose(
        a, P.sum(1), atol=1e-05) 
    np.testing.assert_allclose(
        b, P.sum(0), atol=1e-05) 
    
    # check warn parameter when Dykstra algorithm doesn't converge 
    with pytest.warns(UserWarning):
        ot.lowrank.lowrank_sinkhorn(X_s, X_t, 0.1, a, b, stopThr=0, numItermax=1)



@pytest.mark.parametrize(("alpha, rank"),((0.8,2),(0.5,3),(0.2,4))) 
def test_lowrank_sinkhorn_alpha_warning(alpha,rank):
    # Test warning for value of alpha 
    n = 100
    a = ot.unif(n)
    b = ot.unif(n)

    X_s = np.reshape(1.0 * np.arange(n), (n, 1))
    X_t = np.reshape(1.0 * np.arange(0, n), (n, 1))
    
    with pytest.warns(UserWarning):
        ot.lowrank.lowrank_sinkhorn(X_s, X_t, 0.1, a, b, rank=rank, alpha=alpha, warn=False) # remove warning for lack of convergence 
    


def test_lowrank_sinkhorn_backends(nx):
    # Test low rank sinkhorn for different backends
    n = 100
    a = ot.unif(n)
    b = ot.unif(n)

    X_s = np.reshape(1.0 * np.arange(n), (n, 1))
    X_t = np.reshape(1.0 * np.arange(0, n), (n, 1))

    ab, bb, X_sb, X_tb = nx.from_numpy(a, b, X_s, X_t)

    value, value_linear, lazy_plan, Q, R, g = ot.lowrank.lowrank_sinkhorn(X_sb, X_tb, ab, bb, 0.1)
    P = lazy_plan[:] # default shape for lazy_plan in lowrank_sinkhorn is (ns, nt)

    np.testing.assert_allclose(ab, P.sum(1), atol=1e-05) 
    np.testing.assert_allclose(bb, P.sum(0), atol=1e-05)




