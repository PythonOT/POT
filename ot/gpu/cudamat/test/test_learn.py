import pdb
import numpy as np
import nose
import cudamat as cm

def setup():
    cm.cublas_init()

def teardown():
    cm.cublas_shutdown()

def test_mult_by_sigmoid_deriv():
    m = 256
    n = 128
    c_targets = np.array(np.random.randn(m, n)*10, dtype=np.float32, order='F')
    c_acts = np.array(np.random.rand(m, n), dtype=np.float32, order='F')

    g_targets = cm.CUDAMatrix(c_targets)
    g_acts = cm.CUDAMatrix(c_acts)

    c_targets = c_targets * c_acts * (1. - c_acts)
    cm.learn.mult_by_sigmoid_deriv(g_targets, g_acts)

    assert np.max(np.abs(c_acts - g_acts.asarray())) < 10**-2, "Error in cudamat.learn.mult_by_sigmoid_deriv exceeded threshold"

if __name__ == '__main__':
    nose.runmodule()
