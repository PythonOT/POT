import numpy as np
import nose
import cudamat as cm

def setup():
    cm.cublas_init()

def teardown():
    cm.cublas_shutdown()

def test_reshape():
    m = 256
    n = 1
    cm1 = np.array(np.random.rand(n, m)*10, dtype=np.float64, order='F')
    cm2 = np.array(np.random.rand(m, n)*10, dtype=np.float64, order='F')

    gm1 = cm.CUDAMatrix(cm1)
    gm2 = cm.CUDAMatrix(cm2)

    gm1.reshape((m, n))
    gm2.assign(gm1)
    gm1.reshape((n, m))

    gm1.copy_to_host()
    gm2.copy_to_host()

    assert np.max(np.abs(gm1.numpy_array - gm2.numpy_array.T)) < 10**-2, "Error in CUDAMatrix.reshape exceeded threshold"

def test_T_field():
    m = 256
    n = 128
    cm1 = np.array(np.random.rand(n, m)*10, dtype=np.float64, order='F')
    cm2 = np.array(np.random.rand(m, 1)*10, dtype=np.float64, order='F')
    gm1 = cm.CUDAMatrix(cm1)
    gm2 = cm.CUDAMatrix(cm2)

    # test dot
    gm = cm.dot(gm2.T, gm1.T)
    c = np.dot(cm2.T, cm1.T)
    gm.copy_to_host()

    assert np.max(np.abs(gm.numpy_array - c)) < 10**-2, "Error in CUDAMatrix.dot with TransposedCUDAMatrix exceeded threshold"

    # test add_dot
    cm3 = np.array(np.random.rand(1, n)*10, dtype=np.float64, order='F')
    gm3 = cm.CUDAMatrix(cm3)
    gm3.add_dot(gm2.T, gm1.T)
    c = cm3 + np.dot(cm2.T, cm1.T)
    gm3.copy_to_host()

    assert np.max(np.abs(gm3.numpy_array - c)) < 10**-2, "Error in CUDAMatrix.add_dot TransposedCUDAMatrix exceeded threshold"

    # test add_sums
    gm2.add_sums(gm1.T, axis = 1)
    c = cm2 + np.atleast_2d(cm1.sum(0)).T
    gm2.copy_to_host()

    assert np.max(np.abs(gm2.numpy_array - c)) < 10**-2, "Error in CUDAMatrix.add_sums TransposedCUDAMatrix exceeded threshold"

def test_assign():
    m = 256
    n = 128
    a = np.array(np.random.rand(m, n)*10, dtype=np.float64, order='F')
    b = np.array(np.random.rand(m, n)*10, dtype=np.float64, order='F')

    m1 = cm.CUDAMatrix(a)
    m2 = cm.CUDAMatrix(b)

    m1.assign(m2)
    m1.copy_to_host()

    assert np.max(np.abs(m1.numpy_array - m2.numpy_array)) < 10**-4, "Error in CUDAMatrix.assign exceeded threshold"

def test_assign_scalar():
    m = 256
    n = 128
    a = np.array(np.random.rand(m, n)*10, dtype=np.float64, order='F')

    m1 = cm.CUDAMatrix(a)

    m1.assign(np.pi)
    m1.copy_to_host()

    assert np.max(np.abs(m1.numpy_array - np.pi)) < 10**-4, "Error in CUDAMatrix.assign_scalar exceeded threshold"

def test_get_row_slice():
    m = 256
    n = 128
    start = 11
    end = 54

    a = np.array(np.random.rand(m, n)*10, dtype=np.float64, order='F')
    b = np.array(np.random.rand(end-start, n)*10, dtype=np.float64, order='F')

    c = np.array(a[start:end,:], order='F')

    m1 = cm.CUDAMatrix(a)
    m2 = cm.CUDAMatrix(b)
    m1.get_row_slice(start, end, target = m2)
    m3 = m1.get_row_slice(start, end)
    m2.copy_to_host()
    m3.copy_to_host()

    assert np.max(np.abs(c - m2.numpy_array)) < 10**-4, "Error in CUDAMatrix.get_row_slice exceeded threshold"
    assert np.max(np.abs(c - m3.numpy_array)) < 10**-4, "Error in CUDAMatrix.get_row_slice exceeded threshold"

def test_set_row_slice():
    m = 256
    n = 128
    start = 11
    end = 54

    a = np.array(np.random.rand(m, n)*10, dtype=np.float64, order='F')
    b = np.array(np.random.rand(end-start, n)*10, dtype=np.float64, order='F')

    c = a.copy()
    c[start:end,:] = b

    m1 = cm.CUDAMatrix(a)
    m2 = cm.CUDAMatrix(b)
    m1.set_row_slice(start, end, m2)
    m1.copy_to_host()

    assert np.max(np.abs(c - m1.numpy_array)) < 10**-4, "Error in CUDAMatrix.set_row_slice exceeded threshold"

def test_transpose():
    m = 6
    n = 128

    a = np.array(np.random.rand(m, n)*10, dtype=np.float64, order='F')
    b = np.array(np.random.rand(n, m), dtype=np.float64, order='F')

    c = a.copy().T

    m = cm.CUDAMatrix(a)
    mt1 = cm.CUDAMatrix(b)
    m.transpose(target = mt1)
    mt2 = m.transpose()

    mt1.copy_to_host()
    mt2.copy_to_host()

    assert np.max(np.abs(c - mt1.numpy_array)) < 10**-4, "Error in CUDAMatrix.transpose exceeded threshold"
    assert np.max(np.abs(c - mt2.numpy_array)) < 10**-4, "Error in CUDAMatrix.transpose exceeded threshold"

def test_slice():
    m = 256
    n = 128
    a = np.array(np.random.rand(m, n)*10, dtype=np.float64, order='F')

    c = np.array(a[:,32:64], order='F')

    m1 = cm.CUDAMatrix(a)
    m2 = m1.slice(32, 64)
    m2.copy_to_host()

    assert np.max(np.abs(c - m2.numpy_array)) < 10**-4, "Error in CUDAMatrix.slice exceeded threshold"


def test_add_col_vec():
    m = 250
    n = 120
    a = np.array(np.random.rand(m, n)*10, dtype=np.float64, order='F')
    b = np.array(np.random.rand(m, 1)*10, dtype=np.float64, order='F')
    t = np.array(np.random.rand(m, n)*10, dtype=np.float64, order='F')

    c = a + b

    m1 = cm.CUDAMatrix(a)
    m2 = cm.CUDAMatrix(b)
    m3 = cm.CUDAMatrix(t)

    m1.add_col_vec(m2, target = m3)
    m1.add_col_vec(m2)
    m1.copy_to_host()
    m3.copy_to_host()

    assert np.max(np.abs(c - m1.numpy_array)) < 10**-4, "Error in CUDAMatrix.add_col_vec exceeded threshold"
    assert np.max(np.abs(c - m3.numpy_array)) < 10**-4, "Error in CUDAMatrix.add_col_vec exceeded threshold"

def test_add_col_mult():
    m = 256
    n = 128
    mult = np.pi
    a = np.array(np.random.rand(m, n)*10, dtype=np.float64, order='F')
    b = np.array(np.random.rand(m, 1)*10, dtype=np.float64, order='F')
    t = np.array(np.random.rand(m, n)*10, dtype=np.float64, order='F')

    c = a + mult * b

    m1 = cm.CUDAMatrix(a)
    m2 = cm.CUDAMatrix(b)
    m3 = cm.CUDAMatrix(t)

    m1.add_col_mult(m2, mult, target = m3)
    m1.add_col_mult(m2, mult)
    m1.copy_to_host()
    m3.copy_to_host()

    assert np.max(np.abs(c - m1.numpy_array)) < 10**-4, "Error in CUDAMatrix.add_col_mult exceeded threshold"
    assert np.max(np.abs(c - m3.numpy_array)) < 10**-4, "Error in CUDAMatrix.add_col_mult exceeded threshold"

def test_add_row_vec():
    m = 256
    n = 128
    a = np.array(np.random.rand(m, n)*10, dtype=np.float64, order='F')
    b = np.array(np.random.rand(1, n)*10, dtype=np.float64, order='F')
    t = np.array(np.random.rand(m, n)*10, dtype=np.float64, order='F')

    c = a + b

    m1 = cm.CUDAMatrix(a)
    m2 = cm.CUDAMatrix(b)
    m3 = cm.CUDAMatrix(t)

    m1.add_row_vec(m2, target = m3)
    m1.add_row_vec(m2)
    m1.copy_to_host()
    m3.copy_to_host()

    assert np.max(np.abs(c - m1.numpy_array)) < 10**-4, "Error in CUDAMatrix.add_row_vec exceeded threshold"
    assert np.max(np.abs(c - m3.numpy_array)) < 10**-4, "Error in CUDAMatrix.add_row_vec exceeded threshold"

def test_mult_by_col():
    m = 256
    n = 128
    a = np.array(np.random.rand(m, n)*10, dtype=np.float64, order='F')
    b = np.array(np.random.rand(m, 1)*10, dtype=np.float64, order='F')
    t = np.array(np.random.rand(m, n)*10, dtype=np.float64, order='F')

    c = a * b

    m1 = cm.CUDAMatrix(a)
    m2 = cm.CUDAMatrix(b)
    m3 = cm.CUDAMatrix(t)

    m1.mult_by_col(m2, target = m3)
    m1.mult_by_col(m2)
    m1.copy_to_host()
    m3.copy_to_host()

    assert np.max(np.abs(c - m1.numpy_array)) < 10**-4, "Error in CUDAMatrix.mult_by_col exceeded threshold"
    assert np.max(np.abs(c - m3.numpy_array)) < 10**-4, "Error in CUDAMatrix.mult_by_col exceeded threshold"

def test_mult_by_row():
    m = 256
    n = 128
    a = np.array(np.random.rand(m, n)*10, dtype=np.float64, order='F')
    b = np.array(np.random.rand(1, n)*10, dtype=np.float64, order='F')
    t = np.array(np.random.rand(m, n)*10, dtype=np.float64, order='F')

    c = a * b

    m1 = cm.CUDAMatrix(a)
    m2 = cm.CUDAMatrix(b)
    m3 = cm.CUDAMatrix(t)

    m1.mult_by_row(m2, target = m3)
    m1.mult_by_row(m2)
    m1.copy_to_host()
    m3.copy_to_host()

    assert np.max(np.abs(c - m1.numpy_array)) < 10**-4, "Error in CUDAMatrix.mult_by_row exceeded threshold"
    assert np.max(np.abs(c - m3.numpy_array)) < 10**-4, "Error in CUDAMatrix.mult_by_row exceeded threshold"

def test_div_by_col():
    m = 256
    n = 128
    a = np.array(np.random.rand(m, n)*10, dtype=np.float64, order='F')
    b = np.array(np.random.rand(m, 1)*10, dtype=np.float64, order='F') + 0.1
    t = np.array(np.random.rand(m, n)*10, dtype=np.float64, order='F')

    c = a / b

    m1 = cm.CUDAMatrix(a)
    m2 = cm.CUDAMatrix(b)
    m3 = cm.CUDAMatrix(t)

    m1.div_by_col(m2, target = m3)
    m1.div_by_col(m2)
    m1.copy_to_host()
    m3.copy_to_host()

    assert np.max(np.abs(c - m1.numpy_array)) < 10**-4, "Error in CUDAMatrix.div_by_col exceeded threshold"
    assert np.max(np.abs(c - m3.numpy_array)) < 10**-4, "Error in CUDAMatrix.div_by_col exceeded threshold"

def test_div_by_row():
    m = 256
    n = 128
    a = np.array(np.random.rand(m, n)*10, dtype=np.float64, order='F')
    b = np.array(np.random.rand(1, n)*10, dtype=np.float64, order='F') + 0.1
    t = np.array(np.random.rand(m, n)*10, dtype=np.float64, order='F')

    c = a / b

    m1 = cm.CUDAMatrix(a)
    m2 = cm.CUDAMatrix(b)
    m3 = cm.CUDAMatrix(t)

    m1.div_by_row(m2, target = m3)
    m1.div_by_row(m2)
    m1.copy_to_host()
    m3.copy_to_host()

    assert np.max(np.abs(c - m1.numpy_array)) < 10**-4, "Error in CUDAMatrix.div_by_row exceeded threshold"
    assert np.max(np.abs(c - m3.numpy_array)) < 10**-4, "Error in CUDAMatrix.div_by_row exceeded threshold"

def test_sum():
    m = 256
    n = 128
    a = np.array(np.random.rand(m, n)*10, dtype=np.float64, order='F')
    t1 = np.array(np.random.rand(1, n)*10, dtype=np.float64, order='F')
    t2 = np.array(np.random.rand(m, 1)*10, dtype=np.float64, order='F')

    mult = 0.8
    c1 = np.atleast_2d(a.sum(0)) * mult
    c2 = np.atleast_2d(a.sum(1)).T

    m = cm.CUDAMatrix(a)
    mt1 = cm.CUDAMatrix(t1)
    mt2 = cm.CUDAMatrix(t2)

    m.sum(axis = 0, target = mt1, mult = mult)
    mt1r = m.sum(axis = 0, mult = mult)

    m.sum(axis = 1, target = mt2)
    mt2r = m.sum(axis = 1)

    mt1.copy_to_host()
    mt1r.copy_to_host()
    mt2.copy_to_host()
    mt2r.copy_to_host()

    assert np.max(np.abs(c1 - mt1.numpy_array)) < 10**-3, "Error in CUDAMatrix.sum exceeded threshold"
    assert np.max(np.abs(c1 - mt1r.numpy_array)) < 10**-3, "Error in CUDAMatrix.sum exceeded threshold"
    assert np.max(np.abs(c2 - mt2.numpy_array)) < 10**-3, "Error in CUDAMatrix.sum exceeded threshold"
    assert np.max(np.abs(c2 - mt2r.numpy_array)) < 10**-3, "Error in CUDAMatrix.sum exceeded threshold"

def test_sum_trans():
    m = 256
    n = 128
    a = np.array(np.random.rand(m, n)*10, dtype=np.float64, order='F')
    t1 = np.array(np.random.rand(1, m)*10, dtype=np.float64, order='F')
    t2 = np.array(np.random.rand(n, 1)*10, dtype=np.float64, order='F')

    c1 = np.atleast_2d(a.T.sum(0))
    c2 = np.atleast_2d(a.T.sum(1)).T

    m = cm.CUDAMatrix(a)
    m.set_trans(True)
    mt1 = cm.CUDAMatrix(t1)
    mt2 = cm.CUDAMatrix(t2)

    m.sum(axis = 0, target = mt1)
    mt1r = m.sum(axis = 0)

    m.sum(axis = 1, target = mt2)
    mt2r = m.sum(axis = 1)

    mt1.copy_to_host()
    mt1r.copy_to_host()
    mt2.copy_to_host()
    mt2r.copy_to_host()

    assert np.max(np.abs(c1 - mt1.numpy_array)) < 10**-3, "Error in CUDAMatrix.sum exceeded threshold"
    assert np.max(np.abs(c1 - mt1r.numpy_array)) < 10**-3, "Error in CUDAMatrix.sum exceeded threshold"
    assert np.max(np.abs(c2 - mt2.numpy_array)) < 10**-3, "Error in CUDAMatrix.sum exceeded threshold"
    assert np.max(np.abs(c2 - mt2r.numpy_array)) < 10**-3, "Error in CUDAMatrix.sum exceeded threshold"

def test_mean():
    m = 256
    n = 128
    a = np.array(np.random.rand(m, n)*10, dtype=np.float64, order='F')
    t1 = np.array(np.random.rand(1, n)*10, dtype=np.float64, order='F')
    t2 = np.array(np.random.rand(m, 1)*10, dtype=np.float64, order='F')

    c1 = np.atleast_2d(a.mean(0))
    c2 = np.atleast_2d(a.mean(1)).T

    m = cm.CUDAMatrix(a)
    mt1 = cm.CUDAMatrix(t1)
    mt2 = cm.CUDAMatrix(t2)

    m.mean(axis = 0, target = mt1)
    mt1r = m.mean(axis = 0)

    m.mean(axis = 1, target = mt2)
    mt2r = m.mean(axis = 1)

    mt1.copy_to_host()
    mt1r.copy_to_host()
    mt2.copy_to_host()
    mt2r.copy_to_host()

    assert np.max(np.abs(c1 - mt1.numpy_array)) < 10**-3, "Error in CUDAMatrix.sum exceeded threshold"
    assert np.max(np.abs(c1 - mt1r.numpy_array)) < 10**-3, "Error in CUDAMatrix.sum exceeded threshold"
    assert np.max(np.abs(c2 - mt2.numpy_array)) < 10**-3, "Error in CUDAMatrix.sum exceeded threshold"
    assert np.max(np.abs(c2 - mt2r.numpy_array)) < 10**-3, "Error in CUDAMatrix.sum exceeded threshold"

def test_add_sums():
    m = 256
    n = 128

    a = np.array(np.random.rand(m, n)*10, dtype=np.float64, order='F')
    t1 = np.array(np.random.rand(m, 1)*10, dtype=np.float64, order='F')
    t2 = np.array(np.random.rand(1, n)*10, dtype=np.float64, order='F')

    mult = np.pi
    beta = 0.7

    c1 = beta * t1 + mult * np.atleast_2d(a.sum(1)).T
    c2 = t2 + np.atleast_2d(a.sum(0))

    m = cm.CUDAMatrix(a)
    mt1 = cm.CUDAMatrix(t1)
    mt2 = cm.CUDAMatrix(t2)

    mt1.add_sums(m, axis = 1, mult = np.pi, beta = beta)
    mt2.add_sums(m, axis = 0)

    mt1.copy_to_host()
    mt2.copy_to_host()

    assert np.max(np.abs(c1 - mt1.numpy_array)) < 10**-3, "Error in CUDAMatrix.add_sums exceeded threshold"
    assert np.max(np.abs(c2 - mt2.numpy_array)) < 10**-3, "Error in CUDAMatrix.add_sums exceeded threshold"


def test_less_than():
    m = 256
    n = 128
    a = np.array(np.random.randn(m, n)*10, dtype=np.float64, order='F')
    b = np.array(np.random.randn(m, n)*10, dtype=np.float64, order='F')
    t1 = np.array(np.random.randn(m, n)*10, dtype=np.float64, order='F')
    t2 = np.array(np.random.randn(m, n)*10, dtype=np.float64, order='F')
    v = 0.1

    r1 = 1 * (a < b)
    r2 = 1 * (a < v)

    da = cm.CUDAMatrix(a)
    db = cm.CUDAMatrix(b)
    dt1 = cm.CUDAMatrix(t1)
    dt2 = cm.CUDAMatrix(t2)

    da.less_than(db, target = dt1)
    da.less_than(v, target = dt2)
    da.less_than(db)

    da.copy_to_host()
    dt1.copy_to_host()
    dt2.copy_to_host()

    assert np.max(np.abs(r1 - da.numpy_array)) < 10**-4, "Error in CUDAMatrix.less_than exceeded threshold"
    assert np.max(np.abs(r1 - dt1.numpy_array)) < 10**-4, "Error in CUDAMatrix.less_than exceeded threshold"
    assert np.max(np.abs(r2 - dt2.numpy_array)) < 10**-4, "Error in CUDAMatrix.less_than exceeded threshold"

def test_greater_than():
    m = 256
    n = 128
    a = np.array(np.random.randn(m, n)*10, dtype=np.float64, order='F')
    b = np.array(np.random.randn(m, n)*10, dtype=np.float64, order='F')
    t1 = np.array(np.random.randn(m, n)*10, dtype=np.float64, order='F')
    t2 = np.array(np.random.randn(m, n)*10, dtype=np.float64, order='F')
    v = 0.1

    r1 = 1 * (a > b)
    r2 = 1 * (a > v)

    da = cm.CUDAMatrix(a)
    db = cm.CUDAMatrix(b)
    dt1 = cm.CUDAMatrix(t1)
    dt2 = cm.CUDAMatrix(t2)

    da.greater_than(db, target = dt1)
    da.greater_than(v, target = dt2)
    da.greater_than(db)

    da.copy_to_host()
    dt1.copy_to_host()
    dt2.copy_to_host()

    assert np.max(np.abs(r1 - da.numpy_array)) < 10**-4, "Error in CUDAMatrix.greater_than exceeded threshold"
    assert np.max(np.abs(r1 - dt1.numpy_array)) < 10**-4, "Error in CUDAMatrix.greater_than exceeded threshold"
    assert np.max(np.abs(r2 - dt2.numpy_array)) < 10**-4, "Error in CUDAMatrix.greater_than exceeded threshold"

def test_minimum():
    m = 256
    n = 128
    a = np.array(np.random.randn(m, n)*10, dtype=np.float64, order='F')
    b = np.array(np.random.randn(m, n)*10, dtype=np.float64, order='F')
    t1 = np.array(np.random.randn(m, n)*10, dtype=np.float64, order='F')
    t2 = np.array(np.random.randn(m, n)*10, dtype=np.float64, order='F')
    v = 0.1

    r1 = np.minimum(a, b)
    r2 = np.minimum(a, v)

    da = cm.CUDAMatrix(a)
    db = cm.CUDAMatrix(b)
    dt1 = cm.CUDAMatrix(t1)
    dt2 = cm.CUDAMatrix(t2)

    da.minimum(db, target = dt1)
    da.minimum(v, target = dt2)
    da.minimum(db)

    da.copy_to_host()
    dt1.copy_to_host()
    dt2.copy_to_host()

    assert np.max(np.abs(r1 - da.numpy_array)) < 10**-4, "Error in CUDAMatrix.minimum exceeded threshold"
    assert np.max(np.abs(r1 - dt1.numpy_array)) < 10**-4, "Error in CUDAMatrix.minimum exceeded threshold"
    assert np.max(np.abs(r2 - dt2.numpy_array)) < 10**-4, "Error in CUDAMatrix.minimum exceeded threshold"

def test_maximum():
    m = 256
    n = 128
    a = np.array(np.random.randn(m, n)*10, dtype=np.float64, order='F')
    b = np.array(np.random.randn(m, n)*10, dtype=np.float64, order='F')
    t1 = np.array(np.random.randn(m, n)*10, dtype=np.float64, order='F')
    t2 = np.array(np.random.randn(m, n)*10, dtype=np.float64, order='F')
    v = 0.1

    r1 = np.maximum(a, b)
    r2 = np.maximum(a, v)

    da = cm.CUDAMatrix(a)
    db = cm.CUDAMatrix(b)
    dt1 = cm.CUDAMatrix(t1)
    dt2 = cm.CUDAMatrix(t2)

    da.maximum(db, target = dt1)
    da.maximum(v, target = dt2)
    da.maximum(db)

    da.copy_to_host()
    dt1.copy_to_host()
    dt2.copy_to_host()

    assert np.max(np.abs(r1 - da.numpy_array)) < 10**-4, "Error in CUDAMatrix.maximum exceeded threshold"
    assert np.max(np.abs(r1 - dt1.numpy_array)) < 10**-4, "Error in CUDAMatrix.maximum exceeded threshold"
    assert np.max(np.abs(r2 - dt2.numpy_array)) < 10**-4, "Error in CUDAMatrix.maximum exceeded threshold"

def test_minmax():
    m = 256
    n = 128
    for op in 'min', 'max', 'argmin', 'argmax':
        for sign in (1, -1):
            a = np.array(np.random.randn(m, n)*10*sign, dtype=np.float64, order='F')
            t0 = np.array(np.random.rand(1, n)*10, dtype=np.float64, order='F')
            t1 = np.array(np.random.rand(m, 1)*10, dtype=np.float64, order='F')

            r0 = np.atleast_2d(getattr(a, op)(0))
            r1 = np.atleast_2d(getattr(a, op)(1))

            da = cm.CUDAMatrix(a)
            dr10 = cm.CUDAMatrix(t0)
            dr11 = cm.CUDAMatrix(t1)

            getattr(da, op)(axis = 0, target = dr10)
            getattr(da, op)(axis = 1, target = dr11)
            dr20 = getattr(da, op)(axis = 0)
            dr21 = getattr(da, op)(axis = 1)

            dr10.copy_to_host()
            dr11.copy_to_host()
            dr20.copy_to_host()
            dr21.copy_to_host()

            assert np.max(np.abs(r0 - dr10.numpy_array)) < 10**-4, "Error in CUDAMatrix.%s exceeded threshold" % op
            assert np.max(np.abs(r1 - dr11.numpy_array.T)) < 10**-4, "Error in CUDAMatrix.%s exceeded threshold" % op
            assert np.max(np.abs(r0 - dr20.numpy_array)) < 10**-4, "Error in CUDAMatrix.%s exceeded threshold" % op
            assert np.max(np.abs(r1 - dr21.numpy_array.T)) < 10**-4, "Error in CUDAMatrix.%s exceeded threshold" % op

def test_sign():
    m = 256
    n = 128
    a = np.array(np.random.randn(m, n)*10, dtype=np.float64, order='F')
    a[0,0] = 0.
    a[0,1] = -0.
    t = np.array(np.random.randn(m, n)*10, dtype=np.float64, order='F')

    c = np.sign(a)

    m1 = cm.CUDAMatrix(a)
    m3 = cm.CUDAMatrix(t)

    m2 = m1.sign()
    m1.sign(target = m3)

    m2.copy_to_host()
    m3.copy_to_host()

    assert np.max(np.abs(c - m2.numpy_array)) < 10**-4, "Error in CUDAMatrix.sign exceeded threshold"
    assert np.max(np.abs(c - m3.numpy_array)) < 10**-4, "Error in CUDAMatrix.sign exceeded threshold"

def test_sigmoid():
    m = 256
    n = 128
    a = np.array(np.random.randn(m, n)*10, dtype=np.float64, order='F')
    b = np.array(np.random.randn(m, n)*10, dtype=np.float64, order='F')

    c = 1. / (1. + np.exp(-a))

    m1 = cm.CUDAMatrix(a)
    m2 = cm.CUDAMatrix(b)
    m1.apply_sigmoid(target = m2)
    m1.apply_sigmoid()

    m1.copy_to_host()
    m2.copy_to_host()

    assert np.max(np.abs(c - m1.numpy_array)) < 10**-4, "Error in CUDAMatrix.apply_sigmoid exceeded threshold"
    assert np.max(np.abs(c - m2.numpy_array)) < 10**-4, "Error in CUDAMatrix.apply_sigmoid exceeded threshold"

def test_tanh():
    m = 256
    n = 128
    a = np.array(np.random.randn(m, n)*10, dtype=np.float64, order='F')
    b = np.array(np.random.randn(m, n)*10, dtype=np.float64, order='F')

    c = np.tanh(a)

    m1 = cm.CUDAMatrix(a)
    m2 = cm.CUDAMatrix(b)
    m1.apply_tanh(target = m2)
    m1.apply_tanh()

    m1.copy_to_host()
    m2.copy_to_host()

    assert np.max(np.abs(c - m1.numpy_array)) < 10**-4, "Error in CUDAMatrix.apply_tanh exceeded threshold"
    assert np.max(np.abs(c - m2.numpy_array)) < 10**-4, "Error in CUDAMatrix.apply_tanh exceeded threshold"

def test_soft_threshold():
    m = 256
    n = 128
    a = np.array(np.random.randn(m, n)*10, dtype=np.float64, order='F')
    b = np.array(np.random.randn(m, n)*10, dtype=np.float64, order='F')

    alpha = 0.5
    c = np.sign(a) * np.maximum(0, np.abs(a) - alpha)

    m1 = cm.CUDAMatrix(a)
    m2 = cm.CUDAMatrix(b)
    m1.apply_soft_threshold(alpha, target = m2)
    m1.apply_soft_threshold(alpha)

    m1.copy_to_host()
    m2.copy_to_host()

    assert np.max(np.abs(c - m1.numpy_array)) < 10**-4, "Error in CUDAMatrix.apply_soft_threshold exceeded threshold"
    assert np.max(np.abs(c - m2.numpy_array)) < 10**-4, "Error in CUDAMatrix.apply_soft_threshold exceeded threshold"

def test_log():
    m = 256
    n = 128
    a = np.array(np.random.rand(m, n)*10+0.1, dtype=np.float64, order='F')
    b = np.array(np.random.rand(m, n)*10+0.1, dtype=np.float64, order='F')

    c = np.log(a)

    m1 = cm.CUDAMatrix(a)
    m2 = cm.CUDAMatrix(b)
    cm.log(m1, target = m2)
    cm.log(m1)

    m1.copy_to_host()
    m2.copy_to_host()

    assert np.max(np.abs(c - m1.numpy_array)) < 10**-4, "Error in cudamat.log exceeded threshold"
    assert np.max(np.abs(c - m2.numpy_array)) < 10**-4, "Error in cudamat.log exceeded threshold"

def test_exp():
    m = 256
    n = 128
    a = np.array(np.random.randn(m, n), dtype=np.float64, order='F')
    b = np.array(np.random.randn(m, n), dtype=np.float64, order='F')

    c = np.exp(a)

    m1 = cm.CUDAMatrix(a)
    m2 = cm.CUDAMatrix(b)
    cm.exp(m1, target = m2)
    cm.exp(m1)

    m1.copy_to_host()
    m2.copy_to_host()

    assert np.max(np.abs(c - m1.numpy_array)) < 10**-4, "Error in cudamat.exp exceeded threshold"
    assert np.max(np.abs(c - m2.numpy_array)) < 10**-4, "Error in cudamat.exp exceeded threshold"

def test_gamma():
    m = 256
    n = 128
    a = np.array(np.random.rand(m, n)*5, dtype=np.float64, order='F')
    b = np.array(np.random.rand(m, n)*5, dtype=np.float64, order='F')

    from scipy.special import gamma
    c = gamma(a)

    m1 = cm.CUDAMatrix(a)
    m2 = cm.CUDAMatrix(b)
    cm.gamma(m1, target = m2)
    cm.gamma(m1)

    m1.copy_to_host()
    m2.copy_to_host()

    assert np.max(np.abs(c - m1.numpy_array)) < 10**-2, "Error in cudamat.gamma exceeded threshold"
    assert np.max(np.abs(c - m2.numpy_array)) < 10**-2, "Error in cudamat.gamma exceeded threshold"

def test_lgamma():
    m = 256
    n = 128
    a = np.array(np.random.rand(m, n)*10, dtype=np.float64, order='F')
    b = np.array(np.random.rand(m, n)*10, dtype=np.float64, order='F')

    from scipy.special import gammaln
    c = gammaln(a)

    m1 = cm.CUDAMatrix(a)
    m2 = cm.CUDAMatrix(b)
    cm.lgamma(m1, target = m2)
    cm.lgamma(m1)

    m1.copy_to_host()
    m2.copy_to_host()

    assert np.max(np.abs(c - m1.numpy_array)) < 10**-2, "Error in cudamat.lgamma exceeded threshold " + str(np.max(np.abs(c - m1.numpy_array)))
    assert np.max(np.abs(c - m2.numpy_array)) < 10**-2, "Error in cudamat.lgamma exceeded threshold"

def test_sqrt():
    m = 256
    n = 128
    a = np.array(np.random.rand(m, n)*20, dtype=np.float64, order='F')
    b = np.array(np.random.rand(m, n), dtype=np.float64, order='F')

    c = np.sqrt(a)

    m1 = cm.CUDAMatrix(a)
    m2 = cm.CUDAMatrix(b)
    cm.sqrt(m1, target = m2)
    cm.sqrt(m1)

    m1.copy_to_host()
    m2.copy_to_host()

    assert np.max(np.abs(c - m1.numpy_array)) < 10**-4, "Error in cudamat.sqrt exceeded threshold"
    assert np.max(np.abs(c - m2.numpy_array)) < 10**-4, "Error in cudamat.sqrt exceeded threshold"

def test_pow():
    m = 256
    n = 128
    a = np.array(np.random.randn(m, n)*20, dtype=np.float64, order='F')
    b = np.array(np.random.rand(m, n), dtype=np.float64, order='F')
    p = 2

    c = a**p

    m1 = cm.CUDAMatrix(a)
    m2 = cm.CUDAMatrix(b)
    cm.pow(m1, p, target = m2)
    cm.pow(m1, p)

    m1.copy_to_host()
    m2.copy_to_host()

    assert np.max(np.abs(c - m1.numpy_array)) < 10**-3, "Error in cudamat.pow exceeded threshold"
    assert np.max(np.abs(c - m2.numpy_array)) < 10**-3, "Error in cudamat.pow exceeded threshold"

def test_pow_matrix():
    m = 256
    n = 128
    a = np.array(np.random.rand(m, n)*20, dtype=np.float64, order='F')
    b = np.array(np.random.rand(m, n), dtype=np.float64, order='F')
    p = np.array(np.random.randn(m, n), dtype=np.float64, order='F')


    c = a**p

    m1 = cm.CUDAMatrix(a)
    m2 = cm.CUDAMatrix(b)
    mp = cm.CUDAMatrix(p)
    cm.pow(m1, mp, target = m2)
    cm.pow(m1, mp)

    m1.copy_to_host()
    m2.copy_to_host()

    assert np.max(np.abs(c - m1.numpy_array)) < 10**-2, "Error in cudamat.pow exceeded threshold"
    assert np.max(np.abs(c - m2.numpy_array)) < 10**-2, "Error in cudamat.pow exceeded threshold"

def test_reciprocal():
    m = 256
    n = 128
    a = np.array(np.random.rand(m, n)*10+10**-3, dtype=np.float64, order='F')
    b = np.array(np.random.rand(m, n)*10, dtype=np.float64, order='F')

    c = 1. / a

    m1 = cm.CUDAMatrix(a)
    m2 = cm.CUDAMatrix(b)
    m1.reciprocal(target = m2)
    m1.reciprocal()

    m1.copy_to_host()
    m2.copy_to_host()

    assert np.max(np.abs(c - m1.numpy_array)) < 10**-4, "Error in CUDAMatrix.reciprocal exceeded threshold"
    assert np.max(np.abs(c - m2.numpy_array)) < 10**-4, "Error in CUDAMatrix.reciprocal exceeded threshold"

def test_add_mult():
    m = 256
    n = 128
    alpha = np.pi
    a = np.array(np.random.randn(m, n)*10, dtype=np.float64, order='F')
    b = np.array(np.random.randn(m, n)*10, dtype=np.float64, order='F')

    c = a + np.pi * b

    m1 = cm.CUDAMatrix(a)
    m2 = cm.CUDAMatrix(b)
    m1.add_mult(m2, np.pi)
    m1.copy_to_host()

    assert np.max(np.abs(c - m1.numpy_array)) < 10**-4, "Error in CUDAMatrix.add_mult exceeded threshold"

def test_subtract_mult():
    m = 256
    n = 128
    alpha = np.pi
    a = np.array(np.random.randn(m, n)*10, dtype=np.float64, order='F')
    b = np.array(np.random.randn(m, n)*10, dtype=np.float64, order='F')

    c = a - np.pi * b

    m1 = cm.CUDAMatrix(a)
    m2 = cm.CUDAMatrix(b)
    m1.subtract_mult(m2, np.pi)
    m1.copy_to_host()

    assert np.max(np.abs(c - m1.numpy_array)) < 10**-4, "Error in CUDAMatrix.subtract_mult exceeded threshold"

def test_add():
    m = 256
    n = 128
    a = np.array(np.random.randn(m, n)*10, dtype=np.float64, order='F')
    b = np.array(1.+np.random.rand(m, n)*10, dtype=np.float64, order='F')
    t = np.array(np.empty((m, n)), dtype=np.float64, order='F')

    c = a + b

    m1 = cm.CUDAMatrix(a)
    m2 = cm.CUDAMatrix(b)
    m3 = cm.CUDAMatrix(t)

    m1.add(m2, target = m3)
    m1.add(m2)

    m3.copy_to_host()
    m1.copy_to_host()

    assert np.max(np.abs(c - m3.numpy_array)) < 10**-4, "Error in CUDAMatrix.add exceeded threshold"
    assert np.max(np.abs(c - m1.numpy_array)) < 10**-4, "Error in CUDAMatrix.add exceeded threshold"

def test_subtract():
    m = 256
    n = 128
    a = np.array(np.random.randn(m, n)*10, dtype=np.float64, order='F')
    b = np.array(1.+np.random.rand(m, n)*10, dtype=np.float64, order='F')
    t = np.array(np.empty((m, n)), dtype=np.float64, order='F')

    c = a - b

    m1 = cm.CUDAMatrix(a)
    m2 = cm.CUDAMatrix(b)
    m3 = cm.CUDAMatrix(t)

    m1.subtract(m2, target = m3)
    m1.subtract(m2)

    m3.copy_to_host()
    m1.copy_to_host()

    assert np.max(np.abs(c - m3.numpy_array)) < 10**-4, "Error in CUDAMatrix.subtract exceeded threshold"
    assert np.max(np.abs(c - m1.numpy_array)) < 10**-4, "Error in CUDAMatrix.subtract exceeded threshold"

def test_divide():
    m = 256
    n = 128
    a = np.array(np.random.randn(m, n)*10, dtype=np.float64, order='F')
    b = np.array(1.+np.random.rand(m, n)*10, dtype=np.float64, order='F')
    t = np.array(np.empty((m, n)), dtype=np.float64, order='F')

    c = a / b

    m1 = cm.CUDAMatrix(a)
    m2 = cm.CUDAMatrix(b)
    m3 = cm.CUDAMatrix(t)

    m1.divide(m2, target = m3)
    m1.divide(m2)

    m3.copy_to_host()
    m1.copy_to_host()

    assert np.max(np.abs(c - m3.numpy_array)) < 10**-4, "Error in CUDAMatrix.div exceeded threshold"
    assert np.max(np.abs(c - m1.numpy_array)) < 10**-4, "Error in CUDAMatrix.div exceeded threshold"

def test_mult():
    m = 256
    n = 128
    a = np.array(np.random.randn(m, n)*10, dtype=np.float64, order='F')
    b = np.array(np.random.randn(m, n)*10, dtype=np.float64, order='F')
    t = np.array(np.empty((m, n)), dtype=np.float64, order='F')

    c = a * b

    m1 = cm.CUDAMatrix(a)
    m2 = cm.CUDAMatrix(b)
    m3 = cm.CUDAMatrix(t)

    m1.mult(m2, target = m3)
    m1.mult(m2)

    m3.copy_to_host()
    m1.copy_to_host()

    assert np.max(np.abs(c - m3.numpy_array)) < 10**-4, "Error in CUDAMatrix.multiply exceeded threshold"
    assert np.max(np.abs(c - m1.numpy_array)) < 10**-4, "Error in CUDAMatrix.multiply exceeded threshold"

def test_scalar_mult():
    m = 256
    n = 128
    alpha = np.pi
    a = np.array(np.random.randn(m, n)*10, dtype=np.float64, order='F')
    t = np.array(np.empty((m, n)), dtype=np.float64, order='F')

    c = a * alpha

    m1 = cm.CUDAMatrix(a)
    m2 = cm.CUDAMatrix(t)

    m1.mult(alpha, target = m2)
    m1.mult(alpha)

    m1.copy_to_host()
    m2.copy_to_host()

    assert np.max(np.abs(c - m1.numpy_array)) < 10**-4, "Error in CUDAMatrix.mult exceeded threshold"
    assert np.max(np.abs(c - m2.numpy_array)) < 10**-4, "Error in CUDAMatrix.mult exceeded threshold"

def test_scalar_div():
    m = 256
    n = 128
    alpha = np.pi
    a = np.array(np.random.randn(m, n)*10, dtype=np.float64, order='F')
    t = np.array(np.empty((m, n)), dtype=np.float64, order='F')

    c = a / alpha

    m1 = cm.CUDAMatrix(a)
    m2 = cm.CUDAMatrix(t)

    m1.divide(alpha, target = m2)
    m1.divide(alpha)

    m1.copy_to_host()
    m2.copy_to_host()

    assert np.max(np.abs(c - m1.numpy_array)) < 10**-4, "Error in CUDAMatrix.divide exceeded threshold"
    assert np.max(np.abs(c - m2.numpy_array)) < 10**-4, "Error in CUDAMatrix.divide exceeded threshold"

def test_add_scalar():
    m = 256
    n = 128
    alpha = np.pi
    a = np.array(np.random.randn(m, n)*10, dtype=np.float64, order='F')
    t = np.array(np.empty((m, n)), dtype=np.float64, order='F')

    c = a + alpha

    m1 = cm.CUDAMatrix(a)
    m2 = cm.CUDAMatrix(t)

    m1.add(alpha, target = m2)
    m1.add(alpha)

    m1.copy_to_host()
    m2.copy_to_host()

    assert np.max(np.abs(c - m1.numpy_array)) < 10**-4, "Error in CUDAMatrix.add_scalar exceeded threshold"
    assert np.max(np.abs(c - m2.numpy_array)) < 10**-4, "Error in CUDAMatrix.add_scalar exceeded threshold"

def test_dot():
    m = 128
    k = 256
    n = 64
    a = np.array(np.random.randn(m, k)*10, dtype=np.float64, order='F')
    b = np.array(np.random.randn(k, n)*10, dtype=np.float64, order='F')
    c = np.array(np.random.randn(m, n)*10, dtype=np.float64, order='F')

    alpha = 2.
    beta = 0.3
    r = beta * c + alpha * np.dot(a, b)

    m1 = cm.CUDAMatrix(a)
    m2 = cm.CUDAMatrix(b)
    m3 = cm.CUDAMatrix(c)
    m3 = cm.dot(m1, m2, target = m3, alpha = alpha, beta = beta)
    m3.copy_to_host()

    assert np.max(np.abs(r - m3.numpy_array)) < 10**-2, "Error in CUDAMatrix.dot exceeded threshold"

def test_dot_trans():
    m = 128
    k = 256
    n = 64
    a = np.array(np.random.randn(k, m)*10, dtype=np.float64, order='F')
    b = np.array(np.random.randn(k, n)*10, dtype=np.float64, order='F')

    c = np.dot(a.T, b)

    m1 = cm.CUDAMatrix(a)
    m2 = cm.CUDAMatrix(b)
    m1.set_trans(True);
    m3 = cm.dot(m1, m2)
    m3.copy_to_host()

    assert np.max(np.abs(c - m3.numpy_array)) < 10**-2, "Error in CUDAMatrix.dot exceeded threshold"

def test_dot_vect():
    m = 128
    k = 256
    n = 1
    a = np.array(np.random.randn(m, k)*10, dtype=np.float64, order='F')
    b = np.array(np.random.randn(k, n)*10, dtype=np.float64, order='F')
    A = cm.CUDAMatrix(a)
    B = cm.CUDAMatrix(b)

    c = np.dot(a, b)
    C = cm.dot(A, B)
    assert np.max(np.abs(c - C.asarray())) < 10**-2, "Error in CUDAMatrix.dot exceeded threshold"

    c = np.dot(a.T, b[:m])
    C = cm.dot(A.T, B.slice(0, m))
    assert np.max(np.abs(c - C.asarray())) < 10**-2, "Error in CUDAMatrix.dot exceeded threshold"

    c = np.dot(b.T, a.T)
    C = cm.dot(B.T, A.T)
    assert np.max(np.abs(c - C.asarray())) < 10**-2, "Error in CUDAMatrix.dot exceeded threshold"

    c = np.dot(b[:m].T, a)
    C = cm.dot(B.slice(0, m).reshape((1, m)), A)
    assert np.max(np.abs(c - C.asarray())) < 10**-2, "Error in CUDAMatrix.dot exceeded threshold"

def test_add_dot():
    m = 128
    k = 256
    n = 64
    a = np.array(np.random.randn(m, k)*10, dtype=np.float64, order='F')
    b = np.array(np.random.randn(k, n)*10, dtype=np.float64, order='F')
    c = np.array(np.random.randn(m, n)*10, dtype=np.float64, order='F')

    mult = 2.1
    beta = 0.8
    res = beta * c + mult * np.dot(a, b)

    m1 = cm.CUDAMatrix(a)
    m2 = cm.CUDAMatrix(b)
    m3 = cm.CUDAMatrix(c)
    m3.add_dot(m1, m2, mult = mult, beta = beta)

    m3.copy_to_host()

    assert np.max(np.abs(res - m3.numpy_array)) < 10**-2, "Error in CUDAMatrix.add_dot exceeded threshold"

def test_vdot():
    m = 64
    n = 64
    a = np.array(np.random.randn(m, n), dtype=np.float64, order='F')
    b = np.array(np.random.randn(m, n), dtype=np.float64, order='F')

    true_res = np.vdot(a, b)

    m1 = cm.CUDAMatrix(a)
    m2 = cm.CUDAMatrix(b)

    res = cm.vdot(m1, m2)

    assert np.abs(res - true_res) < 10**-2, "Error in CUDAMatrix.vdot exceeded threshold"

def test_subtract_dot():
    m = 128
    k = 256
    n = 64
    a = np.array(np.random.randn(m, k)*10, dtype=np.float64, order='F')
    b = np.array(np.random.randn(k, n)*10, dtype=np.float64, order='F')
    c = np.array(np.random.randn(m, n)*10, dtype=np.float64, order='F')

    res = c - np.dot(a, b)

    m1 = cm.CUDAMatrix(a)
    m2 = cm.CUDAMatrix(b)
    m3 = cm.CUDAMatrix(c)
    m3.subtract_dot(m1, m2)

    m3.copy_to_host()

    assert np.max(np.abs(res - m3.numpy_array)) < 10**-2, "Error in CUDAMatrix.subtract_dot exceeded threshold"

def test_random():
    cm.CUDAMatrix.init_random(1)
    m1 = cm.CUDAMatrix(np.array(np.empty((128,256)), dtype=np.float64, order='F'))
    m2 = cm.CUDAMatrix(np.array(np.empty((128,256)), dtype=np.float64, order='F'))

    m1.fill_with_rand()
    m1.copy_to_host()
    m2.fill_with_randn()
    m2.copy_to_host()

    assert np.abs(np.mean(m1.numpy_array) - 0.5) < 10**-2, "Error in CUDAMatrix.fill_with_rand threshold"
    assert np.abs(np.mean(m2.numpy_array)) < 10**-2, "Error in CUDAMatrix.fill_with_randn threshold"

def test_euclid_norm():
    m = 256
    n = 128
    a = np.array(np.random.rand(m, n)*10, dtype=np.float64, order='F')

    m = cm.CUDAMatrix(a)

    n1 = np.sqrt(np.sum(a**2))
    n2 = m.euclid_norm()

    assert np.abs(n1-n2) < 10**-2, "Error in CUDAMatrix.euclid_norm exceeded threshold"

def test_manhattan_norm():
    m = 256
    n = 128
    a = np.array(np.random.rand(m, n)*10, dtype=np.float64, order='F')

    m = cm.CUDAMatrix(a)

    n1 = np.sum(np.abs(a), dtype=np.double)
    n2 = m.manhattan_norm()

    assert np.abs(n1-n2) < 2e-2, "Error in CUDAMatrix.manhattan_norm exceeded threshold (%f != %f)" % (n1, n2)

def test_allfinite():
    a = cm.empty((10, 20)).assign(1).divide(0)  # NaN
    b = cm.empty((10, 20)).assign(1e20).mult(1e20)  # Inf
    c = cm.empty((10, 20)).assign(1)  # 1.0

    assert (not a.allfinite()) and (not b.allfinite()) and c.allfinite(), "CUDAMatrix.allfinite does not work"

def test_select_columns():
    m = 256
    n = 128
    k = 8

    s = np.array(np.random.randn(m, n), dtype=np.float64, order='F')
    i_l = [0, 1, 2, 3, 5, 10, 12, 20]
    i = np.array(i_l).T[np.newaxis, :]
    t = np.empty((m, k))

    s_d = cm.CUDAMatrix(s)
    i_d = cm.CUDAMatrix(i)
    t_d = cm.CUDAMatrix(t)

    s_d.select_columns(i_d, t_d)
    res = s[:,i_l]

    assert np.max(np.abs(res - t_d.asarray())) < 10**-4, "Error in CUDAMatrix.select_columns exceeded threshold"


def test_where():
    m = 256
    n = 128
    a = np.array(np.random.randn(m, n)*10, dtype=np.float64, order='F')
    z = np.zeros_like(a)
    res = np.where(a > 0, a, z);

    a_d = cm.CUDAMatrix(a)
    z_d = cm.CUDAMatrix(z)
    res_d = cm.empty(a_d.shape)
    a_d.greater_than(0,  res_d)
    cm.where(res_d, a_d, z_d)
    assert np.abs(res-res_d.asarray()).max() < 1e-2, "Error in cudamat.where"


def test_correlate():
    m = 64
    n = 32
    km = 17
    kn = 11

    a = np.array(np.random.randn(m, n)*10, dtype=np.float64, order='F')
    k = np.array(np.random.randn(km, kn)*10, dtype=np.float64, order='F')

    res = np.zeros_like(a)
    for i in range(len(a)):
        for j in range(len(a[0])):
            for h in range(-(km/2), km/2 + 1):
                for w in range(-(kn/2), kn/2 + 1):
                    if i+h >= 0 and i+h < m and j+w >= 0 and j+w < n:
                        res[i][j] += a[i + h][j + w] * k[km/2 + h][kn/2 + w]

    a_d = cm.CUDAMatrix(a)
    k_d = cm.CUDAMatrix(k)

    res_d = cm.correlate(a_d, k_d)
    assert np.abs(res-res_d.asarray()).max() < 1e-2, "Error in cudamat.correlate"


if __name__ == '__main__':
    nose.runmodule()
