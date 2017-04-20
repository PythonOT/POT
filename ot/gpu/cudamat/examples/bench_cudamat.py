from __future__ import print_function, division
import sys
import numpy as np
import cudamat as cmt
import time
import timeit
from inspect import getmodule, getmembers, isfunction
try: from itertools import ifilter as filter
except: pass

# heat-up time in seconds before starting the benchmark
HEATUP = 2

# shapes used for the small and large test matrix
XS_SHAPE = (400, 256)
XL_SHAPE = (4096, 4096)

# timeit number and repeat parameter
NUM_ITER = 100
NUM_REPEATS = 5

def setup(shape):
    """Creates two matrices and corresponding row/column vectors"""
    mat = cmt.empty(shape).fill_with_randn()
    mat2 = cmt.empty(shape).fill_with_randn()
    col = cmt.empty((shape[0], 1)).assign(0)
    row = cmt.empty((1, shape[1])).assign(0)
    return mat, mat2, col, row

def bench_dot(X, Y, col, row):
    cmt.dot(X.T, Y)

def bench_add(X, Y, col, row):
    X.add(Y)
bench_add.repeats = 5  # 5 times more repetitions than usual

def bench_mult(X, Y, col, row):
    X.mult(Y)

def bench_sigm(X, Y, col, row):
    X.apply_sigmoid()

def bench_colsum(X, Y, col, row):
    X.sum(axis=0, target=row)

def bench_rowsum(X, Y, col, row):
    X.sum(axis=1, target=col)

def bench_addcolsum(X, Y, col, row):
    row.add_sums(X, axis=0, mult=3.2, beta=0.2)

def bench_addrowsum(X, Y, col, row):
    col.add_sums(X, axis=1, mult=3.2, beta=0.2)

def bench_colmax(X, Y, col, row):
    X.max(axis=0, target=row)

def bench_rowmax(X, Y, col, row):
    X.max(axis=1, target=col)

def bench_addcolmult(X, Y, col, row):
    X.add_col_mult(col, mult=3.2)

def heatup(duration):
    """Heat-up the GPU for a while so it enters full-performance mode"""
    t1 = time.time()
    while time.time() - t1 < duration:
        cmt.dot(cmt.empty((200, 200)), cmt.empty((200, 200)))

def main():
    cmt.init()
    cmt.CUDAMatrix.init_random()
    if HEATUP:
        print("heating up for %g seconds..." % HEATUP, end=' ')
        sys.stdout.flush()
        heatup(HEATUP)
        print("done.")
    print("small matrix shape:", XS_SHAPE)
    print("large matrix shape:", XL_SHAPE)
    for funcname, func in filter(lambda f: f[0].startswith('bench_'),
            getmembers(getmodule(main), isfunction)):
        print("%-15s" % funcname[len('bench_'):], end=' ')
        sys.stdout.flush()
        for size, shape, factor in ('small', XS_SHAPE, 10), ('large', XL_SHAPE, 1):
            repeat = NUM_REPEATS * getattr(func, 'repeats', 1)
            time = min(timeit.repeat(\
                    setup="from __main__ import setup, %s\nmats = setup(%s)" % (funcname, shape),
                    stmt="%s(*mats)" % funcname, repeat=repeat,
                    number=NUM_ITER * factor)) / (NUM_ITER * factor)
            print("%.3es (%s) " % (time, size), end=' ')
            sys.stdout.flush()
        print()
    cmt.shutdown()

if __name__=="__main__":
    main()

