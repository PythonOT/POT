"""Helpers for OpenMP support during the build."""

# This code is adapted for a large part from the astropy openmp helpers, which
# can be found at: https://github.com/astropy/extension-helpers/blob/master/extension_helpers/_openmp_helpers.py  # noqa


import os
import sys
import textwrap
import warnings
import subprocess

from distutils.errors import CompileError, LinkError

from pre_build_helpers import compile_test_program


def get_openmp_flag(compiler):
    if hasattr(compiler, 'compiler'):
        compiler = compiler.compiler[0]
    else:
        compiler = compiler.__class__.__name__

    if sys.platform == "win32" and ('icc' in compiler or 'icl' in compiler):
        return ['/Qopenmp']
    elif sys.platform == "win32":
        return ['/openmp']
    elif sys.platform in ("darwin", "linux") and "icc" in compiler:
        return ['-qopenmp']
    elif sys.platform == "darwin" and 'openmp' in os.getenv('CPPFLAGS', ''):
        return []
    # Default flag for GCC and clang:
    return ['-fopenmp']


def check_openmp_support():
    """Check whether OpenMP test code can be compiled and run"""
    
    code = textwrap.dedent(
        """\
        #include <omp.h>
        #include <stdio.h>
        int main(void) {
        #pragma omp parallel
        printf("nthreads=%d\\n", omp_get_num_threads());
        return 0;
        }
        """)

    extra_preargs = os.getenv('LDFLAGS', None)
    if extra_preargs is not None:
        extra_preargs = extra_preargs.strip().split(" ")
        extra_preargs = [
            flag for flag in extra_preargs
            if flag.startswith(('-L', '-Wl,-rpath', '-l'))]

    extra_postargs = get_openmp_flag

    try:
        output = compile_test_program(code,
                                      extra_preargs=extra_preargs,
                                      extra_postargs=extra_postargs)

        if output and 'nthreads=' in output[0]:
            nthreads = int(output[0].strip().split('=')[1])
            openmp_supported = len(output) == nthreads
        elif "PYTHON_CROSSENV" in os.environ:
            # Since we can't run the test program when cross-compiling
            # assume that openmp is supported if the program can be
            # compiled.
            openmp_supported = True
        else:
            openmp_supported = False

    except (CompileError, LinkError, subprocess.CalledProcessError):
        openmp_supported = False
    return openmp_supported
