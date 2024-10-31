"""Helpers for OpenMP support during the build."""

# This code is adapted for a large part from the astropy openmp helpers, which
# can be found at: https://github.com/astropy/extension-helpers/blob/master/extension_helpers/_openmp_helpers.py  # noqa

import os
import sys
import textwrap
import subprocess

from setuptools.errors import CompileError, LinkError

from pre_build_helpers import compile_test_program


def get_openmp_flag(compiler):
    """Get openmp flags for a given compiler"""

    if hasattr(compiler, "compiler"):
        compiler = compiler.compiler[0]
    else:
        compiler = compiler.__class__.__name__

    if sys.platform == "win32" and ("icc" in compiler or "icl" in compiler):
        omp_flag = ["/Qopenmp"]
    elif sys.platform == "win32":
        omp_flag = ["/openmp"]
    elif sys.platform in ("darwin", "linux") and "icc" in compiler:
        omp_flag = ["-qopenmp"]
    elif sys.platform == "darwin" and "openmp" in os.getenv("CPPFLAGS", ""):
        omp_flag = []
    else:
        # Default flag for GCC and clang:
        omp_flag = ["-fopenmp"]
        if sys.platform.startswith("darwin"):
            omp_flag += ["-Xpreprocessor", "-lomp"]
    return omp_flag


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
        """
    )

    extra_preargs = os.getenv("LDFLAGS", None)
    if extra_preargs is not None:
        extra_preargs = extra_preargs.strip().split(" ")
        extra_preargs = [
            flag
            for flag in extra_preargs
            if flag.startswith(("-L", "-Wl,-rpath", "-l"))
        ]

    extra_postargs = get_openmp_flag

    try:
        output, compile_flags = compile_test_program(
            code, extra_preargs=extra_preargs, extra_postargs=extra_postargs
        )

        if output and "nthreads=" in output[0]:
            nthreads = int(output[0].strip().split("=")[1])
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
        compile_flags = []
    return openmp_supported, compile_flags
