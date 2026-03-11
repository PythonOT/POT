#!/usr/bin/env python

# Author: Remi Flamary <remi.flamary@polytechnique.edu>
#
# License: MIT License

import os
import re
import subprocess
import sys

from setuptools import setup
from setuptools.extension import Extension

import numpy
from Cython.Build import cythonize

sys.path.append(os.path.join("ot", "helpers"))
from openmp_helpers import check_openmp_support

# Extract version for the dynamic field declared in pyproject.toml
__version__ = re.search(
    r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]',  # It excludes inline comment too
    open("ot/__init__.py").read(),
).group(1)
# The beautiful part is, I don't even need to check exceptions here.
# If something messes up, let the build process fail noisy, BEFORE my release!

ROOT = os.path.abspath(os.path.dirname(__file__))

# clean cython output if clean is called
if "clean" in sys.argv[1:]:
    for cpp_file in [
        "ot/lp/emd_wrap.cpp",
        "ot/partial/partial_cython.cpp",
    ]:
        if os.path.isfile(cpp_file):
            os.remove(cpp_file)

# add platform dependent optional compilation argument
openmp_supported, flags = check_openmp_support()
compile_args = ["/O2" if sys.platform == "win32" else "-O3"]
link_args = []

if openmp_supported:
    compile_args += flags
    link_args += flags

if sys.platform.startswith("darwin"):
    compile_args.append("-stdlib=libc++")
    sdk_path = subprocess.check_output(["xcrun", "--show-sdk-path"])
    os.environ["CFLAGS"] = '-isysroot "{}"'.format(sdk_path.rstrip().decode("utf-8"))


setup(
    version=__version__,
    ext_modules=cythonize(
        [
            Extension(
                name="ot.lp.emd_wrap",
                sources=[
                    "ot/lp/emd_wrap.pyx",
                    "ot/lp/EMD_wrapper.cpp",
                ],  # cython/c++ src files
                language="c++",
                include_dirs=[numpy.get_include(), os.path.join(ROOT, "ot/lp")],
                extra_compile_args=compile_args,
                extra_link_args=link_args,
            ),
            Extension(
                name="ot.partial.partial_cython",
                sources=["ot/partial/partial_cython.pyx"],
                include_dirs=[numpy.get_include(), os.path.join(ROOT, "ot/partial")],
                extra_compile_args=compile_args,
                language="c++",
            ),
        ]
    ),
)
