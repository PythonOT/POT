#!/usr/bin/env python

# Author: Remi Flamary <remi.flamary@polytechnique.edu>
#
# License: MIT License

import os
import re
import subprocess
import sys

from setuptools import find_packages, setup
from setuptools.extension import Extension

import numpy
from Cython.Build import cythonize

sys.path.append(os.path.join("ot", "helpers"))
from openmp_helpers import check_openmp_support

# dirty but working
__version__ = re.search(
    r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]',  # It excludes inline comment too
    open("ot/__init__.py").read(),
).group(1)
# The beautiful part is, I don't even need to check exceptions here.
# If something messes up, let the build process fail noisy, BEFORE my release!

# thanks PyPI for handling markdown now
ROOT = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(ROOT, "README.md"), encoding="utf-8") as f:
    README = f.read()

# clean cython output is clean is called
if "clean" in sys.argv[1:]:
    if os.path.isfile("ot/lp/emd_wrap.cpp"):
        os.remove("ot/lp/emd_wrap.cpp")

# add platform dependant optional compilation argument
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
    name="POT",
    version=__version__,
    description="Python Optimal Transport Library",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Remi Flamary, Nicolas Courty, Cédric Vincent-Cuaz, POT Contributors",
    author_email="remi.flamary@gmail.com, ncourty@gmail.com, cedvincentcuaz@gmail.com",
    url="https://github.com/PythonOT/POT",
    packages=find_packages(exclude=["benchmarks"]),
    ext_modules=cythonize(
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
        )
    ),
    platforms=["linux", "macosx", "windows"],
    download_url="https://github.com/PythonOT/POT/archive/{}.tar.gz".format(
        __version__
    ),
    license="MIT",
    scripts=[],
    data_files=[],
    install_requires=["numpy>=1.16", "scipy>=1.6"],
    extras_require={
        "backend-numpy": [],  # in requirements.
        "backend-jax": ["jax", "jaxlib"],
        "backend-cupy": [],  # should be installed with conda, not pip
        "backend-tf": ["tensorflow"],
        "backend-torch": ["torch"],
        "cvxopt": ["cvxopt"],  # on it's own to prevent accidental GPL violations
        "dr": ["scikit-learn", "pymanopt", "autograd"],
        "gnn": ["torch", "torch_geometric"],
        "plot": ["matplotlib"],
        "all": [
            "jax",
            "jaxlib",
            "tensorflow",
            "torch",
            "cvxopt",
            "scikit-learn",
            "pymanopt",
            "autograd",
            "torch_geometric",
            "matplotlib",
        ],
    },
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Environment :: Console",
        "Operating System :: OS Independent",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
        "Operating System :: POSIX",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python",
        "Programming Language :: C++",
        "Programming Language :: C",
        "Programming Language :: Cython",
        "Topic :: Utilities",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
