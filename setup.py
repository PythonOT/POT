#!/usr/bin/env python

import os
import re
import subprocess
import shutil
import sys

from distutils.command.clean import clean as Clean
from distutils.sysconfig import customize_compiler
from setuptools import find_packages, setup
from setuptools.extension import Extension

import numpy
from numpy.distutils.ccompiler import new_compiler
from Cython.Build import cythonize


# dirty but working
__version__ = re.search(
    r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]', # excludes inline comment
    open('ot/__init__.py').read()).group(1)
# The beautiful part is, I don't even need to check exceptions here.
# If something messes up, let the build process fail noisy, BEFORE the release!

# PyPI handles markdown now
ROOT = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(ROOT, "README.md"), encoding="utf-8") as f:
    README = f.read()

# Clean command to remove build artifacts
class CleanCommand(Clean):
    description = "Remove build artifacts from the source tree"

    def run(self):
        Clean.run(self)
        # Remove c files if we are not within a sdist package
        cwd = os.path.abspath(os.path.dirname(__file__))
        remove_c_files = not os.path.exists(os.path.join(cwd, "PKG-INFO"))
        if remove_c_files:
            print("Will remove generated .c files")
        if os.path.exists("build"):
            shutil.rmtree("build")
        for dirpath, dirnames, filenames in os.walk('ot'):
            for filename in filenames:
                if any(filename.endswith(suffix) for suffix in
                       (".so", ".pyd", ".dll", ".pyc")):
                    os.unlink(os.path.join(dirpath, filename))
                    continue
                extension = os.path.splitext(filename)[1]
                if remove_c_files and extension in [".c", ".cpp"]:
                    pyx_file = str.replace(filename, extension, ".pyx")
                    if os.path.exists(os.path.join(dirpath, pyx_file)):
                        os.unlink(os.path.join(dirpath, filename))
            for dirname in dirnames:
                if dirname == '__pycache__':
                    shutil.rmtree(os.path.join(dirpath, dirname))

cmdclass = {'clean': CleanCommand}

# Custom compile args to set OpenMP compile flags depending on OS/compiler
def get_openmp_flag(compiler):
    if hasattr(compiler, 'compiler'):
        compiler = compiler.compiler[0]
    else:
        compiler = compiler.__class__.__name__

    if sys.platform == "win32" and ('icc' in compiler or 'icl' in compiler):
        return ['/Qopenmp']
    elif sys.platform == "win32":
        return ['/openmp']
    elif sys.platform == "darwin" and ('icc' in compiler or 'icl' in compiler):
        return ['-openmp']
    elif sys.platform == "darwin" and 'openmp' in os.getenv('CPPFLAGS', ''):
        # -fopenmp can't be passed as compile flag when using Apple-clang.
        # OpenMP support has to be enabled during preprocessing.
        #
        # For example, our macOS wheel build jobs use the following environment
        # variables to build with Apple-clang and the brew installed "libomp":
        #
        # export CPPFLAGS="$CPPFLAGS -Xpreprocessor -fopenmp"
        # export CFLAGS="$CFLAGS -I/usr/local/opt/libomp/include"
        # export CXXFLAGS="$CXXFLAGS -I/usr/local/opt/libomp/include"
        # export LDFLAGS="$LDFLAGS -Wl,-rpath,/usr/local/opt/libomp/lib
        #                          -L/usr/local/opt/libomp/lib -lomp"
        return []
    # Default flag for GCC and clang:
    return ['-fopenmp']

ccompiler = new_compiler()
customize_compiler(ccompiler)

extra_preargs = os.getenv('LDFLAGS', None)
if extra_preargs is not None:
    extra_preargs = extra_preargs.strip().split(" ")
    extra_preargs = [flag for flag in extra_preargs
                     if flag.startswith(('-L', '-Wl,-rpath', '-l'))]
extra_postargs = get_openmp_flag(ccompiler)

compile_args = ("-std=c++11 -Ofast -march=native -fno-signed-zeros "
                "-fno-trapping-math -funroll-loops -openmp").split()
if extra_preargs is not None:
    compile_args += extra_preargs
if extra_postargs is not None:
    compile_args += extra_postargs

setup(
    name='POT',
    version=__version__,
    description='Python Optimal Transport Library',
    long_description=README,
    long_description_content_type='text/markdown',
    author=u'Remi Flamary, Nicolas Courty',
    author_email='remi.flamary@gmail.com, ncourty@gmail.com',
    url='https://github.com/PythonOT/POT',
    packages=find_packages(),
    ext_modules=cythonize(Extension(
        name="ot.lp.emd_wrap",
        sources=["ot/lp/emd_wrap.pyx", "ot/lp/EMD_wrapper.cpp"], # cython/c++ src files
        language="c++",
        include_dirs=[numpy.get_include(), os.path.join(ROOT, 'ot/lp')],
        extra_compile_args=compile_args,
    )),
    platforms=['linux', 'macosx', 'windows'],
    download_url='https://github.com/PythonOT/POT/archive/{}.tar.gz'.format(__version__),
    license='MIT',
    scripts=[],
    data_files=[],
    setup_requires=["numpy>=1.16", "cython>=0.23"],
    install_requires=["numpy>=1.16", "scipy>=1.0"],
    cmdclass=cmdclass,
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Environment :: Console',
        'Operating System :: OS Independent',
        'Operating System :: Linux',
        'Operating System :: MacOS',
        'Operating System :: POSIX',
        'Operating System :: Windows',
        'Programming Language :: Python',
        'Programming Language :: C++',
        'Programming Language :: C',
        'Programming Language :: Cython',
        'Topic :: Utilities',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)
