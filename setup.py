#!/usr/bin/env python

import os
import re
import subprocess
import sys

from distutils.sysconfig import get_python_inc
from setuptools import find_packages, setup, Extension

# dirty but working
__version__ = re.search(
    r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]',  # It excludes inline comment too
    open('ot/__init__.py').read()).group(1)
# The beautiful part is, I don't even need to check exceptions here.
# If something messes up, let the build process fail noisy, BEFORE my release!

# thanks PyPI for handling markdown now
ROOT = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(ROOT, 'README.md'), encoding="utf-8") as f:
    README = f.read()

# clean cython output is clean is called
if 'clean' in sys.argv[1:]:
    if os.path.isfile('ot/lp/emd_wrap.cpp'):
        os.remove('ot/lp/emd_wrap.cpp')

# add platform dependant optional compilation argument
compile_args = ["-O3"]
if sys.platform.startswith('darwin'):
    compile_args.append("-stdlib=libc++")
    sdk_path = subprocess.check_output(['xcrun', '--show-sdk-path'])
    os.environ['CFLAGS'] = '-isysroot "{}"'.format(
        sdk_path.rstrip().decode("utf-8"))

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
    ext_modules=[
        Extension(
            name="ot.lp.emd_wrap",
            sources=["ot/lp/emd_wrap.pyx",
                     "ot/lp/EMD_wrapper.cpp"],  # cython/c++ src files
            language="c++",
            include_dirs=[
                os.path.join(get_python_inc(plat_specific=1), "numpy")
            ],
            extra_compile_args=compile_args,
        )
    ],
    platforms=['linux', 'macosx', 'windows'],
    download_url='https://github.com/PythonOT/POT/archive/{}.tar.gz'.format(
        __version__),
    license='MIT',
    scripts=[],
    data_files=[],
    setup_requires=["numpy>=1.16", "cython>=0.23"],
    install_requires=["numpy>=1.16", "scipy>=1.0"],
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
    ])
