#!/usr/bin/env python

from distutils.core import setup, Extension
import numpy
from Cython.Distutils import build_ext
from Cython.Build import cythonize

import os
import glob

version=0.1

ROOT = os.path.abspath(os.path.dirname(__file__))
README = open(os.path.join(ROOT, 'README.md')).read()

setup(name='python-webgen',
      version=version,
      description='Python Optimal Transport Library',
      long_description=README,
      author=u'Remi Flamary',
      author_email='remi.flamary@gmail.com',
      url='https://github.com/rflamary/POT',
      packages=['ot','ot.emd'],
     ext_modules = cythonize(Extension(
                "ot.emd.emd",                                # the extesion name
                 sources=["ot/emd/emd.pyx", "ot/emd/EMD_wrap.cpp"], # the Cython source and
                                                        # additional C++ source files
                 #extra_compile_args=['-fopenmp'],
                 #extra_link_args=['-fopenmp'],
                 language="c++",                        # generate and compile C++ code,
                 include_dirs=[numpy.get_include(),os.path.join(ROOT,'ot/emd')])),
      platforms=['linux','macosx','windows'],
      license = 'MIT',
      scripts=[],
      data_files=[],
      requires=["numpy (>=1.11)"],
      classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Environment :: Console',
        'Operating System :: OS Independent',
        'Operating System :: MacOS',
        'Operating System :: POSIX',
        'Programming Language :: Python',
        'Topic :: Utilities'
    ]
     )
