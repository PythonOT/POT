#!/usr/bin/env python

from setuptools import setup, find_packages
from codecs import open
from os import path
import numpy
from setuptools.extension import Extension
from Cython.Build import cythonize


here = path.abspath(path.dirname(__file__))



import os
#import glob

version='0.1.3'

ROOT = os.path.abspath(os.path.dirname(__file__))
README = open(os.path.join(ROOT, 'README.md')).read()

setup(name='POT',
      version=version,
      description='Python Optimal Transport Library',
      long_description=README,
      author=u'Remi Flamary, Nicolas Courty',
      author_email='remi.flamary@gmail.com, ncourty@gmail.com',
      url='https://github.com/rflamary/POT',
      packages=find_packages(),
      ext_modules = cythonize(Extension(
                "ot.lp.emd",                                # the extension name
                 sources=["ot/lp/emd.pyx", "ot/lp/EMD_wrap.cpp"], # the Cython source and
                                                        # additional C++ source files
                 language="c++",                        # generate and compile C++ code,
                 include_dirs=[numpy.get_include(),os.path.join(ROOT,'ot/lp')])),
      platforms=['linux','macosx','windows'],
      download_url='https://github.com/rflamary/POT/archive/V0.1.tar.gz',
      license = 'MIT',
      scripts=[],
      data_files=[],
      requires=["numpy (>=1.11)","scipy (>=0.17)"],
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
