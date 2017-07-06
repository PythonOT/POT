#!/usr/bin/env python

from setuptools import setup, find_packages
from codecs import open
from os import path
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy
import re
import os

here = path.abspath(path.dirname(__file__))

# dirty but working
__version__ = re.search(
    r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]',  # It excludes inline comment too
    open('ot/__init__.py').read()).group(1)
# The beautiful part is, I don't even need to check exceptions here.
# If something messes up, let the build process fail noisy, BEFORE my release!

ROOT = os.path.abspath(os.path.dirname(__file__))


# convert markdown readme to rst if pypandoc installed
try:
   import pypandoc
   README = pypandoc.convert('README.md', 'rst')
except (IOError, ImportError):
   README = open(os.path.join(ROOT, 'README.md')).read()


setup(name='POT',
      version=__version__,
      description='Python Optimal Transport Library',
      long_description=README,
      author=u'Remi Flamary, Nicolas Courty',
      author_email='remi.flamary@gmail.com, ncourty@gmail.com',
      url='https://github.com/rflamary/POT',
      packages=find_packages(),
      ext_modules = cythonize(Extension(
                "ot.lp.emd_wrap",                                # the extension name
                 sources=["ot/lp/emd_wrap.pyx", "ot/lp/EMD_wrapper.cpp"], # the Cython source and
                                                        # additional C++ source files
                 language="c++",                        # generate and compile C++ code,
                 include_dirs=[numpy.get_include(),os.path.join(ROOT,'ot/lp')])),
      platforms=['linux','macosx','windows'],
      download_url='https://github.com/rflamary/POT/archive/{}.tar.gz'.format(__version__),
      license = 'MIT',
      scripts=[],
      data_files=[],
      requires=["numpy","scipy","cython","matplotlib"],
      install_requires=["numpy","scipy","cython","matplotlib"],
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
