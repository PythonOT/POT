#!/usr/bin/env python

from setuptools import setup, find_packages
from codecs import open
from os import path
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy
import re
import os
import sys
import subprocess

here = path.abspath(path.dirname(__file__))


os.environ["CC"] = "g++"
os.environ["CXX"] = "g++"

# dirty but working
__version__ = re.search(
    r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]',  # It excludes inline comment too
    open('ot/__init__.py').read()).group(1)
# The beautiful part is, I don't even need to check exceptions here.
# If something messes up, let the build process fail noisy, BEFORE my release!

# thanks Pipy for handling markdown now
ROOT = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(ROOT, 'README.md'), encoding="utf-8") as f:
    README = f.read()

opt_arg = ["-O3"]

# clean cython output is clean is called
if 'clean' in sys.argv[1:]:
    if os.path.isfile('ot/lp/emd_wrap.cpp'):
        os.remove('ot/lp/emd_wrap.cpp')


# add platform dependant optional compilation argument
if sys.platform.startswith('darwin'):
    opt_arg.append("-stdlib=libc++")
    sdk_path = subprocess.check_output(['xcrun', '--show-sdk-path'])
    os.environ['CFLAGS'] = '-isysroot "{}"'.format(sdk_path.rstrip().decode("utf-8"))

setup(name='POT',
      version=__version__,
      description='Python Optimal Transport Library',
      long_description=README,
      long_description_content_type='text/markdown',
      author=u'Remi Flamary, Nicolas Courty',
      author_email='remi.flamary@gmail.com, ncourty@gmail.com',
      url='https://github.com/PythonOT/POT',
      packages=find_packages(),
      ext_modules=cythonize(Extension(
          "ot.lp.emd_wrap",                                # the extension name
          sources=["ot/lp/emd_wrap.pyx", "ot/lp/EMD_wrapper.cpp"],  # the Cython source and
          # additional C++ source files
          language="c++",                        # generate and compile C++ code,
          include_dirs=[numpy.get_include(), os.path.join(ROOT, 'ot/lp')],
          extra_compile_args=opt_arg
      )),
      platforms=['linux', 'macosx', 'windows'],
      download_url='https://github.com/PythonOT/POT/archive/{}.tar.gz'.format(__version__),
      license='MIT',
      scripts=[],
      data_files=[],
      requires=["numpy", "scipy", "cython"],
      setup_requires=["numpy>=1.16", "scipy>=1.0", "cython>=0.23"],
      install_requires=["numpy>=1.16", "scipy>=1.0", "cython>=0.23"],
      classifiers=[
          'Development Status :: 5 - Production/Stable',
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Environment :: Console',
          'Operating System :: OS Independent',
          'Operating System :: MacOS',
          'Operating System :: POSIX',
          'Programming Language :: Python',
          'Programming Language :: C++',
          'Programming Language :: C',
          'Programming Language :: Cython',
          'Topic :: Utilities',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
          'Topic :: Scientific/Engineering :: Mathematics',
          'Topic :: Scientific/Engineering :: Information Analysis',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.4',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
      ]
      )
