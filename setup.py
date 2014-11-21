#!/usr/bin/env python

import re
import sys

from setuptools import setup, find_packages


def version():
    with open('pyqmc/_version.py') as f:
        return f.read()


extra = {}
if sys.version_info >= (3,):
    extra['use_2to3'] = True

setup(name='pyqmc',
      version=version(),
      packages=find_packages(),

      install_requires=['numpy'], #graph_tool

      setup_requires=['setuptools_git >= 0.3', ],

      include_package_data=True,
      exclude_package_data={'': ['README', '.gitignore']},

      zip_safe=True,

      author='Steve Nguyen',
      author_email='steve.nguyen.000@gmail.com',
      description='Python Library for Quasi Metric Control',
      url='https://github.com/poppy-project/pypot',
      license='GNU GENERAL PUBLIC LICENSE Version 3',

      classifiers=[
          "Programming Language :: Python :: 2",
          "Programming Language :: Python :: 3",
          "Topic :: Scientific/Engineering", ],

      **extra
      )
