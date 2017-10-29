#!/usr/bin/env python

from distutils.core import setup

setup(name='GenericML',
      version='1.0',
      description='Generic Machine Learning',
      author='Gabriel Maher',
      author_email='',
      url='',
      packages=['GenericML', 'GenericML.dataset', 'GenericML.model',
        'GenericML.rl'],
     )
