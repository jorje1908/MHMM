#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 00:23:45 2019

@author: george
"""

from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("_utils_cy.pyx")
    
)
