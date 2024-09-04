#!/usr/bin/env python

"""
The setup script for pip. Allows for `pip install -e .` installation.
"""

from setuptools import setup, find_packages

requirements = ['numpy', 'matplotlib', 'torch', 'h5py', 'PyYAML', 'torchvision']
setup_requirements = []
tests_requirements = ['pytest']

setup(
    name='pde_nn',
    packages=find_packages(include=['src']),
    setup_requires=setup_requirements
)
