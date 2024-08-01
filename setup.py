#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from setuptools import setup, find_packages
import setuptools

def get_version() -> str:
    # https://packaging.python.org/guides/single-sourcing-package-version/
    init = open(os.path.join("drones", "__init__.py"), "r").read().split()
    return init[init.index("__version__") + 2][1:-1]

setup(
    name="drones",  # Replace with your own username
    version=get_version(),
    description="drones algorithms of marlbenchmark",
    long_description=open("README.md", encoding="utf8").read(),
    long_description_content_type="text/markdown",
    author="longqianzhao",
    author_email="bwcx@nuaa.edu.cn",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords="multi-agent reinforcement learning platform pytorch",
    python_requires='==3.9.13',
)
