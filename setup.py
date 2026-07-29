"""
Compatibility shim.

All packaging metadata now lives in pyproject.toml, with the version read from
`pandadock.__version__` so there is a single source of truth. This file remains
only so that `python setup.py ...` and older toolchains keep working.
"""

from setuptools import setup

setup()
