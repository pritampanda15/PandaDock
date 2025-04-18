import sys
import os
import json
from setuptools import setup, find_packages
from setuptools.command.install import install
from distutils import log
from urllib.request import urlopen
from packaging import version

# Custom install command to check PyPI and show upgrade warning
class CustomInstallCommand(install):
    """Custom install that warns user if newer version is available on PyPI"""
    def run(self):
        install.run(self)
        try:
            response = urlopen("https://pypi.org/pypi/pandadock/json", timeout=3)
            data = json.loads(response.read().decode())
            latest = data["info"]["version"]
            current = self.distribution.get_version()

            if version.parse(latest) > version.parse(current):
                log.warn("\033[91m" + "="*72)
                log.warn(f"\033[91mWARNING: PandaDock {latest} is available (you have {current})")
                log.warn("\033[91mTo upgrade: pip install --upgrade pandadock")
                log.warn("\033[91mVisit: https://github.com/pritampanda15/PandaDock/releases")
                log.warn("\033[91m" + "="*72 + "\033[0m")
        except Exception:
            pass  # Silent fail

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pandadock",
    version="1.4.0",
    author="Dr. Pritam Kumar Panda",
    author_email="pritam@stanford.edu",
    description="A Python based GPU/CPU-accelerated molecular docking tool for computational drug discovery",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pritampanda15/PandaDock",
    project_urls={
        "Bug Tracker": "https://github.com/pritampanda15/PandaDock/issues",
        "Documentation": "https://github.com/pritampanda15/PandaDock/wiki",
    },
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Chemistry",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "scipy>=1.5.0",
        "matplotlib>=3.3.0",
        "scikit-learn>=0.24.0",
        "requests",
        "packaging",
        "pandas",
        "tabulate",
        "tqdm"
        "seaborn"
    ],
    entry_points={
        'console_scripts': [
            'pandadock=pandadock.main:main',
        ],
    },
    extras_require={
        "gpu": ["torch>=1.9.0"],
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
            "black>=20.8b1",
            "flake8>=3.8.0",
        ],
        "rdkit": ["rdkit>=2020.09.1"],
    },
    include_package_data=True,
    cmdclass={
        'install': CustomInstallCommand,
    },
)
