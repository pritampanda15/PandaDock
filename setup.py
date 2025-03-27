from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pandadock",
    version="0.7.0",
    author="Dr. Pritam Kumar Panda",
    author_email="pritam@stanford.edu",
    description="A GPU-accelerated molecular docking tool for computational drug discovery",
    long_description=long_description,
    long_description_content_type="text/markdown",
    project_urls={
        "Bug Tracker": "https://github.com/pritampanda15/pandadock/issues",
        "Documentation": "https://github.com/pritampanda15/pandadock/wiki",
    },
    url="https://github.com/pritampanda15/pandadock",
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
)
