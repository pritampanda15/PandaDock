from setuptools import setup, find_packages
from pathlib import Path

# Read README for PyPI long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="pandadock",
    version="4.0.2",
    packages=find_packages(),

    # PyPI documentation from README
    long_description=long_description,
    long_description_content_type="text/markdown",

    # Project metadata
    author="Pritam Kumar Panda",
    author_email="pritampanda15@gmail.com",
    description="Molecular docking with SE(3)-equivariant GNN scoring - achieves R=0.88 on PDBbind",
    url="https://github.com/pritampanda15/PandaDock",
    project_urls={
        "Documentation": "https://pandadock.readthedocs.io/",
        "Bug Reports": "https://github.com/pritampanda15/PandaDock/issues",
        "Source": "https://github.com/pritampanda15/PandaDock",
    },

    # Classifiers for PyPI
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="molecular-docking, drug-discovery, graph-neural-network, GNN, SE3-equivariant, binding-affinity, computational-chemistry, bioinformatics",

    # Dependencies
    install_requires=[
        "click>=8.0.0",
        "biopython>=1.80",
        # Note: rdkit must be installed via conda (conda install -c conda-forge rdkit)
        "propka>=3.5.1",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "plotly>=5.0.0",
    ],
    extras_require={
        "torch": [
            "torch>=2.0.0",
        ],
        "ml": [
            "torch>=2.0.0",
            "h5py>=3.7.0",
        ],
        "gnn": [
            "torch>=2.0.0",
            "torch-geometric>=2.4.0",
            "torch-scatter",
            "torch-sparse",
            "pandas>=1.3.0",
        ],
        "conda": [
            "openmm>=8.0.0",
            "pdbfixer>=1.9",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "pandadock-prepare=pandadock.cli:main",
            "pandadock-gridbox=pandadock.gridbox_cli:main",
            "pandadock=pandadock.docking_cli:main",
            "pandadock-flex=pandadock.flex_docking_cli:main",
            "pandadock-report=pandadock.report_cli:main",
            "pandadock-metal=pandadock.metal_docking_cli:main",
            "pandadock-tethered=pandadock.tethered_cli:main",
            "pandadock-ml=pandadock.ml_docking_cli:main",
            "pandadock-gnn=pandadock.gnn.cli:main",
        ],
    },
    python_requires=">=3.8",
    license="MIT",
)
