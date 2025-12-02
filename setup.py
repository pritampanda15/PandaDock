from setuptools import setup, find_packages

setup(
    name="pandadock",
    version="3.0.0",
    packages=find_packages(),
    install_requires=[
        "click>=8.0.0",
        "biopython>=1.80",
        "rdkit>=2022.9.5",
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
        "conda": [
            "openmm>=8.0.0",
            "pdbfixer>=1.9",
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
        ],
    },
    python_requires=">=3.8",
    author="Pritam Kumar Panda @ Stanford University",
    description="GPU-accelerated molecular docking software",
)
