Installation
============

This guide walks through installing **PandaDock v5.0** on macOS, Linux, or Windows (WSL2).

System Requirements
-------------------

**Minimum Requirements (CPU)**

* **Operating System:** macOS (Apple Silicon or Intel), Linux, or Windows (WSL2)
* **Python:** 3.9 or higher (Python 3.10 recommended)
* **RDKit:** Required (most reliably installed via ``conda-forge``)
* **RAM:** 8 GB minimum, 16 GB recommended

**Optional GPU Acceleration**

* **NVIDIA CUDA:** PyTorch with CUDA 11.8+ / 12.x (up to 35× faster on flexible ligands and batched virtual screening)
* **Apple Silicon (MPS):** macOS 13+ on M1 / M2 / M3 / M4 via PyTorch MPS (``--device mps``)

Recommended Installation (Conda + pip)
--------------------------------------

.. code-block:: bash

   # 1. Create a dedicated environment with RDKit
   conda create -n pandadock python=3.10 -y
   conda activate pandadock
   conda install -c conda-forge rdkit -y

   # 2. Install PandaDock v5.0
   pip install pandadock

Install from Source (Development)
---------------------------------

.. code-block:: bash

   git clone https://github.com/pritampanda15/PandaDock.git
   cd PandaDock
   pip install -e .

Optional Feature Extras
-----------------------

.. code-block:: bash

   # PandaDock Studio Apple-designed web GUI (FastAPI + Uvicorn)
   pip install "pandadock[gui]"

   # GPU-accelerated search (CUDA/MPS) and PandaDock-GNN scoring (PyTorch + PyG)
   pip install "pandadock[gnn]"

   # 2D Discovery Studio-style protein-ligand interaction diagrams
   pip install pandamap

   # Or from a local source checkout:
   pip install -e ".[gui,gnn]"

Verify Installation
-------------------

.. code-block:: bash

   pandadock --version
   pandadock list-algorithms
   pandadock-gui --help

You should see ``PandaDock 5.0.0`` and the full list of docking engines and scoring functions.
