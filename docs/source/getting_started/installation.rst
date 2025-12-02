Installation
============

This guide provides detailed instructions for installing PandaDock.

System Requirements
-------------------

**Minimum Requirements (CPU Only)**

* Operating System: Linux, macOS, or Windows (WSL2)
* Python: 3.8 or higher
* RAM: 8 GB minimum, 16 GB recommended
* Disk Space: 5 GB for installation

**Recommended Requirements (GPU Acceleration)**

* GPU: NVIDIA GPU with CUDA Compute Capability 6.0+
* CUDA: Version 11.0 or higher
* GPU Memory: 4 GB minimum, 8+ GB recommended
* NVIDIA Driver: Latest stable version

Quick Install (CPU Only)
-------------------------

.. code-block:: bash

   git clone https://github.com/pritampanda15/PandaDock.git
   cd PandaDock
   pip install -e .

GPU-Accelerated Installation
-----------------------------

.. code-block:: bash

   git clone https://github.com/pritampanda15/PandaDock.git
   cd PandaDock
   pip install -e .

   # Install CUDA support
   pip install cupy-cuda11x  # For CUDA 11
   # OR
   pip install cupy-cuda12x  # For CUDA 12

Verify Installation
-------------------

.. code-block:: bash

   pandadock --version
   pandadock list-algorithms

You should see the version information and list of available algorithms.
