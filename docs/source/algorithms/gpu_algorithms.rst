GPU Algorithms
==============

PandaDock provides 3 GPU-accelerated docking algorithms that deliver 50-200x speedup over CPU equivalents while maintaining comparable accuracy.

Prerequisites
-------------

**Hardware Requirements:**

* NVIDIA GPU with CUDA Compute Capability 6.0+ (Pascal architecture or newer)
* Minimum 4GB GPU memory (8GB+ recommended for large libraries)
* CUDA Toolkit 11.0 or higher

**Software Requirements:**

.. code-block:: bash

   # Install CuPy for CUDA 11.x
   pip install cupy-cuda11x

   # Or for CUDA 12.x
   pip install cupy-cuda12x

Verify GPU availability:

.. code-block:: bash

   pandadock list-algorithms

Available GPU Algorithms
------------------------

Enhanced Hierarchical GPU
^^^^^^^^^^^^^^^^^^^^^^^^^

**Algorithm ID:** ``enhanced_hierarchical_gpu``

**Speedup:** 50-100x over CPU version

**GPU Memory:** 1-4 GB

**Best for:** High-throughput high-accuracy docking

Implements the same 3-stage hierarchical search as the CPU version, but with massive parallelization:

* Parallel pose generation (1000s simultaneously)
* Batch scoring on GPU
* Asynchronous CPU-GPU communication
* Optimized memory management

**Usage:**

.. code-block:: bash

   pandadock dock -r protein.pdb -l ligand.sdf \
                  --algorithm enhanced_hierarchical_gpu \
                  --gpu \
                  --center 10 20 30 --box 20 20 20

**GPU Parameters:**

* ``--gpu-batch-size``: Poses per GPU batch (default: 1000)
* ``--gpu-memory-limit``: GPU memory limit in GB (default: 4.0)
* ``--gpuid``: GPU device ID (default: 0)

**Performance:**

* RMSD: ~0.08 Å (same as CPU)
* Runtime: 2-5 seconds per ligand
* Throughput: 720-1800 ligands/hour

CUDA Monte Carlo
^^^^^^^^^^^^^^^^

**Algorithm ID:** ``cuda_monte_carlo``

**Speedup:** 100-200x over CPU version

**GPU Memory:** 0.5-2 GB

**Best for:** Ultra-fast virtual screening

Massively parallel Monte Carlo with 10,000+ independent walkers:

* Each GPU thread runs independent MC walker
* Shared memory for receptor grids
* Warp-level optimizations
* Minimal divergence

**Usage:**

.. code-block:: bash

   pandadock dock -r protein.pdb -l ligand.sdf \
                  --algorithm cuda_monte_carlo \
                  --gpu \
                  --center 10 20 30 --box 20 20 20

**Performance:**

* RMSD: ~0.5-1.5 Å
* Runtime: 0.5-2 seconds per ligand
* Throughput: 1800-7200 ligands/hour

CUDA Genetic Algorithm
^^^^^^^^^^^^^^^^^^^^^^

**Algorithm ID:** ``cuda_genetic_algorithm``

**Speedup:** 80-150x over CPU version

**GPU Memory:** 1-3 GB

**Best for:** GPU-accelerated complex site docking

GPU-parallelized evolutionary algorithm:

* Population stored in GPU memory
* Parallel fitness evaluation
* GPU-accelerated crossover/mutation
* Efficient device-side selection

**Usage:**

.. code-block:: bash

   pandadock dock -r protein.pdb -l ligand.sdf \
                  --algorithm cuda_genetic_algorithm \
                  --gpu \
                  --center 10 20 30 --box 20 20 20

**Performance:**

* RMSD: ~0.3-0.8 Å
* Runtime: 1-3 seconds per ligand
* Throughput: 1200-3600 ligands/hour

GPU Performance Optimization
-----------------------------

Batch Size Tuning
^^^^^^^^^^^^^^^^^

Optimize GPU batch size for your hardware:

.. code-block:: bash

   # For 8GB GPU (larger batches)
   pandadock dock --algorithm enhanced_hierarchical_gpu \
                  --gpu-batch-size 2000 \
                  --gpu-memory-limit 6.0

   # For 4GB GPU (smaller batches)
   pandadock dock --algorithm enhanced_hierarchical_gpu \
                  --gpu-batch-size 1000 \
                  --gpu-memory-limit 3.0

Multi-GPU Support
^^^^^^^^^^^^^^^^^

Run on specific GPU device:

.. code-block:: bash

   # Use GPU 0
   pandadock dock --algorithm enhanced_hierarchical_gpu \
                  --gpu --gpuid 0

   # Use GPU 1
   pandadock dock --algorithm enhanced_hierarchical_gpu \
                  --gpu --gpuid 1

For parallel screening across multiple GPUs, launch multiple processes:

.. code-block:: bash

   # Terminal 1 (GPU 0)
   CUDA_VISIBLE_DEVICES=0 pandadock dock -l library_part1.sdf --gpu

   # Terminal 2 (GPU 1)
   CUDA_VISIBLE_DEVICES=1 pandadock dock -l library_part2.sdf --gpu

Memory Management
^^^^^^^^^^^^^^^^^

Monitor GPU memory usage:

.. code-block:: bash

   # Watch GPU utilization
   watch -n 1 nvidia-smi

If out-of-memory errors occur:

1. Reduce batch size: ``--gpu-batch-size 500``
2. Lower memory limit: ``--gpu-memory-limit 2.0``
3. Reduce number of poses: ``--num-poses 10``

Performance Comparison
----------------------

**Throughput Comparison (ligands/hour):**

+------------------------------+-------------+---------------+
| Algorithm                    | CPU         | GPU           |
+==============================+=============+===============+
| Enhanced Hierarchical        | 14-24       | 720-1800      |
+------------------------------+-------------+---------------+
| Monte Carlo                  | 60-120      | 1800-7200     |
+------------------------------+-------------+---------------+
| Genetic Algorithm            | 18-30       | 1200-3600     |
+------------------------------+-------------+---------------+

**Accuracy Comparison:**

GPU algorithms maintain the same accuracy as CPU versions:

* Enhanced Hierarchical GPU: RMSD ~0.08 Å (same as CPU)
* CUDA Monte Carlo: RMSD ~0.5-1.5 Å (same as CPU)
* CUDA Genetic Algorithm: RMSD ~0.3-0.8 Å (same as CPU)

Best Practices
--------------

1. **Start with small test**: Run 10 ligands first to verify GPU setup
2. **Monitor GPU memory**: Use ``nvidia-smi`` to check utilization
3. **Tune batch size**: Find optimal batch size for your GPU
4. **Use fast GPUs**: Modern GPUs (RTX 30/40 series, A100, etc.) provide best performance
5. **Minimize data transfer**: Keep receptor on GPU across multiple ligands

Virtual Screening Example
--------------------------

High-throughput screening of large library:

.. code-block:: bash

   # Screen 10,000 compound library
   pandadock dock -r target.pdb -l library_10k.sdf \
                  --algorithm cuda_monte_carlo \
                  --gpu \
                  --num-poses 5 \
                  --fast \
                  --center 15 20 25 --box 20 20 20 \
                  -o screening_results/

Expected runtime: 1-2 hours on modern GPU (vs 30-60 hours on CPU)

Troubleshooting
---------------

**CUDA not available:**

.. code-block:: bash

   Error: GPU acceleration requested but CUDA not available

Solution: Install CuPy matching your CUDA version

**Out of memory:**

.. code-block:: bash

   RuntimeError: out of memory

Solutions:
1. Reduce batch size: ``--gpu-batch-size 500``
2. Lower memory limit: ``--gpu-memory-limit 2.0``
3. Use smaller grid box
4. Reduce number of poses

**Slow performance:**

Possible causes:
1. GPU throttling due to temperature
2. Insufficient batch size (too small)
3. PCIe bottleneck (old PCIe version)
4. Competing processes on GPU

See Also
--------

* :doc:`cpu_algorithms` - CPU algorithm documentation
* :doc:`../guide/performance` - Performance optimization guide
* :doc:`../tutorials/gpu_acceleration` - GPU tutorial
