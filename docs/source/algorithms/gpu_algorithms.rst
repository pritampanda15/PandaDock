GPU-Accelerated Search
======================

PandaDock provides an experimental PyTorch GPU backend (``--device cuda`` or
``--device mps``) implementing the **same** flexible-ligand basin-hopping +
batched L-BFGS algorithm as the CPU search.

.. note::
   The legacy v3.0 CUDA modules (``monte_carlo_gpu``, ``genetic_algorithm_gpu``,
   ``enhanced_hierarchical_gpu``) were removed because they only encoded rigid
   translation and orientation without ligand torsions. Use ``--device cuda``
   or ``--device mps`` on ``pandadock dock`` or ``pandadock screen`` instead.

Single-Ligand GPU Docking
-------------------------

.. code-block:: bash

   pandadock dock -r receptor.pdb -l ligand.sdf \
                  --center 10 12 8 --box 22 22 22 \
                  --device cuda --n-chains 512 -o results/

* On an NVIDIA A10G across 131 paired benchmark complexes, CUDA is **8.5x
  faster overall** (up to **35.4x** on 21+ rotatable bonds) at identical
  best-of-9 accuracy (56.5% vs 56.5%) and a 3.9-point top-1 gap (30.5% vs
  34.4%).
* Numerical parity against the CPU objective is verified on float64 devices
  (coordinates 3.6e-15, energy 1.4e-14). Apple MPS runs in float32 during
  search; final poses are always rescored in float64 on the CPU before
  clustering and output.

Multi-Ligand Virtual Screening (``pandadock screen``)
-----------------------------------------------------

GPU acceleration pays off most when batching multiple ligands together:

.. code-block:: bash

   pandadock screen -r receptor.pdb -l library.sdf \
                    --center 10 12 8 --box 22 22 22 \
                    --device cuda --n-chains 128 --max-batch 64 -o screen_out/

``pandadock screen`` automatically shares signature-keyed affinity grids across
the entire library and buckets ligands by ``(n_torsions, n_atoms)`` to minimize
padding waste.
