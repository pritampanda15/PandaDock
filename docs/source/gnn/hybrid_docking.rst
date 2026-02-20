Hybrid Docking
==============

The hybrid docking workflow combines traditional pose generation with GNN
rescoring for optimal accuracy. This is the **recommended** approach for
production use.

Why Hybrid Docking?
-------------------

Traditional docking algorithms are good at:

* Generating diverse pose conformations
* Sampling the binding site efficiently
* Running quickly

But they struggle with:

* Accurate affinity ranking
* Correlation with experimental data

The GNN excels at:

* Predicting binding affinity
* Ranking poses correctly
* Correlating with experimental pEC50

The hybrid approach combines these strengths.

Workflow
--------

.. code-block:: text

   Phase 1: Pose Generation
   ┌─────────────────────────────────────────┐
   │  Hierarchical Docking (Vina scoring)    │
   │  → Generate 50+ diverse poses           │
   └────────────────────┬────────────────────┘
                        │
   Phase 2: GNN Rescoring
   ┌────────────────────▼────────────────────┐
   │  SE(3)-Equivariant GNN                  │
   │  → Predict pEC50 for each pose          │
   └────────────────────┬────────────────────┘
                        │
   Phase 3: Ranking & Output
   ┌────────────────────▼────────────────────┐
   │  Rank by GNN pEC50                      │
   │  → Output top-K poses                   │
   └─────────────────────────────────────────┘

Basic Usage
-----------

.. code-block:: bash

   pandadock hybrid -r protein.pdb -l ligand.sdf \
                    --center 10 20 30 --box 20 20 20 \
                    -m models/best_model.pt

Full Options
------------

.. code-block:: bash

   pandadock hybrid \
       --receptor protein.pdb \
       --ligand ligand.sdf \
       --center 10 20 30 \
       --box 20 20 20 \
       --model model.pt \
       --output-dir hybrid_results/ \
       --num-poses 50 \
       --top-k 10 \
       --fast                    # Optional: quick mode

Command Options
---------------

+------------------+----------+---------------------------------------------+
| Option           | Default  | Description                                 |
+==================+==========+=============================================+
| ``-r/--receptor``| Required | Receptor PDB file                           |
+------------------+----------+---------------------------------------------+
| ``-l/--ligand``  | Required | Ligand file (SDF/MOL2/PDB)                  |
+------------------+----------+---------------------------------------------+
| ``--center``     | Required | Grid center coordinates (X Y Z)             |
+------------------+----------+---------------------------------------------+
| ``--box``        | Required | Grid box dimensions (X Y Z)                 |
+------------------+----------+---------------------------------------------+
| ``-m/--model``   | Required | GNN model checkpoint                        |
+------------------+----------+---------------------------------------------+
| ``-o/--output``  | hybrid/  | Output directory                            |
+------------------+----------+---------------------------------------------+
| ``-n/--num-poses``| 50      | Poses to generate for rescoring             |
+------------------+----------+---------------------------------------------+
| ``--top-k``      | 10       | Top poses to keep after rescoring           |
+------------------+----------+---------------------------------------------+
| ``--fast``       | False    | Use fast mode (fewer poses)                 |
+------------------+----------+---------------------------------------------+

Output Files
------------

The output directory contains:

* ``hybrid_results.csv``: Rankings with GNN and Vina scores
* ``pose_1_pec50_X.XX.pdb``: Top pose structures
* ``complex_1.pdb``: Protein-ligand complexes

Example output table:

.. code-block:: text

   Rank  GNN pEC50    GNN Energy    Vina Energy    Activity
   1     7.234        -9.87         -8.23          0.92
   2     6.891        -9.41         -7.89          0.85
   3     6.543        -8.93         -8.45          0.78
   ...

Performance
-----------

Typical timings on GPU:

* **Phase 1** (50 poses): 10-30 seconds
* **Phase 2** (rescoring): 1-2 seconds
* **Total**: 15-35 seconds per ligand

Compared to traditional docking alone:

* ~5x better ranking correlation
* Same pose generation quality
* Minimal overhead from GNN

Best Practices
--------------

1. **Generate many poses**: Use ``--num-poses 50`` or more
2. **Keep diverse poses**: The GNN will find the best ones
3. **Use trained model**: Train on similar targets if possible
4. **Check activity probability**: High confidence = reliable prediction

Virtual Screening
-----------------

For screening many ligands:

.. code-block:: bash

   for ligand in ligands/*.sdf; do
       pandadock hybrid -r protein.pdb -l "$ligand" \
                        --center 10 20 30 --box 20 20 20 \
                        -m model.pt -o "results/$(basename $ligand .sdf)"
   done

Or use the Python API for parallelization.

Comparison with Traditional Docking
-----------------------------------

+---------------------------+-----------------+-------------------+
| Metric                    | Traditional     | Hybrid            |
+===========================+=================+===================+
| Affinity Correlation (R)  | 0.12            | **0.67**          |
+---------------------------+-----------------+-------------------+
| Ranking Accuracy          | Low             | **High**          |
+---------------------------+-----------------+-------------------+
| Speed                     | Fast            | Fast              |
+---------------------------+-----------------+-------------------+
| Pose Quality              | Good            | Good              |
+---------------------------+-----------------+-------------------+
