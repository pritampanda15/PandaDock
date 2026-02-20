GNN Prediction
==============

This guide covers using the trained GNN model for binding affinity prediction.

Basic Prediction
----------------

Predict binding affinity for a single protein-ligand complex:

.. code-block:: bash

   pandadock gnn predict -m models/best_model.pt \
                         -p protein.mol2 -l ligand.mol2

Output:

.. code-block:: text

   Loading model...
   Predicting...

   Prediction Results:
     pEC50: 6.234
     Energy: -8.52 kcal/mol
     Activity probability: 0.87
     Predicted active: True

Supported File Formats
----------------------

**Protein files:**

* MOL2 (recommended)
* PDB
* Any format readable by BioPython

**Ligand files:**

* MOL2 (recommended)
* SDF
* PDB
* Any format readable by RDKit

Prediction Options
------------------

.. code-block:: bash

   pandadock gnn predict \
       --model model.pt \
       --protein protein.mol2 \
       --ligand ligand.mol2 \
       --site site.mol2 \        # Optional binding site
       --output results.json     # Save results to file

Output Format
-------------

When using ``--output``, results are saved as JSON:

.. code-block:: json

   {
     "pec50": 6.234,
     "energy": -8.52,
     "activity_prob": 0.87,
     "active": true
   }

Interpreting Results
--------------------

**pEC50**
   Predicted -log10(EC50) value. Higher = stronger binding.
   Typical range: 4-10 (μM to nM affinity).

**Energy**
   Estimated binding energy in kcal/mol.
   Calculated as: energy = -1.366 * pEC50

**Activity Probability**
   Probability that the compound is active (EC50 < threshold).
   Range: 0-1.

**Active**
   Binary classification: True if activity_prob > 0.5.

Batch Prediction
----------------

For multiple complexes, use the benchmark command:

.. code-block:: bash

   pandadock gnn benchmark -m model.pt -d dataset/ -o results/

Or use Python API:

.. code-block:: python

   from pandadock.gnn.scoring import GNNScoring

   scorer = GNNScoring(model_path='model.pt')

   for protein, ligand in complexes:
       result = scorer.predict_affinity(protein, ligand)
       print(f"pEC50: {result['pec50']:.3f}")

Python API
----------

Direct access to the GNN scorer:

.. code-block:: python

   from pandadock.gnn.scoring import GNNScoring

   # Load model
   scorer = GNNScoring(model_path='models/best_model.pt')

   # Predict from files
   result = scorer.predict_affinity(
       protein_file='protein.mol2',
       ligand_file='ligand.mol2',
       site_file='site.mol2'  # optional
   )

   print(f"pEC50: {result['pec50']:.3f}")
   print(f"Energy: {result['energy']:.3f} kcal/mol")

   # Predict from graph (advanced)
   from pandadock.gnn.data.graph_builder import HeterogeneousGraphBuilder

   builder = HeterogeneousGraphBuilder()
   graph = builder.build_graph(protein_mol2, ligand_mol2, site_mol2)
   result = scorer.predict_from_graph(graph)

Performance Notes
-----------------

* **GPU inference**: ~0.02 seconds per complex
* **CPU inference**: ~0.1 seconds per complex
* **Memory**: ~200 MB GPU memory per batch of 32

Troubleshooting
---------------

**"Could not parse molecule"**
   Check file format. MOL2 is most reliable.

**"Model config mismatch"**
   Ensure model was trained with same PandaDock version.

**"CUDA out of memory"**
   Reduce batch size or use CPU (``--cpu``).
