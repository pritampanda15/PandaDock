pandadock gnn - GNN Commands Reference
======================================

The ``pandadock gnn`` command group provides access to the SE(3)-equivariant
Graph Neural Network scoring function.

Synopsis
--------

.. code-block:: bash

   pandadock gnn COMMAND [OPTIONS]

Commands
--------

* ``download-model`` - Download pre-trained model (~82 MB)
* ``train`` - Train GNN model on protein-ligand dataset
* ``predict`` - Predict binding affinity for a complex
* ``benchmark`` - Benchmark model performance on test set
* ``compare`` - Compare GNN against baseline scoring methods
* ``rescore`` - Universal rescorer for poses from ANY docking tool

pandadock gnn download-model
----------------------------

Download the official pre-trained PandaDock-GNN model from GitHub releases.

The model was trained on the combined ULVSH + PDBbind dataset (200 epochs)
and achieves:

* **PDBbind Pearson R: 0.88**
* **ULVSH Test Pearson R: 0.82**
* **ULVSH Activity AUC: 0.94**

**Options:**

``-o, --output PATH``
    Output directory for the model. Default: models/

``-v, --version TEXT``
    Model version to download. Default: latest

``-f, --force``
    Overwrite existing model file

**Example:**

.. code-block:: bash

   # Download to default location
   pandadock gnn download-model

   # Download to custom directory
   pandadock gnn download-model -o /path/to/models/

   # Force re-download
   pandadock gnn download-model --force

**Output:**

The model is saved as ``pandadock_gnn_v3.pt`` in the output directory.

After downloading, use the model with:

.. code-block:: bash

   pandadock gnn predict -m models/pandadock_gnn_v3.pt -p protein.mol2 -l ligand.mol2
   pandadock gnn rescore -m models/pandadock_gnn_v3.pt -r protein.pdb -p poses.sdf
   pandadock hybrid -r protein.pdb -l ligand.sdf -m models/pandadock_gnn_v3.pt --center X Y Z --box X Y Z

pandadock gnn train
-------------------

Train the PandaDock-GNN model on a protein-ligand dataset.

**Required Options:**

``-d, --dataset PATH``
    Path to ULVSH dataset directory

``-o, --output PATH``
    Output directory for checkpoints and logs

**Optional Options:**

``--epochs N``
    Number of training epochs. Default: 100

``--batch-size N``
    Batch size. Default: 32

``--lr FLOAT``
    Learning rate. Default: 1e-4

``--hidden-dim N``
    Hidden dimension. Default: 256

``--num-layers N``
    Number of EGNN layers. Default: 6

``--dropout FLOAT``
    Dropout rate. Default: 0.1

``--split [random|target]``
    Data split strategy. Default: random

``--patience N``
    Early stopping patience. Default: 20

``--gpu / --cpu``
    Use GPU if available. Default: --gpu

``--seed N``
    Random seed for reproducibility. Default: 42

**Example:**

.. code-block:: bash

   pandadock gnn train -d ULVSH/ -o models/ --epochs 100

pandadock gnn predict
---------------------

Predict binding affinity for a protein-ligand complex.

**Required Options:**

``-m, --model PATH``
    Path to trained model checkpoint

``-p, --protein PATH``
    Protein file (MOL2 or PDB)

``-l, --ligand PATH``
    Ligand file (MOL2 or SDF)

**Optional Options:**

``-s, --site PATH``
    Optional binding site MOL2 file

``-o, --output PATH``
    Output JSON file for results

**Example:**

.. code-block:: bash

   pandadock gnn predict -m model.pt -p protein.mol2 -l ligand.mol2

**Output:**

.. code-block:: text

   Prediction Results:
     pEC50: 6.234
     Energy: -8.52 kcal/mol
     Activity probability: 0.87
     Predicted active: True

pandadock gnn benchmark
-----------------------

Benchmark GNN model performance on a test set.

**Required Options:**

``-m, --model PATH``
    Path to trained model checkpoint

``-d, --dataset PATH``
    Path to ULVSH dataset directory

``-o, --output PATH``
    Output directory for results

**Optional Options:**

``--split [train|val|test]``
    Dataset split to evaluate. Default: test

**Example:**

.. code-block:: bash

   pandadock gnn benchmark -m model.pt -d ULVSH/ -o results/

**Output:**

Generates ``metrics.json`` with Pearson R, Spearman rho, RMSE, and MAE.

pandadock gnn compare
---------------------

Compare GNN performance against all baseline scoring methods from the ULVSH
dataset.

**Required Options:**

``-m, --model PATH``
    Path to trained model checkpoint

``-d, --dataset PATH``
    Path to ULVSH dataset directory

``-o, --output PATH``
    Output directory for comparison results

**Optional Options:**

``--split [train|val|test|all]``
    Dataset split to evaluate. Default: test

**Example:**

.. code-block:: bash

   pandadock gnn compare -m model.pt -d ULVSH/ -o comparison/

**Output:**

Generates:

* ``comparison_results.csv`` - Metrics for all methods
* ``comparison_results.json`` - JSON format
* ``comparison_plot.png`` - Bar chart visualization

Example output:

.. code-block:: text

   COMPARISON RESULTS (sorted by Pearson R)
   ======================================================================

   Method               Type            N      Pearson R
   ------------------------------------------------------------
   >>> PandaDock-GNN    ML Scoring    942        0.6705 <<<
   VM2                  ULVSH Baseline 942        0.1452
   PM6                  ULVSH Baseline 939        0.0809
   Hyde                 ULVSH Baseline 942        0.0178
   ...

   PandaDock-GNN Rank: 1/9
   *** PandaDock-GNN achieves BEST performance! ***

pandadock gnn rescore
---------------------

**Universal GNN rescorer for poses from any docking tool.**

This command allows you to rescore docked poses from ANY docking software
(AutoDock Vina, Glide, GOLD, pandadock-flex, pandadock-metal, etc.) using
the SE(3)-equivariant GNN scoring function.

**Required Options:**

``-m, --model PATH``
    Path to trained model checkpoint

``-r, --receptor PATH``
    Receptor file (PDB or MOL2)

``-p, --poses PATH``
    Poses file (multi-conformer SDF from any docking tool)

**Optional Options:**

``-o, --output PATH``
    Output CSV file with ranked poses. Default: rescored_poses.csv

``--output-sdf PATH``
    Output SDF file with poses ranked by GNN score and GNN properties added

``--site-radius FLOAT``
    Radius around ligand centroid to extract binding site (Angstrom). Default: 10.0

**Examples:**

.. code-block:: bash

   # Rescore poses from pandadock-flex
   pandadock gnn rescore -m model.pt -r protein.pdb -p flex_poses.sdf

   # Rescore AutoDock Vina output
   pandadock gnn rescore -m model.pt -r receptor.pdb -p vina_out.sdf -o ranked.csv

   # Get ranked SDF with GNN scores as properties
   pandadock gnn rescore -m model.pt -r protein.pdb -p poses.sdf --output-sdf ranked.sdf

   # Rescore Glide output
   pandadock gnn rescore -m model.pt -r protein.pdb -p glide_poses.sdf -o glide_rescored.csv

**Output CSV Format:**

.. code-block:: text

   pose_name,pose_index,gnn_pKd,gnn_energy,activity_prob,predicted_active,gnn_rank
   pose_3,3,7.234,-9.88,0.92,True,1
   pose_1,1,6.891,-9.41,0.88,True,2
   pose_5,5,6.543,-8.93,0.81,True,3
   ...

**Output SDF Properties:**

When ``--output-sdf`` is specified, each molecule in the output SDF will have:

* ``GNN_pKd`` - Predicted pKd/pKi value
* ``GNN_Energy`` - Predicted binding energy (kcal/mol)
* ``GNN_Activity`` - Activity probability (0-1)
* ``GNN_Rank`` - Rank based on GNN score (1 = best)

**Workflow Example:**

Combine with any docking tool:

.. code-block:: bash

   # Step 1: Run flexible docking with pandadock-flex
   pandadock-flex -r protein.pdb -l ligand.sdf --center 10 20 30 -o flex_output/

   # Step 2: Rescore poses with GNN
   pandadock gnn rescore -m model.pt -r protein.pdb -p flex_output/poses.sdf \\
       -o flex_rescored.csv --output-sdf flex_rescored.sdf

   # Or with AutoDock Vina
   vina --receptor receptor.pdbqt --ligand ligand.pdbqt --out vina_poses.sdf
   pandadock gnn rescore -m model.pt -r receptor.pdb -p vina_poses.sdf

See Also
--------

* :doc:`../gnn/overview` - GNN architecture documentation
* :doc:`../gnn/training` - Training guide
* :doc:`../gnn/prediction` - Prediction guide
* :doc:`../gnn/hybrid_docking` - Hybrid docking workflow
