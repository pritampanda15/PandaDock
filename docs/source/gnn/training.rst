GNN Training
============

This guide covers training the PandaDock-GNN model on protein-ligand datasets.

Prerequisites
-------------

Install the GNN dependencies:

.. code-block:: bash

   pip install -e ".[gnn]"

This installs PyTorch, PyTorch Geometric, and related packages.

Supported Datasets
------------------

PandaDock-GNN supports training on multiple datasets:

* **ULVSH**: 942 compounds, 10 protein targets (pEC50 values)
* **BindingDB**: 8,891 protein-ligand complexes with experimental pK values
* **PDBbind**: 5,316 complexes with pKd/pKi values (v2020 refined set)

Dataset Preparation
-------------------

**ULVSH Format**

PandaDock-GNN is designed for the ULVSH dataset format:

.. code-block:: text

   ULVSH/
   ├── TARGET1/
   │   └── raw/
   │       ├── protein.mol2      # Protein structure
   │       ├── vitro.tsv         # EC50 values
   │       └── COMPOUND_ID/
   │           ├── ligand.mol2   # Ligand structure
   │           └── site.mol2     # Binding site atoms
   ├── TARGET2/
   │   └── ...
   └── ...

The ``vitro.tsv`` file should contain:

.. code-block:: text

   ID         EC50[uM]    Activity
   compound1  0.5         1
   compound2  10.0        0
   ...

**BindingDB Format**

For BindingDB training, prepare a TSV file with protein-ligand complex paths:

.. code-block:: text

   complex_id    protein_file              ligand_file           pK
   complex_1     proteins/1abc.mol2        ligands/lig1.mol2     7.5
   complex_2     proteins/2def.mol2        ligands/lig2.mol2     6.2
   ...

Training Command
----------------

Basic training:

.. code-block:: bash

   pandadock gnn train -d ULVSH/ -o models/ --epochs 100

Full options:

.. code-block:: bash

   pandadock gnn train \
       --dataset ULVSH/ \
       --output models/ \
       --epochs 100 \
       --batch-size 32 \
       --lr 1e-4 \
       --hidden-dim 256 \
       --num-layers 6 \
       --dropout 0.1 \
       --patience 20 \
       --gpu

BindingDB Training
------------------

Train on BindingDB dataset:

.. code-block:: bash

   # BindingDB only training
   python BindingDB_training/train_bindingdb.py \
       --bindingdb BindingDB_training/bindingdb_affinity.tsv \
       --output models/ \
       --epochs 100 \
       --batch-size 16

Combined training with ULVSH (recommended for generalization):

.. code-block:: bash

   # BindingDB + ULVSH combined training
   python BindingDB_training/train_bindingdb.py \
       --bindingdb BindingDB_training/bindingdb_affinity.tsv \
       --ulvsh ULVSH/ \
       --combined \
       --output models/ \
       --epochs 100

**Benchmark Results:**

+---------------------------+----------------+-----------+
| Training Configuration    | Test Pearson R | Test RMSE |
+===========================+================+===========+
| BindingDB Only            | 0.81           | -         |
+---------------------------+----------------+-----------+
| BindingDB + ULVSH         | 0.79           | 0.96      |
+---------------------------+----------------+-----------+

Training Options
----------------

+------------------+----------+---------------------------------------------+
| Option           | Default  | Description                                 |
+==================+==========+=============================================+
| ``--epochs``     | 100      | Number of training epochs                   |
+------------------+----------+---------------------------------------------+
| ``--batch-size`` | 32       | Batch size                                  |
+------------------+----------+---------------------------------------------+
| ``--lr``         | 1e-4     | Learning rate                               |
+------------------+----------+---------------------------------------------+
| ``--hidden-dim`` | 256      | Hidden dimension                            |
+------------------+----------+---------------------------------------------+
| ``--num-layers`` | 6        | Number of EGNN layers                       |
+------------------+----------+---------------------------------------------+
| ``--dropout``    | 0.1      | Dropout rate                                |
+------------------+----------+---------------------------------------------+
| ``--patience``   | 20       | Early stopping patience                     |
+------------------+----------+---------------------------------------------+
| ``--split``      | random   | Data split strategy (random or target)      |
+------------------+----------+---------------------------------------------+
| ``--gpu/--cpu``  | --gpu    | Use GPU if available                        |
+------------------+----------+---------------------------------------------+
| ``--seed``       | 42       | Random seed for reproducibility             |
+------------------+----------+---------------------------------------------+

Training Output
---------------

After training, the output directory contains:

* ``best_model.pt``: Best model checkpoint (by validation loss)
* ``final_model.pt``: Final model checkpoint
* ``training_log.csv``: Training metrics per epoch
* ``training_results.json``: Final metrics summary

Model Checkpoint Format
-----------------------

The checkpoint contains:

.. code-block:: python

   {
       'config': ModelConfig,       # Model configuration
       'state_dict': dict,          # Model weights
       'training_config': dict,     # Training configuration
       'best_metrics': dict,        # Best validation metrics
       'epoch': int                 # Checkpoint epoch
   }

Monitoring Training
-------------------

The training loop outputs:

* Loss values (total, affinity, activity)
* Validation metrics (Pearson R, Spearman ρ, RMSE, MAE)
* Early stopping status
* Best model updates

Example output:

.. code-block:: text

   Epoch 1/100 [====================]
   Train Loss: 0.5234 | Val Loss: 0.4123 | Val R: 0.45

   Epoch 2/100 [====================]
   Train Loss: 0.3421 | Val Loss: 0.3012 | Val R: 0.58
   * New best model saved

   ...

Tips for Better Training
------------------------

1. **Use GPU**: Training is ~10x faster on GPU
2. **Start with default hyperparameters**: They work well for ULVSH
3. **Monitor validation R**: Should steadily increase
4. **Early stopping**: 20 epochs patience is usually sufficient
5. **Batch size**: 32 works well; reduce if memory issues

Evaluating the Model
--------------------

After training, benchmark on test set:

.. code-block:: bash

   pandadock gnn benchmark -m models/best_model.pt \
                           -d ULVSH/ -o results/

Compare against baselines:

.. code-block:: bash

   pandadock gnn compare -m models/best_model.pt \
                         -d ULVSH/ -o comparison/
