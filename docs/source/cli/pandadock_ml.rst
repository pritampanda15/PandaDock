pandadock-ml - ML-Enhanced Docking Command
===========================================

The ``pandadock-ml`` command performs machine learning-enhanced molecular docking with deep learning scoring and pose prediction. It leverages graph neural networks and 3D convolutional networks for state-of-the-art accuracy.

Synopsis
--------

.. code-block:: bash

   pandadock-ml [OPTIONS]

Description
-----------

Performs molecular docking with ML-enhanced scoring:

* **Deep learning scoring function** - Graph Neural Network (GNN) or 3D CNN
* **Pose ranking refinement** - ML-based re-ranking of docked poses
* **Transfer learning** - Pre-trained on PDBBind dataset
* **Uncertainty quantification** - Confidence estimates for predictions
* **Ensemble models** - Multiple models for robust predictions

Best accuracy: R = 0.91 correlation with experimental binding affinities.

Required Options
----------------

``-r, --receptor PATH``
    Receptor PDB file (protein structure)

``-l, --ligand PATH``
    Ligand file (SDF, MOL2, or PDB format)

``--center X Y Z``
    Grid box center coordinates (X Y Z in Angstroms)

``--box X Y Z``
    Grid box dimensions (X Y Z in Angstroms)

ML Model Options
----------------

``--model-type TYPE``
    ML model architecture. Default: ``gnn``

    Options:

    * ``gnn`` - Graph Neural Network (recommended, fastest)
    * ``cnn3d`` - 3D Convolutional Network (higher accuracy, slower)
    * ``hybrid`` - Combined GNN + CNN (best accuracy)
    * ``transformer`` - Transformer-based model (experimental)

``--ml-scoring-mode MODE``
    How to use ML scoring. Default: ``combined``

    Options:

    * ``combined`` - Combine physics-based + ML scoring
    * ``ml_only`` - Use only ML scoring
    * ``refinement`` - Use ML for pose re-ranking only

``--use-ensemble / --no-ensemble``
    Use ensemble of ML models for robust predictions. Default: enabled

    Ensemble averages predictions from 5 models trained on different data splits.

``--model-weights PATH``
    Path to custom model weights (optional)

    Use pre-trained weights or your own fine-tuned model.

ML Feature Options
------------------

``--include-protein-features / --no-protein-features``
    Include protein pocket features. Default: enabled

    Protein features: pocket shape, hydrophobicity, electrostatics

``--include-interaction-features / --no-interaction-features``
    Include protein-ligand interaction features. Default: enabled

    Interaction features: H-bonds, À-stacking, hydrophobic contacts

``--include-pharmacophore / --no-pharmacophore``
    Include pharmacophore features. Default: enabled

``--grid-resolution FLOAT``
    Grid resolution for 3D CNN (Angstroms). Default: 0.5

    Only used with ``--model-type cnn3d``

Docking Algorithm
-----------------

``-a, --algorithm ALGORITHM``
    Docking algorithm for pose generation. Default: ``enhanced_hierarchical_cpu``

    ML scoring can be combined with any docking algorithm.

Scoring Options
---------------

``-s, --scoring FUNCTION``
    Physics-based scoring for initial docking. Default: ``physics_based``

``--ml-weight FLOAT``
    Weight for ML score in combined mode. Default: 0.6

    Final score = (1 - weight) × physics + weight × ML

``--physics-weight FLOAT``
    Weight for physics score in combined mode. Default: 0.4

Uncertainty Quantification
---------------------------

``--estimate-uncertainty / --no-estimate-uncertainty``
    Estimate prediction uncertainty. Default: enabled with ensemble

``--uncertainty-threshold FLOAT``
    Maximum uncertainty for accepting predictions. Default: 1.0

    Predictions with uncertainty > threshold are flagged as low confidence.

``--monte-carlo-dropout / --no-monte-carlo-dropout``
    Use Monte Carlo dropout for uncertainty estimation. Default: disabled

    More accurate but slower uncertainty estimates.

Output Options
--------------

``-o, --output-dir PATH``
    Output directory. Default: ``ml_docking_output``

``-n, --num-poses N``
    Number of poses to generate. Default: 20

``--visualize / --no-visualize``
    Generate visualization plots. Default: enabled

``--save-ml-features``
    Save extracted ML features for analysis

``--save-attention-maps``
    Save attention maps (for GNN/Transformer models)

Performance Options
-------------------

``--cpuworkers N``
    Number of CPU workers. Default: auto-detect

``--gpu``
    Enable GPU acceleration for ML inference

    **Highly recommended** - 10-50x speedup for ML models

``--gpu-batch-size N``
    Batch size for GPU ML inference. Default: 32

``--fast``
    Fast mode with reduced sampling

Examples
--------

Basic ML Docking
^^^^^^^^^^^^^^^^

.. code-block:: bash

   pandadock-ml -r protein.pdb -l ligand.sdf \\
                --center 10 20 30 --box 20 20 20 \\
                -o ml_results/

Uses default GNN model with ensemble scoring.

High-Accuracy ML Docking
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pandadock-ml -r protein.pdb -l ligand.sdf \\
                --center 10 20 30 --box 20 20 20 \\
                --model-type hybrid \\
                --use-ensemble \\
                --algorithm enhanced_hierarchical_cpu \\
                --num-poses 50 \\
                -o high_accuracy_ml/

GPU-Accelerated ML Docking
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pandadock-ml -r target.pdb -l ligands.sdf \\
                --center 10 20 30 --box 20 20 20 \\
                --model-type gnn \\
                --gpu \\
                --gpu-batch-size 64 \\
                -o gpu_ml_docking/

3D CNN Model
^^^^^^^^^^^^

.. code-block:: bash

   pandadock-ml -r protein.pdb -l ligand.sdf \\
                --center 10 20 30 --box 20 20 20 \\
                --model-type cnn3d \\
                --grid-resolution 0.5 \\
                --gpu \\
                -o cnn3d_results/

ML-Only Scoring
^^^^^^^^^^^^^^^

.. code-block:: bash

   pandadock-ml -r protein.pdb -l ligand.sdf \\
                --center 10 20 30 --box 20 20 20 \\
                --ml-scoring-mode ml_only \\
                --model-type gnn \\
                --use-ensemble \\
                -o ml_only/

ML Pose Refinement
^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # First: Standard docking
   pandadock dock -r protein.pdb -l ligands.sdf \\
                  --num-poses 100 \\
                  -o initial_docking/

   # Second: ML re-ranking
   pandadock-ml -r protein.pdb -l ligands.sdf \\
                --ml-scoring-mode refinement \\
                --model-type hybrid \\
                --use-ensemble \\
                -o ml_refined/

With Uncertainty Filtering
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pandadock-ml -r protein.pdb -l library.sdf \\
                --center 10 20 30 --box 20 20 20 \\
                --use-ensemble \\
                --estimate-uncertainty \\
                --uncertainty-threshold 0.8 \\
                -o filtered_predictions/

Only accepts predictions with uncertainty < 0.8

Custom Model Weights
^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pandadock-ml -r kinase.pdb -l inhibitors.sdf \\
                --center 10 20 30 --box 20 20 20 \\
                --model-type gnn \\
                --model-weights kinase_finetuned.pt \\
                -o custom_model/

Target-Specific Fine-Tuned Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Kinase-specific model
   pandadock-ml -r kinase.pdb -l ligands.sdf \\
                --model-type gnn \\
                --model-weights models/kinase_specialist.pt \\
                --center 10 20 30 --box 20 20 20

   # GPCR-specific model
   pandadock-ml -r gpcr.pdb -l ligands.sdf \\
                --model-type gnn \\
                --model-weights models/gpcr_specialist.pt \\
                --center 10 20 30 --box 20 20 20

Output Files
------------

**Structures:**

* ``complex1.pdb, complex2.pdb, ...`` - Protein-ligand complexes
* ``pose1.pdb, pose2.pdb, ...`` - Ligand poses only

**Analysis:**

* ``ml_docking_results.json`` - Complete results with ML scores
* ``ml_predictions.csv`` - ML scores, uncertainties, features
* ``uncertainty_analysis.json`` - Uncertainty quantification results
* ``feature_importance.json`` - ML feature importance
* ``summary.txt`` - Human-readable summary

**ML-Specific:**

* ``attention_maps/`` - Attention visualizations (if requested)
* ``ml_features/`` - Extracted features (if requested)

**Visualizations:**

* ``ml_scores.png`` - ML score distribution
* ``uncertainty_plot.png`` - Uncertainty vs score
* ``feature_importance.png`` - Important features visualization

ML Predictions Output
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: json

   {
     "pose_1": {
       "ml_score": -9.8,
       "physics_score": -8.5,
       "combined_score": -9.2,
       "uncertainty": 0.45,
       "confidence": "high",
       "predicted_pKd": 8.5,
       "predicted_Ki_nM": 3.2,
       "feature_importance": {
         "hydrophobic_contacts": 0.35,
         "hydrogen_bonds": 0.28,
         "shape_complementarity": 0.22,
         "electrostatics": 0.15
       }
     }
   }

Performance Characteristics
----------------------------

**Accuracy:**

* R = 0.91 correlation with experimental data (hybrid model, ensemble)
* R = 0.88 (GNN model)
* R = 0.89 (3D CNN model)

**Speed:**

+------------------+-----------+----------+
| Model Type       | CPU Time  | GPU Time |
+==================+===========+==========+
| GNN              | 0.1-0.2 s | 0.01 s   |
+------------------+-----------+----------+
| 3D CNN           | 0.5-1.0 s | 0.05 s   |
+------------------+-----------+----------+
| Hybrid           | 0.3-0.5 s | 0.02 s   |
+------------------+-----------+----------+
| Ensemble (x5)    | 0.5-2.0 s | 0.05-0.1s|
+------------------+-----------+----------+

**Throughput:**

* CPU: 30-120 ligands/hour
* GPU: 300-600 ligands/hour (10-20x speedup)

ML Models Details
-----------------

Graph Neural Network (GNN)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Architecture:**

* Node features: Atomic properties (element, hybridization, charge)
* Edge features: Bond type, distance, angle
* Graph convolutions: 6 layers
* Attention mechanism: Multi-head attention
* Output: Binding affinity prediction

**Advantages:**

* Fastest ML model
* Rotationally/translationally invariant
* Captures long-range interactions
* Good generalization

3D Convolutional Network (CNN)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Architecture:**

* Input: 3D voxel grid (protein + ligand channels)
* Convolution layers: 8 layers with batch normalization
* Pooling: Max pooling between layers
* Fully connected: 3 dense layers
* Output: Binding affinity

**Advantages:**

* Captures 3D spatial patterns
* Good for shape complementarity
* Handles electrostatics well

Hybrid Model
^^^^^^^^^^^^

Combines GNN + 3D CNN:

* GNN branch: Graph-based features
* CNN branch: Spatial features
* Late fusion: Concatenate features before final layers
* Best accuracy but slower

Uncertainty Quantification
---------------------------

**Methods:**

1. **Ensemble Disagreement**

   Uncertainty = standard deviation across ensemble predictions

2. **Monte Carlo Dropout**

   Multiple forward passes with dropout enabled

3. **Evidential Deep Learning**

   Direct uncertainty estimation (experimental)

**Interpretation:**

* Low uncertainty (<0.5): High confidence
* Medium uncertainty (0.5-1.0): Moderate confidence
* High uncertainty (>1.0): Low confidence, novel chemical space

**Use uncertainty to:**

* Filter unreliable predictions
* Identify compounds requiring experimental validation
* Detect out-of-distribution samples

Best Practices
--------------

When to Use ML Docking
^^^^^^^^^^^^^^^^^^^^^^^

 **Maximum accuracy required** - Lead optimization, critical predictions

 **Novel scaffolds** - ML can capture patterns physics-based scoring misses

 **Large datasets available** - Can fine-tune models

 **GPU available** - Makes ML inference fast

When Not to Use
^^^^^^^^^^^^^^^

 **Ultra-large screening** (>100k compounds) - Too slow even with GPU

 **Very novel chemical space** - May not generalize well

 **No GPU available** and speed critical - Use faster scoring

Optimization Tips
^^^^^^^^^^^^^^^^^

**For maximum accuracy:**

.. code-block:: bash

   --model-type hybrid \\
   --use-ensemble \\
   --estimate-uncertainty

**For maximum speed:**

.. code-block:: bash

   --model-type gnn \\
   --no-ensemble \\
   --gpu \\
   --gpu-batch-size 128

**Balanced:**

.. code-block:: bash

   --model-type gnn \\
   --use-ensemble \\
   --gpu

Troubleshooting
---------------

Slow ML Inference
^^^^^^^^^^^^^^^^^

**Problem:** ML scoring very slow

**Solutions:**

1. Use GPU: ``--gpu``
2. Increase batch size: ``--gpu-batch-size 64``
3. Use simpler model: ``--model-type gnn``
4. Disable ensemble: ``--no-ensemble``

High Uncertainty
^^^^^^^^^^^^^^^^

**Problem:** Many predictions have high uncertainty

**Possible causes:**

* Novel chemical scaffolds not in training data
* Unusual binding modes
* Protein family not well-represented in training

**Solutions:**

* Use physics-based or hybrid scoring as fallback
* Fine-tune model on your target family
* Flag high-uncertainty predictions for manual review

Model Loading Errors
^^^^^^^^^^^^^^^^^^^^^

**Problem:** Cannot load ML model weights

**Solutions:**

1. Verify model file exists
2. Check PyTorch/TensorFlow version compatibility
3. Re-download default models
4. Check file permissions

Out of GPU Memory
^^^^^^^^^^^^^^^^^

**Problem:** GPU out of memory during ML inference

**Solutions:**

1. Reduce batch size: ``--gpu-batch-size 16``
2. Use smaller model: ``--model-type gnn`` (not hybrid)
3. Disable ensemble: ``--no-ensemble``
4. Use CPU inference (remove ``--gpu``)

Fine-Tuning ML Models
----------------------

You can fine-tune models on your own data:

.. code-block:: bash

   # Train on custom dataset
   pandadock-ml-train \\
       --training-data my_protein_ligand_complexes.csv \\
       --model-type gnn \\
       --output-weights custom_model.pt

   # Use fine-tuned model
   pandadock-ml -r protein.pdb -l ligands.sdf \\
                --model-weights custom_model.pt

Exit Status
-----------

Returns 0 on success, non-zero on error.

See Also
--------

* :doc:`pandadock` - Standard docking
* :doc:`pandadock_flex` - Flexible docking
* :doc:`../scoring/hybrid` - Hybrid ML scoring
* :doc:`../algorithms/specialized_modes` - Specialized docking modes
