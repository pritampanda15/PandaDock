Dataset Preparation Guide
=========================

This guide covers how to prepare datasets for training PandaDock-GNN models.
PandaDock supports three dataset formats: ULVSH, PDBbind, and BindingDB.

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
--------

PandaDock-GNN can be trained on:

+------------+------------------+------------------+-------------------+
| Dataset    | Complexes        | Affinity Type    | Best Test R       |
+============+==================+==================+===================+
| PDBbind    | 5,316            | pKd/pKi          | 0.88              |
+------------+------------------+------------------+-------------------+
| ULVSH      | 942              | pEC50            | 0.82              |
+------------+------------------+------------------+-------------------+
| BindingDB  | 8,891+           | pK (various)     | 0.81              |
+------------+------------------+------------------+-------------------+

You can train on any single dataset or combine them for better generalization.

---

ULVSH Dataset
-------------

The ULVSH (Ultra-Large Virtual Screening Hits) dataset contains 942 compounds
across 10 protein targets with experimental EC50 values.

Directory Structure
~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   ULVSH/
   ├── ADRA2B/                    # Target name (one of 10 targets)
   │   ├── raw/
   │   │   └── vitro.tsv          # Experimental binding data
   │   └── minimized/
   │       ├── ZINC000001234567/  # Compound directory (ZINC ID)
   │       │   ├── protein.mol2   # Full protein structure
   │       │   ├── ligand.mol2    # Ligand structure (docked pose)
   │       │   └── site.mol2      # Binding site atoms only
   │       └── ZINC000007654321/
   │           └── ...
   ├── CNR1/
   │   └── ...
   └── DRD4/
       └── ...

Required Files
~~~~~~~~~~~~~~

**vitro.tsv** - Tab or whitespace-separated file with experimental data:

.. code-block:: text

   ID                  EC50[uM]    Active
   ZINC000001234567    0.5         Yes
   ZINC000007654321    15.2        No
   ZINC000009876543    n.d.        No

- ``ID``: Compound identifier (must match directory name in minimized/)
- ``EC50[uM]``: EC50 value in micromolar (use "n.d." for not determined)
- ``Active``: "Yes" or "No" for activity classification

**MOL2 Files** - Standard Tripos MOL2 format with:

- ``protein.mol2``: Full protein structure with all atoms
- ``ligand.mol2``: Ligand in docked/bound pose
- ``site.mol2``: Binding site atoms (typically within 5-10Å of ligand)

Preparing ULVSH Data
~~~~~~~~~~~~~~~~~~~~

1. **Obtain structures**: Get protein-ligand complexes from docking or crystal structures

2. **Extract binding site**:

   .. code-block:: python

      from rdkit import Chem
      import numpy as np

      def extract_binding_site(protein_file, ligand_file, radius=10.0):
          """Extract protein atoms within radius of ligand centroid."""
          # Load structures
          protein = Chem.MolFromMol2File(protein_file)
          ligand = Chem.MolFromMol2File(ligand_file)

          # Get ligand centroid
          ligand_conf = ligand.GetConformer()
          ligand_coords = np.array([ligand_conf.GetAtomPosition(i)
                                    for i in range(ligand.GetNumAtoms())])
          centroid = ligand_coords.mean(axis=0)

          # Filter protein atoms by distance
          # ... (save atoms within radius to site.mol2)

3. **Create vitro.tsv**: Compile experimental binding data from literature or assays

4. **Organize directories**: Follow the structure shown above

Training on ULVSH
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Basic training
   pandadock gnn train -d ULVSH/ -o models/ulvsh_model/

   # With custom parameters
   pandadock gnn train -d ULVSH/ -o models/ \
       --epochs 100 \
       --batch-size 32 \
       --hidden-dim 256 \
       --num-layers 6

---

PDBbind Dataset
---------------

PDBbind is a curated database of protein-ligand complexes with experimentally
measured binding affinities. The v2020 refined set contains 5,316 complexes.

Obtaining PDBbind
~~~~~~~~~~~~~~~~~

1. Register at http://www.pdbbind.org.cn/
2. Download "PDBbind v2020 refined set"
3. Extract to create the directory structure below

Directory Structure
~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   PDBbind/
   ├── PDBbind_v2020/              # or just place files directly in PDBbind/
   │   ├── index/
   │   │   └── INDEX_refined_data.2020   # Binding affinity index
   │   ├── 1a1e/                   # PDB ID directory
   │   │   ├── 1a1e_protein.pdb    # Protein structure
   │   │   ├── 1a1e_pocket.pdb     # Binding pocket (pre-extracted)
   │   │   ├── 1a1e_ligand.mol2    # Ligand structure
   │   │   └── 1a1e_ligand.sdf     # Ligand in SDF format
   │   ├── 1a28/
   │   │   └── ...
   │   └── ...

Required Files
~~~~~~~~~~~~~~

**INDEX_refined_data.2020** - Space-separated index file:

.. code-block:: text

   # PDB   resolution  year  -logKd/Ki  reference
   1a1e   2.00        1998  6.52       // reference info
   1a28   1.90        1997  4.62       // reference info
   ...

- Column 1: PDB ID
- Column 2: Resolution (Å)
- Column 3: Year of structure
- Column 4: Binding affinity as pKd or pKi (-log10 of Kd/Ki in M)

**Structure Files** (per complex):

- ``{pdb_id}_protein.pdb``: Full protein structure
- ``{pdb_id}_pocket.pdb``: Pre-extracted binding pocket (used for training)
- ``{pdb_id}_ligand.mol2``: Ligand structure

Training on PDBbind
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # PDBbind only
   pandadock gnn train -p PDBbind/ -o models/pdbbind_model/

   # Combined with ULVSH (recommended for generalization)
   pandadock gnn train -d ULVSH/ -p PDBbind/ -o models/combined/ --balanced

---

BindingDB Dataset
-----------------

BindingDB is a public database of measured binding affinities. We provide
tools to prepare BindingDB data for PandaDock training.

Obtaining BindingDB Data
~~~~~~~~~~~~~~~~~~~~~~~~

1. Download from https://www.bindingdb.org/bind/index.jsp
2. Select "Download" → "BindingDB_All_2D_NoSmi.tsv.zip" or use the API
3. Filter for entries with 3D structures

TSV File Format
~~~~~~~~~~~~~~~

Create a TSV file with paths to protein and ligand structures:

.. code-block:: text

   complex_id    protein_file              ligand_file           pK       pdb_id
   complex_001   proteins/1abc_protein.mol2   ligands/lig001.mol2   7.52     1abc
   complex_002   proteins/2def_protein.mol2   ligands/lig002.mol2   6.31     2def
   ...

Required columns:

- ``complex_id``: Unique identifier for the complex
- ``protein_file``: Path to protein MOL2 file
- ``ligand_file``: Path to ligand MOL2 file
- ``pK``: Binding affinity as pKd, pKi, or pIC50 (-log10 scale)
- ``pdb_id`` (optional): PDB ID if available

Directory Structure
~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   BindingDB_data/
   ├── bindingdb_affinity.tsv     # Main index file
   ├── proteins/                  # Protein structures
   │   ├── 1abc_protein.mol2
   │   ├── 2def_protein.mol2
   │   └── ...
   └── ligands/                   # Ligand structures
       ├── lig001.mol2
       ├── lig002.mol2
       └── ...

Preparing BindingDB Data
~~~~~~~~~~~~~~~~~~~~~~~~

**Step 1: Download and filter BindingDB**

.. code-block:: python

   import pandas as pd

   # Load BindingDB TSV
   df = pd.read_csv('BindingDB_All.tsv', sep='\t', low_memory=False)

   # Filter for entries with:
   # - Ki, Kd, or IC50 values
   # - Associated PDB structures
   # - Standard conditions

   filtered = df[
       (df['Ki (nM)'].notna() | df['Kd (nM)'].notna() | df['IC50 (nM)'].notna()) &
       (df['PDB ID(s) of Target Chain'].notna())
   ]

   print(f"Filtered to {len(filtered)} entries with binding data and structures")

**Step 2: Obtain 3D structures**

For each entry, you need protein and ligand 3D structures:

.. code-block:: python

   from rdkit import Chem
   from rdkit.Chem import AllChem

   def prepare_ligand(smiles, output_file):
       """Generate 3D ligand structure from SMILES."""
       mol = Chem.MolFromSmiles(smiles)
       mol = Chem.AddHs(mol)
       AllChem.EmbedMolecule(mol, randomSeed=42)
       AllChem.MMFFOptimizeMolecule(mol)
       Chem.MolToMolFile(mol, output_file)

   def download_protein(pdb_id, output_file):
       """Download protein structure from PDB."""
       import urllib.request
       url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
       urllib.request.urlretrieve(url, output_file)

**Step 3: Convert to MOL2 format**

PandaDock requires MOL2 format for atom typing:

.. code-block:: bash

   # Using Open Babel
   obabel protein.pdb -O protein.mol2
   obabel ligand.sdf -O ligand.mol2

**Step 4: Create the TSV index**

.. code-block:: python

   import pandas as pd
   import numpy as np

   def convert_to_pk(value_nm):
       """Convert nM to pK (-log10 M)."""
       if pd.isna(value_nm) or value_nm <= 0:
           return None
       return -np.log10(value_nm * 1e-9)

   # Create index file
   records = []
   for idx, row in filtered.iterrows():
       # Get pK value (prefer Ki > Kd > IC50)
       pk = None
       if pd.notna(row.get('Ki (nM)')):
           pk = convert_to_pk(row['Ki (nM)'])
       elif pd.notna(row.get('Kd (nM)')):
           pk = convert_to_pk(row['Kd (nM)'])
       elif pd.notna(row.get('IC50 (nM)')):
           pk = convert_to_pk(row['IC50 (nM)'])

       if pk and 2.0 <= pk <= 12.0:  # Filter reasonable range
           records.append({
               'complex_id': f'complex_{idx}',
               'protein_file': f'proteins/{row["PDB ID(s) of Target Chain"]}_protein.mol2',
               'ligand_file': f'ligands/lig_{idx}.mol2',
               'pK': pk,
               'pdb_id': row['PDB ID(s) of Target Chain']
           })

   df_index = pd.DataFrame(records)
   df_index.to_csv('bindingdb_affinity.tsv', sep='\t', index=False)

Training on BindingDB
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # BindingDB only
   pandadock gnn train -b bindingdb_affinity.tsv -o models/bindingdb_model/

   # Combined with ULVSH (recommended)
   pandadock gnn train -b bindingdb_affinity.tsv -d ULVSH/ -o models/combined/ \
       --balanced --epochs 100

---

Combined Training
-----------------

Training on multiple datasets improves generalization. Use the ``--balanced``
flag to prevent larger datasets from dominating training.

Recommended Combinations
~~~~~~~~~~~~~~~~~~~~~~~~

**Best accuracy on diverse targets:**

.. code-block:: bash

   pandadock gnn train -d ULVSH/ -p PDBbind/ -o models/ \
       --balanced --epochs 200 --batch-size 32

**Best for BindingDB-like screening:**

.. code-block:: bash

   pandadock gnn train -b bindingdb.tsv -d ULVSH/ -o models/ \
       --balanced --epochs 100

**All datasets combined:**

.. code-block:: bash

   pandadock gnn train -d ULVSH/ -p PDBbind/ -b bindingdb.tsv -o models/ \
       --balanced --epochs 200

.. warning::

   Combining PDBbind with other datasets may reduce performance due to
   affinity scale differences (pKd vs pEC50). Test on your specific use case.

---

Training Options Reference
--------------------------

+------------------+----------+--------------------------------------------------+
| Option           | Default  | Description                                      |
+==================+==========+==================================================+
| ``-d/--dataset`` | None     | Path to ULVSH dataset directory                  |
+------------------+----------+--------------------------------------------------+
| ``-p/--pdbbind`` | None     | Path to PDBbind dataset directory                |
+------------------+----------+--------------------------------------------------+
| ``-b/--bindingdb``| None    | Path to BindingDB TSV file                       |
+------------------+----------+--------------------------------------------------+
| ``-o/--output``  | Required | Output directory for model checkpoints           |
+------------------+----------+--------------------------------------------------+
| ``--epochs``     | 100      | Number of training epochs                        |
+------------------+----------+--------------------------------------------------+
| ``--batch-size`` | 32       | Batch size (reduce if out of memory)             |
+------------------+----------+--------------------------------------------------+
| ``--lr``         | 1e-4     | Learning rate                                    |
+------------------+----------+--------------------------------------------------+
| ``--hidden-dim`` | 256      | Hidden layer dimension                           |
+------------------+----------+--------------------------------------------------+
| ``--num-layers`` | 6        | Number of EGNN message passing layers            |
+------------------+----------+--------------------------------------------------+
| ``--dropout``    | 0.1      | Dropout rate for regularization                  |
+------------------+----------+--------------------------------------------------+
| ``--patience``   | 20       | Early stopping patience (epochs without improve) |
+------------------+----------+--------------------------------------------------+
| ``--balanced``   | False    | Balance sampling across datasets                 |
+------------------+----------+--------------------------------------------------+
| ``--gpu/--cpu``  | --gpu    | Use GPU if available                             |
+------------------+----------+--------------------------------------------------+
| ``--seed``       | 42       | Random seed for reproducibility                  |
+------------------+----------+--------------------------------------------------+

---

Troubleshooting
---------------

**"No valid complexes found"**

- Check that MOL2 files exist at the paths specified
- Verify MOL2 files are valid (not empty, proper format)
- Check that protein and ligand files are paired correctly

**"Edge dimension mismatch"**

- Ensure you're using the latest version of PandaDock
- Edge features should be 23 dimensions (check featurizer)

**Out of memory errors**

- Reduce ``--batch-size`` (try 16 or 8)
- Use ``--cpu`` if GPU memory is limited
- Filter out very large proteins (>1000 atoms)

**Poor correlation (R < 0.5)**

- Check data quality (correct binding affinity values?)
- Ensure structures are properly prepared (hydrogens added?)
- Try training longer (``--epochs 200``)
- Use ``--balanced`` when combining datasets

---

Best Practices
--------------

1. **Data Quality**: Ensure binding affinities are accurate and structures are properly minimized

2. **Structure Preparation**:

   - Add hydrogens to all structures
   - Minimize/optimize ligand geometry
   - Use consistent protonation states

3. **Binding Site Extraction**:

   - Use 10Å radius around ligand centroid
   - Include all residues with atoms in the cutoff

4. **Training**:

   - Start with default hyperparameters
   - Monitor validation R during training
   - Use early stopping to prevent overfitting

5. **Evaluation**:

   - Always evaluate on held-out test set
   - Report Pearson R, Spearman ρ, RMSE, and MAE
   - Compare against baseline methods

---

Example: End-to-End Training
----------------------------

Here's a complete example of preparing data and training a model:

.. code-block:: bash

   # 1. Prepare your data (assuming you have structures ready)
   mkdir -p my_dataset/proteins my_dataset/ligands

   # 2. Create index file
   cat > my_dataset/affinity.tsv << EOF
   complex_id	protein_file	ligand_file	pK
   complex_1	proteins/prot1.mol2	ligands/lig1.mol2	7.5
   complex_2	proteins/prot2.mol2	ligands/lig2.mol2	6.2
   complex_3	proteins/prot3.mol2	ligands/lig3.mol2	8.1
   EOF

   # 3. Train the model
   pandadock gnn train -b my_dataset/affinity.tsv -o my_model/ \
       --epochs 100 --batch-size 16

   # 4. Evaluate on test set
   pandadock gnn benchmark -m my_model/best_model.pt -b my_dataset/affinity.tsv

   # 5. Use for prediction
   pandadock gnn predict -m my_model/best_model.pt \
       -p new_protein.mol2 -l new_ligand.mol2
