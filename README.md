# PandaDock - Molecular Docking with GNN Scoring

---

<p align="center">
  <a href="https://github.com/pritampanda15/PandaDock">
    <img src="https://github.com/pritampanda15/PandaDock/blob/main/PandaDock.png" width="500" alt="PandaDock Logo"/>
  </a>
</p>
<p align="center">
  <a href="https://pypi.org/project/pandadock/">
    <img src="https://img.shields.io/pypi/v/pandadock.svg" alt="PyPI Version">
  </a>
  <a href="https://github.com/pritampanda15/PandaDock/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/pritampanda15/PandaDock" alt="License">
  </a>
  <a href="https://github.com/pritampanda15/PandaDock/stargazers">
    <img src="https://img.shields.io/github/stars/pritampanda15/PandaDock?style=social" alt="GitHub Stars">
  </a>
  <a href="https://github.com/pritampanda15/PandaDock/issues">
    <img src="https://img.shields.io/github/issues/pritampanda15/PandaDock" alt="GitHub Issues">
  </a>
  <a href="https://github.com/pritampanda15/PandaDock/network/members">
    <img src="https://img.shields.io/github/forks/pritampanda15/PandaDock?style=social" alt="GitHub Forks">
  </a>
  <a href="https://pepy.tech/project/pandadock">
    <img src="https://static.pepy.tech/badge/pandadock" alt="Downloads">
  </a>
</p>
<p align="center">
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python 3.8+">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
  </a>
  <a href="https://pandadock.readthedocs.io/">
    <img src="https://readthedocs.org/projects/pandadock/badge/?version=latest" alt="Documentation Status">
  </a>
</p>

---

**SE(3)-Equivariant GNN Scoring for Molecular Docking**

[Installation](#installation) | [Quick Start](#quick-start) | [Documentation](https://pandadock.readthedocs.io/) | [Benchmark](#benchmark-performance) | [Citation](#citation)

</div>

---

## Overview

**PandaDock v4.1** pairs a flexible-ligand conformational search with a novel
SE(3)-equivariant Graph Neural Network (GNN) scoring function that achieves
state-of-the-art correlation with experimental binding affinities (R=0.88 on
PDBbind, R=0.82 on ULVSH, R=0.81 on BindingDB).

**v4.1 replaces the pose search entirely.** Earlier releases did not search
conformational space: they perturbed the input conformer slightly around the box
centre, never varied ligand torsions, and ranked poses by proximity to a
reference point rather than by interaction energy. See
[Pose Prediction](#pose-prediction) for what changed and what it means for
results produced with v4.0.x.

### Key Features

- **PandaCore search**: flexible-ligand Monte Carlo search with quasi-Newton local
  optimization over translation, orientation and ligand torsions, driven by
  precomputed affinity grids with fully analytic gradients
- **PandaDock-GNN**: SE(3)-equivariant scoring achieving **Pearson R = 0.88** on PDBbind
- **Hybrid Docking**: Combined pose generation + GNN rescoring (recommended workflow)
- **Universal Rescorer**: Rescore poses from ANY docking tool (Vina, Glide, GOLD, etc.)
- **Vina-Style Scoring**: AutoDock Vina empirical weights as default scoring
- **Multi-Task Learning**: Joint pKd/pEC50 regression + activity classification
- **Heterogeneous Graphs**: Separate protein/ligand node types with interaction edges
- **Specialized Modes**: Flexible, metal coordination, and tethered docking

---

## Pose Prediction

The conformational search samples position uniformly over the docking box,
orientation uniformly over SO(3), and every rotatable bond as an explicit degree of
freedom. Each Monte Carlo step is followed by an L-BFGS relaxation using analytic
gradients, and accepted or rejected by the Metropolis criterion. Distinct binding
modes are returned after RMSD clustering.

```bash
# Exhaustiveness defaults to a value scaled by ligand flexibility (8-32).
pandadock dock -r receptor.pdb -l ligand.sdf --center 10 12 8 --box 22 22 22

# Reproducible run with an explicit budget
pandadock dock -r receptor.pdb -l ligand.sdf -g grid.json -e 24 --seed 42
```

Sampling has to scale with the number of rotatable bonds. On a 13-DOF ligand,
8 runs converged to a minimum 11 kcal/mol above the global one, while 24 runs found
the global minimum; the default therefore grows with torsion count. Raise `-e`
further for large, flexible ligands.

### Measuring pose accuracy

`benchmarking/redock_benchmark.py` redocks a set of complexes and reports
symmetry-corrected heavy-atom RMSD, per protein family. These scripts are
development tooling and are not shipped in the PyPI package — clone the
repository to use them:

```bash
python benchmarking/redock_benchmark.py --input /data/pdbbind_core --output results/
python benchmarking/redock_benchmark.py --manifest complexes.csv -j 8 --save-poses
```

RMSD is symmetry-corrected and computed without superposition. The harness reports
top-1 and best-of-N success rates separately, because quoting best-of-N as though
it were top-1 substantially overstates accuracy.

> **Note on v4.0.2 and earlier.** The docking algorithms in previous releases did
> not perform a conformational search. They placed the input conformer near the box
> centre with a rotation drawn from a narrow Gaussian, never varied ligand torsions,
> called the scoring function without the ligand (so Vina-style scores were
> uniformly 0.0), and ranked poses by a bonus for proximity to a reference point --
> including a hardcoded ligand coordinate that captured any box placed near it.
> Pose-accuracy figures from those releases measure that bias and should not be
> compared against this one. Affinity results from PandaDock-GNN are unaffected:
> the GNN was trained and evaluated on crystal poses, independently of this code
> path.

---

## Changelog

### 4.1.0

**Pose search rewritten.** `pandadock dock` now performs a real conformational
search. Poses from this release are not comparable to poses from 4.0.x.

Added:
- `pandadock.docking.search` — torsion tree, precomputed Vina affinity grids,
  analytic gradients (verified against finite differences at 2e-10 relative
  error), and Monte Carlo search with L-BFGS local optimization
- `PandaCoreDocker`, registered as the `pandadock` algorithm
- `--exhaustiveness`, `--seed`, `--rigid-ligand` and `--grid-spacing` CLI options;
  exhaustiveness defaults to a value that scales with ligand flexibility
- `pandadock.analysis.rmsd` — symmetry-corrected RMSD without superposition
- `pandadock.preprocessing.complex_splitter` — receptor/ligand splitting with
  ligand identity resolved against the PDB Chemical Component Dictionary
- `benchmarking/` — redocking benchmark, preparation, validation and reporting
  scripts. These were previously absent from the repository because `.gitignore`
  excluded the whole directory.
- `pandadock.ml.vcell` — inspection for SAIR-trained cell-context checkpoints

Fixed:
- The Vina-style scoring function was called without the ligand throughout the
  docking pipeline, so it returned 0.0 for every pose
- A hardcoded ligand coordinate captured any docking box placed within 10 Å of it
- `_rigid_minimization` read coordinates from the input conformer instead of the
  pose it was given, discarding the pose being minimized
- Sub-package versions drifted from the distribution version (`pandadock.docking`
  reported 3.0.0 while the package shipped 4.0.2); the version is now single-sourced

Removed:
- `CrystalGuidedDocker` now raises on construction. It restricted sampling to the
  neighbourhood of a reference pose and rewarded proximity to it, which
  invalidates any benchmark it appears in.
- `MonteCarloDocker`, `GeneticAlgorithmDocker` and `EnhancedHierarchicalDocker`
  are deprecated aliases of `PandaCoreDocker`. Their distinctive code was either
  unreachable or non-functional; each module's docstring records the details.

Reporting:
- Every run now writes `report.html` bundling the plots and run parameters
- New plots: pose scores with the rank-1/rank-2 gap, energy terms split into
  favourable and penalty contributions, a pose-pose RMSD matrix showing whether
  the returned set is one binding mode or several, and a residue interaction
  fingerprint across poses
- PandaMap 2D interaction diagrams are generated automatically when `pandamap` is
  installed (`pip install pandamap`)
- Interactions are analysed for every returned pose rather than only the top one

Output correctness:
- Ligands containing two-letter elements (Cl, Br, I) produced **no** pose or
  complex PDB files at all, while the CLI reported success. BioPython requires
  upper-case element symbols and RDKit supplies capitalised ones.
- `poses.sdf` is now written alongside the PDBs, preserving bond orders and
  formal charges that PDB cannot represent
- Interaction analysis performed no chemistry: it thresholded each ligand atom's
  distance to its nearest receptor atom three times, counting a carbon near a
  carbon as a hydrogen bond, leaving electrostatics permanently zero, and summing
  nested subsets into a double-counted total. It now uses the chemistry-aware
  detectors that were already present in the module but never called.
- Receptor hydrogen-bond typing omitted the protein backbone, so backbone-mediated
  binding — the dominant mode for kinase hinge binders — reported zero hydrogen
  bonds
- Removed the fabricated `binding_affinity_estimate`, which applied invented
  weights to those miscounts and clamped the result to [-15, 5]
- The Boltzmann ensemble applied an unfitted "default calibration" that moved the
  reported dG outside the range of the pose scores it summarises (poses spanning
  -16.3 to -14.5 kcal/mol reported -8.3). The entropy penalty also used a
  hardcoded rotatable bond count of 5 for every ligand.

CLI and specialized docking:
- `pandadock-ml` raised `ImportError: cannot import name 'GPU_AVAILABLE'` on every
  invocation; the name had never been defined. There is no GPU search path, so it
  is now defined as `False`.
- `pandadock-metal` could not complete a run. Its engine called a method
  `DockingEngine` does not expose, registered no algorithms, could not construct
  its own result class, built `Pose` objects without required fields, passed an
  empty array where a receptor structure was needed, and reported violation rates
  above 100% because pose metadata was not truncated alongside the poses.
- Metal parameters fall back to documented built-in values when no AutoDock-format
  parameter file is present, instead of raising at construction. The file has
  never shipped with the package.
- `tests/test_cli.py` covers all nine entry points

Packaging:
- Metadata moved to `pyproject.toml` (PEP 621); `setup.py` is now a shim

---

## Benchmark Performance

### PDBbind v2020 Refined Set (5,316 complexes)

| Metric | Value |
|--------|-------|
| **Pearson R** | **0.88** |
| **Spearman R** | **0.88** |
| **RMSE** | 0.93 pK units |
| **MAE** | 0.68 pK units |
| Within 1.0 pK | 77.5% |
| Within 1.5 pK | 90.5% |

### ULVSH Dataset (942 compounds, 10 protein targets)

| Method | Type | Pearson R | N |
|--------|------|-----------|---|
| **PandaDock-GNN (test)** | **ML Scoring** | **0.82** | 95 |
| **PandaDock-GNN (full)** | **ML Scoring** | **0.67** | 942 |
| VM2 | ULVSH Baseline | 0.15 | 942 |
| PM6 | ULVSH Baseline | 0.08 | 939 |
| Hyde | ULVSH Baseline | 0.02 | 942 |
| Gnina | ULVSH Baseline | 0.01 | 941 |

### BindingDB Dataset (8,891 protein-ligand complexes)

| Training Configuration | Test Pearson R | Test RMSE | N (train) |
|------------------------|----------------|-----------|-----------|
| **BindingDB Only** | **0.81** | - | 7,113 |
| **BindingDB + ULVSH** | **0.79** | 0.96 | 7,866 |
| BindingDB + ULVSH + PDBbind | 0.49 | 1.37 | 12,118 |

**Note:** Combined training with PDBbind shows reduced performance due to affinity scale differences (pKd vs pEC50). For best results, train on datasets with compatible affinity measurements.

**Key Results:**
- PandaDock-GNN achieves **R = 0.88** on PDBbind (5,316 complexes)
- **R = 0.81** on BindingDB test set (889 complexes)
- **5.5x improvement** over the best baseline (VM2) on ULVSH
- Activity classification **AUC = 0.94** on ULVSH test set

---

## Installation

### Prerequisites

- Python 3.8 or higher
- Conda package manager (recommended for RDKit)

### Basic Installation

```bash
# Clone repository
git clone https://github.com/pritampanda15/PandaDock.git
cd PandaDock

# Create conda environment with RDKit
conda create -n pandadock python=3.10
conda activate pandadock
conda install -c conda-forge rdkit

# Install PandaDock
pip install -e .
```

### GNN Installation (Recommended)

```bash
# Install PyTorch and PyTorch Geometric for GNN support
pip install -e ".[gnn]"

# Or manually:
pip install torch torch-geometric torch-scatter torch-sparse
```

For detailed installation instructions, see [INSTALL.md](INSTALL.md).

---

## Quick Start

### Download Pre-trained Model (Recommended)

Get started immediately with the pre-trained model:

```bash
# Download the pre-trained model (~82 MB)
pandadock gnn download-model

# Model is saved to models/pandadock_gnn_v4.pt
```

### Hybrid Docking (Recommended)

The hybrid workflow combines traditional pose generation with GNN rescoring for best accuracy:

```bash
# Using pre-trained model
pandadock hybrid -r protein.pdb -l ligand.sdf \
                 --center 10 20 30 --box 20 20 20 \
                 -m models/pandadock_gnn_v4.pt \
                 -o results/

# Or train your own model first
pandadock gnn train -d ULVSH/ -o models/ --epochs 100
pandadock hybrid -r protein.pdb -l ligand.sdf \
                 --center 10 20 30 --box 20 20 20 \
                 -m models/best_model.pt \
                 -o results/
```

### Flexible-Ligand Docking

```bash
# Every rotatable bond is searched; exhaustiveness scales with flexibility
pandadock dock -r protein.pdb -l ligand.sdf \
               --center 10 20 30 --box 20 20 20 \
               -o results/

# Reproducible run with an explicit sampling budget
pandadock dock -r protein.pdb -l ligand.sdf -g grid.json \
               --exhaustiveness 24 --seed 42 -o results/
```

### GNN Prediction Only

```bash
# Predict binding affinity for a pre-docked complex
pandadock gnn predict -m model.pt -p protein.mol2 -l ligand.mol2
```

### Universal Rescorer (NEW)

Rescore poses from ANY docking tool using the GNN:

```bash
# Rescore poses from AutoDock Vina
pandadock gnn rescore -m model.pt -r receptor.pdb -p vina_out.sdf -o ranked.csv

# Rescore poses from pandadock-flex
pandadock gnn rescore -m model.pt -r protein.pdb -p flex_poses.sdf --output-sdf ranked.sdf

# Rescore poses from Glide, GOLD, or any other tool
pandadock gnn rescore -m model.pt -r protein.pdb -p docked_poses.sdf
```

### Compare Against Baselines

```bash
# Benchmark GNN against all baseline methods
pandadock gnn compare -m model.pt -d ULVSH/ -o comparison/
```

---

## Commands

### Core Commands

| Command | Description |
|---------|-------------|
| `pandadock dock` | Flexible-ligand docking with Vina-style scoring |
| `pandadock hybrid` | Hybrid docking with GNN rescoring (recommended) |

#### `pandadock dock` options

| Option | Default | Description |
|---|---|---|
| `-r, --receptor` | *required* | Receptor PDB file |
| `-l, --ligand` | *required* | Ligand file (SDF/MOL2/PDB) |
| `--center X Y Z` | — | Box centre in Å (or use `-g/--grid-config`) |
| `--box X Y Z` | — | Box dimensions in Å |
| `-s, --scoring` | `vina` | `vina` or `physics_based` |
| `-n, --num-poses` | `20` | Binding modes to return, clustered at 2 Å RMSD |
| `-e, --exhaustiveness` | *auto* | Independent search runs; see below |
| `--seed` | *random* | Seed for reproducible runs |
| `--rigid-ligand` | off | Disable torsional search |
| `--grid-spacing` | `0.375` | Affinity grid spacing in Å |
| `--rescoring` | `none` | `none` or `mmgbsa` |
| `-o, --output-dir` | `docking_output` | Output directory |
| `--fast` | off | Smoke-test only; under-samples badly |

**Exhaustiveness scales with ligand flexibility** unless you set it explicitly,
because a budget that is ample for a rigid fragment leaves a highly rotatable
ligand's search space badly under-explored — and the failure is silent, returning
a confident-looking pose from a local minimum well above the global one.

| Rotatable bonds | 0 | 4 | 8 | 12+ |
|---|---|---|---|---|
| Independent runs | 8 | 16 | 24 | 32 |

Two defaults worth knowing: runs are **not reproducible** unless you pass
`--seed`, and `--fast` drops to exhaustiveness 2, so its poses should not be
reported as results.

### GNN Commands

| Command | Description |
|---------|-------------|
| `pandadock gnn download-model` | **Download pre-trained model (~82 MB)** |
| `pandadock gnn train` | Train GNN model on dataset (ULVSH, PDBbind, or combined) |
| `pandadock gnn predict` | Predict binding affinity for a single complex |
| `pandadock gnn rescore` | **Universal rescorer for poses from ANY docking tool** |
| `pandadock gnn benchmark` | Benchmark model performance on test set |
| `pandadock gnn compare` | Compare against baseline scoring methods |

### Specialized Docking

| Command | Description |
|---------|-------------|
| `pandadock-flex` | Flexible/induced-fit docking |
| `pandadock-metal` | Metal coordination docking |
| `pandadock-tethered` | Constrained docking near reference |

### Utility Tools

| Command | Description |
|---------|-------------|
| `pandadock-prepare` | Prepare ligands (add H, generate 3D) |
| `pandadock-gridbox` | Generate grid box configurations |
| `pandadock-report` | Generate analysis reports |

All nine entry points are covered by `tests/test_cli.py`, which checks that each
imports, responds to `--help`, and only advertises algorithm names that can
actually be constructed.

### Algorithm names

`pandadock` is the flexible-ligand search. The other names are retained so
existing scripts keep working and **all resolve to the same algorithm**:

| Name | Status |
|---|---|
| `pandadock` / `pandacore` | Current flexible-ligand Monte Carlo search |
| `monte_carlo_cpu` | Deprecated alias |
| `genetic_algorithm_cpu` | Deprecated alias |
| `enhanced_hierarchical_cpu` | Deprecated alias |
| `crystal_guided_cpu` | **Removed** — raises on use |

`crystal_guided_cpu` restricted sampling to the neighbourhood of a reference pose
and added an energy bonus for proximity to it. In a redocking benchmark the
reference derives from the answer, so it reported a perturbed copy of the answer
rather than a prediction. Use `pandadock-tethered` if you genuinely need a
reference-constrained search, where the constraint is explicit and recorded in
the output.

---

## Universal GNN Rescorer

The `pandadock gnn rescore` command allows you to rescore docked poses from **any docking software** using the SE(3)-equivariant GNN:

### Supported Input

- **AutoDock Vina** output (SDF/PDBQT converted to SDF)
- **Glide** poses (SDF)
- **GOLD** poses (SDF)
- **pandadock-flex** flexible docking poses
- **pandadock-metal** metal coordination poses
- **pandadock-tethered** constrained poses
- Any multi-conformer SDF file

### Usage

```bash
pandadock gnn rescore -m model.pt -r receptor.pdb -p poses.sdf [OPTIONS]

Options:
  -m, --model PATH      Trained GNN model checkpoint (required)
  -r, --receptor PATH   Receptor PDB or MOL2 file (required)
  -p, --poses PATH      Multi-conformer SDF with poses (required)
  -o, --output PATH     Output CSV with ranked poses (default: rescored_poses.csv)
  --output-sdf PATH     Output SDF with GNN scores as properties
  --site-radius FLOAT   Binding site extraction radius (default: 10 A)
```

### Example Workflow

```bash
# Step 1: Run docking with your preferred tool
vina --receptor protein.pdbqt --ligand ligand.pdbqt --out poses.sdf

# Step 2: Rescore with PandaDock-GNN
pandadock gnn rescore -m model.pt -r protein.pdb -p poses.sdf \
    -o ranked.csv --output-sdf ranked.sdf

# Output CSV columns:
# pose_name, pose_index, gnn_pKd, gnn_energy, activity_prob, predicted_active, gnn_rank
```

### Output SDF Properties

When using `--output-sdf`, each molecule gets these properties:
- `GNN_pKd` - Predicted pKd/pKi value
- `GNN_Energy` - Predicted binding energy (kcal/mol)
- `GNN_Activity` - Activity probability (0-1)
- `GNN_Rank` - Rank based on GNN score (1 = best)

---

## GNN Architecture

PandaDock-GNN uses an SE(3)-equivariant heterogeneous graph neural network:

```
Input: Protein-Ligand Complex
  |
  +-- MOL2/PDB/SDF Parser --> Atom coordinates, types, charges
  |
  +-- Graph Builder --> HeteroData graph
  |   - Protein nodes (56 features)
  |   - Ligand nodes (56 features)
  |   - Interaction edges (23 features, 5A cutoff)
  |
  +-- EGNN Layers x 6 (SE(3)-equivariant message passing)
  |   - Coordinate updates preserve symmetry
  |   - Edge attention mechanism
  |
  +-- Attention Pooling --> Graph-level representation
  |
  +-- Prediction Heads
      - pKd/pEC50 regression
      - Activity classification (sigmoid)
```

**Node Features (56 dims):**
- Atom type one-hot (10)
- SYBYL atom type (16)
- Partial charge (1)
- Hybridization (4)
- Aromaticity, H-bond donor/acceptor (4)
- Residue type (20, protein only)
- Backbone flag (1)

**Edge Features (23 dims):**
- Distance (1)
- Gaussian RBF expansion (16)
- Bond type one-hot (4)
- Interaction type flags (2)

---

## Scoring Functions

| Function | Description | Use Case |
|----------|-------------|----------|
| `vina` | AutoDock Vina empirical scoring (default) | General docking |
| `physics_based` | Lennard-Jones + electrostatics | Detailed energy analysis |

---

## Output Files

### Dock Command

```
docking_output/
+-- complex1.pdb, complex2.pdb, ...   # Protein-ligand complexes
+-- pose1.pdb, pose2.pdb, ...         # Ligand poses only
+-- docking_results.json              # Complete results with energies
+-- interaction_analysis.json         # Detailed interactions
+-- binding_affinities.png            # Affinity distribution
```

### Hybrid Command

```
hybrid_output/
+-- hybrid_results.csv                # Rankings with GNN + Vina scores
+-- pose_1_pec50_X.XX.pdb             # Top poses with pEC50 in filename
+-- complex_1.pdb, ...                # Protein-ligand complexes
```

### Rescore Command

```
rescored_poses.csv                    # Ranked poses with GNN scores
ranked.sdf (optional)                 # SDF with GNN properties
```

---

## Training Your Own GNN Model

PandaDock supports training on three dataset formats: **ULVSH**, **PDBbind**, and **BindingDB**.
For detailed dataset preparation instructions, see the [Dataset Preparation Guide](https://pandadock.readthedocs.io/en/latest/gnn/dataset_preparation.html).

### Dataset Requirements

| Dataset | Format | Key Files |
|---------|--------|-----------|
| ULVSH | Directory | `vitro.tsv` + `protein.mol2`, `ligand.mol2`, `site.mol2` per compound |
| PDBbind | Directory | `INDEX_refined_data.2020` + `{pdb}_pocket.pdb`, `{pdb}_ligand.mol2` |
| BindingDB | TSV file | TSV with `complex_id`, `protein_file`, `ligand_file`, `pK` columns |

### Single Dataset Training

```bash
# Train on ULVSH (942 compounds, 10 targets)
pandadock gnn train -d ULVSH/ -o models/ --epochs 100

# Train on PDBbind (5,316 complexes)
pandadock gnn train -p PDBbind/ -o models/ --epochs 100

# Train on BindingDB (custom TSV file)
pandadock gnn train -b bindingdb_affinity.tsv -o models/ --epochs 100
```

### Combined Dataset Training (Recommended)

Combining datasets improves generalization. Use `--balanced` to prevent larger datasets from dominating:

```bash
# BindingDB + ULVSH (recommended for screening)
pandadock gnn train -b bindingdb.tsv -d ULVSH/ -o models/ \
    --balanced --epochs 100

# ULVSH + PDBbind (recommended for structure-based)
pandadock gnn train -d ULVSH/ -p PDBbind/ -o models/ \
    --balanced --epochs 200

# All three datasets
pandadock gnn train -d ULVSH/ -p PDBbind/ -b bindingdb.tsv -o models/ \
    --balanced --epochs 200 --batch-size 32
```

### Training Options

| Option | Default | Description |
|--------|---------|-------------|
| `--epochs` | 100 | Number of training epochs |
| `--batch-size` | 32 | Batch size (reduce if out of memory) |
| `--hidden-dim` | 256 | Hidden layer dimension |
| `--num-layers` | 6 | Number of EGNN layers |
| `--balanced` | off | Balance sampling across datasets |
| `--patience` | 20 | Early stopping patience |

### Benchmark on Test Set

```bash
pandadock gnn benchmark -m models/best_model.pt -d ULVSH/ -o results/
```

---

## Examples

See the `examples/` directory:

- `examples/basic_docking/` - Simple docking workflow
- `examples/flexible_docking/` - Induced-fit docking
- `examples/metal_docking/` - Metalloprotein docking

---

## Documentation

Full documentation available at [pandadock.readthedocs.io](https://pandadock.readthedocs.io/):

- [Installation Guide](https://pandadock.readthedocs.io/en/latest/installation.html)
- [GNN Overview](https://pandadock.readthedocs.io/en/latest/gnn/overview.html)
- [Training Guide](https://pandadock.readthedocs.io/en/latest/gnn/training.html)
- [Hybrid Docking](https://pandadock.readthedocs.io/en/latest/gnn/hybrid_docking.html)
- [CLI Reference](https://pandadock.readthedocs.io/en/latest/cli/pandadock.html)

---

## Citation

If you use PandaDock in your research, please cite:

```bibtex
@article{panda2024pandadock,
  title={PandaDock: SE(3)-Equivariant Graph Neural Network Scoring for Molecular Docking},
  author={Panda, Pritam Kumar},
  journal={bioRxiv},
  year={2024},
  note={Manuscript in preparation}
}
```

---

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

PandaDock is released under the MIT License. See [LICENSE](LICENSE) for details.

---

## Contact

**Author**: Pritam Kumar Panda
**Affiliation**: Stanford University
**Email**: pritampanda@stanford.edu
**GitHub**: [@pritampanda15](https://github.com/pritampanda15)

---

## Acknowledgments

PandaDock builds upon excellent open-source projects:
- AutoDock Vina (scoring function inspiration)
- PyTorch and PyTorch Geometric (GNN framework)
- RDKit (molecular handling)
- E(n)-Equivariant GNN (Satorras et al. 2021)

---

<div align="center">

**Star this repository if you find it useful!**

[Report Bug](https://github.com/pritampanda15/PandaDock/issues) | [Request Feature](https://github.com/pritampanda15/PandaDock/issues)

</div>
