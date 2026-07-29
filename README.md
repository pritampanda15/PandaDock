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

[Installation](#installation) | [Quick Start](#quick-start) | [Documentation](https://pandadock.readthedocs.io/) | [Benchmark](#affinity-prediction-performance) | [Citation](#citation)

</div>

---

## Overview

**PandaDock** is a molecular docking suite combining a flexible-ligand
conformational search with an SE(3)-equivariant Graph Neural Network scoring
function that reaches Pearson R = 0.88 against experimental binding affinities on
PDBbind.

The search treats every rotatable bond as an explicit degree of freedom. Position
is sampled uniformly over the docking box, orientation uniformly over SO(3), and
each Monte Carlo step is relaxed to a local minimum with a quasi-Newton optimizer
using fully analytic gradients. Scoring runs against precomputed affinity grids,
so a full search costs seconds to minutes rather than hours.

### Key Features

- **Flexible-ligand search** — all rotatable bonds searched, uniform SO(3)
  orientation sampling, L-BFGS relaxation with analytic gradients
- **Grid-accelerated scoring** — AutoDock Vina functional form, precomputed per
  atom type
- **PandaDock-GNN** — SE(3)-equivariant scoring, **R = 0.88** on PDBbind
- **Hybrid workflow** — search with the empirical function, rank with the GNN
- **Universal rescorer** — rescore poses from any docking tool (Vina, Glide, GOLD)
- **Rich output** — poses with bond orders, complexes, interaction analysis,
  plots, PandaMap 2D diagrams and an HTML report
- **Specialized modes** — induced-fit, metal coordination, tethered docking
- **Reproducible** — `--seed` fixes the search; every run records its parameters

---

## Installation

Python 3.8+ is required. RDKit is a hard requirement and is most reliably
installed from conda-forge.

```bash
conda create -n pandadock python=3.10
conda activate pandadock
conda install -c conda-forge rdkit

pip install pandadock
```

From source:

```bash
git clone https://github.com/pritampanda15/PandaDock.git
cd PandaDock
pip install -e .
```

Optional extras:

```bash
pip install -e ".[gnn]"      # PandaDock-GNN (PyTorch, PyTorch Geometric)
pip install pandamap         # 2D protein-ligand interaction diagrams
```

---

## Quick Start

```bash
# 1. Prepare structures (adds hydrogens, assigns protonation)
pandadock-prepare -r receptor.pdb -l ligand.sdf -o prepared/

# 2. Define the binding site
pandadock-gridbox -r receptor.pdb -m similarity \
                  --reference-ligand known_ligand.sdf -o box.json

# 3. Dock
pandadock dock -r receptor.pdb -l ligand.sdf -g box.json -o results/
```

Or give coordinates directly:

```bash
pandadock dock -r receptor.pdb -l ligand.sdf \
               --center 10 12 8 --box 22 22 22 -o results/
```

---

## Docking

### `pandadock dock`

| Option | Default | Description |
|---|---|---|
| `-r, --receptor` | *required* | Receptor PDB file |
| `-l, --ligand` | *required* | Ligand file (SDF/MOL2/PDB) |
| `-g, --grid-config` | — | Grid box JSON from `pandadock-gridbox` |
| `--center X Y Z` | — | Box centre in Å (alternative to `-g`) |
| `--box X Y Z` | — | Box dimensions in Å |
| `-s, --scoring` | `vina` | `vina` or `physics_based` |
| `-n, --num-poses` | `20` | Maximum binding modes returned |
| `-e, --exhaustiveness` | *auto* | Independent search runs |
| `--seed` | *random* | Seed for reproducible runs |
| `--rigid-ligand` | off | Disable torsional search |
| `--grid-spacing` | `0.375` | Affinity grid spacing in Å |
| `--rescoring` | `none` | `none` or `mmgbsa` |
| `-o, --output-dir` | `docking_output` | Output directory |
| `--fast` | off | Reduced sampling for smoke tests only |

### Exhaustiveness

Unless set explicitly, exhaustiveness scales with the number of rotatable bonds:

| Rotatable bonds | 0 | 4 | 8 | 12+ |
|---|---|---|---|---|
| Independent runs | 8 | 16 | 24 | 32 |

A budget that is ample for a rigid fragment leaves a flexible ligand's search
space under-explored, and the failure is silent — the run returns a
confident-looking pose from a local minimum well above the global one. Raise `-e`
for large or highly rotatable ligands.

Two defaults worth knowing: runs are **not reproducible** without `--seed`, and
`--fast` drops to exhaustiveness 2, so its poses should not be reported as
results.

### Number of poses

`--num-poses` is an upper bound. Poses are clustered at 2 Å heavy-atom RMSD, so a
pocket supporting only a few distinct modes returns fewer than requested. The
`pose_diversity.png` plot shows which case you are in.

### Box size

A snug box outperforms a large one. Measured on confirmed search failures,
reducing padding around the ligand from 8 Å to 5 Å lowered median best-of-N RMSD
from 5.11 Å to 2.48 Å — and beat quadrupling exhaustiveness in a larger box,
while running faster. Search volume matters more than sampling budget.

---

## Output

Every run writes:

| File | Contents |
|---|---|
| `report.html` | Run parameters and all plots in one page |
| `poses.sdf` | All poses with bond orders, rank, score, confidence |
| `pose{N}.pdb` | Individual ligand poses |
| `complex{N}.pdb` | Receptor plus ligand, ligand as HETATM chain L |
| `*_poses.json` | Full coordinates and per-term energies |
| `*_summary.json` | Run parameters, ensemble ΔG, runtime |
| `interaction_analysis.json` | Detected interactions, per pose |
| `pose_scores.png` | Scores by rank, with the rank-1/rank-2 gap |
| `energy_components.png` | Energy terms, favourable versus penalty |
| `pose_diversity.png` | Pairwise RMSD between returned poses |
| `interaction_fingerprint.png` | Residues contacted by each pose |
| `pandamap_2d_*.png` | 2D interaction diagram (requires `pandamap`) |

Prefer `poses.sdf` for downstream work: PDB cannot represent bond orders or
formal charges, so viewers infer bonds from distance and routinely mis-assign
aromatic rings.

Interactions are detected with explicit chemistry — donor/acceptor matching
including the protein backbone, hydrophobic typing, electrostatics, π-stacking,
π-cation and metal coordination — not distance thresholds alone.

The score is an empirical docking score in kcal/mol. It ranks poses. It is not a
measured binding free energy and should not be converted to a Kd or IC50 and
reported as a potency prediction; use PandaDock-GNN for affinity.

---

## Hybrid Docking with GNN Rescoring

The recommended workflow: search with the empirical function, rank with the GNN.

```bash
pandadock gnn download-model                    # ~82 MB
pandadock hybrid -r receptor.pdb -l ligand.sdf \
                 --center 10 12 8 --box 22 22 22 \
                 -m models/pandadock_gnn.pt -o results/
```

Produces the same output set as `dock`, ranked by predicted pEC50 with the
empirical score retained alongside for comparison.

### GNN commands

| Command | Description |
|---|---|
| `pandadock gnn download-model` | Download the pre-trained model |
| `pandadock gnn predict` | Predict binding affinity for a complex |
| `pandadock gnn rescore` | Rescore poses from any docking tool |
| `pandadock gnn train` | Train on ULVSH, PDBbind or a combined set |
| `pandadock gnn benchmark` | Evaluate on a test set |
| `pandadock gnn compare` | Compare against baseline scoring methods |

---

## Specialized Docking

```bash
# Induced-fit: refines receptor side chains around the ligand
pandadock-flex -r receptor.pdb -l ligand.sdf --center 10 12 8 --radius 12 -o results/

# Metal coordination: metal-aware scoring with geometry constraints
pandadock-metal dock -r receptor.pdb -l ligand.sdf --center 10 12 8 --box 22 22 22 -o results/

# Tethered: restrains the ligand centroid near a reference pose
pandadock-tethered dock -r receptor.pdb -l ligand.sdf \
                        --ref reference.sdf -t 3.0 -o results/
```

Tethered docking applies a flat-bottom centroid restraint inside the objective:
free movement within the radius, harmonic cost beyond it.

Induced-fit docking redocks into every refined receptor and is substantially
slower than rigid-receptor docking — budget hours rather than minutes for a
single ligand. Refined receptors are written to a temporary directory and removed
when the run ends, including on failure.

Metal parameters fall back to built-in approximations when no AutoDock-format
parameter file is supplied. Those are adequate for identifying coordination
geometry but not for quantitative metal binding energies; pass a parameter file
for that.

---

## Utilities

| Command | Description |
|---|---|
| `pandadock-prepare` | Add hydrogens, assign protonation, generate 3D |
| `pandadock-gridbox` | Define binding sites and write grid box configs |
| `pandadock-report` | Regenerate plots and reports from a results directory |

### Grid box modes

| Mode | Use for |
|---|---|
| `similarity --reference-ligand LIG` | Box centred on a known ligand (redocking) |
| `cavities` | Blind docking — detects pockets without a ligand |
| `residues --residues A:123,A:145` | Box around specified residues |
| `manual --center X Y Z --box X Y Z` | Explicit coordinates |

For redocking or any case where a ligand pose is known, use
`similarity --reference-ligand`. Cavity detection may centre the box on a
neighbouring pocket, placing the true site near the box edge where sampling is
poorer.

---

## Algorithm Names

`pandadock` is the flexible-ligand search. Legacy names are retained for
backwards compatibility and all resolve to the same algorithm:

| Name | Status |
|---|---|
| `pandadock` / `pandacore` | Current flexible-ligand Monte Carlo search |
| `monte_carlo_cpu` | Deprecated alias |
| `genetic_algorithm_cpu` | Deprecated alias |
| `enhanced_hierarchical_cpu` | Deprecated alias |

Docking runs on CPU. The workload parallelises across ligands rather than within
a single search, so throughput scales with core count.

---

## Measuring Pose Accuracy

`benchmarking/` contains a redocking harness. These scripts are development
tooling and are not shipped in the PyPI package — clone the repository to use
them.

```bash
# Split whole PDB entries into receptor/ligand pairs
python benchmarking/prepare_complexes.py --input complexes/ --output prepared/

# Check for ligand leakage before spending compute
python benchmarking/validate_prepared.py --manifest prepared/manifest.csv

# Redock and measure
python benchmarking/redock_benchmark.py --manifest prepared/manifest.csv \
       --output results/ --padding 5 --seed 42 -j 8

# Publication tables (Markdown and LaTeX)
python benchmarking/make_report.py results/redock_results.csv --output results/report
```

RMSD is symmetry-corrected and computed without superposition. Top-1 and
best-of-N success rates are reported separately: quoting best-of-N as though it
were top-1 substantially overstates accuracy.

`analyze_redock.py` separates search failures from ranking failures. A complex
where a sub-2 Å pose was generated but not ranked first is a scoring problem that
more sampling cannot fix, and it is the quantity that tells you whether a learned
rescorer is worth applying.

---
## Affinity Prediction Performance

These are PandaDock-GNN scoring results. For pose-prediction accuracy see
[Measuring Pose Accuracy](#measuring-pose-accuracy) above.

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
