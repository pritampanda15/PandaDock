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
    <img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python 3.9+">
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
function trained on 741,706 co-folded complexes under target-disjoint splits.

The search treats every rotatable bond as an explicit degree of freedom. Position
is sampled uniformly over the docking box, orientation uniformly over SO(3), and
each Monte Carlo step is relaxed to a local minimum with a quasi-Newton optimizer
using fully analytic gradients. Scoring runs against precomputed affinity grids,
so a full search costs seconds to minutes rather than hours.

### Key Features

- **Flexible-ligand search** — all rotatable bonds searched, uniform SO(3)
  orientation sampling, L-BFGS relaxation with analytic gradients
- **Optional GPU search** *(experimental)* — `--device cuda|mps` runs the same
  algorithm batched over thousands of chains, verified against the CPU objective
- **Grid-accelerated scoring** — AutoDock Vina functional form, precomputed per
  atom type
- **PandaDock-GNN** — SE(3)-equivariant affinity scoring, R = 0.53 on PDBbind
  (4,640 complexes, fully held out); see [performance](#affinity-prediction-performance)
- **Separation of concerns** — the empirical function selects poses, the GNN
  estimates affinity for a chosen pose; the GNN should not rescore poses
- **Universal rescorer** — rescore poses from any docking tool (Vina, Glide, GOLD)
- **Rich output** — poses with bond orders, complexes, interaction analysis,
  plots, PandaMap 2D diagrams and an HTML report
- **Specialized modes** *(experimental)* — induced-fit, metal coordination,
  tethered docking; not benchmarked, see [Specialized Docking](#specialized-docking)
- **Reproducible** — `--seed` fixes the search; every run records its parameters

---

## Installation

Python 3.9+ is required. RDKit is a hard requirement and is most reliably
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
| `--device` | `cpu` | Where the search runs: `cpu`, `cuda`, `mps` |
| `--n-chains` | `512` | Parallel search chains when `--device` is not `cpu` |
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

## GPU-Accelerated Search (experimental)

```bash
pandadock dock -r receptor.pdb -l ligand.sdf \
               --center 10 12 8 --box 22 22 22 \
               --device cuda --n-chains 1024 -o results/
```

Requires the `[gnn]` extra for PyTorch. `--device` accepts `cuda`, `mps` (Apple
Silicon) or `cpu`; unlike the removed legacy CUDA modules, this is the same
algorithm as the CPU path rather than a separate engine.

**What runs where.** The conformational search — pose construction, grid
scoring, Monte Carlo, and a batched L-BFGS — runs on the device, advancing
thousands of chains at once instead of one after another. Everything after it
stays on the CPU: minima are returned, rescored in float64, clustered with
symmetry-corrected RMSD, and written out by the same code the CPU path uses. A
`--device` flag changes how poses are found, not what is done with them.

**Numerical agreement.** Against the CPU objective, on float64 devices:

| quantity | agreement |
|---|---|
| coordinates | 3.6e-15 |
| energy (incl. intramolecular) | 1.4e-14 |
| DOF gradient | 1.9e-07 |

The gradient figure is not looseness in the port. The affinity maps are stored
as float32, and the CPU's x-derivative subtracts two raw float32 map values,
losing precision to cancellation; the GPU path upcasts first and is the more
accurate of the two. Apple's MPS backend has no float64 at all and runs the
search in float32, so its tolerances are correspondingly wider.

**Performance, honestly.** Measured on an M-series laptop (MPS), not on a
datacentre GPU, so treat these as directional.

End to end, one ligand through `pandadock dock` at its defaults — the comparison
a user actually experiences:

| | time | best score |
|---|---|---|
| `--device cpu` | 61.2 s | −7.55 |
| `--device mps --n-chains 256` | 23.3 s | −7.34 |

Both returned 9 clustered poses through the identical output path.

Like for like, though — the same batched search on CPU versus on the device —
the picture is more mixed, and it is the honest one to plan against:

| workload | device vs. batched CPU |
|---|---|
| one ligand, few hundred chains | slower — the device is not filled |
| several ligands, mixed sizes | ~1.9× |
| several ligands, size-matched | ~3.6× |

The gain comes from filling the device, so batching *ligands* matters more than
batching chains, and grouping ligands of similar atom and torsion count roughly
doubles it. On a single small ligand the batched search can be slower than
running it batched on the CPU: it is many small sequential kernels, and launch
overhead dominates. The end-to-end win above comes from the device absorbing a
much larger search, not from each step being faster.

**Status.** Numerical parity is tested on every supported device; throughput on
CUDA is not yet characterised, and the 814-complex benchmark in the manuscript
was run on the CPU path. No pose-accuracy claim is made for the GPU path beyond
its agreement with the CPU objective.

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

**Do not use this to pick poses.** Benchmarked against the empirical scoring
function on 50 complexes at 20 poses each, GNN rescoring is clearly worse:

| Selection | Median RMSD | ≤2 Å | ≤1 Å |
|---|---|---|---|
| Empirical scoring function | 2.09 Å | 48% | 24% |
| GNN rescoring | 5.36 Å | 22% | 14% |
| Best pose available | 1.26 Å | 80% | 38% |

The affinity model was trained to rank ligands against each other, not poses
of one ligand against each other, and it does not transfer to the second task.
Use `pandadock dock` to select a pose, and the GNN to estimate the affinity of
a pose already selected. The command below is kept so the comparison above can
be reproduced.

```bash
pandadock-gnn download-model                    # ~82 MB
pandadock hybrid -r receptor.pdb -l ligand.sdf \
                 --center 10 12 8 --box 22 22 22 \
                 -m models/pandadock_gnn.pt -o results/
```

Produces the same output set as `dock`, ranked by predicted pEC50 with the
empirical score retained alongside for comparison.

### GNN commands

| Command | Description |
|---|---|
| `pandadock-gnn download-model` | Download the pre-trained model |
| `pandadock-gnn predict` | Predict binding affinity for a complex |
| `pandadock-gnn rescore` | Rescore poses from any docking tool |
| `pandadock-gnn train` | Train on ULVSH, PDBbind or a combined set |
| `pandadock-gnn benchmark` | Evaluate on a test set |
| `pandadock-gnn compare` | Compare against baseline scoring methods |

---

## Specialized Docking

> **Experimental — not benchmarked.** These modes are scaffolding for future
> development rather than validated methods, and no published PandaDock result
> depends on them. In particular the induced-fit path does not consult a rotamer
> library (side-chain conformers are generated by rotating atom groups about a
> fixed Cartesian axis rather than about residue χ bonds), reports receptor RMSD
> as a hard-coded constant, and accepts several options — loop refinement, the
> refinement-cycle count — that do not currently affect the calculation. The
> metal module falls back to unvalidated parameters for metals outside its
> curated set, which it reports at runtime. Treat any output as indicative only.

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

Induced-fit docking redocks into every refined receptor, so cost scales with the
number of poses carried into refinement: budget hours rather than minutes for a
single ligand. Use `--initial-poses-to-retain` to control that trade-off. Output
matches `dock`: `poses.sdf`, ligand poses, and complexes with the refined
receptor. Refined receptors are written to a temporary directory and removed
when the run ends, including on failure. The IFD score is the binding energy plus
a weighted receptor-strain penalty, so a pose requiring more side-chain
rearrangement scores worse than an equivalent one that requires none.

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

> **These numbers were revised downward in August 2026.** Earlier releases of
> this README reported R = 0.88 on PDBbind. That figure came from a benchmarking
> script that scored 50 alphabetically-selected complexes rather than the full
> set and applied a rescaling fitted to the same labels it then correlated
> against. It does not reproduce and has been withdrawn. The numbers below are
> measured on the full dataset with a fixed prediction pipeline and no
> label-informed transform.

### SAIR (released model, 741,706 complexes, target-disjoint splits)

| Evaluation | N | Pearson R |
|---|---|---|
| Held-out SAIR test split | 90,219 | 0.407 |
| Experimental crystal structures | 202 | 0.467 |
| PDBbind v2020 refined set | 4,640 | 0.531 |

PDBbind is fully independent of this model — it appears nowhere in SAIR
training — so 0.531 is a genuine out-of-distribution result.

### PDBbind v2020 refined set (n = 4,640, native crystal poses)

| Model | Pearson R | Spearman ρ |
|---|---|---|
| SAIR (independent) | 0.531 | 0.523 |
| ULVSH+BindingDB+PDBbind *(trained on this data)* | 0.387 | 0.382 |
| BindingDB (independent) | 0.277 | 0.281 |
| BindingDB+ULVSH (independent) | −0.079 | −0.022 |
| **PDBbind-only, own held-out test (n = 624)** | **0.690** | **0.674** |

A model trained on PDBbind alone under a target-disjoint split reaches R = 0.690
on complexes it never saw. Rows marked *trained on this data* are scored partly
on their own training complexes and are not held-out results.

### A note on within-target correlation

Pooled correlation across many targets is not evidence of ligand ranking: 84.4%
of PDBbind's label variance lies *between* targets, so a predictor that emits
each target's mean affinity and ignores the ligand entirely scores R = 0.92.
Measured per target, the model's median within-target R is 0.31 on PDBbind, 0.24
on its own SAIR test split, and 0.36 on an independent 30-compound series. Treat
these as the honest estimate of ligand-ranking ability.

### ULVSH — withdrawn

Previously reported here as R = 0.82 affinity / AUC 0.94 classification. That
dataset stores assay results in target-specific columns, and the loader falls
back to a censored default for anything it cannot parse: **96.1% of compounds
share a single label** and only 37 of 943 carry a distinct measured value. Any
correlation over that distribution measures separation from the floor, not
affinity ranking. The result is withdrawn and ULVSH is not recommended as an
affinity benchmark. Training on it also compresses model output — the spread of
predicted affinity falls from 2.25 pK units to 0.58 on a held-out series.

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
@article {Panda2026.08.19.745667,
  author = {Panda, Pritam Kumar},
  title = {PandaDock: An Open-Source Molecular Docking Platform with Flexible-Ligand Search and Equivariant Neural Scoring},
  elocation-id = {2026.08.19.745667},
  year = {2026},
  doi = {10.64898/2026.08.19.745667},
  publisher = {Cold Spring Harbor Laboratory},
  abstract = {We present PandaDock, an open-source molecular docking platform implementing flexible-ligand conformational search with analytic gradients, a precomputed affinity grid engine, specialized modules for induced-fit, metal-coordination and tethered docking, and an SE(3)-equivariant graph neural network scoring function trained at scale. Ligand flexibility is represented as a torsion tree and pose parameters are optimized by Monte Carlo with Metropolis acceptance refined by L-BFGS, with rotational gradients obtained in closed form through the derivative of the SO(3) exponential map rather than by finite differences. Affinity grids are built by a blocked neighbor-selection scheme that is exact and 5.6-9.7x faster than dense evaluation, and may be cached across ligands sharing a receptor and site, reducing a six-ligand series from 29.3 s to 10.4 s. On 814 protein-ligand complexes spanning 14 target families, PandaDock recovers a pose within 2 Angstroms of the crystal geometry in 33.7\% of cases at rank 1 and in 57.0\% of cases within the returned ensemble. The GNN scoring function is trained on 741,706 co-folded complexes from SAIR under target-disjoint splits, reaching a Pearson r of 0.407 on 90,219 held-out complexes and transferring to 202 independent crystal structures with measured Ki, Kd, IC50 or EC50 at r = 0.467. We report the model against three controls, a target-mean predictor, a ligand-descriptor-only baseline, and within-target correlations, and document both where it performs and where it does not, including its unsuitability for pose rescoring. On an independent 30-compound series against a single GABAA receptor target, PandaDock{\textquoteright}s empirical scoring function ranks 8th of 25 methods evaluated, ahead of every AutoDock Vina and Vinardo configuration tested, while the GNN scores below Vina, consistent with the within-target ceiling identified on SAIR. At full scale on the PDBbind v2020 refined set (n = 4,640, native crystal poses), the fully independent SAIR model reaches r = 0.531, and a dedicated model trained on PDBbind alone under a target-disjoint split reaches r = 0.690 on its own held-out test complexes, the strongest evidence in this work that PandaDock{\textquoteright}s affinity predictions generalize. PandaDock is distributed under an open-source license at https://github.com/pritampanda15/PandaDock with a complete command-line interface and a reproducible benchmarking harness.Competing Interest StatementThe authors have declared no competing interest.},
  URL = {https://www.biorxiv.org/content/early/2026/08/20/2026.08.19.745667},
  eprint = {https://www.biorxiv.org/content/early/2026/08/20/2026.08.19.745667.full.pdf},
  journal = {bioRxiv}
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
**Email**: pritam@stanford.edu
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
