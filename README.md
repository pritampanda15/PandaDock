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

**PandaDock v5.0 — Flexible Molecular Docking, Covalent & Series SAR, PoseBusters Validation, Studio GUI & SE(3)-Equivariant GNN Scoring**

[Official v5.0 User Manual (PDF)](docs/PandaDock_v5.0_User_Manual.pdf) | [ReadTheDocs Documentation](https://pandadock.readthedocs.io/) | [CLI Commands](#cli-commands) | [Installation](#installation) | [Quick Start](#quick-start) | [What's New in v5.0](#whats-new-in-v50) | [Benchmark](#affinity-prediction-performance) | [Citation](#citation)

</div>

> **New to PandaDock or looking for a step-by-step walkthrough?**
> - **Read the Official Plain-Language User Manual (24-page A4 PDF with clickable TOC):** [`docs/PandaDock_v5.0_User_Manual.pdf`](docs/PandaDock_v5.0_User_Manual.pdf) — explains every step from preparing & protonating proteins/ligands, stripping vs. keeping pocket waters, and defining grid boxes (including clicking the X-ray ligand on a complex) to all 48 CLI commands and PandaDock Studio.
> - **Browse the Full Online Documentation on ReadTheDocs:** [https://pandadock.readthedocs.io/](https://pandadock.readthedocs.io/) — guides, Studio GUI & PandaLens 3D/2D walkthroughs, tutorials, API reference, and troubleshooting.

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
  orientation sampling, L-BFGS relaxation with analytic gradients, optional
  non-aromatic ring conformer sampling (`--sample-rings`), and multi-threaded
  CPU chains (`--workers`)
- **Optional GPU search** *(experimental)* — `--device cuda|mps` runs the same
  algorithm batched over thousands of chains, verified against the CPU objective.
  On the 815-complex benchmark an A10G is **8.5× faster overall and 35× on the
  most flexible ligands**, at identical best-of-9 accuracy — see
  [CPU vs GPU](#benchmark-cpu-vs-gpu-on-the-redocking-set)
- **Virtual screening** — `pandadock screen` batches a library onto the device,
  reusing grids across ligands and bucketing them by size
- **Grid-accelerated & physics scoring** — AutoDock Vina (`vina`), Vinardo
  (`vinardo`), aligned `physics_based`, and fast vectorized differential pocket
  MM-GBSA rescoring (`--rescoring mmgbsa`), with AutoDock Vina `XS_TYPE_Met_D`
  catalytic metal handling
- **PandaDock-GNN** — SE(3)-equivariant affinity scoring, R = 0.53 on PDBbind
  (4,640 complexes, fully held out); checkpoint metadata (`strip_hydrogens`,
  `site_radius`) auto-applied on load; see [performance](#affinity-prediction-performance)
- **Separation of concerns** — the empirical function selects poses, the GNN
  estimates affinity for a chosen pose; the GNN should not rescore poses
- **Universal rescorer** — rescore poses from any docking tool (Vina, Glide, GOLD)
- **Publication-grade reports** — poses with bond orders, complexes, interaction
  analysis, 300-DPI diagnostics (scoring funnel, MDS pose landscape, stacked
  contact occupancies, ligand atom RMSF & torsions, residue hotspots), PandaMap
  2D diagrams, built-in **PoseBusters** chemistry/geometry checks, an editorial
  HTML report (`report.html`), and a multi-page A4 vector brochure PDF (`report.pdf`)
- **v5.0 Medicinal-Chemistry Suite** — built-in **PoseBusters audit**
  (`pandadock validate-poses`), automatic **pH 7.4 protonation/tautomer/stereo
  enumeration** (`--enumerate-states`), **differentiable geometric / H-bond /
  metal / MCS restraints** (`--restraint`, CPU & GPU L-BFGS), **displaceable &
  bridging structural waters** (`--waters keep|strip|displace`), **congeneric
  series & SAR brochure PDF** (`pandadock series`), **covalent warhead docking**
  (`pandadock covalent`, 8 warheads), **few-shot per-target weight calibration**
  (`pandadock calibrate` + `--weights-file`), **dual-head pose-RMSD GNN v5**,
  and **one-command Dock-to-MD handoff** (`pandadock to-md` for OpenMM, GROMACS,
  and PandaInsight)
- **Specialized modes** *(experimental)* — induced-fit (`pandadock-flex` with
  χ₁/χ₂ side-chain refinement), metal coordination (`pandadock-metal` with
  bundled 31-metal `AD4_parameters_full.dat`), tethered docking
  (`pandadock-tethered`); see [Specialized Docking](#specialized-docking)
- **Reproducible** — `--seed` fixes the search; every run records its parameters

---

## CLI Commands

PandaDock **v5.0.0** ships **10 top-level CLI entry points** totaling **48 leaf commands** (including 8 GNN subcommands mirrored under both `pandadock gnn` and `pandadock-gnn`, plus `pandadock gui` / `pandadock-gui`):

### 1. `pandadock` (18 commands)

| Command | Purpose |
|---|---|
| `pandadock dock` | Flexible-ligand docking (Vina / Vinardo / Physics-Based, CPU/GPU, `--restraint`, `--waters`, `--enumerate-states`, `--weights-file`, `--rescoring mmgbsa`, `--export-md`) |
| `pandadock series` *(v5.0)* | Congeneric series / MCS core-tethered docking with multi-score consensus & SAR brochure PDF/HTML |
| `pandadock covalent` *(v5.0)* | Covalent warhead docking (8 warheads) anchored to a reactive residue + PandaMap + PDF/HTML report |
| `pandadock validate-poses` *(v5.0)* | Built-in 11-point PoseBusters chemistry & geometry validity audit |
| `pandadock calibrate` *(v5.0)* | Per-target few-shot Vina weight calibration (Ridge LOOCV) |
| `pandadock to-md` *(v5.0)* | One-command Dock-to-MD handoff (OpenMM / GROMACS / PandaInsight) |
| `pandadock gui` *(v5.0)* | Launch PandaDock Studio Apple-designed web GUI (3Dmol + plots + PDF/HTML) |
| `pandadock screen` | Batched GPU/CPU virtual screening of ligand libraries with shared grids |
| `pandadock hybrid` | Docking followed by PandaDock-GNN rescoring |
| `pandadock list-algorithms` | List available docking algorithms and scoring functions |
| `pandadock gnn train` | Train SE(3)-equivariant PandaDock-GNN on protein–ligand datasets |
| `pandadock gnn train-sair` | Train PandaDock-GNN on SAIR shard cache |
| `pandadock gnn predict` | Predict binding affinity (and v5 pose RMSD) for a complex |
| `pandadock gnn rescore` | Universal GNN rescorer for poses from any docking tool |
| `pandadock gnn benchmark` | Benchmark GNN model performance on a test set |
| `pandadock gnn compare` | Compare GNN against baseline scoring functions |
| `pandadock gnn download-model` | Download pre-trained PandaDock-GNN checkpoint |
| `pandadock gnn list-algorithms` | List available PandaDock-GNN models and commands |

### 2. Standalone Tools (9 entry points, 30 commands)

| Entry Point | Commands | Purpose |
|---|---|---|
| **`pandadock-gui`** *(v5.0)* | `pandadock-gui` (1) | Launch PandaDock Studio GUI directly |
| **`pandadock-prepare`** | `pandadock-prepare` (1) | Prepare receptor & ligand structures (with `--enumerate-states --ph`) |
| **`pandadock-gridbox`** | `pandadock-gridbox` (1) | Define binding site grid boxes (`similarity`, `cavities`, `residues`, `manual`) |
| **`pandadock-flex`** | `pandadock-flex` (1) | Flexible / induced-fit docking (χ₁/χ₂ side-chain refinement) |
| **`pandadock-metal`** | `analyze-protein`, `dock`, `list-metals`, `prepare-ligand`, `validate` (5) | Metalloprotein coordination docking (31 bundled metals via `AD4_parameters_full.dat`) |
| **`pandadock-tethered`** | `analyze`, `dock`, `extract`, `validate` (4) | Tethered / crystal-reference pose validation workflows |
| **`pandadock-report`** | `combined`, `pandamap`, `plots`, `scoring` (4) | Regenerate 300-DPI plots, PandaMap 2D/3D maps, scoring breakdowns, and PDF/HTML reports |
| **`pandadock-ml`** | `dock`, `extract-features`, `hybrid-dock`, `list-models`, `train` (5) | Classical ML-enhanced docking & feature pipelines |
| **`pandadock-gnn`** | `benchmark`, `compare`, `download-model`, `list-algorithms`, `predict`, `rescore`, `train`, `train-sair` (8) | Dedicated GNN CLI (mirrors `pandadock gnn *`) |

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
pip install -e ".[gui]"      # PandaDock Studio Apple-designed web GUI
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

Or launch **PandaDock Studio** (Apple-designed GUI with interactive 3D viewer, plots, PDF & HTML reports):

```bash
pandadock-gui --port 8765 --workspace .
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
| `-s, --scoring` | `vina` | `vina`, `vinardo`, or `physics_based` |
| `-n, --num-poses` | `20` | Maximum binding modes returned |
| `-e, --exhaustiveness` | *auto* | Independent search runs |
| `--workers` | `1` | Parallel CPU worker threads across search runs |
| `--sample-rings` | off | Sample non-aromatic ring conformers (ETKDGv3) |
| `--seed` | *random* | Seed for reproducible runs |
| `--rigid-ligand` | off | Disable torsional search |
| `--grid-spacing` | `0.375` | Affinity grid spacing in Å |
| `--device` | `cpu` | Where the search runs: `cpu`, `cuda`, `mps` |
| `--n-chains` | `512` | Parallel search chains when `--device` is not `cpu` |
| `--rescoring` | `none` | `none` or `mmgbsa` (differential 12 Å pocket MM-GBSA) |
| `--restraint` | — | Repeatable differentiable distance restraint (e.g. `A:ZN:301:ZN<2.2:donor`) |
| `--restraints-file` | — | JSON file of distance / pharmacophore restraints |
| `--waters` | `strip` | Structural pocket water protocol: `strip`, `keep`, or `displace` |
| `--enumerate-states` | off | Enumerate pH protonation, tautomer, and unassigned stereo microspecies |
| `--ph` | `7.4` | Target pH for `--enumerate-states` Henderson–Hasselbalch penalties |
| `--weights-file` | — | Target-calibrated Vina weights JSON from `pandadock calibrate` |
| `--export-md` | `none` | Export top pose as MD starter package: `openmm`, `gromacs`, or `both` |
| `-o, --output-dir` | `docking_output` | Output directory |
| `--fast` | off | Reduced sampling for smoke tests only |

---

## What's New in v5.0

PandaDock v5.0 adds an end-to-end medicinal-chemistry workflow built directly on
the analytic-gradient search and Panda-family reporting engine:

### 1. Congeneric Series & SAR Brochure (`pandadock series`)

Dock an analog series sharing one cached receptor affinity grid (`GridCache`),
optionally tethering the Maximum Common Substructure (MCS) core to a 3D
reference ligand, score every top pose with **Vina + Vinardo + pocket MM-GBSA +
PoseBusters**, and export a multi-page A4 **Series SAR Brochure PDF**
(`series_report.pdf`), `series_report.html`, `series_results.csv`, and an
MCS-highlighted 2D analog grid:

```bash
pandadock series -r receptor.pdb -l analogs.sdf \
                 --center 126.5 119.2 146.1 --box 22 22 22 \
                 --reference-pose lead.sdf --workers 4 -o series_out/
```

### 2. Covalent Warhead Docking (`pandadock covalent`)

Reaction-SMARTS covalent adduct assembly for **8 medicinal-chemistry warheads**
(`michael_acceptor`, `chloroacetamide`, `nitrile`, `epoxide`, `boronic_acid`,
`aldehyde`, `sulfonyl_fluoride`, `disulfide`) anchored to a reactive receptor
nucleophile (`CYS`, `SER`, `THR`, `LYS`, `TYR`, `HIS`) as the fixed root of the
`TorsionTree`, writing PDB `LINK`-annotated complexes and full PDF/HTML reports:

```bash
pandadock covalent -r 6LU7_mpro.pdb -l nirmatrelvir.smi \
                   --residue A:CYS:145 --warhead auto -o covalent_out/
```

### 3. Differentiable Restraints & Displaceable Waters

Flat-bottom harmonic restraints work identically on CPU (`L-BFGS-B`) and
GPU/MPS (`batched_lbfgs`) with exact analytic gradients, while `--waters displace`
explicitly compares stripped vs. conserved crystallographic pocket waters:

```bash
# Metal-coordination restraint (Carbonic Anhydrase II + Zn + AZM)
pandadock dock -r 3HS4.pdb -l AZM.sdf --center -4.2 -4.1 17.0 --box 18 18 18 \
               --restraint "A:ZN:301:ZN<2.2:donor" -o restrained_out/

# Displaceable / bridging crystallographic waters
pandadock dock -r 3HS4_waters.pdb -l AZM.sdf --center -4.2 -4.1 17.0 --box 18 18 18 \
               --waters displace -o water_out/
```

### 4. Built-In PoseBusters Audit, pH State Enumeration, Calibration & MD Handoff

```bash
# Audit poses from ANY docking tool (PandaDock, Vina, Glide, Boltz-2, AF3)
pandadock validate-poses -p poses.sdf -r receptor.pdb -o audit.json

# Automatic pH 7.4 protonation, tautomer, and unassigned stereo enumeration
pandadock dock -r receptor.pdb -l ligand.sdf --center 10 12 8 --box 22 22 22 \
               --enumerate-states --ph 7.4 -o states_out/

# Per-target few-shot Vina weight calibration (Ridge LOOCV)
pandadock calibrate -r receptor.pdb -p series_poses.sdf -y labels.csv -o weights.json
pandadock dock -r receptor.pdb -l new_analog.sdf --center 10 12 8 --box 22 22 22 \
               --weights-file weights.json -o calibrated_out/

# One-command Dock-to-MD handoff (OpenMM + GROMACS + PandaInsight scripts)
pandadock to-md -r receptor.pdb -p poses.sdf --engine both --duration-ns 25 -o md_pkg/
```

### 5. PandaDock Studio GUI (`pandadock-gui` / `pandadock gui`)

Launch the local Apple-HIG web workspace to inspect 3D receptors and docked
ligands (WebGL 3Dmol.js with live grid box, Cartoon/Surface/Binding Pocket
styles, pose switcher, and PNG snapshots), configure & run any v5.0 workflow
with live SSE console streaming, browse all 300-DPI diagnostic plots, and view
embedded **PDF** and **HTML** reports in one unified window:

```bash
pandadock-gui --port 8765 --workspace .
# or:
pandadock gui --port 8765
```

Real-complex validation outputs across **6X3V** (GABA\_A), **3HS4** (Carbonic
Anhydrase II + Zn + crystal waters), and **6LU7** (SARS-CoV-2 Mpro `CYS A:145`
+ Nirmatrelvir) are archived in [`test/v5_validation/VALIDATION_SUMMARY.md`](test/v5_validation/VALIDATION_SUMMARY.md).

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

### Benchmark: CPU vs GPU on the redocking set

Run on the full 815-complex benchmark on an **NVIDIA A10G**, and against a CPU
arm on the same machine with four dedicated cores each so neither starved the
other. The CPU arm was stopped after 131 complexes — at its rate the full set
would have taken about 6.5 days — so the head-to-head below is over those 131,
paired complex by complex.

**Full CUDA run, all 814 complexes that docked:**

| | |
|---|---|
| Wall time | **20.2 h** (median 84.4 s/complex) |
| Median top-1 RMSD | 4.37 Å |
| top-1 ≤ 2 Å | 30.2% |
| best-of-9 ≤ 2 Å | 50.5% |

**Paired head-to-head, 131 complexes, same box:**

| | CPU | CUDA (A10G) | Speedup |
|---|---:|---:|---:|
| total wall time | 25.25 h | **2.96 h** | **8.5×** |
| median per complex | 537.6 s | **76.7 s** | 7.0× |
| slowest complex | 7457 s | **211 s** | 35× |

**The speedup tracks ligand flexibility**, because CPU cost scales with
exhaustiveness × torsion count while the batched search absorbs that into one
wider batch:

| Ligand flexibility | CPU | CUDA | Speedup |
|---|---:|---:|---:|
| 0–2 torsions | 120 s | 48 s | 2.5× |
| 3–8 torsions | 575 s | 81 s | 7.1× |
| 9–20 torsions | 1801 s | 126 s | 14.3× |
| **21+ torsions** | **7457 s** | **211 s** | **35.4×** |

CUDA wall time spans 48–211 s across the whole range; the CPU spans 120–7457 s.
The worst complex in the paired set took **over two hours on CPU and 3.5 minutes
on the GPU**.

**Accuracy, on the same 131 complexes:**

| | CPU | CUDA |
|---|---:|---:|
| **best-of-9 ≤ 2 Å** | **56.5%** | **56.5%** |
| top-1 ≤ 2 Å | 34.4% | 30.5% |
| median top-1 RMSD | 4.21 Å | 5.05 Å |

**Best-of-9 is identical.** The GPU generates the same correct poses; it ranks
them slightly worse, costing 3.9 points of top-1 accuracy. That is a ranking
difference, not a sampling one, and it is the honest cost of the speedup as
configured here (512 chains × 8 basin hops against the CPU's auto
exhaustiveness).

Raising `--n-chains` does **not** obviously close that gap, which is worth
recording because it is the intuitive fix. Measured on 14 complexes at 512,
2048 and 4096 chains: wall time rose 1.21x and 1.62x, while top-1 within 2 A
went 50.0%, 42.9%, 50.0% — no trend. Per complex the rigid ligands are
bit-identical across all three settings, and the flexible ones swing
non-monotonically (one 7-torsion ligand gave 3.23, 12.48 and 1.56 A). More
chains changes the random stream as much as it deepens the search, so at this
sample size the run-to-run variance swamps any systematic effect. Establishing
whether more chains help would take a far larger matched run; it should not be
assumed in the meantime.

For reference, the manuscript's CPU figures on the full set are 33.7% top-1 and
57.0% best-of-N, which the paired CPU arm here reproduces closely.

To reproduce:

```bash
python benchmarking/redock_benchmark.py \
    --manifest benchmark_prepared/manifest.csv \
    --device cuda --n-chains 512 \
    --output results_cuda/
```

### Other measurements

End to end, one ligand through `pandadock dock` at its defaults:

| | time | best score |
|---|---|---|
| `--device cpu` | 61.2 s | −7.55 |
| `--device mps --n-chains 256` | 23.3 s | −7.34 |

Like for like — the same batched search on CPU versus on the device — a single
small ligand is *slower* on the device, because it is many small sequential
kernels and launch overhead dominates. The gain comes from filling the device:

| workload | device vs. batched CPU |
|---|---|
| one ligand, few hundred chains | slower — the device is not filled |
| several ligands, mixed sizes | ~1.9× |
| several ligands, size-matched | ~3.6× |

So batching *ligands* matters more than batching chains, and grouping ligands of
similar atom and torsion count roughly doubles the gain — which is what
`pandadock screen` does automatically.

**Status.** Numerical parity is tested on every supported device, in float64 on
CPU and CUDA. The full 815-complex benchmark has now been run on an A10G; the
manuscript's reported figures remain the CPU ones. The GPU path matches CPU
best-of-9 accuracy and trails it by 3.9 points on top-1 pose ranking at the
settings benchmarked above.

---

## Output

Every run writes:

| File | Contents |
|---|---|
| `report.pdf` | Multi-page A4 vector brochure report (cover silhouette, quality checks, all figures, tables, methods & references) |
| `report.html` | Self-contained editorial HTML summary with embedded branding and all figures |
| `poses.sdf` | All poses with bond orders, rank, score, confidence |
| `pose{N}.pdb` | Individual ligand poses |
| `complex{N}.pdb` | Receptor plus ligand, ligand as HETATM chain L |
| `*_poses.json` | Full coordinates and per-term energies |
| `*_summary.json` | Run parameters, ensemble ΔG, runtime |
| `interaction_analysis.json` | Detected interactions, per pose |
| `protein_silhouette.png` | 2D PCA-projected receptor Cα trace with bound top ligand pose |
| `pose_scores.png` | 3-panel energetics: ranked scores, 298 K Boltzmann populations & score-vs-RMSD funnel |
| `energy_components.png` | Per-pose energy terms, favourable (negative) versus penalty (positive) |
| `pose_diversity.png` | Pairwise heavy-atom RMSD matrix and 2D MDS conformational landscape |
| `interaction_fingerprint.png` | Stacked per-residue contact occupancies by type and residue-by-pose heatmap |
| `ligand_dynamics.png` | Per-heavy-atom RMSF across poses (anchors vs mobile tails) and rotatable torsion histograms |
| `residue_hotspots.png` | Empirical distance-weighted contact proxy scores per pocket residue |
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

Both `pandadock gnn` and standalone `pandadock-gnn` expose the same 8 subcommands:

| Command | Description |
|---|---|
| `pandadock-gnn download-model` | Download or `--list` pre-trained models |
| `pandadock-gnn predict` | Predict binding affinity for a complex (stdout or `-o json`) |
| `pandadock-gnn rescore` | Rescore poses from any docking tool |
| `pandadock-gnn train` | Train on ULVSH, PDBbind or a combined set |
| `pandadock-gnn train-sair` | Train on target-disjoint SAIR shard caches (with optional `--target-grouped` & `--rank-weight`) |
| `pandadock-gnn benchmark` | Evaluate on a test set |
| `pandadock-gnn compare` | Compare against baseline scoring methods |
| `pandadock-gnn list-algorithms` | List available GNN scoring models and training entry points |

---

## Specialized Docking

> **Experimental — not yet benchmarked on the full 815-complex set.** No
> manuscript result depends on these modes, though all three have been updated
> to use real physical geometry and consistent kcal/mol scoring scales.

```bash
# Induced-fit: refines receptor side chains around the ligand
pandadock-flex -r receptor.pdb -l ligand.sdf --center 10 12 8 --radius 12 -o results/

# Metal coordination: metal-aware scoring with bundled AD4 parameters (31 metals)
pandadock-metal analyze-protein -r metalloprotein.pdb
pandadock-metal dock -r metalloprotein.pdb -l ligand.sdf --center 10 12 8 --box 22 22 22 -o results/

# Tethered: restrains the ligand centroid near a reference pose
pandadock-tethered dock -r receptor.pdb -l ligand.sdf \
                        --ref reference.pdb -t 3.0 -o results/
```

Tethered docking applies a flat-bottom centroid restraint inside the objective:
free movement within the radius, harmonic cost beyond it.

Induced-fit docking (`pandadock-flex`) samples real side-chain χ₁/χ₂ rotations
(keeping backbone and Cβ atoms fixed) with Metropolis Monte Carlo, computes
actual heavy-atom receptor RMSD and relative MMFF94 ligand conformational
strain, redocks with Vina scoring, and ranks poses with a calibrated IFD score
(`binding + 0.05 * receptor_strain + 0.05 * ligand_strain + interaction_bonus`).
Use `--initial-poses-to-retain` to control runtime. Temporary refined receptors
are cleaned up automatically on exit or failure.

Metal docking (`pandadock-metal`) ships `AD4_parameters_full.dat` (31 AutoDock4
metal ion types including Zn, Mg, Fe, Ca, Mn, Cu, Co, Ni, Pt, Pd, Au, Mo, Gd)
with case-insensitive lookup, AutoDock Vina `XS_TYPE_Met_D` catalytic metal
handling during grid search, open-coordination-site accounting (e.g. recognizing
a 3-His catalytic Zn site has 1 open coordination site), and nearest-donor
geometric constraints. Supply `--metal-params` to override with custom
AutoDock-format parameters.

---

## Virtual Screening

```bash
pandadock screen -r receptor.pdb -l library.sdf \
                 --center 10 12 8 --box 22 22 22 \
                 --device cuda --n-chains 128 -o results/
```

`--ligands` takes a multi-molecule SDF or a directory of ligand files. This is
where batching pays: one ligand never fills a device, and a library is many such
runs.

Two things happen automatically. Affinity grids are built once per atom
*signature* and reused across the library, so grid construction becomes a
per-campaign cost rather than a per-ligand one. And ligands are bucketed by
torsion and atom count before batching, because everything in a batch is padded
to the batch maximum — mixing a 9-atom fragment with a 48-atom ligand wastes
most of the work on padding, and size-matched batches measured roughly twice the
throughput of mixed ones.

| Option | Default | Description |
|---|---|---|
| `-r, --receptor` | *required* | Receptor PDB, shared by every ligand |
| `-l, --ligands` | *required* | Multi-molecule SDF, or a directory |
| `--device` | *auto* | `cuda`, `mps`, `cpu` |
| `--n-chains` | `128` | Search chains per ligand |
| `--n-steps` | `8` | Basin-hopping steps |
| `--max-batch` | `64` | Ligands per batch; bounds device memory |
| `--top` | *all* | Write poses for the top N only |
| `--seed` | *random* | Reproducible runs |

Output is deliberately thin — writing a full report per ligand would dominate
the runtime of the thing this command exists to speed up:

```
screening_results.csv     rank, ligand, score, torsion count
poses/{ligand}.sdf        best pose per ligand, score in a tag
```

Scores rank ligands against each other. They are docking scores in kcal/mol, not
measured binding free energies, and should not be converted to a Kd — use
`pandadock-gnn` for affinity. Note also that docking scores correlate with
ligand size, so a ranked list of mixed-size compounds will favour the larger
ones; compare within a series where you can.

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

On CPU, `--workers N` parallelises independent Monte Carlo runs within a single
docking job, and virtual screening also scales across ligands; on GPU,
`--device cuda|mps` batches chains and ligands directly on the accelerator.

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
