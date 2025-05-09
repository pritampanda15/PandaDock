# 🐼 PandaDock: Version: 2.0.2

**Python-based Molecular Docking Platform for Drug Discovery, Bioinformatics, and Computational Chemistry**.

<p align="center">
  <a href="https://github.com/pritampanda15/PandaDock">
    <img src="https://github.com/pritampanda15/PandaDock/blob/main/logo/pandadock-logo.svg" width="1000" alt="PandaDock Logo"/>
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

---

## 🚀 Overview

**PandaDock** is a powerful, Python-based molecular docking toolkit combining:

- Traditional docking strategies
- Physics-based scoring functions (MM-GBSA-inspired)
- CHARMm-style dynamics and sampling
- CPU **and** GPU acceleration
- Automatic flexible residue detection
- Comprehensive reporting with interactive HTML visualization

It is designed for high-accuracy drug discovery, computational chemistry, and next-generation bioinformatics workflows.

---

## ✨ Key Features

- 🔬 **Flexible Input Parsing**: Supports PDB, MOL, SDF files
- 🎯 **Binding Site Definition**: Manual or automatic CASTp like pocket detection
- 📊 **Multiple Scoring Functions**:
  - Basic (VDW + H-bond)
  - Enhanced (VDW + H-bond + electrostatics + desolvation + hydrophobic)
  - Physics-based (full MM-GBSA inspired energy decomposition)
- 🔥 **Hardware Acceleration**:
  - Native GPU (PyTorch/CuPy) and multi-core CPU parallelization
- 🧬 **Advanced Search Algorithms**:
  - Genetic Algorithm (Parallelized)
  - Monte Carlo Simulated Annealing
  - PANDADOCK (Simulated Annealing + Final Minimization)
  - Random Search, Gradient-based, Replica-Exchange
- 🧪 **Flexible Residue Docking**: Auto-flex and custom-flex options
- 📈 **Batch Screening**: High-throughput screening with scoring, filtering, and summaries
- 🧩 **Comprehensive Reports**:
  - Energy component breakdown
  - Interaction analysis
  - Pose clustering
  - RMSD validation
  - Full HTML visualization
  - Binding Affinity calculations 
  - All score plots
- 🛠️ **Extensible Python API** for custom workflows and integrations
- **Virtual Screening Option**

---
## Installation

### Prerequisites

PandaDock requires Python 3.6+ and the following dependencies:
- NumPy, SciPy, Matplotlib
- scikit-learn
- RDKit (optional but recommended)
- PyTorch or CuPy (optional for GPU acceleration)

### Install via pip

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies (recommended)
pip install rdkit
pip install torch  # For GPU acceleration

# Install PandaDock
pip install pandadock
```

### Install via conda

```bash
# First install RDKit with conda
conda install -c conda-forge rdkit

# Then install PandaDock
pip install pandadock
```

### GPU Acceleration (Optional)

> *Tip*: Install `torch` with CUDA for GPU acceleration:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
### PyTorch with CUDA support
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
### OR CuPy
```bash
pip install cupy-cuda11x
```

## 🔥 Quick Start
**🚀 Pro Tip:**  
Always provide an active site center using `-s X Y Z`.  
It significantly **boosts docking speed, precision, and reliability** in PandaDock.
If GPU is available then it would run smoothly and efficiently or if you have multi-core CPU

```bash
# Simple run (default Genetic Algorithm)
pandadock -p protein.pdb -l ligand.sdf -s -15.7 -17.7 8.1 --grid-radius 10.0 --grid-spacing 0.375  -i 10 -a genetic --use-gpu

# Physics-Based docking 
pandadock -p protein.pdb -l ligand.sdf -a genetic --physics-based -s -15.7 -17.7 8.1 --grid-radius 10.0 --grid-spacing 0.375  --use-gpu

# Physics-Based docking (PANDADOCK + MMGBSA scoring)
pandadock -p protein.pdb -l ligand.sdf -a pandadock --physics-based
pandadock -p protein.pdb -l ligand.sdf -a pandadock --physics-based -s -15.7 -17.7 8.1 --grid-radius 10.0 --grid-spacing 0.375 

# Use GPU
pandadock -p protein.pdb -l ligand.sdf --use-gpu --physics-based -s -15.7 -17.7 8.1 --grid-radius 10.0 --grid-spacing 0.375 

# Automatic algorithm selection
pandadock -p protein.pdb -l ligand.sdf --auto-algorithm -s -15.7 -17.7 8.1 --grid-radius 10.0 --grid-spacing 0.375 

# Virtual Screening
pandadock --virtual-screening --ligand-library ligands/ --parallel-screening -p receptor.pdb -s -15.7 -17.7 8.1 --use-gpu -a random --num-modes 9 --vs-exhaustiveness 1 --screening-processes 36 --top-hits 10 --grid-spacing 0.3
```
#### Running Modes

PandaDock offers flexible running modes to balance between speed and accuracy:

- **Fast Mode**: `--fast-mode` - Quick docking with minimal enhancements
- **Default Mode**: Basic docking with standard scoring
- **Enhanced Mode**: Use any combination of enhancement flags:
  - `--enhanced-scoring` - More accurate scoring with electrostatics
  - `--local-opt` - Fine-tune top poses for better results
  - `--exhaustiveness 5` - Run multiple independent searches
  - `--prepare-molecules` - Optimize molecule geometry before docking
  - `--flex-residues` - Use flexible residues
- **Physics Mode**: More advanced algorithm
- **Pandadock**: Pandadock algorithm with conformer generation
- **VirtualScreening**: Run virtual screening with your choice of algorithm

---

## 📊 Result Outputs

# 📂 Example: **Docking Output Structure**

After running PandaDock, a comprehensive report is automatically generated:

```
├── binding_affinity_report.csv
├── binding_affinity_report.txt
├── complex_pose_1_score:-14.25.pdb
├── complex_pose_2_score:-14.07.pdb
├── complex_pose_3_score:-13.94.pdb
├── complex_pose_4_score:-12.35.pdb
├── detailed_docking_report.txt
├── docking_report.html
├── docking_report.json
├── docking_results.csv
├── docking_scores.txt
├── energy_breakdown.csv
├── parameters.txt
├── plots/
│   ├── combined_metrics_vs_rank.png
│   ├── energy_breakdown.png
│   ├── energy_stacked.png
│   ├── ic50_vs_deltag.png
│   ├── kd_vs_deltag.png
│   ├── score_analysis.png
│   ├── score_distribution.png
│   ├── score_improvement.png
│   ├── score_plot.png
│   ├── score_rank.png
├── pose_1_score:-14.2.pdb
├── pose_2_score:-14.1.pdb
├── pose_3_score:-13.9.pdb
├── pose_4_score:-12.3.pdb
├── sphere.pdb
└── status.json
```
Even generates a grid sphere where you define your grid center
Example:
<p align="center">
  <img src="https://github.com/pritampanda15/PandaDock/blob/main/example_outputs/plots/sphere.png" width="350">
</p>


### Each docking run generates:
- 📄 **Reports** in CSV, TXT, JSON, and interactive HTML formats
- 📈 **Analysis plots** (binding energy breakdowns, score distributions, ranking curves)
- 🧬 **Pose files** (.pdb) for each predicted ligand conformation
- 🛠️ **Parameters** and **status** tracking for reproducibility

One example output in detailed_docking_report.txt is appened below:
```bash
ALGORITHM DETAILS
-----------------
Algorithm: Genetic
Population Size: 100
Scoring Function: Advanced
Hardware Acceleration: GPU (ID: 0)
GPU Precision: float32
Exhaustiveness: 1
RESULTS SUMMARY
--------------
Total Poses Generated: 4
Best Score: -14.2475
Worst Score: -12.3464
Average Score: -13.6515

Score Standard Deviation: 0.7612

TOP 10 POSES
--------------
Rank	Score	File
ENERGY COMPONENT BREAKDOWN (TOP 10 POSES)
----------------------------------------
Pose       Clash    Desolvat    Electros     Entropy      H-Bond    Hydropho    Van der 
----------------------------------------------------------------------------------------
   1       -0.46       -0.00       -0.09       -0.12       -0.09       -0.09       -0.14
   2       -0.93       -0.00       -0.19       -0.23       -0.19       -0.19       -0.28
   3       -1.39       -0.01       -0.28       -0.35       -0.28       -0.28       -0.42
   4       -1.86       -0.01       -0.37       -0.46       -0.37       -0.37       -0.56

BINDING AFFINITY ESTIMATION
---------------------------
Pose 1: Score = -14.25 kcal/mol
  Estimated ΔG: -14.25 kcal/mol
  Estimated Kd: 3.60e-11 M
  Estimated IC50: 7.20e-11 M

Pose 2: Score = -14.07 kcal/mol
  Estimated ΔG: -14.07 kcal/mol
  Estimated Kd: 4.87e-11 M
  Estimated IC50: 9.75e-11 M

Pose 3: Score = -13.94 kcal/mol
  Estimated ΔG: -13.94 kcal/mol
  Estimated Kd: 6.02e-11 M
  Estimated IC50: 1.20e-10 M

Pose 4: Score = -12.35 kcal/mol
  Estimated ΔG: -12.35 kcal/mol
  Estimated Kd: 8.91e-10 M
  Estimated IC50: 1.78e-09 M
════════════════════════════════════════════════════════════════════════════════
    Binding Affinity Report
================================================================================

Pose         ΔG (kcal/mol)              Kd (M)            IC50 (M)
------------------------------------------------------------------
1                   -14.25            3.60e-11            7.20e-11
2                   -14.07            4.87e-11            9.75e-11
3                   -13.94            6.02e-11            1.20e-10
4                   -12.35            8.91e-10            1.78e-09

```

## 📈 Example Docking Plots

<p align="center">
  <img src="https://github.com/pritampanda15/PandaDock/blob/main/example_outputs/plots/combined_metrics_vs_rank.png" width="220">
  <img src="https://github.com/pritampanda15/PandaDock/blob/main/example_outputs/plots/energy_breakdown.png" width="220">
  <img src="https://github.com/pritampanda15/PandaDock/blob/main/example_outputs/plots/energy_stacked.png" width="220">
  <img src="https://github.com/pritampanda15/PandaDock/blob/main/example_outputs/plots/ic50_vs_deltag.png" width="220">
</p>

<p align="center">
  <img src="https://github.com/pritampanda15/PandaDock/blob/main/example_outputs/plots/ic50_vs_rank.png" width="220">
  <img src="https://github.com/pritampanda15/PandaDock/blob/main/example_outputs/plots/kd_vs_deltag.png" width="220">
  <img src="https://github.com/pritampanda15/PandaDock/blob/main/example_outputs/plots/kd_vs_rank.png" width="220">
  <img src="https://github.com/pritampanda15/PandaDock/blob/main/example_outputs/plots/score_analysis.png" width="220">
</p>

<p align="center">
  <img src="https://github.com/pritampanda15/PandaDock/blob/main/example_outputs/plots/score_distribution.png" width="220">
  <img src="https://github.com/pritampanda15/PandaDock/blob/main/example_outputs/plots/score_improvement.png" width="220">
  <img src="https://github.com/pritampanda15/PandaDock/blob/main/example_outputs/plots/score_plot.png" width="220">
  <img src="https://github.com/pritampanda15/PandaDock/blob/main/example_outputs/plots/score_rank.png" width="220">
</p>

---



# 📖 **Docking Command Examples**

## 🎯 Basic Docking (Random / Genetic / Auto-Algorithm)

| Description | Command |
|:------------|:--------|
| **Random Search** | `pandadock -p protein.pdb -l ligand.sdf -s -15.7 -17.7 8.1 -a random -i 100` |
| **Genetic Algorithm (default)** | `pandadock -p protein.pdb -l ligand.sdf -s -15.7 -17.7 8.1 -a genetic -i 100` |
| **Auto Algorithm Selection** | `pandadock -p protein.pdb -l ligand.sdf -s -15.7 -17.7 8.1 --auto-algorithm` |

---

## 🧪 Physics-Based Docking

| Description | Command |
|:------------|:--------|
| **Physics-Based Scoring (CPU)** | `pandadock -p protein.pdb -l ligand.sdf -s -15.7 -17.7 8.1 -a genetic --physics-based` |
| **Physics-Based Scoring (GPU accelerated)** | `pandadock -p protein.pdb -l ligand.sdf -s -15.7 -17.7 8.1 -a genetic --physics-based --use-gpu` |

---

## 🧬 Monte Carlo Sampling + Docking

| Description | Command |
|:------------|:--------|
| **Monte Carlo Sampling + Genetic Optimization** | `pandadock -p protein.pdb -l ligand.sdf -s -15.7 -17.7 8.1 -a genetic --monte-carlo` |
| **Monte Carlo with Physics-Based Scoring** | `pandadock -p protein.pdb -l ligand.sdf -s -15.7 -17.7 8.1 -a genetic --monte-carlo --physics-based` |

---

## 🧊 PANDADOCK Algorithm (Simulated Annealing + Final Minimization)

| Description | Command |
|:------------|:--------|
| **PANDADOCK Docking (default parameters)** | `pandadock -p protein.pdb -l ligand.sdf -s -15.7 -17.7 8.1 -a pandadock` |
| **PANDADOCK Docking with Physics-Based Scoring** | `pandadock -p protein.pdb -l ligand.sdf -s -15.7 -17.7 8.1 -a pandadock --physics-based` |
| **PANDADOCK using Grid-Based Energies** | `pandadock -p protein.pdb -l ligand.sdf -s -15.7 -17.7 8.1 -a pandadock --use-grid` |
| **Fine-tune PANDADOCK MD Parameters** | `pandadock -p protein.pdb -l ligand.sdf -s -15.7 -17.7 8.1 -a pandadock --high-temp 500 --target-temp 300 --cooling-factor 0.95` |

---

## ⚡ Hardware Acceleration Examples (GPU/CPU Hybrid)

| Description | Command |
|:------------|:--------|
| **GPU-Only Docking** | `pandadock -p protein.pdb -l ligand.sdf -s -15.7 -17.7 8.1 -a genetic --use-gpu` |
| **Hybrid CPU/GPU Workload (80% GPU)** | `pandadock -p protein.pdb -l ligand.sdf -s -15.7 -17.7 8.1 -a genetic --use-gpu --workload-balance 0.8` |
| **Float64 Precision on GPU** | `pandadock -p protein.pdb -l ligand.sdf -s -15.7 -17.7 8.1 --use-gpu --gpu-precision float64` |

---

## 🧠 Advanced Search Algorithms

| Description | Command |
|:------------|:--------|
| **Gradient-Based Search** | `pandadock -p protein.pdb -l ligand.sdf -s -15.7 -17.7 8.1 --advanced-search gradient` |
| **Replica Exchange Docking** | `pandadock -p protein.pdb -l ligand.sdf -s -15.7 -17.7 8.1 --advanced-search replica-exchange` |
| **ML-Guided Docking (Random Forest Surrogate)** | `pandadock -p protein.pdb -l ligand.sdf -s -15.7 -17.7 8.1 --advanced-search ml-guided --surrogate-model rf` |
| **Fragment-Based Docking** | `pandadock -p protein.pdb -l ligand.sdf -s -15.7 -17.7 8.1 --advanced-search fragment-based` |
| **Hybrid GA + L-BFGS Docking** | `pandadock -p protein.pdb -l ligand.sdf -s -15.7 -17.7 8.1 --advanced-search hybrid` |

---

## 📑 Reporting Options (Recommended after docking)

| Description | Command |
|:------------|:--------|
| **Generate Detailed HTML Report** | `pandadock -p protein.pdb -l ligand.sdf -s -15.7 -17.7 8.1 -a genetic --generate-analysis-report --analysis-report-format html` |
| **Save Report as Text/CSV/JSON** | `pandadock -p protein.pdb -l ligand.sdf -s -15.7 -17.7 8.1 -a genetic --report-format all` |

---


# 🚀 PandaDock Docking Command Examples

<details>
<summary>🎯 <strong>Basic Docking (Random / Genetic / Auto)</strong></summary>

```bash
# Random Search Docking
pandadock -p protein.pdb -l ligand.sdf -s -15.7 -17.7 8.1 -a random -i 100

# Genetic Algorithm Docking (Default)
pandadock -p protein.pdb -l ligand.sdf -s -15.7 -17.7 8.1 -a genetic -i 100

# Auto Algorithm Selection
pandadock -p protein.pdb -l ligand.sdf -s -15.7 -17.7 8.1 --auto-algorithm
```
</details>

---

<details>
<summary>🧪 <strong>Physics-Based Docking (High Accuracy)</strong></summary>

```bash
# CPU Physics-Based Docking
pandadock -p protein.pdb -l ligand.sdf -s -15.7 -17.7 8.1 -a genetic --physics-based

# GPU-Accelerated Physics-Based Docking
pandadock -p protein.pdb -l ligand.sdf -s -15.7 -17.7 8.1 -a genetic --physics-based --use-gpu
```
</details>

---

<details>
<summary>🧬 <strong>Monte Carlo Sampling + Docking</strong></summary>

```bash
# Monte Carlo Sampling with Genetic Optimization
pandadock -p protein.pdb -l ligand.sdf -s -15.7 -17.7 8.1 -a genetic --monte-carlo

# Monte Carlo + Physics-Based Scoring
pandadock -p protein.pdb -l ligand.sdf -s -15.7 -17.7 8.1 -a genetic --monte-carlo --physics-based
```
</details>

---

<details>
<summary>🧊 <strong>PANDADOCK Algorithm (Simulated Annealing)</strong></summary>

```bash
# PANDADOCK Docking (Default Parameters)
pandadock -p protein.pdb -l ligand.sdf -s -15.7 -17.7 8.1 -a pandadock

# PANDADOCK + Physics-Based Scoring
pandadock -p protein.pdb -l ligand.sdf -s -15.7 -17.7 8.1 -a pandadock --physics-based

# PANDADOCK Using Grid-Based Energies
pandadock -p protein.pdb -l ligand.sdf -s -15.7 -17.7 8.1 -a pandadock --use-grid

# Fine-Tune PANDADOCK MD Parameters
pandadock -p protein.pdb -l ligand.sdf -s -15.7 -17.7 8.1 -a pandadock --high-temp 500 --target-temp 300 --cooling-factor 0.95
```
</details>

---

<details>
<summary>⚡ <strong>Hardware Acceleration (CPU/GPU Hybrid)</strong></summary>

```bash
# GPU-Only Docking
pandadock -p protein.pdb -l ligand.sdf -s -15.7 -17.7 8.1 -a genetic --use-gpu

# CPU/GPU Hybrid Workload Balancing
pandadock -p protein.pdb -l ligand.sdf -s -15.7 -17.7 8.1 -a genetic --use-gpu --workload-balance 0.8

# GPU Float64 Precision Docking
pandadock -p protein.pdb -l ligand.sdf -s -15.7 -17.7 8.1 --use-gpu --gpu-precision float64
```
</details>

---

<details>
<summary>🧠 <strong>Advanced Search Algorithms (Gradient / Replica Exchange / ML-Guided / Fragment-Based / Hybrid)</strong></summary>

```bash
# Gradient-Based Search
pandadock -p protein.pdb -l ligand.sdf -s -15.7 -17.7 8.1 --advanced-search gradient

# Replica Exchange Monte Carlo
pandadock -p protein.pdb -l ligand.sdf -s -15.7 -17.7 8.1 --advanced-search replica-exchange

# ML-Guided Docking (Random Forest Surrogate)
pandadock -p protein.pdb -l ligand.sdf -s -15.7 -17.7 8.1 --advanced-search ml-guided --surrogate-model rf

# Fragment-Based Docking
pandadock -p protein.pdb -l ligand.sdf -s -15.7 -17.7 8.1 --advanced-search fragment-based

# Hybrid Genetic Algorithm + L-BFGS Docking
pandadock -p protein.pdb -l ligand.sdf -s -15.7 -17.7 8.1 --advanced-search hybrid
```
</details>

---

<details>
<summary>📑 <strong>Optional Reporting & Analysis</strong></summary>

```bash
# Generate Detailed HTML Analysis Report
pandadock -p protein.pdb -l ligand.sdf -s -15.7 -17.7 8.1 --generate-analysis-report --analysis-report-format html

# Save Results as Text, CSV, and JSON
pandadock -p protein.pdb -l ligand.sdf -s -15.7 -17.7 8.1 --report-format all
```
</details>

---

# 🏁 **Recommended Best Practices**

✅ Always add `--prepare-molecules` to automatically protonate and minimize molecules.  
✅ Use `--enhanced-scoring` or `--physics-based` for better electrostatics (even in non-physics mode).  
✅ Use `--local-opt` to refine poses with local energy minimization.  
✅ Add `--generate-analysis-report` for detailed visual inspection after docking.

---


## PANDADOCK Algorithm

PANDADOCK is a CHARMm-inspired docking algorithm that combines conformational sampling with molecular dynamics simulation. The protocol includes:

1. **Conformer Generation**: Multiple ligand conformers are generated by systematic rotation of rotatable bonds
2. **Random Orientation**: Each conformer is positioned in the binding site with multiple initial orientations
3. **Simulated Annealing**: Poses are refined using simulated annealing molecular dynamics
4. **Final Minimization**: A final energy minimization optimizes the poses
5. **Scoring and Ranking**: All poses are scored and ranked using the physics-based scoring function

## 🛠️ Command Line Options

> (**Over 70+ options for full control!**)

| Category | Main Flags |
|:--|:--|
| Basic | `-p, -l, -o, -a, -s, -r` |
| Physics | `--physics-based, --mmff-minimization` |
| Hardware | `--use-gpu, --gpu-id, --cpu-workers, --workload-balance` |
| Flexibility | `--flex-residues`, `--auto-flex`, `--max-flex-residues` |
| Search Methods | `--random`, `--genetic`, `--monte-carlo`, `--pandadock`, `--advanced-search` |
| Advanced | `--gradient-step`, `--n-replicas`, `--surrogate-model`, `--fragment-min-size` |
| Reporting | `--report-format`, `--detailed-energy`, `--generate-analysis-report` |
| Clustering | `--cluster-poses`, `--rmsd-cutoff` |
| Validation | `--reference`, `--exact-alignment` |

[▶️ See full list below](#command-line-arguments)

---
### Basic Options
| Argument | Description |
|----------|-------------|
| `-h, --help` | Show help message and exit |
| `-p, --protein PROTEIN` | Path to protein PDB file |
| `-l, --ligand LIGAND` | Path to ligand MOL/SDF file |
| `-o, --output OUTPUT` | Output directory for docking results |
| `-a, --algorithm {random,genetic,pandadock}` | Docking algorithm to use (default: genetic) |
| `-i, --iterations ITERATIONS` | Number of iterations/generations (default: 100) |
| `-s, --site X Y Z` | Active site center coordinates |
| `-r, --radius RADIUS` | Active site radius in Angstroms (default: 10.0) |
| `--detect-pockets` | Automatically detect binding pockets |
| `--grid-spacing GRID_SPACING` | Grid spacing in Å for spherical grid sampling (default: 0.375 Å) |
| `--grid-radius GRID_RADIUS` | Grid radius in Å around the binding site for spherical sampling (default: 10.0 Å) |
| `--fast-mode` | Run with minimal enhancements for quick results |
| `--enhanced` | Use enhanced algorithms for more accurate (but slower) results |
| `--enhanced-scoring` | Use enhanced scoring function with electrostatics |
| `--prepare-molecules` | Prepare protein and ligand before docking (recommended) |
| `--population-size POPULATION_SIZE` | Population size for genetic algorithm (default: 100) |
| `--exhaustiveness EXHAUSTIVENESS` | Number of independent docking runs (default: 1) |
| `--local-opt` | Enable local optimization on top poses (default: disabled) |
| `--ph PH` | pH for protein preparation (default: 7.4) |
| `--physics-based` | Use full physics-based scoring (very slow but most accurate) |
| `--mmff-minimization` | Use MMFF94 force field minimization (requires RDKit) |
| `--monte-carlo` | Use Monte Carlo sampling instead of genetic algorithm |
| `--mc-steps MC_STEPS` | Number of Monte Carlo steps (default: 1000) |
| `--temperature TEMPERATURE` | Temperature for Monte Carlo simulation in Kelvin (default: 300K) |
| `--tethered-docking` | Use tethered scoring with reference structure |
| `--tether-weight TETHER_WEIGHT` | Weight for tethered scoring (higher = stronger tethering) |
| `--reference REFERENCE` | Reference ligand structure for validation |
| `--exact-alignment` | Align docked pose exactly to reference structure |
| `--auto-algorithm` | Automatically select the best docking algorithm based on your system |
| `--flex-residues FLEX_RESIDUES [FLEX_RESIDUES ...]` | Specify flexible residue IDs (e.g., A_42 B_57) |
| `--max-flex-bonds MAX_FLEX_BONDS` | Maximum rotatable bonds per residue (default: 3) |
| `--auto-flex` | Automatically detect flexible residues in the binding site |
| `--max-flex-residues MAX_FLEX_RESIDUES` | Maximum number of flexible residues to detect in auto mode (default: 5) |

### PandaDock Algorithm Options
| Argument | Description |
|----------|-------------|
| `--high-temp HIGH_TEMP` | High temperature for pandadock MD simulations (K) |
| `--target-temp TARGET_TEMP` | Target temperature for pandadock cooling (K) |
| `--num-conformers NUM_CONFORMERS` | Number of ligand conformers to generate in pandadock |
| `--num-orientations NUM_ORIENTATIONS` | Number of orientations to try for each conformer in pandadock |
| `--md-steps MD_STEPS` | Number of MD steps for simulated annealing in pandadock |
| `--minimize-steps MINIMIZE_STEPS` | Number of minimization steps for final refinement in pandadock |
| `--use-grid` | Use grid-based energy calculations in pandadock |
| `--cooling-factor COOLING_FACTOR` | Cooling factor for simulated annealing (applies to PANDADOCK and Monte Carlo) |

### Reporting Options
| Argument | Description |
|----------|-------------|
| `--report-format {text,csv,json,html,all}` | Report format (default: all) |
| `--report-name REPORT_NAME` | Custom name for the report files |
| `--detailed-energy` | Include detailed energy component breakdown in reports |
| `--skip-plots` | Skip generating plots for reports |

### Hardware Acceleration
| Argument | Description |
|----------|-------------|
| `--use-gpu` | Use GPU acceleration if available |
| `--gpu-id GPU_ID` | GPU device ID to use (default: 0) |
| `--gpu-precision {float32,float64}` | Numerical precision for GPU calculations (default: float32) |
| `--cpu-workers CPU_WORKERS` | Number of CPU workers for parallel processing (default: all cores) |
| `--cpu-affinity` | Set CPU affinity for better performance |
| `--workload-balance WORKLOAD_BALANCE` | GPU/CPU workload balance (0.0-1.0, higher values assign more work to GPU) |
| `--auto-tune` | Automatically tune hardware parameters for best performance |

### Advanced Search Algorithms
| Argument | Description |
|----------|-------------|
| `--advanced-search {gradient,replica-exchange,ml-guided,fragment-based,hybrid}` | Advanced search algorithm to use |
| `--gradient-step GRADIENT_STEP` | Step size for gradient calculation in gradient-based search |
| `--convergence-threshold CONVERGENCE_THRESHOLD` | Convergence threshold for gradient-based search |
| `--n-replicas N_REPLICAS` | Number of replicas for replica exchange |
| `--replica-temperatures REPLICA_TEMPERATURES [REPLICA_TEMPERATURES ...]` | Temperatures for replicas (e.g., 300 400 500 600) |
| `--exchange-steps EXCHANGE_STEPS` | Number of exchange attempts in replica exchange |
| `--surrogate-model {rf,gp,nn}` | Surrogate model type for ML-guided search |
| `--exploitation-factor EXPLOITATION_FACTOR` | Exploitation vs exploration balance (0-1) for ML-guided search |
| `--fragment-min-size FRAGMENT_MIN_SIZE` | Minimum fragment size for fragment-based docking |
| `--growth-steps GROWTH_STEPS` | Number of fragment growth steps |
| `--ga-iterations GA_ITERATIONS` | Genetic algorithm iterations in hybrid search |
| `--lbfgs-iterations LBFGS_ITERATIONS` | L-BFGS iterations in hybrid search |
| `--top-n-for-local TOP_N_FOR_LOCAL` | Top N poses to optimize with L-BFGS in hybrid search |

### Pose Clustering and Analysis
| Argument | Description |
|----------|-------------|
| `--cluster-poses` | Perform clustering of docking poses |
| `--clustering-method {hierarchical,dbscan}` | Method for clustering poses |
| `--rmsd-cutoff RMSD_CUTOFF` | RMSD cutoff for pose clustering |
| `--analyze-interactions` | Generate interaction fingerprints and analysis |
| `--interaction-types {hbond,hydrophobic,ionic,aromatic,halogen}` | Interaction types to include in analysis |
| `--classify-modes` | Classify binding modes of docking poses |
| `--discover-modes` | Automatically discover binding modes from results |
| `--n-modes N_MODES` | Number of binding modes to discover |
| `--energy-decomposition` | Perform energy decomposition analysis |
| `--per-residue-energy` | Calculate per-residue energy contributions |
| `--generate-analysis-report` | Generate comprehensive docking report |
| `--analysis-report-format {html,pdf,txt}` | Format for analysis report |
| `--analysis-report-sections {summary,clusters,interactions,energetics}` | Sections to include in the analysis report |

## Physics-Based Features

PandaDock includes physics-based molecular modeling capabilities that significantly enhance the accuracy of docking results:

- **MMFF94 Force Field Minimization**: Full molecular mechanics energy minimization
- **Enhanced Electrostatics**: Poisson-Boltzmann inspired model with distance-dependent dielectric
- **Implicit Solvation (GB/SA)**: Generalized Born model with surface area-based nonpolar term
- **Monte Carlo Sampling**: Enhanced conformational sampling with Metropolis criterion
- **Physics-Based Scoring**: Complete MM-GBSA inspired scoring function

## 🧠 Advanced Features

- **Physics-Based Energy Decomposition**:
  - VDW, Electrostatics, H-Bond, Desolvation, Hydrophobic, Clash, Entropy
- **Replica-Exchange Docking** (Multiple temperatures)
- **Gradient Descent (L-BFGS-B)** Local refinement
- **ML-Guided Docking** (Experimental)
- **Rotamer sampling for flexible residues**
- **Tethered docking** (if a reference ligand is provided)

---

## Python API Example

Yes! You can absolutely showcase **multiple PandaDock Python APIs** in your README or docs for different use cases.

Here’s a well-structured **section** with **three API blocks** you can include:

---

## 🐼 Python API Examples

### 🔬 Basic Docking with Genetic Algorithm

```python
from pandadock.protein import Protein
from pandadock.ligand import Ligand
from pandadock.unified_scoring import EnhancedScoringFunction
from pandadock.search import GeneticAlgorithm
from pandadock.utils import save_docking_results

protein = Protein("protein.pdb")
ligand = Ligand("ligand.sdf")
protein.define_active_site([10.0, 20.0, 30.0], 12.0)

scoring = EnhancedScoringFunction()
search = GeneticAlgorithm(scoring, max_iterations=1000, population_size=100)

results = search.search(protein, ligand)
save_docking_results(results, "docking_results")
```

---

### ⚗️ Physics-Based Scoring with PANDADOCK Algorithm

```python
from pandadock import Protein, Ligand
from pandadock.unified_scoring import PhysicsScoringFunction
from pandadock.pandadock import PandaDockSearch

protein = Protein("protein.pdb")
ligand = Ligand("ligand.sdf")
protein.auto_detect_pocket()  # Auto-detect binding pocket

scoring = PhysicsScoringFunction()
search = PandaDockSearch(scoring, md_steps=100, minimize_steps=50)

results = search.search(protein, ligand)
```

---

### 🧬 Flexible Residue Docking (Auto Flex)

```python
from pandadock import Protein, Ligand
from pandadock.unified_scoring import EnhancedScoringFunction
from pandadock.search import GeneticAlgorithm

protein = Protein("protein.pdb")
ligand = Ligand("ligand.sdf")
protein.define_active_site([35.0, 22.0, 18.0], 10.0)
protein.auto_define_flexible_residues(max_residues=5)

scoring = EnhancedScoringFunction()
search = GeneticAlgorithm(scoring, max_iterations=500)

results = search.search(protein, ligand)
```

---

## 📚 Batch Screening API Example

```python
from pandadock.batch_screening import run
from pandadock.utils import setup_logging

# Define batch screening configuration
config = {
    "protein": "protein.pdb",                     # Path to protein file
    "ligand_library": "ligands/",                  # Folder containing many ligands (MOL/SDF)
    "output_dir": "batch_screening_results",       # Output folder
    "screening_params": {
        "hardware": {
            "use_gpu": True,                       # Use GPU if available
            "gpu_id": 0,
            "gpu_precision": "float32"
        },
        "scoring": "enhanced",                      # Use 'basic', 'enhanced', or 'physics-based'
        "algorithm": "genetic",                     # Docking algorithm: 'genetic', 'random', 'pandadock', etc.
        "iterations": 500,
        "population_size": 50,
        "detect_pockets": True,                     # Auto detect binding site
        "prepare_molecules": True                   # Auto prepare ligands
    }
}

# Setup logger (optional)
setup_logging("batch_screening_results")

# Run batch screening
results = run(config)

# Access results
for ligand_name, info in results.items():
    print(f"Ligand: {ligand_name}, Best Score: {info['score']:.2f}, Runtime: {info['runtime']:.2f} sec")
```

---

## Acknowledgments

PandaDock incorporates concepts from several established molecular docking and computational chemistry packages:

- CDOCKER (Wu et al. 2003)
- AutoDock Vina (Trott & Olson, 2010)
- DOCK (Allen et al., 2015)
- RDKit (http://www.rdkit.org)

---
## 📜 License

**MIT License** — free for academic and commercial use.

---

## 📝 Citation

If you use **PandaDock** in your research:

> **Pritam Kumar Panda** (2025). *PandaDock: Python-Based Molecular Docking*. GitHub.  
> [https://github.com/pritampanda15/PandaDock](https://github.com/pritampanda15/PandaDock)

---

## 📬 Contact

- GitHub: [@pritampanda15](https://github.com/pritampanda15)
- Email: [pritam@stanford.edu](mailto:pritam@stanford.edu)
- Issue Tracker: [Open an Issue](https://github.com/pritampanda15/PandaDock/issues)

---

## 🛡️ Disclaimer

> PandaDock is intended for research purposes.  
> Always verify docking predictions through experimental validation.

**Dock Smarter. Discover Faster**
