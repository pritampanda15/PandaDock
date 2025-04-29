# 🐼 PandaDock

        ██████╗  █████╗ ███╗   ██╗██████╗  █████╗ ██████╗  ██████╗  ██████╗██╗  ██╗
        ██╔══██╗██╔══██╗████╗  ██║██╔══██╗██╔══██╗██╔══██╗██╔═══██╗██╔════╝██║ ██╔╝
        ██████╔╝███████║██╔██╗ ██║██║  ██║███████║██║  ██║██║   ██║██║     █████╔╝ 
        ██╔═══╝ ██╔══██║██║╚██╗██║██║  ██║██╔══██║██║  ██║██║   ██║██║     ██╔═██╗ 
        ██║     ██║  ██║██║ ╚████║██████╔╝██║  ██║██████╔╝╚██████╔╝╚██████╗██║  ██╗
        ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═══╝╚═════╝ ╚═╝  ╚═╝╚═════╝  ╚═════╝  ╚═════╝╚═╝  ╚═╝                                                                        
        
PandaDock Molecular Docking Suite 🚀
Version: 2.0.0

**Python-based Molecular Docking Platform for Drug Discovery, Bioinformatics, and Computational Chemistry**.

![PandaDock Logo](https://github.com/pritampanda15/PandaDock/blob/main/logo/pandadock-logo.svg)
[![PyPI Version](https://img.shields.io/pypi/v/pandadock.svg)](https://pypi.org/project/pandadock/)
[![License](https://img.shields.io/github/license/pritampanda15/PandaDock)](https://github.com/pritampanda15/PandaDock/blob/main/LICENSE)
[![Stars](https://img.shields.io/github/stars/pritampanda15/PandaDock?style=social)](https://github.com/pritampanda15/PandaDock/stargazers)
[![Issues](https://img.shields.io/github/issues/pritampanda15/PandaDock)](https://github.com/pritampanda15/PandaDock/issues)
[![GitHub forks](https://img.shields.io/github/forks/pritampanda15/PandaDock?style=social)](https://github.com/pritampanda15/PandaDock/network/members)
[![Downloads](https://static.pepy.tech/badge/pandadock)](https://pepy.tech/project/pandadock)

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
- 🎯 **Binding Site Definition**: Manual or automatic pocket detection
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
- 🛠️ **Extensible Python API** for custom workflows and integrations

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
# PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# OR CuPy
pip install cupy-cuda11x
```

## 🔥 Quick Start

```bash
# Simple run (default Genetic Algorithm)
pandadock -p protein.pdb -l ligand.sdf -o results/

# Physics-Based docking (PANDADOCK + MMGBSA scoring)
pandadock -p protein.pdb -l ligand.sdf -a pandadock --physics-based

# Use GPU
pandadock -p protein.pdb -l ligand.sdf --use-gpu --physics-based

# Automatic algorithm selection
pandadock -p protein.pdb -l ligand.sdf --auto-algorithm
```

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

## Python API Example

```python
from pandadock.protein import Protein
from pandadock.ligand import Ligand
from pandadock.unified_scoring import EnhancedScoringFunction
from pandadock.search import GeneticAlgorithm
from pandadock.utils import save_docking_results

# Load protein and ligand
protein = Protein("protein.pdb")
ligand = Ligand("ligand.sdf")

# Define active site (optional)
protein.define_active_site([10.0, 20.0, 30.0], 12.0)

# Create scoring function
scoring_function = EnhancedScoringFunction()

# Create search algorithm
search_algorithm = GeneticAlgorithm(
    scoring_function, 
    max_iterations=1000,
    population_size=100
)

# Perform docking
results = search_algorithm.search(protein, ligand)

# Save results
save_docking_results(results, "docking_results")
```

## 🧠 Advanced Features

- **Physics-Based Energy Decomposition**:
  - VDW, Electrostatics, H-Bond, Desolvation, Hydrophobic, Clash, Entropy
- **Replica-Exchange Docking** (Multiple temperatures)
- **Gradient Descent (L-BFGS-B)** Local refinement
- **ML-Guided Docking** (Experimental)
- **Rotamer sampling for flexible residues**
- **Tethered docking** (if a reference ligand is provided)

---

## 📊 Result Outputs

| Output | Description |
|:-------|:------------|
| `poses/` | Top-ranked ligand poses (PDB files) |
| `plots/` | Score distributions, RMSD plots |
| `validation_report.txt` | RMSD vs reference ligand |
| `energy_breakdown.txt` | Energy components |
| `report.html` | Interactive full docking report |

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
╔════════════════════════════════════════════╗
║   🐼  Dock Smarter. Discover Faster.       ║
╚════════════════════════════════════════════╝