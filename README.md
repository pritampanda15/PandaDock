# PandaDock: Python-based molecular docking software for bioinformatics and drug design with physics-based approach

PandaDock is a Python-based molecular docking package designed for bioinformatics and drug design applications. It provides a command-line interface for molecular docking simulations, with support for multiple scoring functions and search algorithms.

![PandaDock Logo](logo/pandadock-logo.svg)

## Features

- 🔬 PDB and MOL/SDF file parsing
- 🎯 Active site definition and automatic pocket detection
- 📊 Multiple scoring functions:
  - Basic: van der Waals and hydrogen bonds
  - Enhanced: electrostatics, desolvation, and hydrophobic interactions
  - Physics-based: MM-GBSA inspired scoring with full molecular mechanics
- 🧬 Advanced search algorithms:
  - Random search
  - Genetic algorithm with local optimization
  - Monte Carlo sampling with simulated annealing
- 🔄 Molecule preparation with hydrogen addition and energy minimization
- Flexible residues docking
- 📏 Validation against reference structures with RMSD calculations
- 📈 Comprehensive results visualization and analysis
- ⚡ Hardware acceleration:
  - Multi-core CPU parallelization
  - GPU acceleration for scoring functions
  - Hybrid CPU/GPU workload balancing

## Installation

### Prerequisites

PandaDock requires Python 3.6+ and the following dependencies:
- NumPy
- SciPy
- Matplotlib
- scikit-learn
- RDKit (optional but recommended)
- Open Babel (optional for additional molecule preparation)
- PyTorch or CuPy (optional for GPU acceleration)

### Install from GitHub

```bash
# Clone the repository
git clone https://github.com/pritampanda15/PandaDock.git

# Navigate to the directory
cd PandaDock

# Install the package
pip install -e .
```
### Install via pip

```bash
# Create virtual environment
python -m venv .venv

#Activate the environment
source .venv/bin/activate

#install rdkit (Recommended)
pip install rdkit #in the virual environment 

#install torch for gpu
pip install torch 

#install openbabel
pip install openbabel

# Then install PandaDock
pip install pandadock
```
```bash
conda install -c conda-forge rdkit
```
### Install via pip in conda

```bash
# First install RDKit with conda
conda install -c conda-forge rdkit

# Then install PandaDock
pip install pandadock
```
### Install Open Babel (Optional)

For advanced molecule preparation:

```bash
conda install -c conda-forge openbabel
```

### Install GPU Acceleration (Optional)

For GPU-accelerated docking:

```bash
# PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# OR CuPy
pip install cupy-cuda11x
```
### Option 2: Use extras_require

```python
extras_require={
    "gpu": ["torch>=1.9.0"],
    "full": ["torch>=1.9.0"],  # Users can install with: pip install pandadock[full]
},
```
## Version 1.0 Latest Improvements (UPDATE)
```bash
## New Core Features

1. **Reference-Guided Docking**
   - `--reference` flag to guide docking based on known ligand poses
   - `--exact-alignment` option for direct superimposition with reference structures
   - `--no-local-optimization` to preserve exact alignment without refinement

2. **Flexible Residue Handling**
   - `--flex-residues` option for manual selection of flexible protein side chains
   - `--auto-flex` for automatic detection of flexible residues in binding sites
   - `--max-flex-residues` to control number of automatically detected flexible residues
   - `--max-flex-bonds` to limit rotatable bonds per residue

3. **Enhanced Search Algorithms**
   - Improved rigid docking with multi-stage optimization
   - Aggressive atom-by-atom refinement for better clash resolution
   - Advanced clash detection and energy minimization

4. **Hardware Acceleration**
   - GPU acceleration with `--use-gpu` flag
   - Hybrid CPU/GPU workload balancing
   - Automatic hardware tuning with `--auto-tune`

## Refinement Techniques

1. **Clash Resolution Methods**
   - Gentle clash relief to maintain structural integrity
   - Atom-by-atom adjustment for problematic interactions
   - Systematic directional shifts to escape local minima

2. **Local Optimization Improvements**
   - Multi-stage optimization with decreasing step sizes
   - Enhanced rotational sampling with diagonal axes
   - Retreat-and-approach strategy for complex binding sites

## Usage Combinations

1. **For Maximum Accuracy**
   ```
   pandadock -p protein.pdb -l ligand.sdf -s X Y Z --enhanced --enhanced-scoring --local-opt --prepare-molecules --use-gpu --auto-tune
   ```

2. **For Reference-Based Docking**
   ```
   pandadock -p protein.pdb -l ligand.sdf -s X Y Z --reference ref_ligand.sdf --exact-alignment --no-local-optimization
   ```

3. **For Flexible Side Chain Modeling**
   ```
   pandadock -p protein.pdb -l ligand.sdf -s X Y Z --enhanced --auto-flex --max-flex-residues 5 --max-flex-bonds 4
   ```

4. **For Multiple Independent Runs**
   ```
   pandadock -p protein.pdb -l ligand.sdf -s X Y Z --enhanced --exhaustiveness 8 --population-size 200
   ```

## Output Enhancements

1. **Complex Visualization**
   - Generates complex PDB files showing both ligand and flexible residues
   - Improved score plotting with detailed statistics

2. **Detailed Reporting**
   - Comprehensive docking results text files
   - RMSD calculations for reference-based docking
   - Energy breakdown for physics-based scoring

These enhancements make PandaDock a more versatile molecular docking platform with advanced capabilities for structure-based drug design, virtual screening, and computational chemistry research.

```
### Command Line Interface

***Note: Remeber to change the --iteration or -i value to smaller number or else it will take hours to optimize it if you set it for blind docking using CPU<br> ***

```bash
1. Quick and Simple Docking
pandadock -p protein.pdb -l ligand.sdf -s X Y Z --fast-mode #This provides fast results with minimal computation time.

2. Standard Accurate Docking
pandadock -p protein.pdb -l ligand.sdf -s X Y Z --enhanced-scoring --local-opt --prepare-molecules #This is more accurate than the quick mode but takes more time.

3. High-Accuracy Docking with Hardware Acceleration
pandadock -p protein.pdb -l ligand.sdf -s X Y Z --enhanced --enhanced-scoring --local-opt --prepare-molecules --use-gpu --auto-tune  #This combination leverages GPU acceleration with enhanced algorithms for accuracy.

4. Reference-Guided Docking for Known Binding Modes
pandadock -p protein.pdb -l ligand.sdf -s X Y Z --reference ref_ligand.sdf --enhanced-scoring --local-opt #Use when you have a reference crystal structure to guide the docking.

5. Exact Alignment to Reference Structure
pandadock -p protein.pdb -l ligand.sdf -s X Y Z --reference ref_ligand.sdf --exact-alignment --no-local-optimization #For exact superimposition with a reference ligand without any adjustments.

6. Exact Alignment with Refinement
pandadock -p protein.pdb -l ligand.sdf -s X Y Z --reference ref_ligand.sdf --exact-alignment #For exact superimposition with a reference ligand plus refinement for better scoring.

7. Flexible Side Chains in Binding Site
pandadock -p protein.pdb -l ligand.sdf -s X Y Z --enhanced --auto-flex --local-opt #Automatically detects and allows flexibility in binding site residues.

8. Manual Flexible Side Chains
pandadock -p protein.pdb -l ligand.sdf -s X Y Z --flex-residues A_123 B_45 --max-flex-bonds 4 #Manually specify which residues should be flexible during docking.

9. Physics-Based Docking
pandadock -p protein.pdb -l ligand.sdf -s X Y Z --physics-based --mmff-minimization --local-opt #Use more accurate physical models for scoring and optimization.

10. Monte Carlo Sampling (Alternative to Genetic Algorithm)
pandadock -p protein.pdb -l ligand.sdf -s X Y Z --monte-carlo --mc-steps 2000 --temperature 300 #Use Monte Carlo sampling instead of genetic algorithm for different search behavior.

11. Exhaustive Search for Better Pose Diversity
pandadock -p protein.pdb -l ligand.sdf -s X Y Z --enhanced --exhaustiveness 8 --population-size 200 #Run multiple independent docking runs for better exploration of pose space.

12. Automatic Binding Site Detection
pandadock -p protein.pdb -l ligand.sdf --detect-pockets --enhanced-scoring --local-opt #Automatically detect binding pockets instead of specifying coordinates.

13. Maximum Performance on CPU
pandadock -p protein.pdb -l ligand.sdf -s X Y Z --enhanced --cpu-workers 8 --cpu-affinity #Optimize for CPU performance with multiple cores and CPU affinity.

14. Maximum Performance on GPU
pandadock -p protein.pdb -l ligand.sdf -s X Y Z --enhanced --use-gpu --gpu-id 0 --gpu-precision float32 --workload-balance 0.9 #Maximize GPU usage for fastest performance on compatible hardware.

15. Hybrid CPU/GPU with Auto-Tuning
pandadock -p protein.pdb -l ligand.sdf -s X Y Z --enhanced --use-gpu --auto-tune --cpu-workers 4 #Let PandaDock automatically balance CPU and GPU workloads for optimal performance.



```

### Running Modes

PandaDock offers flexible running modes to balance between speed and accuracy:

- **Fast Mode**: `--fast-mode` - Quick docking with minimal enhancements
- **Default Mode**: Basic docking with standard scoring
- **Enhanced Mode**: Use any combination of enhancement flags:
  - `--enhanced-scoring` - More accurate scoring with electrostatics
  - `--local-opt` - Fine-tune top poses for better results
  - `--exhaustiveness 5` - Run multiple independent searches
  - `--prepare-molecules` - Optimize molecule geometry before docking
  - `--flex-residues` - Use flexible residues

### Physics-Based Docking

```bash
# Use MMFF94 minimization (requires RDKit)
pandadock -p protein.pdb -l ligand.mol --mmff-minimization

# Enhanced electrostatics with solvation
pandadock -p protein.pdb -l ligand.mol --enhanced-scoring

# Full physics-based scoring (MM-GBSA inspired)
pandadock -p protein.pdb -l ligand.mol --physics-based

# Monte Carlo sampling with simulated annealing
pandadock -p protein.pdb -l ligand.mol --monte-carlo --mc-steps 2000 --temperature 300

# Combined approach for best accuracy
pandadock -p protein.pdb -l ligand.mol --physics-based --mmff-minimization --local-opt
```

### Hardware Acceleration

```bash
# Enable GPU acceleration (if available)
pandadock -p protein.pdb -l ligand.mol --use-gpu

# Specify CPU workers for parallelization
pandadock -p protein.pdb -l ligand.mol --cpu-workers 8

# Auto-tune hardware settings for optimal performance
pandadock -p protein.pdb -l ligand.mol --auto-tune

# Combine hardware acceleration with physics-based methods
pandadock -p protein.pdb -l ligand.mol --use-gpu --physics-based --mmff-minimization
```

### Python API

```python
from pandadock.protein import Protein
from pandadock.ligand import Ligand
from pandadock.scoring import CompositeScoringFunction, EnhancedScoringFunction
from pandadock.search import GeneticAlgorithm
from pandadock.utils import save_docking_results
from pandadock.preparation import prepare_protein, prepare_ligand

# Prepare molecules (optional)
prepared_protein = prepare_protein("protein.pdb", add_hydrogens=True, ph=7.4)
prepared_ligand = prepare_ligand("ligand.mol", minimize=True)

# Load protein and ligand
protein = Protein(prepared_protein)
ligand = Ligand(prepared_ligand)

# Define active site (optional)
protein.define_active_site([10.0, 20.0, 30.0], 12.0)

# Create scoring function (choose basic or enhanced)
scoring_function = EnhancedScoringFunction()  # or CompositeScoringFunction()

# Create search algorithm
search_algorithm = GeneticAlgorithm(
    scoring_function, 
    max_iterations=1000,
    population_size=100,
    mutation_rate=0.3
)

# Perform docking
results = search_algorithm.search(protein, ligand)

# Apply local optimization (optional)
optimized_results = []
for i, (pose, score) in enumerate(sorted(results, key=lambda x: x[1])[:5]):
    opt_pose, opt_score = search_algorithm._local_optimization(pose, protein)
    optimized_results.append((opt_pose, opt_score))

# Save results
save_docking_results(optimized_results, "docking_results")
```

## Command Line Arguments

| Argument | Description |
|----------|-------------|
| `-p, --protein` | Path to protein PDB file |
| `-l, --ligand` | Path to ligand MOL/SDF file |
| `-o, --output` | Output directory for results (default: docking_results) |
| `-a, --algorithm` | Docking algorithm: 'random' or 'genetic' (default: genetic) |
| `-i, --iterations` | Number of iterations/generations (default: 1000) |
| `-s, --site` | Active site center coordinates (x y z) |
| `-r, --radius` | Active site radius in Angstroms (default: 10.0) |
| `--detect-pockets` | Automatically detect binding pockets |
| `--flex-residues` | Specify flexible residue IDs (e.g., A_42 B_57) |
| `--max-flex-bonds` | Maximum rotatable bonds per residue (default: 3) |
| `--auto-flex`      |     Automatically detect flexible residues in the binding site |
|  `--max-flex-residues` | Maximum number of flexible residues to detect in auto mode (default: 5) |

### Enhancement Options
| Argument | Description |
|----------|-------------|
| `--fast-mode` | Run with minimal enhancements for quick results |
| `--enhanced` | Use enhanced algorithms for more accurate (but slower) results |
| `--enhanced-scoring` | Use enhanced scoring function with electrostatics |
| `--prepare-molecules` | Prepare protein and ligand before docking |
| `--population-size` | Population size for genetic algorithm (default: 100) |
| `--exhaustiveness` | Number of independent docking runs (default: 1) |
| `--local-opt` | Perform local optimization on top poses |
|  `--no-local-optimization` | Disable local optimization of poses (keep exact alignment)|
| `--ph` | pH for protein preparation (default: 7.4) |
| `--reference` | Reference ligand structure for validation |

### Physics-Based Options
| Argument | Description |
|----------|-------------|
| `--physics-based` | Use full physics-based scoring (MM-GBSA inspired) |
| `--mmff-minimization` | Use MMFF94 force field minimization (requires RDKit) |
| `--monte-carlo` | Use Monte Carlo sampling instead of genetic algorithm |
| `--mc-steps` | Number of Monte Carlo steps (default: 1000) |
| `--temperature` | Temperature for Monte Carlo simulation in Kelvin (default: 300K) |

### Hardware Acceleration Options
| Argument | Description |
|----------|-------------|
| `--use-gpu` | Use GPU acceleration if available |
| `--gpu-id` | GPU device ID to use (default: 0) |
| `--gpu-precision` | Numerical precision for GPU: 'float32' or 'float64' (default: 'float32') |
| `--cpu-workers` | Number of CPU workers for parallel processing (default: all cores) |
| `--cpu-affinity` | Set CPU affinity for better performance |
| `--workload-balance` | GPU/CPU workload balance (0.0-1.0, higher values assign more work to GPU) |
| `--auto-tune` | Automatically tune hardware parameters for best performance |

```

## Physics-Based Features

PandaDock now includes physics-based molecular modeling capabilities that significantly enhance the accuracy of docking results. These advanced features provide more realistic molecular interactions based on established physical principles.

- **MMFF94 Force Field Minimization**: Full molecular mechanics energy minimization using the Merck Molecular Force Field.
- **Enhanced Electrostatics**: Poisson-Boltzmann inspired model with distance-dependent dielectric and salt screening effects.
- **Implicit Solvation (GB/SA)**: Generalized Born model with surface area-based nonpolar term to account for solvent effects.
- **Monte Carlo Sampling**: Enhanced conformational sampling with Metropolis criterion and simulated annealing capability.
- **Physics-Based Scoring**: Complete MM-GBSA inspired scoring that combines molecular mechanics with implicit solvation.

## How These Methods Compare to Commercial Software

PandaDock's physics-based features are inspired by approaches used in commercial packages like Discovery Studio:

| Feature | PandaDock Implementation | Commercial Equivalent |
|---------|--------------------------|----------------------|
| Force Field | MMFF94 via RDKit | CHARMM, CDOCKER |
| Electrostatics | Modified PB model | Poisson-Boltzmann solver |
| Solvation | Simplified GB/SA | MM-GBSA |
| Sampling | Monte Carlo with annealing | Simulated annealing |
| Minimization | Gradient-based optimization | Energy minimization |

## Hardware Acceleration

PandaDock implements hardware acceleration to significantly improve performance:

- **Multi-core CPU Parallelization**: Distributes workload across available CPU cores
- **GPU Acceleration**: Uses CUDA-capable GPUs via PyTorch or CuPy for compute-intensive calculations
- **Hybrid CPU/GPU Mode**: Intelligently balances workload between CPU and GPU resources
- **Auto-tuning**: Automatically detects and configures optimal hardware settings

### Performance Considerations

The physics-based methods are significantly more computationally intensive than standard docking approaches:

- **Standard docking**: Seconds to minutes
- **Enhanced scoring**: Minutes
- **MMFF minimization**: Minutes to tens of minutes
- **Physics-based scoring**: Tens of minutes to hours
- **Monte Carlo sampling**: Hours

GPU acceleration can provide significant speedups (5-20x) for scoring functions, especially when using physics-based methods.

## Understanding Results

PandaDock generates several output files in the specified directory:

1. **Docking poses (PDB files)**: Top-scoring ligand conformations
2. **Score plot (PNG)**: Visualization of score distribution
3. **Docking results (TXT)**: Detailed results with scores and statistics
4. **Validation report (TXT)**: RMSD analysis if reference structure provided

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use PandaDock in your research, please cite:

```
Pritam Kumar Panda. (2025). PandaDock: Python Molecular Docking. GitHub repository. https://github.com/pritampanda15/PandaDock
```

## Acknowledgments

- Thanks to me who crafted this after dealing with commercial software. 
- Inspiration from established docking software like CDOCKER, AutoDock, Vina, etc.
