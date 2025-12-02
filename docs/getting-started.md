# Getting Started with PandaDock

This guide will help you get up and running with PandaDock for molecular docking simulations.

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- CUDA toolkit (optional, for GPU acceleration)

### Required Dependencies

PandaDock automatically installs the following dependencies:

- **RDKit**: Molecular informatics toolkit
- **BioPython**: Biological sequence analysis
- **NumPy**: Numerical computing
- **SciPy**: Scientific computing
- **OpenMM**: Molecular dynamics engine (for advanced scoring)
- **PyTorch**: Machine learning framework (for ML features)

### Installation Methods

#### From PyPI (Recommended)

```bash
pip install pandadock
```

#### From Source

```bash
git clone https://github.com/pandadock/pandadock.git
cd pandadock
pip install -e .
```

#### With GPU Support

```bash
pip install pandadock[gpu]
```

### Verify Installation

```bash
pandadock-dock --version
pandadock-dock --help
```

## Basic Concepts

### Input Files

PandaDock requires:

1. **Receptor (Protein)**: PDB format
2. **Ligand**: SDF, MOL2, or PDB format
3. **Grid Configuration**: JSON format (optional)

### Output Files

PandaDock generates:

- **Docked Poses**: PDB files for each pose
- **Complex Structures**: Receptor-ligand complexes
- **Interaction Analysis**: JSON reports
- **Binding Affinities**: Energy calculations
- **Visualizations**: PNG plots and charts

## Your First Docking Simulation

### Step 1: Prepare Input Files

Ensure you have:
- `protein.pdb`: Your target protein structure
- `ligand.sdf`: Your small molecule ligand

### Step 2: Run Basic Docking

```bash
pandadock-dock -r protein.pdb -l ligand.sdf -o my_first_docking/
```

This command will:
1. Automatically detect binding sites
2. Generate grid boxes
3. Perform docking with the default algorithm
4. Analyze interactions
5. Generate visualizations

### Step 3: Examine Results

```bash
ls my_first_docking/
```

You'll find:
- `pose1.pdb`, `pose2.pdb`, etc.: Top-ranked poses
- `complex1.pdb`, `complex2.pdb`, etc.: Complete complexes
- `interaction_analysis.json`: Detailed interaction report
- `binding_affinities.png`: Energy distribution plot
- `docking_summary.json`: Complete docking results

## Common Workflows

### 1. Standard Protein-Ligand Docking

```bash
# Basic docking with default settings
pandadock-dock -r receptor.pdb -l ligand.sdf -o results/

# Specify algorithm and scoring function
pandadock-dock -r receptor.pdb -l ligand.sdf \
  -a enhanced_hierarchical_cpu \
  -s physics_based \
  -o results/

# Generate more poses
pandadock-dock -r receptor.pdb -l ligand.sdf -n 20 -o results/
```

### 2. High-Throughput Screening

```bash
# Fast screening mode
pandadock-dock -r receptor.pdb -l ligand_library.sdf \
  --fast \
  -a enhanced_hierarchical_cpu \
  -o screening_results/

# GPU-accelerated screening
pandadock-dock -r receptor.pdb -l ligand_library.sdf \
  -a cuda_monte_carlo \
  --gpu \
  --gpu-batch-size 1000 \
  -o gpu_screening/
```

### 3. Precise Docking with Known Binding Site

```bash
# Manual grid specification
pandadock-dock -r receptor.pdb -l ligand.sdf \
  --center 25.0 30.0 40.0 \
  --box 20 20 20 \
  -a genetic_algorithm_cpu \
  -s precision_score \
  -o precise_docking/
```

### 4. Flexible Receptor Docking

```bash
# Include receptor flexibility
pandadock-flex -r receptor.pdb -l ligand.sdf \
  --flexible-residues "GLU123,ASP456,TYR789" \
  -o flexible_docking/
```

### 5. Metal Coordination Docking

```bash
# Docking with metal constraints
pandadock-metal -r metalloprotein.pdb -l ligand.sdf \
  --metal-ions ZN,MG \
  -o metal_docking/
```

## Grid Box Configuration

### Automatic Detection

```bash
# Detect all possible binding sites
pandadock-gridbox detect -r protein.pdb -o cavities/

# Use detected cavity for docking
pandadock-dock -r protein.pdb -l ligand.sdf \
  --grid-config cavities/cavity_1.json \
  -o docking_cavity1/
```

### Manual Grid Definition

```bash
# Create custom grid around known site
pandadock-gridbox manual \
  --center 25.0 30.0 40.0 \
  --dimensions 20 20 20 \
  -o custom_grid.json

# Use custom grid
pandadock-dock -r protein.pdb -l ligand.sdf \
  --grid-config custom_grid.json \
  -o custom_docking/
```

### Ligand-Based Grid

```bash
# Generate grid based on reference ligand
pandadock-gridbox ligand-based \
  -r protein.pdb \
  -l reference_ligand.sdf \
  --expansion 5.0 \
  -o ligand_based_grid.json
```

## Algorithm Selection Guide

### For Speed (High-Throughput Screening)

```bash
# Fastest: Enhanced Hierarchical with fast mode
pandadock-dock -r receptor.pdb -l ligands.sdf \
  -a enhanced_hierarchical_cpu \
  --fast \
  -o fast_screening/

# GPU acceleration for maximum speed
pandadock-dock -r receptor.pdb -l ligands.sdf \
  -a cuda_monte_carlo \
  --gpu \
  -o gpu_screening/
```

### For Accuracy (Lead Optimization)

```bash
# Most accurate: Genetic Algorithm with precision scoring
pandadock-dock -r receptor.pdb -l ligand.sdf \
  -a genetic_algorithm_cpu \
  -s precision_score \
  --ensemble \
  -n 50 \
  -o accurate_docking/
```

### For Known Crystal Structures

```bash
# Crystal-guided docking for high accuracy
pandadock-dock -r receptor.pdb -l ligand.sdf \
  -a crystal_guided_cpu \
  --crystal-reference 25.0 30.0 40.0 \
  -o crystal_guided/
```

## Scoring Function Selection

### Physics-Based (Default)

```bash
pandadock-dock -r receptor.pdb -l ligand.sdf -s physics_based
```
- Most accurate for binding affinity prediction
- Considers electrostatics, van der Waals, and solvation
- Slower but more reliable

### Empirical (Fast)

```bash
pandadock-dock -r receptor.pdb -l ligand.sdf -s empirical
```
- Fast scoring for high-throughput screening
- Good for ranking and filtering
- Less accurate absolute energies

### Precision Score (Most Accurate)

```bash
pandadock-dock -r receptor.pdb -l ligand.sdf -s precision_score
```
- Highest accuracy for binding affinity
- Includes quantum mechanical corrections
- Slowest but most reliable

### Hybrid Approach

```bash
pandadock-dock -r receptor.pdb -l ligand.sdf -s hybrid
```
- Combines multiple scoring approaches
- Balanced speed and accuracy
- Good for general use

## Output Analysis

### Interaction Analysis

```python
import json

# Load interaction analysis
with open('results/interaction_analysis.json', 'r') as f:
    interactions = json.load(f)

print(f"Total interactions: {interactions['total_interactions']}")
print(f"Hydrogen bonds: {interactions['interaction_types']['hydrogen_bonds']}")
print(f"Hydrophobic contacts: {interactions['interaction_types']['hydrophobic_contacts']}")
```

### Binding Affinity Analysis

```python
# Load docking results
with open('results/docking_summary.json', 'r') as f:
    results = json.load(f)

for i, pose in enumerate(results['poses'][:5]):
    print(f"Pose {i+1}: {pose['energy']:.2f} kcal/mol")
```

### Visualization

```bash
# Generate comprehensive plots
pandadock-report -i results/ -t "My Docking Study"

# Create publication-quality figures
pandadock-report -i results/ --publication-plots -o figures/
```

## Troubleshooting

### Common Issues

1. **"No poses generated"**
   - Check grid box size and position
   - Verify ligand and protein formats
   - Try different algorithms

2. **GPU not detected**
   - Install CUDA toolkit
   - Verify GPU compatibility
   - Check nvidia-smi output

3. **High memory usage**
   - Reduce batch size for GPU algorithms
   - Use fast mode for initial screening
   - Limit number of poses generated

### Performance Optimization

```bash
# Optimize CPU usage
pandadock-dock -r receptor.pdb -l ligand.sdf \
  --cpuworkers 8 \
  -o optimized_cpu/

# Optimize GPU usage
pandadock-dock -r receptor.pdb -l ligand.sdf \
  -a cuda_monte_carlo \
  --gpu-batch-size 500 \
  --gpu-memory-limit 4.0 \
  -o optimized_gpu/
```

## Next Steps

- Explore [Algorithm Documentation](algorithms/index.md) for detailed algorithm descriptions
- Read [Scoring Functions](scoring/index.md) to understand energy calculations
- Check [Tutorials](tutorials/index.md) for advanced workflows
- Review [Performance Guide](performance/index.md) for optimization tips

## Getting Help

- Check the [FAQ](faq.md) for common questions
- Browse [Tutorials](tutorials/index.md) for examples
- Visit [GitHub Issues](https://github.com/pandadock/pandadock/issues) for bug reports
- Join [Discussions](https://github.com/pandadock/pandadock/discussions) for community support