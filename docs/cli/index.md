# Command Line Interface

PandaDock provides a comprehensive command-line interface for all molecular docking operations. The CLI is designed for both interactive use and automated workflows.

## Main Commands Overview

| Command | Purpose | Usage |
|---------|---------|-------|
| `pandadock-dock` | Main docking interface | Standard protein-ligand docking |
| `pandadock-gridbox` | Grid box generation | Binding site detection and setup |
| `pandadock-flex` | Flexible docking | Receptor flexibility and induced fit |
| `pandadock-metal` | Metal coordination | Metalloprotein docking |
| `pandadock-report` | Analysis and reporting | Results visualization and analysis |
| `pandadock-tethered` | Validation | Crystal pose reproduction |
| `pandadock-prepare` | Preprocessing | Ligand and protein preparation |

## pandadock-dock

The main docking command for standard protein-ligand docking simulations.

### Basic Syntax

```bash
pandadock-dock [OPTIONS]
```

### Required Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `-r, --receptor PATH` | Receptor PDB file | `-r protein.pdb` |
| `-l, --ligand PATH` | Ligand file (SDF/MOL2/PDB) | `-l ligand.sdf` |

### Grid Configuration

Choose one of these options to define the binding site:

#### Option 1: Grid Configuration File
```bash
pandadock-dock -r protein.pdb -l ligand.sdf --grid-config grid.json
```

#### Option 2: Manual Grid Definition
```bash
pandadock-dock -r protein.pdb -l ligand.sdf \
  --center 25.0 30.0 40.0 \
  --box 20 20 20
```

#### Option 3: Automatic Detection
```bash
# Grid will be automatically detected
pandadock-dock -r protein.pdb -l ligand.sdf
```

### Algorithm and Scoring Options

| Parameter | Options | Default | Description |
|-----------|---------|---------|-------------|
| `-a, --algorithm` | See [algorithm list](#available-algorithms) | `enhanced_hierarchical_cpu` | Docking algorithm |
| `-s, --scoring` | `physics_based`, `empirical`, `precision_score`, `hybrid`, `gpu_precision`, `gpu_mmgbsa` | `physics_based` | Scoring function |

### Performance Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--gpu` | Flag | False | Enable GPU acceleration |
| `--gpu-batch-size INTEGER` | Number | 1000 | GPU batch size |
| `--gpu-memory-limit FLOAT` | GB | Auto | GPU memory limit |
| `--cpuworkers INTEGER` | Number | Auto | CPU thread count |
| `--fast` | Flag | False | Fast screening mode |

### Output Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `-o, --output-dir PATH` | Directory | `./results` | Output directory |
| `-n, --num-poses INTEGER` | Number | 20 | Number of poses to generate |
| `--visualize` | Flag | False | Generate visualization plots |

### Advanced Options

| Parameter | Type | Description |
|-----------|------|-------------|
| `--ensemble` | Flag | Enable Boltzmann ensemble averaging |
| `--rescoring` | Choice | `none`, `mmgbsa` |
| `--gpuid INTEGER` | GPU device ID (default: 0) |

### Examples

#### Basic Docking
```bash
pandadock-dock -r protein.pdb -l ligand.sdf \
  --center 25.0 30.0 40.0 --box 20 20 20 \
  -o results/
```

#### High-Throughput Screening
```bash
pandadock-dock -r protein.pdb -l compound_library.sdf \
  -a enhanced_hierarchical_cpu --fast \
  --cpuworkers 8 \
  -o screening_results/
```

#### GPU-Accelerated Docking
```bash
pandadock-dock -r protein.pdb -l ligand.sdf \
  -a cuda_monte_carlo --gpu \
  --gpu-batch-size 2000 \
  --center 25.0 30.0 40.0 --box 20 20 20 \
  -o gpu_results/
```

#### Precision Docking
```bash
pandadock-dock -r protein.pdb -l ligand.sdf \
  -a genetic_algorithm_cpu \
  -s precision_score \
  --ensemble \
  --rescoring mmgbsa \
  -n 50 \
  -o precision_results/
```

## pandadock-gridbox

Grid box generation and binding site detection utilities.

### Commands

#### detect
Automatic cavity and binding site detection:

```bash
pandadock-gridbox detect [OPTIONS]
```

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `-r, --receptor PATH` | Required | Receptor PDB file |
| `-o, --output-dir PATH` | Optional | Output directory |
| `--min-volume FLOAT` | 50.0 | Minimum cavity volume (Å³) |
| `--probe-radius FLOAT` | 1.4 | Probe radius for detection (Å) |
| `--grid-spacing FLOAT` | 0.5 | Grid spacing for analysis (Å) |

**Example:**
```bash
pandadock-gridbox detect -r protein.pdb -o cavities/
```

#### manual
Create custom grid box configuration:

```bash
pandadock-gridbox manual [OPTIONS]
```

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `--center X Y Z` | Required | Grid center coordinates |
| `--dimensions X Y Z` | Required | Grid box dimensions |
| `-o, --output PATH` | Required | Output JSON file |

**Example:**
```bash
pandadock-gridbox manual \
  --center 25.0 30.0 40.0 \
  --dimensions 20 20 20 \
  -o custom_grid.json
```

#### ligand-based
Generate grid based on reference ligand:

```bash
pandadock-gridbox ligand-based [OPTIONS]
```

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `-r, --receptor PATH` | Required | Receptor PDB file |
| `-l, --ligand PATH` | Required | Reference ligand file |
| `--expansion FLOAT` | 5.0 | Grid expansion around ligand (Å) |
| `-o, --output PATH` | Required | Output JSON file |

**Example:**
```bash
pandadock-gridbox ligand-based \
  -r protein.pdb -l reference.sdf \
  --expansion 7.0 \
  -o ligand_grid.json
```

## pandadock-flex

Flexible receptor docking with induced fit capabilities.

### Basic Syntax

```bash
pandadock-flex [OPTIONS]
```

### Key Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `-r, --receptor PATH` | Receptor PDB file | `-r protein.pdb` |
| `-l, --ligand PATH` | Ligand file | `-l ligand.sdf` |
| `--flexible-residues LIST` | Flexible residue IDs | `--flexible-residues "GLU123,ASP456"` |
| `--flexibility-radius FLOAT` | Radius around binding site (Å) | `--flexibility-radius 8.0` |
| `--side-chain-flexibility` | Enable side chain sampling | Flag |
| `--backbone-flexibility` | Enable backbone sampling | Flag |

### Examples

#### Side Chain Flexibility
```bash
pandadock-flex -r protein.pdb -l ligand.sdf \
  --side-chain-flexibility \
  --flexibility-radius 6.0 \
  --center 25.0 30.0 40.0 --box 20 20 20 \
  -o flex_results/
```

#### Specific Residue Flexibility
```bash
pandadock-flex -r protein.pdb -l ligand.sdf \
  --flexible-residues "ASP123,GLU456,TYR789" \
  --center 25.0 30.0 40.0 --box 20 20 20 \
  -o specific_flex/
```

#### Full Induced Fit
```bash
pandadock-flex -r protein.pdb -l ligand.sdf \
  --side-chain-flexibility \
  --backbone-flexibility \
  --flexibility-radius 8.0 \
  --refinement-cycles 5 \
  -o induced_fit/
```

## pandadock-metal

Specialized docking for metalloproteins and metal coordination.

### Basic Syntax

```bash
pandadock-metal [OPTIONS]
```

### Key Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `-r, --receptor PATH` | Metalloprotein PDB file | `-r metalloprotein.pdb` |
| `-l, --ligand PATH` | Ligand file | `-l ligand.sdf` |
| `--metal-ions LIST` | Metal ion types | `--metal-ions ZN,MG,CA` |
| `--coordination-geometry` | Preferred geometry | `--coordination-geometry octahedral` |
| `--distance-constraints` | Metal-ligand distances | `--distance-constraints 2.0-2.5` |

### Available Metal Types

| Metal | Symbol | Common Coordination | Typical Distance Range |
|-------|--------|---------------------|----------------------|
| Zinc | ZN | Tetrahedral, Octahedral | 1.9-2.3 Å |
| Magnesium | MG | Octahedral | 1.9-2.1 Å |
| Calcium | CA | Irregular | 2.2-2.6 Å |
| Iron | FE | Octahedral | 1.9-2.4 Å |
| Copper | CU | Square Planar, Tetrahedral | 1.9-2.3 Å |
| Manganese | MN | Octahedral | 1.9-2.4 Å |

### Examples

#### Zinc Metalloprotein
```bash
pandadock-metal -r zinc_protein.pdb -l inhibitor.sdf \
  --metal-ions ZN \
  --coordination-geometry tetrahedral \
  --distance-constraints 2.0-2.2 \
  -o zinc_docking/
```

#### Multi-Metal System
```bash
pandadock-metal -r enzyme.pdb -l substrate.sdf \
  --metal-ions ZN,MG \
  --allow-bridging \
  -o multi_metal/
```

## pandadock-report

Analysis and visualization of docking results.

### Basic Syntax

```bash
pandadock-report [OPTIONS]
```

### Key Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `-i, --input PATH` | Results directory | `-i docking_results/` |
| `-t, --title STRING` | Report title | `-t "My Docking Study"` |
| `--publication-plots` | High-quality figures | Flag |
| `--format` | Output format | `--format pdf` |

### Generated Outputs

- **Binding affinity plots**: Energy distribution and rankings
- **Interaction analysis**: Hydrogen bonds, hydrophobic contacts
- **Pose clustering**: RMSD-based pose grouping
- **Energy components**: Detailed energy breakdown
- **3D visualizations**: Molecular graphics and binding poses

### Examples

#### Basic Report
```bash
pandadock-report -i results/ -t "Protein-Ligand Docking Study"
```

#### Publication-Quality Figures
```bash
pandadock-report -i results/ \
  --publication-plots \
  --format pdf \
  --dpi 300 \
  -o publication_figures/
```

#### Custom Analysis
```bash
pandadock-report -i results/ \
  --include-clustering \
  --include-pharmacophore \
  --energy-components \
  -o detailed_analysis/
```

## pandadock-tethered

Validation and crystal pose reproduction tools.

### Basic Syntax

```bash
pandadock-tethered [OPTIONS]
```

### Commands

#### validate
Validate docking against known crystal structures:

```bash
pandadock-tethered validate [OPTIONS]
```

#### analyze
Analyze crystal pose reproduction accuracy:

```bash
pandadock-tethered analyze [OPTIONS]
```

### Key Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `-i, --input PATH` | Crystal structure or docking results | `-i crystal.pdb` |
| `-l, --ligand-id STRING` | Ligand identifier | `-l LIG` |
| `--tether-radius FLOAT` | Tethering radius (Å) | `--tether-radius 2.0` |
| `--rmsd-threshold FLOAT` | Success RMSD threshold | `--rmsd-threshold 2.0` |

### Examples

#### Crystal Pose Validation
```bash
pandadock-tethered validate \
  -i crystal_complex.pdb \
  -l LIG \
  --tether-radius 2.0 \
  -o validation_results/
```

#### Docking Accuracy Analysis
```bash
pandadock-tethered analyze \
  -i docking_results/ \
  --reference crystal.pdb \
  --ligand-id LIG \
  -o accuracy_analysis/
```

## pandadock-prepare

Preprocessing utilities for ligands and proteins.

### Basic Syntax

```bash
pandadock-prepare [OPTIONS]
```

### Commands

#### ligand
Prepare ligands for docking:

```bash
pandadock-prepare ligand -i input.sdf -o prepared.sdf
```

#### protein
Prepare proteins for docking:

```bash
pandadock-prepare protein -i protein.pdb -o prepared.pdb
```

### Ligand Preparation Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--add-hydrogens` | Add hydrogen atoms | True |
| `--generate-3d` | Generate 3D coordinates | True |
| `--optimize-geometry` | Optimize molecular geometry | True |
| `--assign-charges` | Assign partial charges | True |
| `--enumerate-tautomers` | Generate tautomeric forms | False |
| `--enumerate-stereoisomers` | Generate stereoisomers | False |

### Protein Preparation Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--add-hydrogens` | Add hydrogen atoms | True |
| `--remove-waters` | Remove water molecules | False |
| `--remove-heterogens` | Remove heterogens | False |
| `--repair-structure` | Fix structural issues | True |
| `--optimize-side-chains` | Optimize side chain conformations | True |

## Available Algorithms

Use `pandadock list-algorithms` to see all available algorithms:

### CPU Algorithms
- `enhanced_hierarchical_cpu`: Enhanced hierarchical search (recommended)
- `monte_carlo_cpu`: Monte Carlo with simulated annealing
- `genetic_algorithm_cpu`: Evolutionary algorithm
- `hierarchical_cpu`: Multi-resolution hierarchical search
- `crystal_guided_cpu`: Crystal structure-guided docking

### GPU Algorithms
- `enhanced_hierarchical_gpu`: GPU-accelerated hierarchical search
- `cuda_monte_carlo`: CUDA Monte Carlo implementation
- `cuda_genetic_algorithm`: CUDA genetic algorithm

## Global Options

These options work with all PandaDock commands:

| Option | Description |
|--------|-------------|
| `--version` | Show version information |
| `--help` | Show help message |
| `--verbose` | Enable verbose output |
| `--quiet` | Suppress non-essential output |
| `--debug` | Enable debug logging |
| `--config PATH` | Use custom configuration file |

## Configuration Files

### Grid Configuration (JSON)
```json
{
  "center": [25.0, 30.0, 40.0],
  "dimensions": [20.0, 20.0, 20.0],
  "spacing": 0.375,
  "receptor_file": "protein.pdb"
}
```

### Algorithm Configuration (JSON)
```json
{
  "algorithm": "enhanced_hierarchical_cpu",
  "parameters": {
    "num_conformers": 5,
    "coarse_samples": 100,
    "energy_threshold": 50.0
  },
  "scoring": {
    "function": "physics_based",
    "ensemble": true
  }
}
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PANDADOCK_GPU` | Enable GPU by default | False |
| `PANDADOCK_WORKERS` | Default CPU worker count | Auto |
| `PANDADOCK_CONFIG` | Default config file path | None |
| `CUDA_VISIBLE_DEVICES` | Available GPU devices | All |

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 2 | Invalid arguments |
| 3 | File not found |
| 4 | GPU not available |
| 5 | Insufficient memory |

## Troubleshooting

### Common Issues

#### "Command not found"
```bash
# Check installation
pip show pandadock

# Reinstall if needed
pip install --upgrade pandadock
```

#### GPU not detected
```bash
# Check CUDA installation
nvidia-smi

# Verify GPU support
pandadock-dock --version
```

#### Out of memory
```bash
# Reduce batch size
pandadock-dock --gpu-batch-size 100

# Use CPU fallback
pandadock-dock --cpuworkers 4
```

### Getting Help

- Use `--help` with any command for detailed options
- Check the [FAQ](../faq.md) for common questions
- Report issues on [GitHub](https://github.com/pandadock/pandadock/issues)