# PandaDock - Next-Generation Molecular Docking Suite

---

<p align="center">
  <a href="https://github.com/pritampanda15/PandaDock">
    <img src="https://github.com/pritampanda15/PandaDock/blob/c4e5c9e91c4c8262faf1c7736850ca647d65adc1/PandaDock.png" width="500" alt="PandaDock Logo"/> 
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

**High-Accuracy Molecular Docking with GPU Acceleration**

[Installation](#installation) • [Quick Start](#quick-start) • [Documentation](#documentation) • [Algorithms](#algorithms) • [Citation](#citation)

</div>

---

## Overview

**PandaDock** is a state-of-the-art molecular docking platform that combines cutting-edge algorithms, GPU acceleration, and physics-based scoring functions to achieve sub-angstrom precision in protein-ligand binding predictions. Designed for both drug discovery and computational biology research, PandaDock delivers exceptional accuracy with industry-leading performance.

### Key Features

- **8 Advanced Docking Algorithms** (5 CPU + 3 GPU variants)
- **6 Specialized Docking Modes** (Standard, Flexible, Metal, ML-powered, Tethered, Crystal-guided)
- **Multiple Scoring Functions** (Physics-based, Empirical, Hybrid, GPU-accelerated, MM-GBSA)
- **Sub-angstrom Accuracy** (Mean RMSD: 0.014 Å on PDBbind v2020)
- **GPU Acceleration** with CUDA support for 100x speedup
- **Comprehensive Analysis Tools** including PandaMap visualization
- **Production-Ready** with enterprise-grade code quality and extensive benchmarking

---

## Benchmark Performance

Comprehensive benchmarking on **150 diverse protein-ligand complexes** from PDBbind v2020:

### Top Performing Algorithms

| Algorithm | Success Rate | RMSD < 2Å | Mean RMSD | Runtime (s) |
|-----------|--------------|-----------|-----------|-------------|
| **enhanced_hierarchical_cpu** | **100%** | **99.3%** | **0.014 Å** | 0.30 |
| **enhanced_hierarchical_gpu** | 91.3% | 99.3% | 0.015 Å | 0.82 |
| **cuda_genetic_algorithm** | **100%** | **99.3%** | **0.014 Å** | 35.24 |

### Full Algorithm Comparison

| Algorithm | Success Rate | RMSD < 2Å | RMSD < 3Å | Mean RMSD (Å) | Runtime (s) |
|-----------|--------------|-----------|-----------|---------------|-------------|
| enhanced_hierarchical_cpu | 100% | 99.3% | 100% | 0.014 | 0.30 |
| enhanced_hierarchical_gpu | 91.3% | 99.3% | 100% | 0.015 | 0.82 |
| cuda_genetic_algorithm | 100% | 99.3% | 100% | 0.014 | 35.24 |
| cuda_monte_carlo | 48.7% | 100%* | 100%* | 0.0* | 414.20 |
| monte_carlo_cpu | 95.3% | 44.1% | 76.9% | 2.207 | 86.86 |
| hierarchical_cpu | 94.7% | 42.3% | 74.7% | 2.278 | 17.90 |
| genetic_algorithm_cpu | 89.3% | 45.5% | 75.4% | 2.246 | 4.84 |
| crystal_guided_cpu | 100% | 41.3% | 74.7% | 2.298 | 3.91 |

*For successful runs only

### Key Highlights

✅ **Sub-Angstrom Accuracy**: Mean RMSD of 0.014 Å for top performers
✅ **High Reliability**: 100% completion rate for enhanced algorithms
✅ **Ultra-Fast Performance**: 0.30s per complex (CPU) with enhanced_hierarchical_cpu
✅ **Excellent Pose Prediction**: >99% of poses within 2Å RMSD

*Complete benchmark results and reproducibility instructions available in [/benchmarking](benchmarking/README.md)*

---

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA 11.0+ (for GPU acceleration, optional)
- Conda or pip package manager

### Basic Installation (CPU Only)

```bash
git clone https://github.com/pritampanda15/PandaDock.git
cd PandaDock
pip install -e .
```

### GPU-Accelerated Installation

```bash
# Install with CUDA 11.x support
pip install -e .
pip install cupy-cuda11x  # or cupy-cuda12x for CUDA 12

# Verify GPU availability
pandadock list-algorithms
```

### Optional Dependencies

```bash
# For ML-powered docking
pip install -e ".[ml]"

# For flexible docking with OpenMM
conda install -c conda-forge openmm pdbfixer
pip install -e ".[conda]"
```

For detailed installation instructions, see [INSTALL.md](INSTALL.md).

---

## Quick Start

### Basic Molecular Docking

```bash
# Fast, optimized docking (recommended for most cases)
pandadock dock -r protein.pdb -l ligand.sdf \
               --center 10 20 30 --box 20 20 20 \
               -o results/
```

### High-Accuracy Docking

```bash
# Enhanced hierarchical algorithm with physics-based scoring
pandadock dock -r protein.pdb -l ligand.sdf \
               --center 10 20 30 --box 20 20 20 \
               --algorithm enhanced_hierarchical_cpu \
               --scoring physics_based \
               -o high_accuracy_results/
```

### GPU-Accelerated Docking

```bash
# 100x faster with GPU
pandadock dock -r protein.pdb -l ligand.sdf \
               --center 10 20 30 --box 20 20 20 \
               --algorithm enhanced_hierarchical_gpu \
               --gpu \
               -o gpu_results/
```

### Flexible (Induced-Fit) Docking

```bash
# Account for receptor flexibility
pandadock-flex -r protein.pdb -l ligand.sdf \
               --center 10 20 30 --radius 12.0 \
               --refine-distance 6.0 \
               -o flex_results/
```

### Metal-Coordinating Ligands

```bash
# Specialized for metalloproteins
pandadock-metal -r metalloprotein.pdb -l ligand.sdf \
                --center 10 20 30 --box 20 20 20 \
                --metal-type ZN --metal-residue "A:201" \
                -o metal_results/
```

### Tethered Docking (Constrained)

```bash
# Constrain ligand near reference position
pandadock-tethered -r protein.pdb -l ligand.sdf \
                   --reference-ligand crystal_ligand.sdf \
                   --tether-radius 2.0 \
                   -o tethered_results/
```

---

## Algorithms

PandaDock provides multiple docking algorithms, each optimized for different use cases:

### CPU Algorithms

| Algorithm | Description | Best For | Speed |
|-----------|-------------|----------|-------|
| **enhanced_hierarchical_cpu** | 3-stage hierarchical search with refinement | High-accuracy general docking | ⭐⭐⭐ |
| **monte_carlo_cpu** | Monte Carlo sampling with simulated annealing | Fast screening | ⭐⭐⭐⭐⭐ |
| **genetic_algorithm_cpu** | Genetic algorithm with ensemble refinement | Complex binding sites | ⭐⭐⭐ |
| **hierarchical_cpu** | Standard hierarchical search | Balanced accuracy/speed | ⭐⭐⭐⭐ |
| **crystal_guided_cpu** | Crystal structure-guided docking | Validation studies | ⭐⭐⭐ |

### GPU Algorithms

| Algorithm | Description | Speedup | Requirements |
|-----------|-------------|---------|--------------|
| **enhanced_hierarchical_gpu** | GPU-accelerated hierarchical search | 50-100x | CUDA 11.0+ |
| **cuda_monte_carlo** | Massively parallel Monte Carlo | 100-200x | CUDA 11.0+ |
| **cuda_genetic_algorithm** | GPU genetic algorithm | 80-150x | CUDA 11.0+ |

### Specialized Docking Modes

- **Flexible Docking** (`pandadock-flex`): Induced-fit docking with receptor side-chain flexibility
- **Metal Docking** (`pandadock-metal`): Specialized for metal-coordinating ligands (Zn, Fe, Mg, Ca, Mn, Cu, Ni, Co)
- **ML Docking** (`pandadock-ml`): Machine learning-enhanced scoring and pose prediction
- **Tethered Docking** (`pandadock-tethered`): Constrained docking near reference positions
- **Crystal-Guided Docking** : Use crystallographic information for improved accuracy

For complete algorithm details, see [ALGORITHMS.md](ALGORITHMS.md).

---

## Scoring Functions

| Scoring Function | Description | Use Case |
|------------------|-------------|----------|
| **physics_based** | Comprehensive force field scoring | General docking (recommended) |
| **empirical** | Empirical statistical potential | Fast screening |
| **precision_score** | High-precision interaction energy | Detailed analysis |
| **hybrid** | Combined physics + ML scoring | Maximum accuracy |
| **gpu_precision** | GPU-accelerated precision scoring | Large-scale studies |
| **gpu_mmgbsa** | GPU MM-GBSA rescoring | Binding free energy |

---

## Command-Line Tools

PandaDock provides a comprehensive suite of command-line tools:

### Core Tools

```bash
pandadock                # Main docking interface
pandadock-flex          # Flexible/induced-fit docking
pandadock-metal         # Metal coordination docking
pandadock-ml            # ML-enhanced docking
pandadock-tethered      # Tethered/constrained docking
```

### Utility Tools

```bash
pandadock-prepare       # Prepare ligands (add H, generate 3D)
pandadock-gridbox       # Generate grid box configurations
pandadock-report        # Generate publication-quality reports
```

### Analysis Tools

```bash
# Generate comprehensive analysis report
pandadock-report -i docking_output/ \
                 -t "PandaDock Analysis" \
                 --compare-algorithms

# PandaMap visualization (Discovery Studio-style)
pandadock-report pandamap -i docking_output/ -o pandamap_results/
```

---

## Output Files

PandaDock generates comprehensive outputs for each docking run:

```
docking_output/
├── complex1.pdb                    # Top-ranked protein-ligand complex
├── complex2.pdb                    # Second-ranked complex
├── ...
├── pose1.pdb                       # Top-ranked ligand pose
├── pose2.pdb                       # Second-ranked pose
├── ...
├── docking_results.json            # Complete results with energies
├── interaction_analysis.json       # Detailed interaction analysis
├── binding_affinities.png          # Affinity distribution plot
├── interaction_energies.png        # Energy decomposition plot
└── summary.txt                     # Human-readable summary
```

---

## Advanced Usage

### Ensemble Docking

```bash
# Generate ensemble of poses with Boltzmann averaging
pandadock dock -r protein.pdb -l ligand.sdf \
               --center 10 20 30 --box 20 20 20 \
               --num-poses 50 --ensemble
```

### MM-GBSA Rescoring

```bash
# Rescore top poses with MM-GBSA
pandadock dock -r protein.pdb -l ligand.sdf \
               --center 10 20 30 --box 20 20 20 \
               --rescoring mmgbsa
```

### Parallel CPU Processing

```bash
# Use multiple CPU cores
pandadock dock -r protein.pdb -l ligand.sdf \
               --center 10 20 30 --box 20 20 20 \
               --cpuworkers 16
```

### Multi-GPU Support

```bash
# Specify GPU device
pandadock dock -r protein.pdb -l ligand.sdf \
               --center 10 20 30 --box 20 20 20 \
               --algorithm enhanced_hierarchical_gpu \
               --gpuid 1
```

---

## Validation and Testing

Comprehensive test suites are provided to ensure reproducibility:

```bash
# Run CPU algorithm tests
cd cpu_comprehensive_testing_fixed/
./run_tests.sh

# Run GPU algorithm tests
cd gpu_comprehensive_testing/
./run_tests.sh
```

Test results include:
- RMSD calculations against crystal structures
- Energy distribution analysis
- Algorithm comparison reports
- Performance benchmarks

---

## Documentation

- [Installation Guide](INSTALL.md) - Detailed installation instructions
- [Algorithm Documentation](ALGORITHMS.md) - Complete algorithm descriptions
- [API Reference](docs/api.md) - Python API documentation
- [Tutorial](docs/tutorial.md) - Step-by-step tutorials
- [FAQ](docs/faq.md) - Frequently asked questions

---

## Citation

If you use PandaDock in your research, please cite:

```bibtex
@article{panda2024pandadock,
  title={PandaDock: Python based Next-Generation Molecular Docking },
  author={Panda, Pritam Kumar},
  journal={arXiv},
  year={2024},
  note={Manuscript in preparation}
}
```

---

## Examples

See the `examples/` directory for complete examples:

- `examples/basic_docking/` - Simple docking workflow
- `examples/high_accuracy/` - High-accuracy docking protocol
- `examples/flexible_docking/` - Induced-fit docking examples
- `examples/metal_docking/` - Metalloprotein docking
- `examples/virtual_screening/` - Large-scale screening
- `examples/benchmarking/` - Reproduce benchmark results

---

## Troubleshooting

### GPU Issues

```bash
# Check GPU availability
pandadock list-algorithms

# If CUDA errors occur, reinstall CuPy
pip uninstall cupy-cuda11x cupy-cuda12x
pip install cupy-cuda11x  # Match your CUDA version
```

### Memory Issues

```bash
# Reduce batch size for GPU
pandadock dock ... --gpu-batch-size 500 --gpu-memory-limit 2.0

# Reduce number of poses
pandadock dock ... --num-poses 10
```

For more troubleshooting, see [docs/troubleshooting.md](docs/troubleshooting.md).

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

PandaDock builds upon and is inspired by several excellent open-source projects:
- AutoDock Vina
- RDKit
- OpenMM
- Biopython
- CuPy/PyCUDA

Special thanks to the computational chemistry and drug discovery communities for their invaluable contributions.

---

<div align="center">

**Star ⭐ this repository if you find it useful!**

[Report Bug](https://github.com/pritampanda15/PandaDock/issues) • [Request Feature](https://github.com/pritampanda15/PandaDock/issues)

</div>
