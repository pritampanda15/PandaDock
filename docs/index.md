# PandaDock: Advanced Molecular Docking Platform

PandaDock is a comprehensive, high-performance molecular docking platform designed for drug discovery and computational chemistry. It provides state-of-the-art algorithms, GPU acceleration, and extensive analysis capabilities for protein-ligand docking simulations.

## Key Features

- **Multiple Docking Algorithms**: CPU and GPU implementations of Monte Carlo, Genetic Algorithm, Hierarchical, and Crystal-Guided docking
- **Advanced Scoring Functions**: Physics-based, empirical, precision scoring, hybrid approaches, and GPU-accelerated scoring
- **Flexible Docking**: Support for flexible receptor and side-chain sampling
- **Metal Coordination**: Specialized algorithms for metal-containing binding sites
- **Machine Learning Integration**: ML-enhanced pose ranking and hybrid ML-physics pipelines
- **GPU Acceleration**: CUDA-powered algorithms for high-throughput screening
- **Comprehensive Analysis**: Interaction analysis, visualization, and reporting tools
- **Grid Box Tools**: Automated cavity detection and binding site identification
- **Preprocessing Pipeline**: Automated protein and ligand preparation

## Architecture Overview

PandaDock is built with a modular architecture that separates core algorithms, scoring functions, visualization, and analysis components:

```
PandaDock/
├── docking/           # Core docking algorithms and engines
├── scoring/           # Scoring function implementations
├── visualization/     # Molecular visualization and plotting
├── analysis/          # Interaction analysis and metrics
├── gridbox/           # Binding site detection and grid generation
├── preprocessing/     # Molecule preparation and optimization
├── ml/               # Machine learning models and pipelines
└── utils/            # Utility functions and I/O operations
```

## Quick Start

### Installation

```bash
pip install pandadock
```

### Basic Docking

```bash
# Standard protein-ligand docking
pandadock-dock -r protein.pdb -l ligand.sdf -o results/

# With specific algorithm and scoring
pandadock-dock -r protein.pdb -l ligand.sdf -a enhanced_hierarchical_cpu -s physics_based -o results/

# GPU-accelerated docking
pandadock-dock -r protein.pdb -l ligand.sdf -a cuda_monte_carlo --gpu -o results/
```

### Grid Box Generation

```bash
# Automatic cavity detection
pandadock-gridbox detect -r protein.pdb -o cavity_grids/

# Ligand-based grid generation
pandadock-gridbox ligand-based -r protein.pdb -l reference_ligand.sdf -o grid.json
```

## Performance Characteristics

| Algorithm | Speed | Accuracy | GPU Support | Best Use Case |
|-----------|-------|----------|-------------|---------------|
| Enhanced Hierarchical CPU | Very Fast | High | No | General purpose, fast screening |
| Monte Carlo CPU | Fast | High | No | Thorough sampling |
| Genetic Algorithm CPU | Medium | Very High | No | Complex binding sites |
| Crystal Guided CPU | Fast | Very High | Yes | Known crystal structures |
| CUDA Monte Carlo | Very Fast | High | Yes | High-throughput screening |
| CUDA Genetic Algorithm | Fast | Very High | Yes | GPU-accelerated complex docking |

## Documentation Structure

- **[Getting Started](getting-started.md)**: Installation, setup, and first steps
- **[Algorithms](algorithms/index.md)**: Detailed documentation of all docking algorithms
- **[Scoring Functions](scoring/index.md)**: Comprehensive guide to scoring approaches
- **[Command Line Interface](cli/index.md)**: Complete CLI reference
- **[Grid Box Tools](gridbox/index.md)**: Binding site detection and grid generation
- **[Preprocessing](preprocessing/index.md)**: Molecule preparation workflows
- **[Analysis & Visualization](analysis/index.md)**: Results analysis and plotting
- **[GPU Computing](gpu/index.md)**: CUDA acceleration and performance optimization
- **[Machine Learning](ml/index.md)**: ML-enhanced docking and pose ranking
- **[Flexible Docking](flexible/index.md)**: Receptor flexibility and side-chain sampling
- **[Metal Coordination](metal/index.md)**: Metalloprotein docking capabilities
- **[API Reference](api/index.md)**: Python API documentation
- **[Tutorials](tutorials/index.md)**: Step-by-step examples and workflows
- **[Performance Guide](performance/index.md)**: Optimization tips and benchmarks

## Citation

If you use PandaDock in your research, please cite:

```bibtex
@software{pandadock2024,
  title={PandaDock: Advanced Molecular Docking Platform},
  author={PandaDock Development Team},
  year={2024},
  url={https://github.com/pandadock/pandadock}
}
```

## License

PandaDock is released under the MIT License. See [LICENSE](LICENSE) for details.

## Support

- **Documentation**: https://pandadock.readthedocs.io/
- **Issues**: https://github.com/pandadock/pandadock/issues
- **Discussions**: https://github.com/pandadock/pandadock/discussions