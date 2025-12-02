# Docking Algorithms

PandaDock provides a comprehensive suite of molecular docking algorithms, each optimized for different use cases. All algorithms support both CPU and GPU execution (where applicable) and can be combined with various scoring functions.

## Algorithm Overview

| Algorithm | Type | Speed | Accuracy | GPU Support | Best For |
|-----------|------|-------|----------|-------------|----------|
| [Enhanced Hierarchical CPU](enhanced-hierarchical-cpu.md) | Hierarchical Search | Very Fast | High | No | General purpose, screening |
| [Monte Carlo CPU](monte-carlo-cpu.md) | Stochastic Sampling | Fast | High | No | Thorough exploration |
| [Genetic Algorithm CPU](genetic-algorithm-cpu.md) | Evolutionary | Medium | Very High | No | Complex binding sites |
| [Hierarchical CPU](hierarchical-cpu.md) | Multi-resolution | Fast | High | No | Systematic sampling |
| [Crystal Guided CPU](crystal-guided-cpu.md) | Structure-guided | Fast | Very High | No | Known crystal structures |
| [Enhanced Hierarchical GPU](enhanced-hierarchical-gpu.md) | Hierarchical Search | Very Fast | High | Yes | GPU-accelerated screening |
| [Monte Carlo GPU](monte-carlo-gpu.md) | Stochastic Sampling | Very Fast | High | Yes | High-throughput GPU |
| [Genetic Algorithm GPU](genetic-algorithm-gpu.md) | Evolutionary | Fast | Very High | Yes | Complex sites + GPU |

## Algorithm Selection Guide

### By Use Case

#### High-Throughput Virtual Screening
- **Primary**: Enhanced Hierarchical CPU with `--fast` mode
- **Alternative**: Monte Carlo GPU for maximum throughput
- **Characteristics**: Speed prioritized, good accuracy for filtering

#### Lead Optimization
- **Primary**: Genetic Algorithm CPU with precision scoring
- **Alternative**: Crystal Guided CPU if reference structure available
- **Characteristics**: Maximum accuracy, thorough sampling

#### Fragment-Based Drug Design
- **Primary**: Monte Carlo CPU with ensemble averaging
- **Alternative**: Hierarchical CPU for systematic exploration
- **Characteristics**: Good sampling of small molecule conformations

#### Allosteric Site Discovery
- **Primary**: Hierarchical CPU with large grid boxes
- **Alternative**: Monte Carlo CPU with extended sampling
- **Characteristics**: Thorough exploration of protein surface

#### Metal-Containing Active Sites
- **Primary**: Crystal Guided CPU with metal constraints
- **Alternative**: Enhanced Hierarchical CPU with specialized scoring
- **Characteristics**: Geometric constraints handling

### By Computational Resources

#### Single CPU Core
```bash
pandadock-dock -a enhanced_hierarchical_cpu --fast
```
- Fastest single-threaded performance
- Good for quick initial screening

#### Multi-Core CPU
```bash
pandadock-dock -a monte_carlo_cpu --cpuworkers 8
```
- Excellent parallel scaling
- Balances speed and accuracy

#### GPU Available
```bash
pandadock-dock -a cuda_monte_carlo --gpu --gpu-batch-size 1000
```
- Maximum throughput for large libraries
- Requires CUDA-compatible GPU

#### High-Memory Systems
```bash
pandadock-dock -a genetic_algorithm_cpu -n 100 --ensemble
```
- Can generate and analyze many poses
- Best accuracy for complex systems

## Common Parameters

All algorithms support these common parameters:

### Basic Parameters
- `num_poses`: Number of final poses to generate (default: 20)
- `num_conformers`: Ligand conformations to sample (varies by algorithm)
- `energy_threshold`: Energy cutoff for pose acceptance (default: 100.0 kcal/mol)

### Sampling Parameters
- `max_attempts`: Maximum sampling attempts (algorithm-dependent)
- `temperature`: Simulated annealing temperature (where applicable)
- `convergence_threshold`: Stopping criteria for optimization

### Performance Parameters
- `cpuworkers`: Number of CPU threads (CPU algorithms)
- `gpu_batch_size`: GPU batch size (GPU algorithms)
- `gpu_memory_limit`: GPU memory limit in GB

### Quality Parameters
- `ensemble`: Enable Boltzmann ensemble averaging
- `refinement_steps`: Local optimization steps
- `precision_mode`: Higher accuracy at computational cost

## Algorithm-Specific Features

### Enhanced Hierarchical CPU
- **Unique Features**:
  - Ultra-fast mode for screening
  - Crystal-guided pose generation
  - Adaptive grid sampling
- **Parameters**:
  - `fast`: Enable fast screening mode
  - `coarse_samples`: Initial sampling density
  - `refinement_levels`: Hierarchical refinement stages

### Monte Carlo CPU
- **Unique Features**:
  - Simulated annealing optimization
  - Temperature scheduling
  - Conformational clustering
- **Parameters**:
  - `temperature_schedule`: Annealing temperatures [300, 200, 100]
  - `poses_per_conformer`: Sampling density per conformer
  - `clustering_rmsd`: RMSD threshold for pose clustering

### Genetic Algorithm CPU
- **Unique Features**:
  - Population-based evolution
  - Crossover and mutation operators
  - Multi-objective optimization
- **Parameters**:
  - `population_size`: Number of individuals (default: 150)
  - `generations`: Evolution cycles (default: 300)
  - `mutation_rate`: Mutation probability (default: 0.1)
  - `crossover_rate`: Crossover probability (default: 0.8)

### Hierarchical CPU
- **Unique Features**:
  - Multi-resolution grid sampling
  - Progressive refinement
  - Systematic exploration
- **Parameters**:
  - `grid_levels`: Resolution levels [coarse, medium, fine]
  - `samples_per_level`: Sampling density at each level
  - `refinement_radius`: Local optimization radius

### Crystal Guided CPU
- **Unique Features**:
  - Reference structure guidance
  - Pharmacophore constraints
  - Biased sampling around known sites
- **Parameters**:
  - `crystal_reference`: Reference coordinates [x, y, z]
  - `guidance_weight`: Bias strength (default: 0.5)
  - `max_deviation`: Maximum deviation from reference (default: 5.0 Å)

## GPU Algorithms

### CUDA Requirements
- CUDA toolkit 11.0 or higher
- GPU compute capability 6.0 or higher
- Minimum 4GB GPU memory

### GPU-Specific Parameters
- `gpu_batch_size`: Number of poses processed simultaneously
- `gpu_memory_limit`: Maximum GPU memory usage (GB)
- `gpuid`: GPU device ID for multi-GPU systems

### Performance Optimization
```bash
# Optimize GPU batch size for your hardware
pandadock-dock -a cuda_monte_carlo --gpu --gpu-batch-size 2000

# Multiple GPU support
pandadock-dock -a cuda_genetic_algorithm --gpu --gpuid 0
```

## Algorithm Combinations

### Sequential Screening
```bash
# Stage 1: Fast screening
pandadock-dock -a enhanced_hierarchical_cpu --fast -n 5 -o stage1/

# Stage 2: Refined docking of top hits
pandadock-dock -a genetic_algorithm_cpu -n 20 -o stage2/
```

### Ensemble Docking
```bash
# Multiple algorithms for consensus
pandadock-dock -a monte_carlo_cpu -o mc_results/
pandadock-dock -a genetic_algorithm_cpu -o ga_results/
pandadock-dock -a hierarchical_cpu -o hier_results/
```

### Cross-Validation
```bash
# Algorithm comparison study
pandadock-dock -a enhanced_hierarchical_cpu -o enhanced_results/
pandadock-dock -a crystal_guided_cpu --crystal-reference 25 30 40 -o crystal_results/
```

## Best Practices

### Algorithm Selection
1. **Start with Enhanced Hierarchical CPU** for initial exploration
2. **Use Crystal Guided** when reference structures are available
3. **Apply Genetic Algorithm** for final refinement of promising compounds
4. **Consider GPU algorithms** for large-scale screening (>1000 compounds)

### Parameter Tuning
1. **Begin with default parameters** for initial testing
2. **Increase sampling** for difficult binding sites
3. **Enable ensemble averaging** for better affinity estimates
4. **Use fast mode** for preliminary screening

### Validation
1. **Test on known active compounds** to validate setup
2. **Compare multiple algorithms** for consensus
3. **Analyze binding poses** for chemical reasonableness
4. **Validate energy rankings** against experimental data

## Performance Benchmarks

### Typical Runtime (Single Ligand)
- Enhanced Hierarchical CPU (fast): 0.1-0.5 seconds
- Monte Carlo CPU: 1-5 seconds
- Genetic Algorithm CPU: 5-30 seconds
- CUDA Monte Carlo: 0.05-0.2 seconds
- CUDA Genetic Algorithm: 0.5-2 seconds

### Scaling (1000 Ligands)
- Enhanced Hierarchical CPU: 2-10 minutes
- Monte Carlo CPU: 20-80 minutes
- GPU algorithms: 1-10 minutes (depending on hardware)

### Memory Requirements
- CPU algorithms: 1-4 GB RAM
- GPU algorithms: 2-8 GB GPU memory + 2-4 GB RAM

## Troubleshooting

### Common Issues
1. **No poses generated**: Increase energy threshold or expand grid box
2. **Poor pose quality**: Try different algorithm or increase sampling
3. **GPU out of memory**: Reduce batch size or memory limit
4. **Slow performance**: Enable fast mode or use GPU acceleration

### Debug Mode
```bash
# Enable detailed logging
pandadock-dock -a monte_carlo_cpu --debug -o debug_results/
```

This will provide detailed information about:
- Sampling statistics
- Energy evaluations
- Convergence criteria
- Performance metrics