# PandaDock Benchmarking Suite

Benchmarking tools for evaluating PandaDock-GNN performance on molecular docking datasets.

## Directory Structure

```
benchmarking/
├── README.md                     # This file
├── run_benchmark.py              # Main benchmark runner
├── generate_publication_plots.py # Generate publication figures
│
├── ULVSH/                        # ULVSH dataset benchmarks
│   ├── benchmark_ulvsh.py        # Full ULVSH benchmark (dock + hybrid)
│   ├── benchmark_pandadock.py    # PandaDock algorithm comparison
│   ├── benchmark_dock_ulvsh.py   # Docking-only benchmark
│   ├── benchmark_hybrid_site.py  # Hybrid docking with site extraction
│   ├── validate_ulvsh_full.py    # Validation script
│   └── ULVSH_benchmark_results.md # Benchmark results
│
├── PDBbind/                      # PDBbind dataset benchmarks
│   ├── benchmark_pdbbind.py      # Full PDBbind benchmark
│   ├── benchmark_hybrid_dock.py  # Hybrid docking benchmark
│   ├── benchmark_combined_model.py # Combined model benchmark
│   ├── prepare_pdbbind_core.py   # PDBbind data preparation
│   ├── prepare_ligands_comprehensive.py # Ligand preparation
│   ├── prepare_benchmark_simple.py # Simple benchmark prep
│   ├── run_pdbbind_benchmark.py  # PDBbind benchmark runner
│   ├── analyze_pdbbind_results.py # Results analysis
│   └── validate_pdbbind_full.py  # Validation script
│
├── BindingDB/                    # BindingDB dataset benchmarks
│   ├── bindingdb_dataset.py      # BindingDB PyTorch Dataset class
│   ├── prepare_bindingdb.py      # Data preparation script
│   ├── train_bindingdb.py        # Training script for BindingDB
│   ├── run_experiments.sh        # Run training experiments
│   └── bindingdb_affinity.tsv    # Example affinity data
│
└── Analysis/                     # Analysis utilities
    ├── analyze_binding_affinity_correlation.py # Correlation analysis
    ├── analyze_results.py        # General results analysis
    ├── compare_affinity_methods.py # Method comparison
    └── monitor_benchmark.py      # Benchmark monitoring
```

## Quick Start

### ULVSH Benchmark

```bash
# Run full ULVSH benchmark (dock + hybrid + GNN)
python benchmarking/ULVSH/benchmark_ulvsh.py \
    --ulvsh-dir ULVSH/ \
    --model models/best_model.pt \
    --output results/ulvsh_benchmark/

# Validate GNN on ULVSH
python benchmarking/ULVSH/validate_ulvsh_full.py \
    --model models/best_model.pt \
    --ulvsh-dir ULVSH/
```

### PDBbind Benchmark

```bash
# Prepare PDBbind data
python benchmarking/PDBbind/prepare_pdbbind_core.py \
    --pdbbind-dir PDBbind/ \
    --output benchmarking/prepared/

# Run PDBbind benchmark
python benchmarking/PDBbind/benchmark_pdbbind.py \
    --pdbbind-dir PDBbind/ \
    --model models/best_model.pt \
    --output results/pdbbind_benchmark/
```

### BindingDB Benchmark

```bash
# Prepare BindingDB data (if using custom data)
python benchmarking/BindingDB/prepare_bindingdb.py \
    --input raw_bindingdb.tsv \
    --output benchmarking/BindingDB/bindingdb_affinity.tsv

# Train on BindingDB
python benchmarking/BindingDB/train_bindingdb.py \
    --bindingdb benchmarking/BindingDB/bindingdb_affinity.tsv \
    --output models/bindingdb_model/

# Train on BindingDB + ULVSH combined
python benchmarking/BindingDB/train_bindingdb.py \
    --bindingdb benchmarking/BindingDB/bindingdb_affinity.tsv \
    --ulvsh ULVSH/ \
    --combined \
    --output models/combined_model/

# Or use the main CLI (recommended)
pandadock gnn train -b benchmarking/BindingDB/bindingdb_affinity.tsv -o models/
pandadock gnn train -b benchmarking/BindingDB/bindingdb_affinity.tsv -d ULVSH/ -o models/ --balanced
```

### Analysis

```bash
# Analyze binding affinity correlation
python benchmarking/Analysis/analyze_binding_affinity_correlation.py \
    --results results/benchmark_results.csv \
    --output results/analysis/

# Generate publication plots
python benchmarking/generate_publication_plots.py \
    --results results/ \
    --output figures/
```

## Benchmark Results (v4.0)

### PandaDock-GNN Performance

| Dataset | Test Pearson R | Test RMSE | N Complexes |
|---------|----------------|-----------|-------------|
| PDBbind | 0.88 | 0.93 pK | 5,316 |
| ULVSH | 0.82 | 0.32 pEC50 | 942 |
| BindingDB | 0.81 | 0.96 pK | 8,891 |

### BindingDB Training Configurations

| Training Data | Test Pearson R | Test RMSE | N (train) |
|---------------|----------------|-----------|-----------|
| BindingDB Only | 0.81 | 0.96 | 7,113 |
| BindingDB + ULVSH | 0.79 | 0.96 | 7,866 |

### Comparison with Baselines (ULVSH)

| Method | Pearson R | Type |
|--------|-----------|------|
| **PandaDock-GNN** | **0.82** | ML Scoring |
| VM2 | 0.15 | Free energy |
| PM6 | 0.08 | Semi-empirical |
| Gnina | 0.01 | ML Scoring |
| Hyde | 0.02 | Empirical |

## Requirements

- PandaDock with GNN dependencies: `pip install -e ".[gnn]"`
- Trained model checkpoint
- ULVSH or PDBbind dataset

## See Also

- [Dataset Preparation Guide](../docs/source/gnn/dataset_preparation.rst)
- [GNN Training Guide](../docs/source/gnn/training.rst)
- [Main README](../README.md)
