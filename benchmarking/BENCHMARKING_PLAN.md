# PandaDock Benchmarking Plan for Publication

## Objective
Comprehensive benchmarking of PandaDock against established docking tools for academic publication.

## Dataset Selection

### Core Dataset (290 complexes)
- **Source**: PDBbind Core Set 2020
- **Why**: Most cited benchmark, diverse protein families, experimental binding affinities
- **Download**: http://www.pdbbind.org.cn/download.php
- **Metrics**: RMSD, binding affinity correlation, runtime

### Category-Specific Datasets

#### 1. Kinases (50 complexes)
- **Source**: KLIFS database or PDB
- **Why**: Large drug target class, ATP-binding site challenges
- **Selection criteria**: Diverse kinase families, resolution < 2.5Å

#### 2. Metal-Binding Proteins (50 complexes)
- **Source**: MetalPDB or manual curation from PDB
- **Why**: Showcase PandaDock's metal coordination handling
- **Selection criteria**: Zn, Mg, Ca, Fe coordination, experimentally validated

**Total: 390 complexes**

## Comparison Tools

### Must Include (Free/Accessible)
1. **AutoDock Vina 1.2** - Gold standard baseline
2. **Smina** - Vina fork with additional scoring
3. **AutoDock-GPU** - GPU performance comparison
4. **rDock** - Open-source, consensus docking

### Optional (If Available)
5. **Glide SP** (Schrödinger) - Commercial standard
6. **GOLD** (CCDC) - Academic license

## Evaluation Metrics

### Pose Prediction Quality
```
- Success Rate (RMSD < 2.0Å) - PRIMARY METRIC
- Top-1, Top-3, Top-5 success rates
- RMSD distribution (box plots)
- Symmetry-corrected RMSD
```

### Binding Affinity Prediction
```
- Pearson correlation coefficient (predicted vs experimental pKd/pKi)
- Spearman rank correlation
- RMSE of binding affinity
- Scatter plots with confidence intervals
```

### Virtual Screening Performance (DUD-E subset)
```
- Enrichment Factor at 1%, 5%, 10%
- AUC (ROC curve)
- BEDROC score
- Early recognition capability
```

### Computational Performance
```
- Mean runtime per complex (seconds)
- Success rate (completed without errors)
- GPU memory usage (for GPU algorithms)
- Throughput (complexes/hour)
- CPU vs GPU speedup
```

## Experimental Design

### Phase 1: Redocking (Pose Prediction)
- Use co-crystallized ligand starting positions
- Metric: RMSD from crystal structure
- Generates: Success rates, RMSD distributions

### Phase 2: Cross-Docking (Binding Affinity)
- Use different conformer as starting point
- Metric: Correlation with experimental binding affinity
- Generates: Pearson R, Spearman ρ, scatter plots

### Phase 3: Virtual Screening (Optional)
- Use DUD-E decoys (50:1 ratio actives:decoys)
- Metric: Enrichment factors, AUC
- Generates: ROC curves, enrichment plots

## PandaDock Algorithms to Test

```
CPU Algorithms:
  - hierarchical_cpu_physics_based (high accuracy)
  - evolutionary_search_cpu (balanced)

GPU Algorithms:
  - cuda_genetic_algorithm (fast convergence)
  - cuda_monte_carlo (thorough sampling)
  - enhanced_hierarchical_gpu (best of both)
```

## Expected Results Table

| Method | Success Rate (%) | Pearson R | Mean Time (s) | GPU Speedup |
|--------|-----------------|-----------|---------------|-------------|
| Vina | ~50-60 | 0.5-0.6 | 120 | N/A |
| AutoDock-GPU | ~55-65 | 0.45-0.55 | 15 | 8x |
| Smina | ~52-62 | 0.52-0.62 | 150 | N/A |
| **PandaDock Hierarchical CPU** | **?** | **?** | 180 | N/A |
| **PandaDock CUDA GA** | **?** | **?** | 25 | **7x** |
| **PandaDock CUDA MC** | **?** | **?** | 35 | **5x** |

## Figures for Publication

### Figure 1: Success Rate Comparison
- Bar chart: Success rates across all methods
- Error bars: 95% confidence intervals
- Grouped by dataset (Core Set, Kinases, Metal-binding)

### Figure 2: Binding Affinity Correlation
- Scatter plots: Predicted vs experimental (6 panels, one per method)
- Regression lines with Pearson R values
- Highlight outliers

### Figure 3: Performance Analysis
- Panel A: Runtime distribution (violin plots)
- Panel B: Success rate vs runtime (scatter)
- Panel C: GPU speedup (bar chart)

### Figure 4: Category-Specific Performance
- Heatmap: Success rates by protein family
- Shows where PandaDock excels

### Supplementary Figures
- ROC curves for virtual screening
- RMSD distributions per category
- Convergence plots for optimization algorithms

## Timeline

```
Week 1-2:   Dataset preparation (PDBbind Core Set cleaning)
Week 3-4:   Category-specific dataset curation (kinases, metals)
Week 5-8:   Run all benchmarks (PandaDock + comparison tools)
Week 9-10:  Data analysis and statistical tests
Week 11-12: Figure generation and manuscript writing
```

## Statistical Analysis

- **Wilcoxon signed-rank test** for paired comparisons (same complexes)
- **Mann-Whitney U test** for unpaired comparisons
- **Bonferroni correction** for multiple comparisons
- **Bootstrap confidence intervals** (n=1000) for correlation coefficients

## Target Journals

**Tier 1 (High Impact):**
- Journal of Chemical Information and Modeling (JCIM)
- Journal of Medicinal Chemistry
- Nature Communications (if prospective validation included)

**Tier 2 (Solid):**
- Journal of Computer-Aided Molecular Design (JCAMD)
- Molecules
- Bioinformatics

## Key Selling Points for PandaDock

1. **GPU Acceleration** - Faster than Vina, competitive with AutoDock-GPU
2. **Metal Coordination** - Better handling than most tools
3. **Multiple Algorithms** - User can choose speed vs accuracy
4. **Open Source** - Free alternative to Glide/GOLD
5. **Modern Stack** - Python/CuPy vs outdated C++ tools

## Data Repository

All benchmark results will be deposited at:
- GitHub repository: benchmarking/ directory
- Zenodo DOI: For permanent archival
- Supporting Information: Raw data tables

## Reproducibility Checklist

- [ ] All dataset PDB IDs listed
- [ ] Exact software versions documented
- [ ] Random seeds specified
- [ ] All parameters in methods section
- [ ] Scripts available on GitHub
- [ ] Docker container for exact environment
