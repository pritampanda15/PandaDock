# PandaDock Benchmarking Suite

Comprehensive benchmarking tools for comparing PandaDock against other docking software for academic publication.

## Quick Start

### 1. Prepare Benchmark Dataset

Download and prepare PDBbind Core Set (290 complexes):

```bash
# Install dependencies
pip install biopython rdkit pandas requests tqdm

# Prepare dataset (test with 10 complexes first)
python benchmarking/prepare_pdbbind_core.py \
  --output benchmarking/pdbbind_core_set \
  --max-complexes 10

# Full dataset (290 complexes)
python benchmarking/prepare_pdbbind_core.py \
  --output benchmarking/pdbbind_core_set
```

This will:
- Download PDBbind index file
- Filter by quality (resolution < 2.5Å, reasonable affinity)
- Download PDB structures
- Extract receptors and ligands
- Create metadata CSV with experimental binding affinities

**Output structure:**
```
benchmarking/pdbbind_core_set/
├── benchmark_metadata.csv          # Binding affinities, resolution, etc.
├── pdbs/                           # Original PDB files
├── receptors/                      # Prepared receptor PDB files
└── ligands/                        # Ligand SDF files
```

### 2. Install Comparison Tools

**Required:**
```bash
# AutoDock Vina
conda install -c conda-forge vina

# Smina (Vina fork)
# Download from: https://sourceforge.net/projects/smina/
# Or: conda install -c conda-forge smina
```

**Optional (if available):**
- AutoDock-GPU: https://github.com/ccsb-scripps/AutoDock-GPU
- Glide (Schrödinger suite - commercial)
- GOLD (CCDC - academic license)

### 3. Run Benchmark

```bash
# Test run (10 complexes, 2 algorithms)
python benchmarking/run_benchmark_comparison.py \
  --benchmark-dir benchmarking/pdbbind_core_set \
  --output-dir benchmarking/results_test \
  --algorithms pandadock_hierarchical_cpu_physics_based vina \
  --n-jobs 4

# Full benchmark (all algorithms)
python benchmarking/run_benchmark_comparison.py \
  --benchmark-dir benchmarking/pdbbind_core_set \
  --output-dir benchmarking/results_full \
  --algorithms \
    pandadock_hierarchical_cpu_physics_based \
    pandadock_cuda_genetic_algorithm \
    pandadock_cuda_monte_carlo \
    vina \
    smina \
  --n-jobs 8
```

**Expected runtime:**
- Test (10 complexes × 2 algorithms): ~30 minutes
- Full (290 complexes × 5 algorithms): ~24-48 hours (depends on hardware)

### 4. Analyze Results

```bash
# Generate all figures and statistics
python benchmarking/analyze_results.py \
  --results benchmarking/results_full/benchmark_results.csv \
  --metadata benchmarking/pdbbind_core_set/benchmark_metadata.csv \
  --output-dir benchmarking/analysis
```

**Generates:**
- `figures/fig1_success_rates.png` - Bar chart of success rates (RMSD < 2Å, < 3Å)
- `figures/fig2_rmsd_distribution.png` - Violin plots of RMSD distributions
- `figures/fig3_affinity_correlation.png` - Predicted vs experimental binding affinity
- `figures/fig4_runtime_comparison.png` - Runtime comparison box plots
- `summary_statistics.csv` - Comprehensive summary table
- `statistical_tests.csv` - Pairwise statistical significance tests

## Benchmarking Strategy

### Dataset Composition

**Tier 1: Standard Benchmark (Essential)**
- PDBbind Core Set 2020: 290 diverse complexes
- Most cited benchmark in docking literature
- Covers diverse protein families and ligand types
- Experimental binding affinities available

**Tier 2: Category-Specific (Optional)**
- Kinases: 50-100 complexes from KLIFS
- Metal-binding proteins: 50 complexes (showcase PandaDock strength)
- GPCRs: 30-50 complexes from GPCRdb

**Recommendation for Publication:**
- **Good paper**: 200-300 complexes (Core Set only)
- **Strong paper**: 400-500 complexes (Core Set + 1-2 categories)
- **Exceptional paper**: 500+ complexes + prospective validation

### Evaluation Metrics

#### Pose Prediction (Primary)
- **Success Rate (RMSD < 2Å)** - Gold standard metric
- Top-1, Top-3, Top-5 success rates
- RMSD distribution analysis
- Symmetry-corrected RMSD

#### Binding Affinity Prediction (Secondary)
- Pearson correlation (linear relationship)
- Spearman correlation (rank-order)
- RMSE of predicted vs experimental pKd

#### Computational Performance
- Mean runtime per complex
- Completion rate (didn't crash/timeout)
- GPU speedup (vs CPU baseline)

#### Virtual Screening (Optional)
- Enrichment Factor at 1%, 5%, 10%
- AUC (ROC curve)
- BEDROC score

### Statistical Tests

- **Wilcoxon signed-rank test**: Paired comparisons (same complexes)
- **Mann-Whitney U test**: Unpaired comparisons
- **Bonferroni correction**: Multiple comparison adjustment
- **Bootstrap confidence intervals**: For correlation coefficients

## Output Interpretation

### Success Rate Table

| Algorithm | Success Rate (RMSD < 2Å) | Mean RMSD | Pearson R | Runtime (s) |
|-----------|--------------------------|-----------|-----------|-------------|
| Vina | 55-65% | 2.1-2.5 | 0.5-0.6 | 120 |
| PandaDock Hierarchical | **60-70%** | **1.9-2.3** | **0.55-0.65** | 180 |
| PandaDock CUDA GA | **58-68%** | **2.0-2.4** | **0.52-0.62** | **25** |

**Interpretation:**
- **Success rate > 50%**: Acceptable for publication
- **Success rate > 60%**: Competitive with state-of-art
- **Success rate > 70%**: Exceptional performance
- **Pearson R > 0.5**: Reasonable affinity prediction
- **Pearson R > 0.6**: Strong affinity prediction

### Statistical Significance

p-value interpretation:
- **p < 0.05**: Statistically significant difference
- **p < 0.01**: Highly significant
- **p < 0.001**: Very highly significant

After Bonferroni correction for 10 comparisons:
- Significant at α = 0.05/10 = 0.005

## Advanced Usage

### Custom Category Benchmarks

Create kinase-specific benchmark:

```python
# benchmarking/prepare_kinase_set.py
import pandas as pd
from pathlib import Path

# Download kinase structures from KLIFS database
# Filter by quality criteria
# Extract and prepare for docking

# Save in same format as PDBbind Core Set
```

### Adding New Comparison Tools

To add a new docking tool, edit `run_benchmark_comparison.py`:

```python
def run_custom_tool(self, pdb_id, receptor_file, ligand_file, box):
    """Run custom docking tool"""
    # Prepare inputs
    # Run tool via subprocess
    # Parse output
    # Calculate RMSD
    # Return result dict
    return {
        'pdb_id': pdb_id,
        'algorithm': 'custom_tool',
        'success': True,
        'runtime': runtime,
        'best_score': score,
        'rmsd': rmsd
    }
```

### Parallel Execution

Use GNU Parallel for distributed execution:

```bash
# Split dataset into chunks
split -l 50 benchmark_metadata.csv chunk_

# Run each chunk in parallel
parallel -j 4 python benchmarking/run_benchmark_comparison.py \
  --benchmark-dir benchmarking/pdbbind_core_set \
  --output-dir benchmarking/results_{#} \
  --metadata {} ::: chunk_*

# Combine results
cat benchmarking/results_*/benchmark_results.csv > combined_results.csv
```

## Publication Checklist

### Methods Section Must Include:

- [ ] Dataset description (PDBbind Core Set 2020, N complexes)
- [ ] Quality criteria (resolution < 2.5Å, etc.)
- [ ] Comparison tools with exact versions (Vina 1.2.3, etc.)
- [ ] All parameters used (exhaustiveness, box size, etc.)
- [ ] Evaluation metrics (RMSD cutoffs, correlation methods)
- [ ] Statistical tests (Wilcoxon, Bonferroni correction)
- [ ] Hardware specifications (CPU, GPU models)
- [ ] Random seeds (for reproducibility)

### Data Availability:

- [ ] All PDB IDs listed (supplementary table)
- [ ] Raw results CSV deposited (GitHub/Zenodo)
- [ ] Analysis scripts available (GitHub)
- [ ] Docker container for reproducibility

### Key Figures:

1. **Success rate comparison** (bar chart) - Shows PandaDock competitive with/better than Vina
2. **RMSD distribution** (violin plot) - Shows distribution of pose quality
3. **Affinity correlation** (scatter plot) - Shows scoring function quality
4. **Runtime comparison** (box plot) - Shows computational efficiency (GPU advantage)

### Supplementary Materials:

- Complete results table (all complexes)
- Statistical test results (all pairwise comparisons)
- Parameter sensitivity analysis
- Convergence plots for optimization algorithms
- Failure case analysis

## Target Journals

### Tier 1 (High Impact)
- **Journal of Chemical Information and Modeling (JCIM)** - IF ~5.6
  - Focus: Computational chemistry, docking, scoring
  - Requires: Strong benchmark (300+ complexes), comparison to 3+ tools

- **Journal of Medicinal Chemistry** - IF ~7.3
  - Focus: Drug discovery, medicinal chemistry
  - Requires: Prospective validation OR solve specific hard problem

- **Nature Communications** - IF ~16.6
  - Focus: Broad impact, novel methodology
  - Requires: Exceptional results + prospective validation + experimental confirmation

### Tier 2 (Solid Specialty Journals)
- **Journal of Computer-Aided Molecular Design (JCAMD)** - IF ~3.0
  - Focus: Docking, structure-based design
  - Requires: Thorough benchmark (200+ complexes)

- **Molecules** - IF ~4.6
  - Focus: Computational chemistry, open access
  - Requires: Solid benchmark, comparison to established tools

- **Bioinformatics** - IF ~5.8
  - Focus: Computational methods, software tools
  - Requires: Novel algorithm + benchmark + usable software

## Tips for Strong Publication

### What Makes a Paper Strong:

1. **Large, diverse dataset**: 400+ complexes across protein families
2. **Fair comparison**: Compare to 3-4 established tools (Vina, Glide, etc.)
3. **Complete metrics**: RMSD, affinity correlation, runtime, success rate
4. **Statistical rigor**: Proper tests, multiple comparison correction
5. **Identify niche**: Where does PandaDock excel? (metals? flexibility? speed?)
6. **Ablation studies**: Compare CPU vs GPU, different algorithms
7. **User-friendly**: Easy to install, good documentation, examples

### What Reviewers Look For:

- **Reproducibility**: Can they run your benchmarks?
- **Fair comparison**: Same box size, exhaustiveness, etc. for all tools
- **Statistical significance**: Not just "looks better", but *provably* better
- **Honest limitations**: What doesn't work well?
- **Practical utility**: When should users choose PandaDock?

### Common Pitfalls to Avoid:

- ❌ Too small dataset (< 100 complexes)
- ❌ Cherry-picked examples (show all results)
- ❌ Unfair comparison (different parameters for different tools)
- ❌ Missing error bars / confidence intervals
- ❌ No statistical tests
- ❌ Comparing to outdated tools (Vina 1.1 vs 1.2)
- ❌ Ignoring failed dockings in success rate calculation

## Expected Timeline

**Week 1-2**: Dataset preparation
- Download and clean PDBbind Core Set
- Prepare category-specific sets (if applicable)
- Quality control checks

**Week 3-4**: Method validation
- Test all tools on 10-20 complexes
- Verify RMSD calculations
- Debug any issues

**Week 5-8**: Full benchmark execution
- Run all algorithms on full dataset
- Monitor for failures
- Collect runtime statistics

**Week 9-10**: Analysis
- Generate all figures
- Run statistical tests
- Identify interesting patterns

**Week 11-12**: Manuscript preparation
- Write methods section
- Create supplementary materials
- Prepare data repository

**Total: ~3 months for thorough benchmark + publication**

## Support

For questions or issues:
- GitHub Issues: https://github.com/YOUR_REPO/issues
- Documentation: See main README.md
- Email: your.email@institution.edu

## Citation

If you use this benchmarking suite, please cite:

```bibtex
@article{pandadock2025,
  title={PandaDock: GPU-Accelerated Molecular Docking with Metal Coordination Support},
  author={Your Name},
  journal={Journal Name},
  year={2025},
  doi={...}
}
```
