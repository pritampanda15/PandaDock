# Binding Affinity Correlation Analysis Report

**Generated:** 2025-12-08 00:14:15

---

## Dataset Summary

- **Total docking runs:** 1200
- **Successful docking runs:** 1079
- **Complexes with binding affinity data:** 150
- **Algorithms tested:** 8

## Correlation Statistics

Correlation between docking scores and experimental binding affinity (-log Ki/Kd):

| Algorithm | N | Pearson R | Spearman ρ | Kendall τ | RMSE |
|-----------|---|-----------|------------|-----------|------|
| crystal_guided_cpu | 150 | nan | nan | nan | nan |
| cuda_genetic_algorithm | 150 | -0.064 | -0.079 | -0.044 | 1.459 |
| cuda_monte_carlo | 73 | -0.088 | 0.088 | 0.064 | 1.475 |
| enhanced_hierarchical_cpu | 150 | 0.033 | 0.035 | 0.025 | 1.391 |
| enhanced_hierarchical_gpu | 137 | -0.193 | -0.153 | -0.097 | 1.544 |
| genetic_algorithm_cpu | 134 | -0.010 | 0.011 | 0.006 | 1.421 |
| hierarchical_cpu | 142 | -0.066 | -0.060 | -0.049 | 1.460 |
| monte_carlo_cpu | 143 | -0.265 | -0.246 | -0.181 | 1.591 |

## Interpretation

### Correlation Coefficients:
- **|R| > 0.7:** Strong correlation
- **0.4 < |R| < 0.7:** Moderate correlation
- **0.2 < |R| < 0.4:** Weak correlation
- **|R| < 0.2:** Very weak/no correlation

### Statistical Significance:
- **p < 0.001:** Highly significant (***)
- **p < 0.01:** Significant (**)
- **p < 0.05:** Marginally significant (*)
- **p ≥ 0.05:** Not significant

## Best Performing Algorithms

**Best Pearson correlation:** monte_carlo_cpu (R=-0.265)

**Best Spearman correlation:** monte_carlo_cpu (ρ=-0.246)

## Generated Files

- `correlation_statistics.csv` - Detailed correlation statistics
- `correlation_scatter_plots.png/pdf` - Scatter plots for all algorithms
- `correlation_heatmap.png/pdf` - Heatmap of correlation coefficients
- `performance_by_affinity_range.png/pdf` - Performance across affinity ranges
- `merged_data.csv` - Complete merged dataset with docking results and binding affinities

