#!/usr/bin/env python3
"""
Analyze Correlation Between Docking Scores and Experimental Binding Affinities

This script compares docking scores from PandaDock benchmarking with experimental
binding affinity data (Ki/Kd values) from PDBbind v2020.

Usage:
    python analyze_binding_affinity_correlation.py \
        --results benchmarking/comprehensive_benchmark/benchmark_results.csv \
        --pdbbind-index benchmarking/full_benchmark/PDBbind_v2020/index/INDEX_general_PL_data.2020 \
        --output benchmarking/comprehensive_benchmark/binding_affinity_analysis

Author: PandaDock Benchmarking Suite
Date: December 2025
"""

import os
import sys
import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality plotting defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9


class PDBbindIndexParser:
    """Parse PDBbind index file to extract binding affinity data"""

    def __init__(self, index_file: Path):
        self.index_file = index_file
        self.data = self.parse_index()

    def parse_index(self) -> pd.DataFrame:
        """
        Parse PDBbind INDEX file

        Format: PDB_code  resolution  year  -logKd/Ki  Kd/Ki_value  reference  ligand_name
        Example: 1a9q  2.80  1996  6.30  Kd=500nM      // 1a9q.pdf (CPA)
        """
        records = []

        with open(self.index_file, 'r') as f:
            for line in f:
                line = line.strip()

                # Skip comments and headers
                if line.startswith('#') or not line or line.startswith('='):
                    continue

                # Parse line
                parts = line.split()
                if len(parts) < 5:
                    continue

                try:
                    pdb_id = parts[0].lower()
                    resolution = float(parts[1])
                    year = int(parts[2])
                    pK = float(parts[3])  # -log(Kd/Ki)
                    affinity_str = parts[4]  # e.g., "Kd=500nM" or "Ki=1.2uM"

                    # Extract affinity value and unit
                    affinity_value, affinity_unit, affinity_type = self._parse_affinity(affinity_str)

                    records.append({
                        'pdb_id': pdb_id,
                        'resolution': resolution,
                        'year': year,
                        'pK': pK,  # -log(Kd/Ki) in M
                        'affinity_value': affinity_value,
                        'affinity_unit': affinity_unit,
                        'affinity_type': affinity_type,  # Kd, Ki, or IC50
                        'affinity_nM': self._convert_to_nM(affinity_value, affinity_unit)
                    })

                except (ValueError, IndexError) as e:
                    continue

        df = pd.DataFrame(records)
        print(f"Parsed {len(df)} complexes from PDBbind index")

        return df

    def _parse_affinity(self, affinity_str: str) -> Tuple[float, str, str]:
        """
        Parse affinity string like 'Kd=500nM' or 'Ki=1.2uM'

        Returns:
            (value, unit, type) e.g., (500, 'nM', 'Kd')
        """
        # Handle special cases
        if '>' in affinity_str or '<' in affinity_str or '~' in affinity_str:
            affinity_str = affinity_str.replace('>', '').replace('<', '').replace('~', '')

        # Extract type (Kd, Ki, IC50)
        if 'IC50' in affinity_str:
            affinity_type = 'IC50'
        elif 'Kd' in affinity_str:
            affinity_type = 'Kd'
        elif 'Ki' in affinity_str:
            affinity_type = 'Ki'
        else:
            affinity_type = 'Unknown'

        # Extract value and unit
        # Match patterns like: 500nM, 1.2uM, 0.5mM, etc.
        match = re.search(r'([\d.]+)(pM|nM|uM|mM|M)', affinity_str)
        if match:
            value = float(match.group(1))
            unit = match.group(2)
            return value, unit, affinity_type

        return None, None, affinity_type

    def _convert_to_nM(self, value: Optional[float], unit: Optional[str]) -> Optional[float]:
        """Convert affinity to nM"""
        if value is None or unit is None:
            return None

        conversion = {
            'pM': 0.001,
            'nM': 1.0,
            'uM': 1000.0,
            'mM': 1e6,
            'M': 1e9
        }

        return value * conversion.get(unit, 1.0)


class BindingAffinityCorrelationAnalyzer:
    """Analyze correlation between docking scores and experimental binding affinities"""

    def __init__(self, results_file: Path, pdbbind_index: Path, output_dir: Path):
        self.results_file = results_file
        self.pdbbind_index = pdbbind_index
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load data
        self.docking_results = pd.read_csv(results_file)
        self.pdbbind_parser = PDBbindIndexParser(pdbbind_index)
        self.pdbbind_data = self.pdbbind_parser.data

        # Merge data
        self.merged_data = self.merge_datasets()

    def merge_datasets(self) -> pd.DataFrame:
        """Merge docking results with PDBbind binding affinity data"""

        # Filter successful docking results only
        successful = self.docking_results[self.docking_results['success'] == True].copy()

        print(f"\nMerging datasets:")
        print(f"  Successful docking runs: {len(successful)}")
        print(f"  PDBbind complexes: {len(self.pdbbind_data)}")

        # Merge on pdb_id
        merged = successful.merge(
            self.pdbbind_data,
            on='pdb_id',
            how='inner'
        )

        print(f"  Merged complexes: {merged['pdb_id'].nunique()}")
        print(f"  Total merged rows: {len(merged)}")

        return merged

    def calculate_correlations(self) -> Dict[str, pd.DataFrame]:
        """Calculate correlation statistics for each algorithm"""

        algorithms = self.merged_data['algorithm'].unique()
        correlation_results = {}

        print(f"\n{'='*70}")
        print("CORRELATION ANALYSIS: Docking Score vs Experimental Binding Affinity")
        print(f"{'='*70}")

        for algo in algorithms:
            algo_data = self.merged_data[self.merged_data['algorithm'] == algo].copy()

            if len(algo_data) < 3:
                print(f"\n{algo}: Insufficient data (n={len(algo_data)})")
                continue

            # Get scores and experimental pK values
            scores = algo_data['best_score'].values
            pK_values = algo_data['pK'].values

            # Calculate correlations
            pearson_r, pearson_p = stats.pearsonr(scores, pK_values)
            spearman_rho, spearman_p = stats.spearmanr(scores, pK_values)
            kendall_tau, kendall_p = stats.kendalltau(scores, pK_values)

            # Calculate RMSE
            # Normalize scores for RMSE calculation
            scores_norm = (scores - scores.mean()) / scores.std()
            pK_norm = (pK_values - pK_values.mean()) / pK_values.std()
            rmse = np.sqrt(np.mean((scores_norm - pK_norm)**2))

            correlation_results[algo] = {
                'algorithm': algo,
                'n_complexes': len(algo_data),
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'spearman_rho': spearman_rho,
                'spearman_p': spearman_p,
                'kendall_tau': kendall_tau,
                'kendall_p': kendall_p,
                'rmse': rmse
            }

            print(f"\n{algo} (n={len(algo_data)}):")
            print(f"  Pearson R:   {pearson_r:7.3f} (p={pearson_p:.2e})")
            print(f"  Spearman ρ:  {spearman_rho:7.3f} (p={spearman_p:.2e})")
            print(f"  Kendall τ:   {kendall_tau:7.3f} (p={kendall_p:.2e})")
            print(f"  RMSE (norm): {rmse:7.3f}")

        # Save correlation summary
        correlation_df = pd.DataFrame(correlation_results.values())
        correlation_df = correlation_df.sort_values('pearson_r', key=abs, ascending=False)
        correlation_df.to_csv(self.output_dir / 'correlation_statistics.csv', index=False)

        print(f"\n{'='*70}")
        print(f"Correlation statistics saved to: {self.output_dir / 'correlation_statistics.csv'}")

        return correlation_results

    def plot_correlation_scatter(self):
        """Generate scatter plots of docking score vs experimental pK"""

        algorithms = self.merged_data['algorithm'].unique()
        n_algos = len(algorithms)

        # Create subplot grid
        n_cols = 3
        n_rows = (n_algos + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_algos > 1 else [axes]

        for idx, algo in enumerate(sorted(algorithms)):
            ax = axes[idx]
            algo_data = self.merged_data[self.merged_data['algorithm'] == algo]

            if len(algo_data) < 3:
                ax.text(0.5, 0.5, f'{algo}\nInsufficient data',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            # Scatter plot
            ax.scatter(algo_data['pK'], algo_data['best_score'],
                      alpha=0.6, s=50, edgecolors='black', linewidth=0.5)

            # Add regression line
            z = np.polyfit(algo_data['pK'], algo_data['best_score'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(algo_data['pK'].min(), algo_data['pK'].max(), 100)
            ax.plot(x_line, p(x_line), 'r--', alpha=0.8, linewidth=2)

            # Calculate correlation
            r, p_val = stats.pearsonr(algo_data['pK'], algo_data['best_score'])

            # Labels and title
            ax.set_xlabel('Experimental pK (-log Ki/Kd)', fontweight='bold')
            ax.set_ylabel('Docking Score (kcal/mol)', fontweight='bold')
            ax.set_title(f'{algo}\nR={r:.3f}, p={p_val:.2e}, n={len(algo_data)}')
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(n_algos, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()

        # Save
        output_file = self.output_dir / 'correlation_scatter_plots.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.savefig(output_file.with_suffix('.pdf'), bbox_inches='tight')
        plt.close()

        print(f"Correlation scatter plots saved to: {output_file}")

    def plot_correlation_heatmap(self, correlation_results: Dict):
        """Generate heatmap of correlation coefficients"""

        # Prepare data for heatmap
        algorithms = sorted(correlation_results.keys())
        metrics = ['pearson_r', 'spearman_rho', 'kendall_tau']
        metric_labels = ['Pearson R', 'Spearman ρ', 'Kendall τ']

        data = np.array([[correlation_results[algo][metric] for metric in metrics]
                        for algo in algorithms])

        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, len(algorithms) * 0.5 + 2))

        im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)

        # Set ticks
        ax.set_xticks(np.arange(len(metrics)))
        ax.set_yticks(np.arange(len(algorithms)))
        ax.set_xticklabels(metric_labels)
        ax.set_yticklabels(algorithms)

        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

        # Add text annotations
        for i in range(len(algorithms)):
            for j in range(len(metrics)):
                text = ax.text(j, i, f'{data[i, j]:.3f}',
                             ha="center", va="center", color="black", fontsize=9)

        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Correlation Coefficient', rotation=270, labelpad=20)

        ax.set_title('Correlation: Docking Score vs Experimental Binding Affinity',
                    fontweight='bold', pad=20)

        plt.tight_layout()

        # Save
        output_file = self.output_dir / 'correlation_heatmap.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.savefig(output_file.with_suffix('.pdf'), bbox_inches='tight')
        plt.close()

        print(f"Correlation heatmap saved to: {output_file}")

    def analyze_by_affinity_range(self):
        """Analyze performance across different binding affinity ranges"""

        # Define affinity ranges based on pK
        bins = [0, 4, 6, 8, 10, 15]
        labels = ['Very Weak\n(pK<4)', 'Weak\n(4≤pK<6)', 'Moderate\n(6≤pK<8)',
                 'Strong\n(8≤pK<10)', 'Very Strong\n(pK≥10)']

        self.merged_data['affinity_range'] = pd.cut(
            self.merged_data['pK'], bins=bins, labels=labels, include_lowest=True
        )

        # Analyze each algorithm
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))

        # Plot 1: Success rate by affinity range
        ax1 = axes[0]
        pivot_success = pd.crosstab(
            self.merged_data['algorithm'],
            self.merged_data['affinity_range'],
            normalize='columns'
        )
        pivot_success.plot(kind='bar', ax=ax1, width=0.8)
        ax1.set_xlabel('Algorithm', fontweight='bold')
        ax1.set_ylabel('Proportion', fontweight='bold')
        ax1.set_title('Algorithm Distribution Across Binding Affinity Ranges', fontweight='bold')
        ax1.legend(title='Affinity Range', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3, axis='y')
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

        # Plot 2: Mean RMSD by affinity range
        ax2 = axes[1]
        pivot_rmsd = self.merged_data.groupby(['algorithm', 'affinity_range'])['rmsd_angstrom'].mean().unstack()
        pivot_rmsd.plot(kind='bar', ax=ax2, width=0.8)
        ax2.set_xlabel('Algorithm', fontweight='bold')
        ax2.set_ylabel('Mean RMSD (Å)', fontweight='bold')
        ax2.set_title('Mean RMSD Across Binding Affinity Ranges', fontweight='bold')
        ax2.legend(title='Affinity Range', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.axhline(y=2.0, color='r', linestyle='--', alpha=0.7, label='Success threshold (2Å)')
        ax2.grid(True, alpha=0.3, axis='y')
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

        plt.tight_layout()

        # Save
        output_file = self.output_dir / 'performance_by_affinity_range.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.savefig(output_file.with_suffix('.pdf'), bbox_inches='tight')
        plt.close()

        print(f"Affinity range analysis saved to: {output_file}")

    def generate_summary_report(self, correlation_results: Dict):
        """Generate comprehensive text summary report"""

        report_file = self.output_dir / 'BINDING_AFFINITY_CORRELATION_REPORT.md'

        with open(report_file, 'w') as f:
            f.write("# Binding Affinity Correlation Analysis Report\n\n")
            f.write(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")

            # Dataset summary
            f.write("## Dataset Summary\n\n")
            f.write(f"- **Total docking runs:** {len(self.docking_results)}\n")
            f.write(f"- **Successful docking runs:** {len(self.docking_results[self.docking_results['success']])}\n")
            f.write(f"- **Complexes with binding affinity data:** {self.merged_data['pdb_id'].nunique()}\n")
            f.write(f"- **Algorithms tested:** {self.merged_data['algorithm'].nunique()}\n\n")

            # Correlation results
            f.write("## Correlation Statistics\n\n")
            f.write("Correlation between docking scores and experimental binding affinity (-log Ki/Kd):\n\n")
            f.write("| Algorithm | N | Pearson R | Spearman ρ | Kendall τ | RMSE |\n")
            f.write("|-----------|---|-----------|------------|-----------|------|\n")

            for algo in sorted(correlation_results.keys()):
                results = correlation_results[algo]
                f.write(f"| {algo} | {results['n_complexes']} | "
                       f"{results['pearson_r']:.3f} | {results['spearman_rho']:.3f} | "
                       f"{results['kendall_tau']:.3f} | {results['rmse']:.3f} |\n")

            f.write("\n")

            # Interpretation
            f.write("## Interpretation\n\n")
            f.write("### Correlation Coefficients:\n")
            f.write("- **|R| > 0.7:** Strong correlation\n")
            f.write("- **0.4 < |R| < 0.7:** Moderate correlation\n")
            f.write("- **0.2 < |R| < 0.4:** Weak correlation\n")
            f.write("- **|R| < 0.2:** Very weak/no correlation\n\n")

            f.write("### Statistical Significance:\n")
            f.write("- **p < 0.001:** Highly significant (***)\n")
            f.write("- **p < 0.01:** Significant (**)\n")
            f.write("- **p < 0.05:** Marginally significant (*)\n")
            f.write("- **p ≥ 0.05:** Not significant\n\n")

            # Best performers
            best_pearson = max(correlation_results.items(),
                             key=lambda x: abs(x[1]['pearson_r']))
            best_spearman = max(correlation_results.items(),
                              key=lambda x: abs(x[1]['spearman_rho']))

            f.write("## Best Performing Algorithms\n\n")
            f.write(f"**Best Pearson correlation:** {best_pearson[0]} (R={best_pearson[1]['pearson_r']:.3f})\n\n")
            f.write(f"**Best Spearman correlation:** {best_spearman[0]} (ρ={best_spearman[1]['spearman_rho']:.3f})\n\n")

            # Files generated
            f.write("## Generated Files\n\n")
            f.write("- `correlation_statistics.csv` - Detailed correlation statistics\n")
            f.write("- `correlation_scatter_plots.png/pdf` - Scatter plots for all algorithms\n")
            f.write("- `correlation_heatmap.png/pdf` - Heatmap of correlation coefficients\n")
            f.write("- `performance_by_affinity_range.png/pdf` - Performance across affinity ranges\n")
            f.write("- `merged_data.csv` - Complete merged dataset with docking results and binding affinities\n\n")

        # Save merged data
        self.merged_data.to_csv(self.output_dir / 'merged_data.csv', index=False)

        print(f"\nSummary report saved to: {report_file}")

    def run_full_analysis(self):
        """Run complete binding affinity correlation analysis"""

        print(f"\n{'='*70}")
        print("BINDING AFFINITY CORRELATION ANALYSIS")
        print(f"{'='*70}\n")

        # Calculate correlations
        correlation_results = self.calculate_correlations()

        # Generate plots
        print(f"\nGenerating visualizations...")
        self.plot_correlation_scatter()
        self.plot_correlation_heatmap(correlation_results)
        self.analyze_by_affinity_range()

        # Generate report
        self.generate_summary_report(correlation_results)

        print(f"\n{'='*70}")
        print("ANALYSIS COMPLETE!")
        print(f"{'='*70}")
        print(f"\nAll results saved to: {self.output_dir}")
        print(f"\nKey files:")
        print(f"  - BINDING_AFFINITY_CORRELATION_REPORT.md (summary report)")
        print(f"  - correlation_statistics.csv (detailed stats)")
        print(f"  - correlation_scatter_plots.png (visual comparison)")
        print(f"  - merged_data.csv (complete dataset)")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze correlation between docking scores and experimental binding affinities"
    )
    parser.add_argument(
        '--results',
        type=Path,
        required=True,
        help='Path to benchmark_results.csv file'
    )
    parser.add_argument(
        '--pdbbind-index',
        type=Path,
        required=True,
        help='Path to PDBbind INDEX_general_PL_data.2020 file'
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output directory for analysis results'
    )

    args = parser.parse_args()

    # Run analysis
    analyzer = BindingAffinityCorrelationAnalyzer(
        results_file=args.results,
        pdbbind_index=args.pdbbind_index,
        output_dir=args.output
    )

    analyzer.run_full_analysis()


if __name__ == '__main__':
    main()
