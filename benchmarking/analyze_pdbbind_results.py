#!/usr/bin/env python3
"""
Comprehensive Analysis and Visualization for PDBbind Benchmarking Results

Generates publication-quality figures and statistical analysis for research papers.

Features:
- Success rate comparison (RMSD < 2Å, < 3Å)
- RMSD distribution analysis (violin/box plots)
- Runtime performance comparison
- Statistical significance testing (Wilcoxon, Mann-Whitney U)
- Per-protein-family analysis (optional)
- Publication-ready figures (high-DPI PNG/PDF)

Usage:
    python analyze_pdbbind_results.py \\
        --results benchmarking/pdbbind_results/benchmark_results.csv \\
        --output benchmarking/pdbbind_results/analysis

Author: PandaDock Benchmarking Suite
Date: December 2025
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib.patches import Rectangle

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'


class PDBbindResultsAnalyzer:
    """Analyze and visualize PDBbind benchmarking results"""

    def __init__(self, results_file: Path, output_dir: Path):
        self.results_file = results_file
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.figures_dir = output_dir / "figures"
        self.figures_dir.mkdir(exist_ok=True)
        self.tables_dir = output_dir / "tables"
        self.tables_dir.mkdir(exist_ok=True)

        # Load results
        print(f"Loading results from {results_file}")
        self.results = pd.read_csv(results_file)

        # Filter successful runs
        self.successful = self.results[
            (self.results['success'] == True) &
            (self.results['rmsd_angstrom'].notna())
        ]

        print(f"Total docking runs: {len(self.results)}")
        print(f"Successful runs with RMSD: {len(self.successful)}")
        print(f"Algorithms: {self.results['algorithm'].nunique()}")
        print()

    def calculate_summary_statistics(self) -> pd.DataFrame:
        """Calculate comprehensive summary statistics"""
        summary_data = []

        for algorithm in self.results['algorithm'].unique():
            alg_data = self.results[self.results['algorithm'] == algorithm]
            alg_successful = self.successful[self.successful['algorithm'] == algorithm]

            # Basic metrics
            total = len(alg_data)
            successful = len(alg_successful)
            completion_rate = (successful / total * 100) if total > 0 else 0

            # RMSD metrics
            if len(alg_successful) > 0:
                rmsd_values = alg_successful['rmsd_angstrom']

                success_2a = (rmsd_values < 2.0).sum()
                success_3a = (rmsd_values < 3.0).sum()

                rate_2a = (success_2a / len(alg_successful) * 100)
                rate_3a = (success_3a / len(alg_successful) * 100)

                mean_rmsd = rmsd_values.mean()
                median_rmsd = rmsd_values.median()
                std_rmsd = rmsd_values.std()
                q1_rmsd = rmsd_values.quantile(0.25)
                q3_rmsd = rmsd_values.quantile(0.75)
            else:
                success_2a = success_3a = 0
                rate_2a = rate_3a = 0
                mean_rmsd = median_rmsd = std_rmsd = q1_rmsd = q3_rmsd = None

            # Runtime metrics
            runtime_data = alg_data[alg_data['runtime_seconds'].notna()]
            if len(runtime_data) > 0:
                mean_runtime = runtime_data['runtime_seconds'].mean()
                median_runtime = runtime_data['runtime_seconds'].median()
                std_runtime = runtime_data['runtime_seconds'].std()
                total_runtime_hours = runtime_data['runtime_seconds'].sum() / 3600
            else:
                mean_runtime = median_runtime = std_runtime = total_runtime_hours = None

            summary_data.append({
                'Algorithm': algorithm,
                'Total Runs': total,
                'Successful': successful,
                'Completion Rate (%)': round(completion_rate, 2),
                'Success <2Å Count': success_2a,
                'Success <2Å Rate (%)': round(rate_2a, 2),
                'Success <3Å Count': success_3a,
                'Success <3Å Rate (%)': round(rate_3a, 2),
                'Mean RMSD (Å)': round(mean_rmsd, 3) if mean_rmsd else None,
                'Median RMSD (Å)': round(median_rmsd, 3) if median_rmsd else None,
                'Std RMSD (Å)': round(std_rmsd, 3) if std_rmsd else None,
                'Q1 RMSD (Å)': round(q1_rmsd, 3) if q1_rmsd else None,
                'Q3 RMSD (Å)': round(q3_rmsd, 3) if q3_rmsd else None,
                'Mean Runtime (s)': round(mean_runtime, 2) if mean_runtime else None,
                'Median Runtime (s)': round(median_runtime, 2) if median_runtime else None,
                'Std Runtime (s)': round(std_runtime, 2) if std_runtime else None,
                'Total Runtime (h)': round(total_runtime_hours, 2) if total_runtime_hours else None,
            })

        df = pd.DataFrame(summary_data)
        return df

    def perform_statistical_tests(self) -> pd.DataFrame:
        """Perform pairwise statistical significance tests"""
        algorithms = self.successful['algorithm'].unique()

        if len(algorithms) < 2:
            print("WARNING: Need at least 2 algorithms for statistical tests")
            return pd.DataFrame()

        test_results = []

        for i, alg1 in enumerate(algorithms):
            for alg2 in algorithms[i+1:]:
                # Get RMSD values for common complexes
                alg1_data = self.successful[self.successful['algorithm'] == alg1]
                alg2_data = self.successful[self.successful['algorithm'] == alg2]

                # Find common PDB IDs
                common_ids = set(alg1_data['pdb_id']) & set(alg2_data['pdb_id'])

                if len(common_ids) < 5:
                    continue

                # Get RMSD values for common complexes
                rmsd1 = alg1_data[alg1_data['pdb_id'].isin(common_ids)].sort_values('pdb_id')['rmsd_angstrom']
                rmsd2 = alg2_data[alg2_data['pdb_id'].isin(common_ids)].sort_values('pdb_id')['rmsd_angstrom']

                # Wilcoxon signed-rank test (paired)
                try:
                    wilcoxon_stat, wilcoxon_p = stats.wilcoxon(rmsd1, rmsd2)
                except:
                    wilcoxon_stat = wilcoxon_p = None

                # Mann-Whitney U test (unpaired)
                try:
                    mannwhitney_stat, mannwhitney_p = stats.mannwhitneyu(rmsd1, rmsd2, alternative='two-sided')
                except:
                    mannwhitney_stat = mannwhitney_p = None

                # Effect size (Cohen's d)
                try:
                    mean_diff = rmsd1.mean() - rmsd2.mean()
                    pooled_std = np.sqrt((rmsd1.std()**2 + rmsd2.std()**2) / 2)
                    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
                except:
                    cohens_d = None

                test_results.append({
                    'Algorithm 1': alg1,
                    'Algorithm 2': alg2,
                    'N Common': len(common_ids),
                    'Mean RMSD 1': round(rmsd1.mean(), 3),
                    'Mean RMSD 2': round(rmsd2.mean(), 3),
                    'Mean Difference': round(mean_diff, 3) if mean_diff else None,
                    'Wilcoxon p-value': round(wilcoxon_p, 4) if wilcoxon_p else None,
                    'Mann-Whitney p-value': round(mannwhitney_p, 4) if mannwhitney_p else None,
                    "Cohen's d": round(cohens_d, 3) if cohens_d else None,
                    'Significant (p<0.05)': wilcoxon_p < 0.05 if wilcoxon_p else False
                })

        df = pd.DataFrame(test_results)
        return df

    def plot_success_rates(self):
        """Bar chart of success rates"""
        summary = self.calculate_summary_statistics()

        fig, ax = plt.subplots(figsize=(12, 6))

        algorithms = summary['Algorithm']
        x = np.arange(len(algorithms))
        width = 0.35

        bars1 = ax.bar(x - width/2, summary['Success <2Å Rate (%)'],
                      width, label='RMSD < 2.0 Å', alpha=0.8, color='#2E86AB')
        bars2 = ax.bar(x + width/2, summary['Success <3Å Rate (%)'],
                      width, label='RMSD < 3.0 Å', alpha=0.8, color='#A23B72')

        ax.set_xlabel('Algorithm', fontsize=12, fontweight='bold')
        ax.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
        ax.set_title('Docking Success Rates on PDBbind v2020', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(algorithms, rotation=45, ha='right')
        ax.legend(frameon=True, shadow=True)
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                           f'{height:.1f}%',
                           ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        output_file = self.figures_dir / "fig1_success_rates.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.savefig(self.figures_dir / "fig1_success_rates.pdf")
        print(f"✓ Saved: {output_file}")
        plt.close()

    def plot_rmsd_distribution(self):
        """Violin plot of RMSD distributions"""
        # Filter outliers for better visualization
        plot_data = self.successful[self.successful['rmsd_angstrom'] < 10.0].copy()

        fig, ax = plt.subplots(figsize=(14, 7))

        # Create violin plot
        algorithms = sorted(plot_data['algorithm'].unique())
        positions = range(len(algorithms))

        parts = ax.violinplot(
            [plot_data[plot_data['algorithm'] == alg]['rmsd_angstrom'].values
             for alg in algorithms],
            positions=positions,
            showmeans=True,
            showmedians=True,
            widths=0.7
        )

        # Customize violin colors
        for pc in parts['bodies']:
            pc.set_facecolor('#3A86FF')
            pc.set_alpha(0.7)

        # Add threshold lines
        ax.axhline(y=2.0, color='red', linestyle='--', linewidth=2,
                  alpha=0.7, label='2.0 Å Success Threshold')
        ax.axhline(y=3.0, color='orange', linestyle='--', linewidth=2,
                  alpha=0.7, label='3.0 Å Success Threshold')

        ax.set_xlabel('Algorithm', fontsize=12, fontweight='bold')
        ax.set_ylabel('RMSD (Å)', fontsize=12, fontweight='bold')
        ax.set_title('RMSD Distribution by Algorithm (PDBbind v2020)',
                    fontsize=14, fontweight='bold')
        ax.set_xticks(positions)
        ax.set_xticklabels(algorithms, rotation=45, ha='right')
        ax.legend(loc='upper right', frameon=True, shadow=True)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, min(10, plot_data['rmsd_angstrom'].max() + 1))

        plt.tight_layout()
        output_file = self.figures_dir / "fig2_rmsd_distribution.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.savefig(self.figures_dir / "fig2_rmsd_distribution.pdf")
        print(f"✓ Saved: {output_file}")
        plt.close()

    def plot_rmsd_boxplot(self):
        """Box plot of RMSD distributions with outliers"""
        plot_data = self.successful[self.successful['rmsd_angstrom'] < 15.0].copy()

        fig, ax = plt.subplots(figsize=(14, 7))

        algorithms = sorted(plot_data['algorithm'].unique())
        data_to_plot = [plot_data[plot_data['algorithm'] == alg]['rmsd_angstrom'].values
                       for alg in algorithms]

        bp = ax.boxplot(data_to_plot, labels=algorithms, patch_artist=True,
                       showmeans=True, meanline=True)

        # Customize colors
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8',
                 '#F7DC6F', '#BB8FCE', '#85C1E2']
        for patch, color in zip(bp['boxes'], colors * (len(algorithms) // len(colors) + 1)):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Add threshold lines
        ax.axhline(y=2.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='2.0 Å')
        ax.axhline(y=3.0, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='3.0 Å')

        ax.set_xlabel('Algorithm', fontsize=12, fontweight='bold')
        ax.set_ylabel('RMSD (Å)', fontsize=12, fontweight='bold')
        ax.set_title('RMSD Box Plot - PDBbind v2020', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        ax.legend(loc='upper right')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        output_file = self.figures_dir / "fig3_rmsd_boxplot.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.savefig(self.figures_dir / "fig3_rmsd_boxplot.pdf")
        print(f"✓ Saved: {output_file}")
        plt.close()

    def plot_runtime_comparison(self):
        """Runtime comparison across algorithms"""
        runtime_data = self.results[self.results['runtime_seconds'].notna()].copy()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        algorithms = sorted(runtime_data['algorithm'].unique())

        # Plot 1: Mean runtime with error bars
        means = []
        stds = []
        for alg in algorithms:
            alg_runtime = runtime_data[runtime_data['algorithm'] == alg]['runtime_seconds']
            means.append(alg_runtime.mean())
            stds.append(alg_runtime.std())

        x = np.arange(len(algorithms))
        bars = ax1.bar(x, means, yerr=stds, alpha=0.8, capsize=5, color='#8E44AD')

        ax1.set_xlabel('Algorithm', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Mean Runtime (seconds)', fontsize=12, fontweight='bold')
        ax1.set_title('Average Runtime per Complex', fontsize=13, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(algorithms, rotation=45, ha='right')
        ax1.grid(axis='y', alpha=0.3)

        # Add value labels
        for i, (bar, mean) in enumerate(zip(bars, means)):
            ax1.text(bar.get_x() + bar.get_width()/2., mean + stds[i] + 2,
                    f'{mean:.1f}s',
                    ha='center', va='bottom', fontsize=9)

        # Plot 2: Runtime distribution box plot
        data_to_plot = [runtime_data[runtime_data['algorithm'] == alg]['runtime_seconds'].values
                       for alg in algorithms]

        bp = ax2.boxplot(data_to_plot, labels=algorithms, patch_artist=True)

        colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6',
                 '#1ABC9C', '#E67E22', '#34495E']
        for patch, color in zip(bp['boxes'], colors * (len(algorithms) // len(colors) + 1)):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax2.set_xlabel('Algorithm', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Runtime (seconds)', fontsize=12, fontweight='bold')
        ax2.set_title('Runtime Distribution', fontsize=13, fontweight='bold')
        ax2.set_xticklabels(algorithms, rotation=45, ha='right')
        ax2.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        output_file = self.figures_dir / "fig4_runtime_comparison.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.savefig(self.figures_dir / "fig4_runtime_comparison.pdf")
        print(f"✓ Saved: {output_file}")
        plt.close()

    def plot_success_vs_runtime(self):
        """Scatter plot: Success rate vs Runtime"""
        summary = self.calculate_summary_statistics()

        fig, ax = plt.subplots(figsize=(10, 7))

        # Filter valid data
        plot_data = summary[
            (summary['Success <2Å Rate (%)'].notna()) &
            (summary['Mean Runtime (s)'].notna())
        ]

        ax.scatter(plot_data['Mean Runtime (s)'],
                  plot_data['Success <2Å Rate (%)'],
                  s=200, alpha=0.7, c='#E74C3C', edgecolors='black', linewidth=2)

        # Add labels
        for _, row in plot_data.iterrows():
            ax.annotate(row['Algorithm'],
                       (row['Mean Runtime (s)'], row['Success <2Å Rate (%)']),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=9, alpha=0.8)

        ax.set_xlabel('Mean Runtime per Complex (seconds)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Success Rate - RMSD < 2.0 Å (%)', fontsize=12, fontweight='bold')
        ax.set_title('Performance Trade-off: Accuracy vs Speed', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)

        plt.tight_layout()
        output_file = self.figures_dir / "fig5_success_vs_runtime.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.savefig(self.figures_dir / "fig5_success_vs_runtime.pdf")
        print(f"✓ Saved: {output_file}")
        plt.close()

    def create_summary_report(self):
        """Create comprehensive summary report"""
        print("\n" + "="*70)
        print("GENERATING ANALYSIS REPORT")
        print("="*70)

        # Calculate statistics
        summary_stats = self.calculate_summary_statistics()
        statistical_tests = self.perform_statistical_tests()

        # Save tables
        summary_file = self.tables_dir / "summary_statistics.csv"
        summary_stats.to_csv(summary_file, index=False)
        print(f"\n✓ Summary statistics: {summary_file}")

        if len(statistical_tests) > 0:
            tests_file = self.tables_dir / "statistical_tests.csv"
            statistical_tests.to_csv(tests_file, index=False)
            print(f"✓ Statistical tests: {tests_file}")

        # Print summary to console
        print("\n" + "="*70)
        print("SUMMARY STATISTICS")
        print("="*70)
        print(summary_stats.to_string(index=False))

        if len(statistical_tests) > 0:
            print("\n" + "="*70)
            print("PAIRWISE STATISTICAL TESTS")
            print("="*70)
            print(statistical_tests.to_string(index=False))

        # Generate all figures
        print("\n" + "="*70)
        print("GENERATING FIGURES")
        print("="*70)

        self.plot_success_rates()
        self.plot_rmsd_distribution()
        self.plot_rmsd_boxplot()
        self.plot_runtime_comparison()
        self.plot_success_vs_runtime()

        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70)
        print(f"Output directory: {self.output_dir}")
        print(f"Figures: {self.figures_dir}")
        print(f"Tables: {self.tables_dir}")
        print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze PDBbind benchmarking results and generate publication figures"
    )

    parser.add_argument(
        "-r", "--results",
        type=Path,
        required=True,
        help="Path to benchmark_results.csv"
    )

    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("benchmarking/pdbbind_results/analysis"),
        help="Output directory for analysis (default: benchmarking/pdbbind_results/analysis)"
    )

    args = parser.parse_args()

    if not args.results.exists():
        print(f"ERROR: Results file not found: {args.results}")
        return 1

    analyzer = PDBbindResultsAnalyzer(args.results, args.output)
    analyzer.create_summary_report()

    return 0


if __name__ == "__main__":
    exit(main())
