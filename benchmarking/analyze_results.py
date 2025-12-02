#!/usr/bin/env python3
"""
Analyze benchmark results and generate publication-quality figures

Creates:
- Success rate comparison (bar chart)
- RMSD distribution (violin plots)
- Correlation plots (predicted vs experimental affinity)
- Runtime comparison
- Statistical significance tests
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from typing import Dict, List

# Set publication style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("colorblind")

class BenchmarkAnalyzer:
    """Analyze and visualize benchmark results"""

    def __init__(self, results_file: Path, metadata_file: Path, output_dir: Path):
        self.results = pd.read_csv(results_file)
        self.metadata = pd.read_csv(metadata_file)
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Merge results with metadata
        self.data = self.results.merge(self.metadata, on='pdb_id', how='left')

    def calculate_success_rates(self) -> pd.DataFrame:
        """Calculate success rates (RMSD < 2Å) for each algorithm"""
        success_data = []

        for algorithm in self.data['algorithm'].unique():
            alg_data = self.data[
                (self.data['algorithm'] == algorithm) &
                (self.data['success'] == True) &
                (self.data['rmsd'].notna())
            ]

            total = len(alg_data)
            if total == 0:
                continue

            success_2a = (alg_data['rmsd'] < 2.0).sum()
            success_3a = (alg_data['rmsd'] < 3.0).sum()

            success_data.append({
                'algorithm': algorithm,
                'total': total,
                'success_2A': success_2a,
                'success_3A': success_3a,
                'rate_2A': success_2a / total * 100,
                'rate_3A': success_3a / total * 100,
                'mean_rmsd': alg_data['rmsd'].mean(),
                'median_rmsd': alg_data['rmsd'].median(),
                'mean_runtime': alg_data['runtime'].mean()
            })

        return pd.DataFrame(success_data)

    def plot_success_rates(self, save_path: Path = None):
        """Bar chart of success rates"""
        success_df = self.calculate_success_rates()

        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(success_df))
        width = 0.35

        bars1 = ax.bar(x - width/2, success_df['rate_2A'], width,
                      label='RMSD < 2.0 Å', alpha=0.8)
        bars2 = ax.bar(x + width/2, success_df['rate_3A'], width,
                      label='RMSD < 3.0 Å', alpha=0.8)

        ax.set_xlabel('Algorithm', fontsize=12)
        ax.set_ylabel('Success Rate (%)', fontsize=12)
        ax.set_title('Docking Success Rates', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(success_df['algorithm'], rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%',
                       ha='center', va='bottom', fontsize=9)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")

        plt.show()

    def plot_rmsd_distribution(self, save_path: Path = None):
        """Violin plot of RMSD distributions"""
        # Filter successful dockings with valid RMSD
        plot_data = self.data[
            (self.data['success'] == True) &
            (self.data['rmsd'].notna()) &
            (self.data['rmsd'] < 10.0)  # Remove outliers
        ]

        fig, ax = plt.subplots(figsize=(12, 6))

        # Create violin plot
        algorithms = plot_data['algorithm'].unique()
        positions = range(len(algorithms))

        parts = ax.violinplot(
            [plot_data[plot_data['algorithm'] == alg]['rmsd'].values for alg in algorithms],
            positions=positions,
            showmeans=True,
            showmedians=True
        )

        # Add horizontal lines at 2Å and 3Å thresholds
        ax.axhline(y=2.0, color='red', linestyle='--', alpha=0.5, label='2.0 Å threshold')
        ax.axhline(y=3.0, color='orange', linestyle='--', alpha=0.5, label='3.0 Å threshold')

        ax.set_xlabel('Algorithm', fontsize=12)
        ax.set_ylabel('RMSD (Å)', fontsize=12)
        ax.set_title('RMSD Distribution by Algorithm', fontsize=14, fontweight='bold')
        ax.set_xticks(positions)
        ax.set_xticklabels(algorithms, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")

        plt.show()

    def plot_affinity_correlation(self, save_path: Path = None):
        """Scatter plots of predicted vs experimental binding affinity"""
        # Filter data with scores and experimental affinity
        plot_data = self.data[
            (self.data['success'] == True) &
            (self.data['best_score'].notna()) &
            (self.data['pKd'].notna())
        ]

        algorithms = plot_data['algorithm'].unique()
        n_algs = len(algorithms)

        # Create subplots
        n_cols = 3
        n_rows = (n_algs + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_algs > 1 else [axes]

        for idx, algorithm in enumerate(algorithms):
            ax = axes[idx]
            alg_data = plot_data[plot_data['algorithm'] == algorithm]

            # Calculate correlation
            pearson_r, pearson_p = stats.pearsonr(alg_data['pKd'], alg_data['best_score'])
            spearman_r, spearman_p = stats.spearmanr(alg_data['pKd'], alg_data['best_score'])

            # Scatter plot
            ax.scatter(alg_data['pKd'], alg_data['best_score'], alpha=0.6)

            # Regression line
            z = np.polyfit(alg_data['pKd'], alg_data['best_score'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(alg_data['pKd'].min(), alg_data['pKd'].max(), 100)
            ax.plot(x_line, p(x_line), "r--", alpha=0.8)

            ax.set_xlabel('Experimental pKd', fontsize=10)
            ax.set_ylabel('Predicted Score', fontsize=10)
            ax.set_title(f'{algorithm}\nPearson R = {pearson_r:.3f}, Spearman ρ = {spearman_r:.3f}',
                        fontsize=10)
            ax.grid(alpha=0.3)

        # Hide unused subplots
        for idx in range(n_algs, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")

        plt.show()

    def plot_runtime_comparison(self, save_path: Path = None):
        """Box plot of runtime comparison"""
        plot_data = self.data[
            (self.data['success'] == True) &
            (self.data['runtime'].notna())
        ]

        fig, ax = plt.subplots(figsize=(10, 6))

        # Box plot
        algorithms = plot_data['algorithm'].unique()
        runtime_data = [plot_data[plot_data['algorithm'] == alg]['runtime'].values
                       for alg in algorithms]

        bp = ax.boxplot(runtime_data, labels=algorithms, patch_artist=True)

        # Color boxes
        colors = sns.color_palette("colorblind", len(algorithms))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_xlabel('Algorithm', fontsize=12)
        ax.set_ylabel('Runtime (seconds)', fontsize=12)
        ax.set_title('Runtime Comparison', fontsize=14, fontweight='bold')
        ax.set_xticklabels(algorithms, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)

        # Log scale if large range
        runtime_range = plot_data['runtime'].max() / plot_data['runtime'].min()
        if runtime_range > 100:
            ax.set_yscale('log')
            ax.set_ylabel('Runtime (seconds, log scale)', fontsize=12)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")

        plt.show()

    def statistical_tests(self) -> pd.DataFrame:
        """Perform statistical significance tests between algorithms"""
        algorithms = self.data['algorithm'].unique()
        results = []

        for i, alg1 in enumerate(algorithms):
            for alg2 in algorithms[i+1:]:
                data1 = self.data[
                    (self.data['algorithm'] == alg1) &
                    (self.data['success'] == True) &
                    (self.data['rmsd'].notna())
                ]['rmsd']

                data2 = self.data[
                    (self.data['algorithm'] == alg2) &
                    (self.data['success'] == True) &
                    (self.data['rmsd'].notna())
                ]['rmsd']

                # Wilcoxon signed-rank test (paired)
                # Only if same complexes were docked
                common_pdbs = set(
                    self.data[(self.data['algorithm'] == alg1) & (self.data['rmsd'].notna())]['pdb_id']
                ).intersection(
                    self.data[(self.data['algorithm'] == alg2) & (self.data['rmsd'].notna())]['pdb_id']
                )

                if len(common_pdbs) > 10:
                    # Get paired data
                    paired1 = self.data[
                        (self.data['algorithm'] == alg1) &
                        (self.data['pdb_id'].isin(common_pdbs))
                    ].sort_values('pdb_id')['rmsd']

                    paired2 = self.data[
                        (self.data['algorithm'] == alg2) &
                        (self.data['pdb_id'].isin(common_pdbs))
                    ].sort_values('pdb_id')['rmsd']

                    stat, p_value = stats.wilcoxon(paired1, paired2)
                    test_type = 'Wilcoxon (paired)'
                else:
                    # Mann-Whitney U test (unpaired)
                    stat, p_value = stats.mannwhitneyu(data1, data2)
                    test_type = 'Mann-Whitney (unpaired)'

                results.append({
                    'algorithm_1': alg1,
                    'algorithm_2': alg2,
                    'test': test_type,
                    'statistic': stat,
                    'p_value': p_value,
                    'significant': 'Yes' if p_value < 0.05 else 'No',
                    'n_comparisons': len(common_pdbs) if test_type.startswith('Wilcoxon') else min(len(data1), len(data2))
                })

        return pd.DataFrame(results)

    def generate_summary_table(self, save_path: Path = None) -> pd.DataFrame:
        """Generate comprehensive summary table"""
        summary = self.calculate_success_rates()

        # Add statistical test results
        if save_path:
            # Save main summary
            summary.to_csv(save_path.parent / 'summary_statistics.csv', index=False)

            # Save statistical tests
            stat_tests = self.statistical_tests()
            stat_tests.to_csv(save_path.parent / 'statistical_tests.csv', index=False)

            print(f"Saved summary tables to {save_path.parent}")

        return summary

    def generate_all_figures(self):
        """Generate all publication figures"""
        print("\nGenerating publication figures...")

        figures_dir = self.output_dir / "figures"
        figures_dir.mkdir(exist_ok=True)

        # Figure 1: Success rates
        print("Creating Figure 1: Success rates...")
        self.plot_success_rates(figures_dir / "fig1_success_rates.png")

        # Figure 2: RMSD distribution
        print("Creating Figure 2: RMSD distribution...")
        self.plot_rmsd_distribution(figures_dir / "fig2_rmsd_distribution.png")

        # Figure 3: Affinity correlation
        print("Creating Figure 3: Binding affinity correlation...")
        self.plot_affinity_correlation(figures_dir / "fig3_affinity_correlation.png")

        # Figure 4: Runtime comparison
        print("Creating Figure 4: Runtime comparison...")
        self.plot_runtime_comparison(figures_dir / "fig4_runtime_comparison.png")

        # Generate summary tables
        print("Creating summary tables...")
        self.generate_summary_table(figures_dir / "summary.csv")

        print(f"\nAll figures saved to: {figures_dir}")

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Analyze benchmark results")
    parser.add_argument("-r", "--results", type=Path, required=True,
                       help="Results CSV file (from run_benchmark_comparison.py)")
    parser.add_argument("-m", "--metadata", type=Path, required=True,
                       help="Metadata CSV file")
    parser.add_argument("-o", "--output-dir", type=Path, required=True,
                       help="Output directory for figures")

    args = parser.parse_args()

    analyzer = BenchmarkAnalyzer(args.results, args.metadata, args.output_dir)
    analyzer.generate_all_figures()

if __name__ == "__main__":
    main()
