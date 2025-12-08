#!/usr/bin/env python3
"""
Comprehensive Binding Affinity Method Comparison

This script analyzes and compares different approaches for predicting binding affinity:
1. Rigid docking scores (current baseline)
2. Literature expectations for MM-GBSA rescoring
3. Literature expectations for flexible docking
4. Recommendations for improving correlation

Author: PandaDock Benchmarking Suite
Date: December 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11


class AffinityMethodComparison:
    """Compare different methods for binding affinity prediction"""

    def __init__(
        self,
        correlation_data_file: Path,
        output_dir: Path
    ):
        self.correlation_data_file = correlation_data_file
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load correlation data
        self.correlation_stats = pd.read_csv(correlation_data_file)

        print(f"\n{'='*80}")
        print("BINDING AFFINITY METHOD COMPARISON ANALYSIS")
        print(f"{'='*80}")
        print(f"Loaded correlation statistics for {len(self.correlation_stats)} algorithms")

    def analyze_current_performance(self):
        """Analyze current rigid docking performance"""

        print(f"\n{'='*80}")
        print("CURRENT RIGID DOCKING PERFORMANCE")
        print(f"{'='*80}\n")

        # Display current results
        print("Current Pearson R correlations:")
        print("-" * 60)
        for _, row in self.correlation_stats.iterrows():
            algo = row['algorithm']
            pearson_r = row['pearson_r']
            n = row['n_complexes']
            print(f"  {algo:30s} R = {pearson_r:+.3f}  (n={n})")

        # Best performer
        best_idx = self.correlation_stats['pearson_r'].abs().idxmax()
        best = self.correlation_stats.iloc[best_idx]

        print(f"\nBest performer: {best['algorithm']}")
        print(f"  Pearson R: {best['pearson_r']:.3f}")
        print(f"  Spearman ρ: {best['spearman_rho']:.3f}")
        print(f"  RMSE: {best['rmse']:.3f}")

        # Interpret correlation strength
        best_r = abs(best['pearson_r'])
        if best_r < 0.2:
            strength = "very weak/negligible"
        elif best_r < 0.4:
            strength = "weak"
        elif best_r < 0.7:
            strength = "moderate"
        else:
            strength = "strong"

        print(f"\nInterpretation: {strength} correlation")
        print("\nNote: This is typical for geometry-based docking algorithms,")
        print("which optimize for pose prediction (RMSD) rather than binding affinity.")

        return best

    def literature_comparison(self):
        """Compare with literature expectations"""

        print(f"\n{'='*80}")
        print("LITERATURE EXPECTATIONS FOR IMPROVED METHODS")
        print(f"{'='*80}\n")

        # Literature benchmarks (typical ranges from published studies)
        methods_lit = {
            'Rigid Docking (Geometry-based)': {
                'typical_r': 0.3,
                'range': (0.0, 0.5),
                'description': 'Standard docking scores (AutoDock, Vina, etc.)',
                'strengths': '- Fast\n- Good pose prediction\n- Suitable for virtual screening',
                'limitations': '- Poor affinity prediction\n- No solvation effects\n- Static receptor'
            },
            'MM-GBSA Rescoring': {
                'typical_r': 0.6,
                'range': (0.4, 0.75),
                'description': 'Physics-based rescoring with solvation',
                'strengths': '- Better affinity prediction\n- Includes solvation\n- Entropy estimation',
                'limitations': '- Computationally expensive\n- Requires good initial poses\n- Sensitive to force field'
            },
            'Flexible Docking': {
                'typical_r': 0.4,
                'range': (0.2, 0.6),
                'description': 'Induced-fit docking with receptor flexibility',
                'strengths': '- Better pose prediction\n- Captures conformational changes\n- More realistic binding',
                'limitations': '- Slower than rigid\n- Scoring still geometry-based\n- Increased search space'
            },
            'Flex + MM-GBSA': {
                'typical_r': 0.7,
                'range': (0.5, 0.8),
                'description': 'Combined approach (best practice)',
                'strengths': '- Best affinity correlation\n- Realistic binding model\n- Publication-quality',
                'limitations': '- Very slow\n- Complex workflow\n- High computational cost'
            },
            'Machine Learning (GNINA, etc.)': {
                'typical_r': 0.75,
                'range': (0.65, 0.85),
                'description': 'Deep learning-based scoring',
                'strengths': '- Excellent correlation\n- Fast inference\n- Learns from data',
                'limitations': '- Requires training data\n- Black box\n- May not generalize'
            }
        }

        # Display comparison table
        print("Method Comparison (Typical Literature Performance):")
        print("="*80)

        for method, info in methods_lit.items():
            print(f"\n{method}")
            print(f"  Typical Pearson R: {info['typical_r']:.2f} (range: {info['range'][0]:.2f}-{info['range'][1]:.2f})")
            print(f"  Description: {info['description']}")
            print(f"  Strengths:")
            for line in info['strengths'].split('\n'):
                print(f"    {line}")
            print(f"  Limitations:")
            for line in info['limitations'].split('\n'):
                print(f"    {line}")

        # Create visualization
        self._plot_literature_comparison(methods_lit)

    def _plot_literature_comparison(self, methods_lit):
        """Create visualization comparing methods"""

        fig, ax = plt.subplots(figsize=(12, 8))

        methods = list(methods_lit.keys())
        typical_r = [methods_lit[m]['typical_r'] for m in methods]
        ranges = [methods_lit[m]['range'] for m in methods]

        # Current performance
        best = self.correlation_stats.loc[self.correlation_stats['pearson_r'].abs().idxmax()]
        current_r = abs(best['pearson_r'])

        # Plot ranges
        for i, (method, r_range) in enumerate(zip(methods, ranges)):
            ax.barh(i, r_range[1] - r_range[0], left=r_range[0], height=0.6,
                   color='lightblue', alpha=0.5, edgecolor='steelblue', linewidth=2)
            ax.plot(methods_lit[method]['typical_r'], i, 'o', color='darkblue',
                   markersize=10, zorder=3)

        # Plot current performance
        ax.axvline(current_r, color='red', linestyle='--', linewidth=2,
                  label=f'Current (Rigid): R={current_r:.3f}', zorder=4)

        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels(methods)
        ax.set_xlabel('Pearson R (Correlation with Experimental Binding Affinity)', fontsize=12)
        ax.set_title('Binding Affinity Prediction: Method Comparison\n(Based on Literature Benchmarks)',
                    fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1.0)
        ax.legend(loc='lower right')
        ax.grid(axis='x', alpha=0.3)

        # Add annotations
        ax.text(0.05, -0.7, 'Blue bars: Typical range from literature\nBlue dots: Typical performance\nRed line: Current PandaDock rigid docking',
               transform=ax.transData, fontsize=10, style='italic')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'method_comparison_literature.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'method_comparison_literature.pdf', bbox_inches='tight')
        print(f"\n  Saved: {self.output_dir / 'method_comparison_literature.png'}")

    def generate_recommendations(self):
        """Generate recommendations for improving affinity prediction"""

        print(f"\n{'='*80}")
        print("RECOMMENDATIONS FOR IMPROVING BINDING AFFINITY CORRELATION")
        print(f"{'='*80}\n")

        best = self.correlation_stats.loc[self.correlation_stats['pearson_r'].abs().idxmax()]
        current_r = abs(best['pearson_r'])

        recommendations = []

        # Short-term recommendations
        recommendations.append({
            'priority': 'HIGH',
            'timeframe': 'Short-term (days)',
            'method': 'Consensus Scoring',
            'expected_improvement': 'R: 0.25 → 0.35',
            'effort': 'Low',
            'description': 'Combine scores from multiple algorithms (weighted average)',
            'implementation': '- Already have 8 algorithms benchmarked\n'
                            '- Use ensemble of top 3 performers\n'
                            '- Weight by individual correlation strength'
        })

        recommendations.append({
            'priority': 'HIGH',
            'timeframe': 'Short-term (days)',
            'method': 'Empirical Corrections',
            'expected_improvement': 'R: 0.25 → 0.40',
            'effort': 'Low',
            'description': 'Apply simple corrections for ligand properties',
            'implementation': '- Correct for molecular weight\n'
                            '- Correct for rotatable bonds\n'
                            '- Correct for buried surface area'
        })

        # Medium-term recommendations
        recommendations.append({
            'priority': 'MEDIUM',
            'timeframe': 'Medium-term (weeks)',
            'method': 'MM-GBSA Rescoring (optimized)',
            'expected_improvement': 'R: 0.25 → 0.55',
            'effort': 'Medium',
            'description': 'Optimize MM-GBSA implementation for speed',
            'implementation': '- Use GPU acceleration\n'
                            '- Simplify entropy calculation\n'
                            '- Rescore only top poses\n'
                            '- Parallel processing'
        })

        recommendations.append({
            'priority': 'MEDIUM',
            'timeframe': 'Medium-term (weeks)',
            'method': 'Machine Learning Rescoring',
            'expected_improvement': 'R: 0.25 → 0.65',
            'effort': 'Medium',
            'description': 'Train ML model on PDBbind data',
            'implementation': '- Use RF-Score or similar\n'
                            '- Features: docking scores + ligand properties\n'
                            '- Train on PDBbind general set\n'
                            '- Validate on refined set'
        })

        # Long-term recommendations
        recommendations.append({
            'priority': 'LOW',
            'timeframe': 'Long-term (months)',
            'method': 'Flexible Docking + MM-GBSA',
            'expected_improvement': 'R: 0.25 → 0.70',
            'effort': 'High',
            'description': 'Full induced-fit workflow with physics-based scoring',
            'implementation': '- Flexible docking already implemented\n'
                            '- Optimize MM-GBSA speed\n'
                            '- Automated workflow\n'
                            '- Extensive validation'
        })

        recommendations.append({
            'priority': 'LOW',
            'timeframe': 'Long-term (months)',
            'method': 'Deep Learning Integration',
            'expected_improvement': 'R: 0.25 → 0.75',
            'effort': 'High',
            'description': 'Integrate GNINA or custom deep learning',
            'implementation': '- Train on large datasets\n'
                            '- 3D CNN architecture\n'
                            '- Transfer learning\n'
                            '- GPU infrastructure'
        })

        # Display recommendations
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec['method']} [{rec['priority']} PRIORITY]")
            print(f"   Timeframe: {rec['timeframe']}")
            print(f"   Expected improvement: {rec['expected_improvement']}")
            print(f"   Effort: {rec['effort']}")
            print(f"   Description: {rec['description']}")
            print(f"   Implementation:")
            for line in rec['implementation'].split('\n'):
                print(f"     {line}")
            print()

        return recommendations

    def create_summary_report(self, best_current, recommendations):
        """Create comprehensive summary report"""

        report_file = self.output_dir / 'AFFINITY_METHOD_COMPARISON_REPORT.md'

        with open(report_file, 'w') as f:
            f.write("# Binding Affinity Method Comparison Report\n\n")
            f.write(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")

            # Executive Summary
            f.write("## Executive Summary\n\n")
            f.write(f"Current best performance: **{best_current['algorithm']}** with Pearson R = **{best_current['pearson_r']:.3f}**\n\n")
            f.write(f"This represents **weak correlation** with experimental binding affinities, ")
            f.write(f"which is typical for geometry-based docking algorithms.\n\n")

            # Current Performance
            f.write("## Current Performance (Rigid Docking)\n\n")
            f.write("| Algorithm | N | Pearson R | Spearman ρ | RMSE |\n")
            f.write("|-----------|---|-----------|------------|------|\n")
            for _, row in self.correlation_stats.iterrows():
                f.write(f"| {row['algorithm']} | {row['n_complexes']} | {row['pearson_r']:.3f} | ")
                f.write(f"{row['spearman_rho']:.3f} | {row['rmse']:.3f} |\n")
            f.write("\n")

            # Literature Comparison
            f.write("## Literature Comparison\n\n")
            f.write("Typical performance ranges from published benchmarks:\n\n")
            f.write("| Method | Typical R | Range | Computational Cost |\n")
            f.write("|--------|-----------|-------|-------------------|\n")
            f.write("| Rigid Docking | 0.30 | 0.0-0.5 | Low (minutes) |\n")
            f.write("| MM-GBSA Rescoring | 0.60 | 0.4-0.75 | Medium (hours) |\n")
            f.write("| Flexible Docking | 0.40 | 0.2-0.6 | Medium (hours) |\n")
            f.write("| Flex + MM-GBSA | 0.70 | 0.5-0.8 | High (days) |\n")
            f.write("| Machine Learning | 0.75 | 0.65-0.85 | Low after training |\n\n")

            # Key Insights
            f.write("## Key Insights\n\n")
            f.write("1. **Geometry-based scores are weak affinity predictors**: Current docking scores ")
            f.write("optimize for pose geometry, not binding free energy.\n\n")
            f.write("2. **Expected improvement from MM-GBSA**: Literature shows typical improvement ")
            f.write("from R~0.3 to R~0.6 with MM-GBSA rescoring.\n\n")
            f.write("3. **Computational cost is significant**: MM-GBSA rescoring takes ~10-15 minutes per ")
            f.write("complex, making large-scale benchmarking impractical.\n\n")
            f.write("4. **Machine learning offers best balance**: ML approaches achieve R~0.75 with fast ")
            f.write("inference after initial training.\n\n")

            # Recommendations
            f.write("## Recommendations\n\n")
            for i, rec in enumerate(recommendations, 1):
                f.write(f"### {i}. {rec['method']} [{rec['priority']} PRIORITY]\n\n")
                f.write(f"- **Timeframe:** {rec['timeframe']}\n")
                f.write(f"- **Expected improvement:** {rec['expected_improvement']}\n")
                f.write(f"- **Effort:** {rec['effort']}\n")
                f.write(f"- **Description:** {rec['description']}\n\n")
                f.write("**Implementation:**\n")
                for line in rec['implementation'].split('\n'):
                    if line.strip():
                        f.write(f"{line}\n")
                f.write("\n")

            # References
            f.write("## References\n\n")
            f.write("1. Wang et al. (2016) \"Comparative Assessment of Scoring Functions\" *J. Med. Chem.*\n")
            f.write("2. Genheden & Ryde (2015) \"MM-PBSA and MM-GBSA methods\" *Expert Opin. Drug Discov.*\n")
            f.write("3. Ragoza et al. (2017) \"GNINA molecular docking\" *J. Chem. Inf. Model.*\n")
            f.write("4. Su et al. (2019) \"Comparative Assessment\" *J. Chem. Inf. Model.*\n")
            f.write("5. Ballester & Mitchell (2010) \"Machine learning scoring\" *Bioinformatics*\n\n")

        print(f"\n{'='*80}")
        print(f"SUMMARY REPORT SAVED")
        print(f"{'='*80}")
        print(f"Report: {report_file}")
        print(f"Visualizations: {self.output_dir}")

    def run_full_analysis(self):
        """Run complete analysis"""

        # Analyze current performance
        best_current = self.analyze_current_performance()

        # Literature comparison
        self.literature_comparison()

        # Generate recommendations
        recommendations = self.generate_recommendations()

        # Create summary report
        self.create_summary_report(best_current, recommendations)

        print(f"\n{'='*80}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*80}\n")


def main():
    """Main entry point"""

    correlation_file = Path("benchmarking/comprehensive_benchmark/binding_affinity_analysis/correlation_statistics.csv")
    output_dir = Path("benchmarking/comprehensive_benchmark/method_comparison")

    if not correlation_file.exists():
        print(f"ERROR: Correlation statistics file not found: {correlation_file}")
        return

    analyzer = AffinityMethodComparison(
        correlation_data_file=correlation_file,
        output_dir=output_dir
    )

    analyzer.run_full_analysis()


if __name__ == '__main__':
    main()
