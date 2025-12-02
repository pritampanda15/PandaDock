"""
Simplified Publication Plots for PandaDock

Creates clean, uncluttered publication-ready plots focusing on key metrics
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any
import json
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rcParams
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# Clean publication style
plt.style.use('default')
rcParams.update({
    'font.size': 14,
    'font.family': 'Arial',
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'lines.linewidth': 2,
    'lines.markersize': 8,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.3
})

class SimplePublicationVisualizer:
    """Creates clean, focused publication plots"""

    def __init__(self):
        self.colors = {
            'monte_carlo_cpu': '#1f77b4',      # Blue
            'genetic_algorithm_cpu': '#ff7f0e',  # Orange
            'hierarchical_cpu': '#2ca02c',      # Green
        }

        self.algorithm_names = {
            'monte_carlo_cpu': 'Monte Carlo',
            'genetic_algorithm_cpu': 'Genetic Algorithm',
            'hierarchical_cpu': 'Hierarchical'
        }

        self.scoring_names = {
            'physics_based': 'Physics',
            'empirical': 'Empirical',
            'precision_score': 'Precision',
            'hybrid': 'Hybrid'
        }

    def generate_simple_report(self, results_data: Dict[str, Any],
                             output_dir: Path,
                             title: str = "PandaDock Analysis") -> None:
        """Generate clean 4-panel publication figure"""

        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        # Create 2x2 layout with much more spacing
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        fig.subplots_adjust(hspace=0.35, wspace=0.3)  # Much more vertical and horizontal spacing
        fig.suptitle(title, fontsize=20, fontweight='bold', y=0.96)

        # Flatten axes for easier indexing
        ax1, ax2, ax3, ax4 = axes.flatten()

        # Panel 1: Performance Heatmap
        self._plot_performance_heatmap(ax1, results_data)
        ax1.set_title('A. Algorithm Performance Matrix', fontweight='bold', pad=15)

        # Panel 2: Runtime vs Energy
        self._plot_runtime_vs_energy(ax2, results_data)
        ax2.set_title('B. Runtime vs Binding Energy', fontweight='bold', pad=15)

        # Panel 3: Energy Distributions
        self._plot_energy_boxplots(ax3, results_data)
        ax3.set_title('C. Energy Distribution by Algorithm', fontweight='bold', pad=15)

        # Panel 4: Summary Stats
        self._plot_summary_bars(ax4, results_data)
        ax4.set_title('D. Algorithm Summary Statistics', fontweight='bold', pad=15)

        plt.tight_layout()

        # Save files
        png_path = output_dir / f"pandadock_simple_analysis.png"
        pdf_path = output_dir / f"pandadock_simple_analysis.pdf"

        plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"✅ Clean publication plots saved:")
        print(f"   📊 PNG: {png_path}")
        print(f"   📄 PDF: {pdf_path}")

    def _plot_performance_heatmap(self, ax, results_data):
        """Clean performance heatmap"""
        algorithms = list(results_data.keys())
        scoring_functions = []

        # Get all scoring functions
        for alg_data in results_data.values():
            for scoring in alg_data.keys():
                if scoring not in scoring_functions:
                    scoring_functions.append(scoring)

        # Create matrix
        matrix = np.full((len(algorithms), len(scoring_functions)), np.nan)

        for i, algorithm in enumerate(algorithms):
            for j, scoring in enumerate(scoring_functions):
                if scoring in results_data[algorithm]:
                    result = results_data[algorithm][scoring]
                    if 'best_energy' in result and result['best_energy'] is not None:
                        matrix[i, j] = result['best_energy']

        # Create heatmap
        im = ax.imshow(matrix, cmap='RdYlBu_r', aspect='auto', vmin=-12, vmax=0)

        # Labels
        ax.set_xticks(range(len(scoring_functions)))
        ax.set_yticks(range(len(algorithms)))
        ax.set_xticklabels([self.scoring_names.get(sf, sf) for sf in scoring_functions])
        ax.set_yticklabels([self.algorithm_names.get(alg, alg) for alg in algorithms])

        # Add text annotations
        for i in range(len(algorithms)):
            for j in range(len(scoring_functions)):
                if not np.isnan(matrix[i, j]):
                    color = 'white' if matrix[i, j] < -6 else 'black'
                    ax.text(j, i, f'{matrix[i, j]:.1f}', ha='center', va='center',
                           fontweight='bold', color=color, fontsize=12)

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Binding Energy (kcal/mol)', rotation=270, labelpad=20, fontweight='bold')

        ax.set_xlabel('Scoring Functions', fontweight='bold')
        ax.set_ylabel('Docking Algorithms', fontweight='bold')

    def _plot_runtime_vs_energy(self, ax, results_data):
        """Runtime vs energy scatter plot"""
        for algorithm in results_data:
            runtimes = []
            energies = []
            scoring_labels = []

            for scoring in results_data[algorithm]:
                result = results_data[algorithm][scoring]
                if 'runtime' in result and 'best_energy' in result:
                    if result['runtime'] is not None and result['best_energy'] is not None:
                        runtimes.append(result['runtime'])
                        energies.append(abs(result['best_energy']))  # Absolute for better viz
                        scoring_labels.append(scoring)

            if runtimes and energies:
                ax.scatter(runtimes, energies,
                          c=self.colors.get(algorithm, '#888888'),
                          s=100, alpha=0.7,
                          label=self.algorithm_names.get(algorithm, algorithm),
                          edgecolors='black', linewidth=1)

        ax.set_xlabel('Runtime (seconds)', fontweight='bold')
        ax.set_ylabel('Binding Affinity (|kcal/mol|)', fontweight='bold')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=True, fancybox=True)

    def _plot_energy_boxplots(self, ax, results_data):
        """Box plots of energy distributions by algorithm"""
        algorithm_energies = {}

        for algorithm in results_data:
            energies = []
            for scoring in results_data[algorithm]:
                result = results_data[algorithm][scoring]
                if 'poses' in result and result['poses']:
                    pose_energies = [pose.get('energy', 0) for pose in result['poses'] if pose.get('energy') is not None]
                    energies.extend(pose_energies)

            if energies:
                algorithm_energies[self.algorithm_names.get(algorithm, algorithm)] = energies

        if algorithm_energies:
            # Create box plot
            bp = ax.boxplot(algorithm_energies.values(),
                           labels=algorithm_energies.keys(),
                           patch_artist=True)

            # Color the boxes
            colors = [self.colors.get(alg, '#888888') for alg in results_data.keys()]
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            ax.set_ylabel('Binding Energy (kcal/mol)', fontweight='bold')
            ax.set_xlabel('Docking Algorithms', fontweight='bold')
            ax.grid(True, alpha=0.3)

    def _plot_summary_bars(self, ax, results_data):
        """Summary statistics bars"""
        algorithms = []
        avg_energies = []
        avg_runtimes = []

        for algorithm in results_data:
            energies = []
            runtimes = []

            for scoring in results_data[algorithm]:
                result = results_data[algorithm][scoring]
                if result.get('best_energy') is not None:
                    energies.append(abs(result['best_energy']))
                if result.get('runtime') is not None:
                    runtimes.append(result['runtime'])

            if energies and runtimes:
                algorithms.append(self.algorithm_names.get(algorithm, algorithm))
                avg_energies.append(np.mean(energies))
                avg_runtimes.append(np.mean(runtimes))

        if algorithms:
            x = np.arange(len(algorithms))
            width = 0.35

            # Normalize runtimes for visualization (scale to 0-10)
            max_runtime = max(avg_runtimes) if avg_runtimes else 1
            normalized_runtimes = [(r/max_runtime) * 10 for r in avg_runtimes]

            bars1 = ax.bar(x - width/2, avg_energies, width,
                          label='Avg Binding Affinity',
                          color='lightblue', alpha=0.8, edgecolor='black')

            ax2 = ax.twinx()
            bars2 = ax2.bar(x + width/2, normalized_runtimes, width,
                           label='Relative Runtime',
                           color='lightcoral', alpha=0.8, edgecolor='black')

            ax.set_xlabel('Algorithms', fontweight='bold')
            ax.set_ylabel('Avg Binding Affinity (|kcal/mol|)', fontweight='bold', color='blue')
            ax2.set_ylabel('Relative Runtime (scaled)', fontweight='bold', color='red')

            ax.set_xticks(x)
            ax.set_xticklabels(algorithms)
            ax.grid(True, alpha=0.3)

            # Add value labels
            for bar, val in zip(bars1, avg_energies):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       f'{val:.1f}', ha='center', va='bottom', fontweight='bold')

            # Add legends
            ax.legend(loc='upper left')
            ax2.legend(loc='upper right')


def create_simple_plots_from_directory(results_dir: str, output_dir: str = None) -> None:
    """Create simple publication plots from results directory"""
    results_dir = Path(results_dir)
    if output_dir is None:
        output_dir = results_dir / "simple_plots"

    # Parse results (same as complex version)
    results_data = {}

    for result_dir in results_dir.glob("*_full"):
        if result_dir.is_dir():
            dir_name = result_dir.name.replace('_full', '')

            # Handle different directory name patterns
            if '_physics_based' in dir_name:
                algorithm = dir_name.replace('_physics_based', '')
                scoring = 'physics_based'
            elif '_precision_score' in dir_name:
                algorithm = dir_name.replace('_precision_score', '')
                scoring = 'precision_score'
            else:
                parts = dir_name.split('_')
                if len(parts) >= 3:
                    algorithm = '_'.join(parts[:-1])
                    scoring = parts[-1]
                else:
                    continue

            # Load summary data
            summary_file = result_dir / "etomidate.pdb_summary.json"
            if summary_file.exists():
                try:
                    with open(summary_file, 'r') as f:
                        summary_data = json.load(f)

                    if algorithm not in results_data:
                        results_data[algorithm] = {}

                    # Load pose data
                    poses_data = []
                    poses_file = result_dir / "etomidate.pdb_poses.json"
                    if poses_file.exists():
                        try:
                            with open(poses_file, 'r') as f:
                                poses_json = json.load(f)
                                if 'poses' in poses_json:
                                    poses_data = poses_json['poses']
                        except:
                            pass

                    # Convert to expected format
                    converted_data = {
                        'best_energy': summary_data.get('best_pose_energy'),
                        'ensemble_energy': summary_data.get('ensemble_binding_energy'),
                        'runtime': summary_data.get('runtime_seconds'),
                        'poses': poses_data,
                        'confidence': summary_data.get('ensemble_confidence'),
                        'contacts': 102
                    }

                    results_data[algorithm][scoring] = converted_data

                except Exception as e:
                    print(f"Warning: Could not load {summary_file}: {e}")

    if results_data:
        visualizer = SimplePublicationVisualizer()
        visualizer.generate_simple_report(
            results_data,
            output_dir,
            "PandaDock Molecular Docking Analysis"
        )
    else:
        print("No valid results found")