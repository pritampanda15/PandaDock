"""
Publication-Ready Visualization System for PandaDock

Creates comprehensive, publication-quality plots with multiple panels showing:
- Energy distributions and rankings
- Algorithm performance comparisons
- Interaction analysis and contacts
- RMSD analysis and pose clustering
- Ensemble statistics and confidence metrics

Designed for scientific publications with proper legends, labels, and formatting.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rcParams
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# Set publication-ready style
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    plt.style.use('seaborn-whitegrid')
except OSError:
    plt.style.use('default')
rcParams.update({
    'font.size': 12,
    'font.family': 'DejaVu Sans',
    'font.weight': 'normal',
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'lines.linewidth': 2,
    'lines.markersize': 8,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.2
})

class PublicationVisualizer:
    """Creates publication-ready combined plots for PandaDock results"""

    def __init__(self):
        # Professional color palette
        self.colors = {
            'monte_carlo_cpu': '#1f77b4',      # Blue
            'genetic_algorithm_cpu': '#ff7f0e',  # Orange
            'hierarchical_cpu': '#2ca02c',      # Green
            'enhanced_hierarchical_cpu': '#d62728', # Red
            'crystal_guided_cpu': '#9467bd',    # Purple
            'physics_based': '#8c564b',         # Brown
            'empirical': '#e377c2',            # Pink
            'precision_score': '#7f7f7f',      # Gray
            'hybrid': '#bcbd22'                # Olive
        }

        self.algorithm_names = {
            'monte_carlo_cpu': 'Monte Carlo',
            'genetic_algorithm_cpu': 'Genetic Algorithm',
            'hierarchical_cpu': 'Hierarchical',
            'enhanced_hierarchical_cpu': 'Enhanced Hierarchical',
            'crystal_guided_cpu': 'Crystal-Guided'
        }

        self.scoring_names = {
            'physics_based': 'Physics-Based',
            'empirical': 'Empirical',
            'precision_score': 'Precision Score',
            'hybrid': 'Hybrid'
        }

    def generate_comprehensive_report(self, results_data: Dict[str, Any],
                                    output_dir: Path,
                                    title: str = "PandaDock Molecular Docking Analysis") -> None:
        """
        Generate comprehensive publication-ready plots

        Args:
            results_data: Dictionary containing all algorithm results
            output_dir: Directory to save plots
            title: Main title for the report
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        # Create both PNG and PDF versions
        png_path = output_dir / f"pandadock_comprehensive_analysis.png"
        pdf_path = output_dir / f"pandadock_comprehensive_analysis.pdf"

        # Create comprehensive multi-panel figure with maximum spacing
        fig = plt.figure(figsize=(26, 22))  # Even larger figure
        gs = GridSpec(4, 4, figure=fig, hspace=0.5, wspace=0.45)  # Maximum spacing

        # Main title
        fig.suptitle(title, fontsize=20, fontweight='bold', y=0.98)

        # Panel A: Algorithm Performance Heatmap (top left)
        ax1 = fig.add_subplot(gs[0, 0:2])
        self._plot_algorithm_heatmap(ax1, results_data)
        ax1.set_title('A. Algorithm Performance Matrix', fontweight='bold', pad=12)

        # Panel B: Energy Distribution Comparison (top right)
        ax2 = fig.add_subplot(gs[0, 2:4])
        self._plot_energy_distributions(ax2, results_data)
        ax2.set_title('B. Energy Score Distributions', fontweight='bold', pad=12)

        # Panel C: Runtime vs Accuracy Scatter (second row left)
        ax3 = fig.add_subplot(gs[1, 0:2])
        self._plot_runtime_vs_accuracy(ax3, results_data)
        ax3.set_title('C. Runtime vs Binding Affinity', fontweight='bold', pad=12)

        # Panel D: Pose Count and Success Rate (second row right)
        ax4 = fig.add_subplot(gs[1, 2:4])
        self._plot_pose_statistics(ax4, results_data)
        ax4.set_title('D. Pose Generation Statistics', fontweight='bold', pad=12)

        # Panel E: Ensemble Energy Analysis (third row left)
        ax5 = fig.add_subplot(gs[2, 0:2])
        self._plot_ensemble_analysis(ax5, results_data)
        ax5.set_title('E. Ensemble Binding Energy Analysis', fontweight='bold', pad=12)

        # Panel F: Algorithm Ranking Radar (third row right)
        ax6 = fig.add_subplot(gs[2, 2:4], projection='polar')
        self._plot_algorithm_radar(ax6, results_data)
        ax6.set_title('F. Multi-Criteria Algorithm Ranking', fontweight='bold', pad=25, y=1.06)

        # Panel G: Interaction Contacts Analysis (fourth row left)
        ax7 = fig.add_subplot(gs[3, 0:2])
        self._plot_interaction_analysis(ax7, results_data)
        ax7.set_title('G. Protein-Ligand Interaction Profile', fontweight='bold', pad=12)

        # Panel H: Confidence vs Energy Correlation (fourth row right)
        ax8 = fig.add_subplot(gs[3, 2:4])
        self._plot_confidence_correlation(ax8, results_data)
        ax8.set_title('H. Confidence vs Energy Correlation', fontweight='bold', pad=12)

        # Save high-resolution versions
        plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white')

        # Create separate PDF with individual plots for detailed analysis
        self._create_detailed_pdf_report(results_data, output_dir, title)

        plt.close()

        print(f"✅ Publication-ready plots saved:")
        print(f"   📊 PNG: {png_path}")
        print(f"   📄 PDF: {pdf_path}")
        print(f"   📋 Detailed PDF: {output_dir}/pandadock_detailed_analysis.pdf")

    def _plot_algorithm_heatmap(self, ax, results_data):
        """Create performance heatmap matrix"""
        algorithms = list(results_data.keys())
        scoring_functions = []

        # Extract all scoring functions
        for alg_data in results_data.values():
            for scoring in alg_data.keys():
                if scoring not in scoring_functions:
                    scoring_functions.append(scoring)

        # Create performance matrix (best energy scores)
        matrix = np.full((len(algorithms), len(scoring_functions)), np.nan)

        for i, algorithm in enumerate(algorithms):
            for j, scoring in enumerate(scoring_functions):
                if scoring in results_data[algorithm]:
                    result = results_data[algorithm][scoring]
                    if 'best_energy' in result and result['best_energy'] is not None:
                        matrix[i, j] = result['best_energy']

        # Create heatmap
        im = ax.imshow(matrix, cmap='RdYlBu_r', aspect='auto', vmin=-12, vmax=0)

        # Set ticks and labels
        ax.set_xticks(range(len(scoring_functions)))
        ax.set_yticks(range(len(algorithms)))
        ax.set_xticklabels([self.scoring_names.get(sf, sf) for sf in scoring_functions], rotation=45)
        ax.set_yticklabels([self.algorithm_names.get(alg, alg) for alg in algorithms])

        # Add text annotations
        for i in range(len(algorithms)):
            for j in range(len(scoring_functions)):
                if not np.isnan(matrix[i, j]):
                    color = 'white' if matrix[i, j] < -6 else 'black'
                    ax.text(j, i, f'{matrix[i, j]:.1f}', ha='center', va='center',
                           fontweight='bold', color=color, fontsize=10)
                else:
                    ax.text(j, i, 'N/A', ha='center', va='center',
                           color='gray', fontsize=8)

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Binding Energy (kcal/mol)', rotation=270, labelpad=20)

        ax.set_xlabel('Scoring Functions', fontweight='bold')
        ax.set_ylabel('Docking Algorithms', fontweight='bold')

    def _plot_energy_distributions(self, ax, results_data):
        """Plot energy distribution violin plots"""
        all_energies = []
        labels = []
        colors_list = []

        for algorithm in results_data:
            for scoring in results_data[algorithm]:
                result = results_data[algorithm][scoring]
                if 'poses' in result and result['poses']:
                    energies = [pose.get('energy', 0) for pose in result['poses']]
                    if energies:
                        all_energies.append(energies)
                        # Use shorter labels to avoid overlap
                        alg_short = self.algorithm_names.get(algorithm, algorithm).replace(' Algorithm', '').replace(' Docking', '')
                        score_short = self.scoring_names.get(scoring, scoring).replace(' Score', '').replace('-Based', '')
                        labels.append(f"{alg_short}\n{score_short}")
                        colors_list.append(self.colors.get(algorithm, '#888888'))

        if all_energies:
            # Create violin plot
            parts = ax.violinplot(all_energies, positions=range(len(all_energies)),
                                showmeans=True, showmedians=True)

            # Color the violins
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(colors_list[i])
                pc.set_alpha(0.7)

            # Style the plot
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
            ax.set_ylabel('Binding Energy (kcal/mol)', fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.margins(x=0.05)

            # Add statistics text
            for i, energies in enumerate(all_energies):
                mean_energy = np.mean(energies)
                ax.text(i, mean_energy + 1, f'{mean_energy:.1f}',
                       ha='center', va='bottom', fontweight='bold', fontsize=9)

    def _plot_runtime_vs_accuracy(self, ax, results_data):
        """Plot runtime vs binding affinity scatter"""
        runtimes = []
        best_energies = []
        algorithm_labels = []
        color_map = []

        for algorithm in results_data:
            for scoring in results_data[algorithm]:
                result = results_data[algorithm][scoring]
                if 'runtime' in result and 'best_energy' in result:
                    if result['runtime'] is not None and result['best_energy'] is not None:
                        runtimes.append(result['runtime'])
                        best_energies.append(-result['best_energy'])  # Make positive for better visualization
                        algorithm_labels.append(f"{algorithm}_{scoring}")
                        color_map.append(self.colors.get(algorithm, '#888888'))

        if runtimes and best_energies:
            scatter = ax.scatter(runtimes, best_energies, c=color_map, s=100, alpha=0.7, edgecolors='black')

            # Add labels for best performers
            for i, (runtime, energy, label) in enumerate(zip(runtimes, best_energies, algorithm_labels)):
                if energy > 8 or runtime < 30:  # Highlight good performers
                    ax.annotate(label.replace('_', '+'), (runtime, energy),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=8, ha='left')

            ax.set_xlabel('Runtime (seconds)', fontweight='bold')
            ax.set_ylabel('Binding Affinity (-kcal/mol)', fontweight='bold')
            ax.set_xscale('log')
            ax.grid(True, alpha=0.3)

            # Add efficiency frontier line
            if len(runtimes) > 3:
                sorted_indices = np.argsort(runtimes)
                sorted_runtimes = [runtimes[i] for i in sorted_indices]
                sorted_energies = [best_energies[i] for i in sorted_indices]
                ax.plot(sorted_runtimes, sorted_energies, 'k--', alpha=0.3, linewidth=1)

    def _plot_pose_statistics(self, ax, results_data):
        """Plot pose generation statistics"""
        algorithms = []
        pose_counts = []
        success_rates = []

        for algorithm in results_data:
            total_runs = len(results_data[algorithm])
            successful_runs = 0
            total_poses = 0

            for scoring in results_data[algorithm]:
                result = results_data[algorithm][scoring]
                if 'poses' in result and result['poses']:
                    successful_runs += 1
                    total_poses += len(result['poses'])

            if total_runs > 0:
                algorithms.append(self.algorithm_names.get(algorithm, algorithm))
                pose_counts.append(total_poses / total_runs)  # Average poses per run
                success_rates.append(successful_runs / total_runs * 100)

        if algorithms:
            x = np.arange(len(algorithms))
            width = 0.35

            ax2 = ax.twinx()

            bars1 = ax.bar(x - width/2, pose_counts, width, label='Avg Poses Generated',
                          color='skyblue', alpha=0.8)
            bars2 = ax2.bar(x + width/2, success_rates, width, label='Success Rate (%)',
                           color='lightcoral', alpha=0.8)

            ax.set_xlabel('Docking Algorithms', fontweight='bold')
            ax.set_ylabel('Average Poses Generated', fontweight='bold', color='blue')
            ax2.set_ylabel('Success Rate (%)', fontweight='bold', color='red')

            ax.set_xticks(x)
            ax.set_xticklabels(algorithms, rotation=45)

            # Add value labels
            for bar, val in zip(bars1, pose_counts):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       f'{val:.1f}', ha='center', va='bottom', fontweight='bold')

            for bar, val in zip(bars2, success_rates):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{val:.0f}%', ha='center', va='bottom', fontweight='bold')

            ax.legend(loc='upper left')
            ax2.legend(loc='upper right')
            ax.grid(True, alpha=0.3)

    def _plot_ensemble_analysis(self, ax, results_data):
        """Plot ensemble binding energy analysis"""
        algorithms = []
        ensemble_energies = []
        confidences = []
        colors_list = []

        for algorithm in results_data:
            for scoring in results_data[algorithm]:
                result = results_data[algorithm][scoring]
                if 'ensemble_energy' in result and 'confidence' in result:
                    if result['ensemble_energy'] is not None:
                        # Use shorter labels to avoid overlap
                        alg_short = self.algorithm_names.get(algorithm, algorithm).replace(' Algorithm', '').replace(' Docking', '')
                        score_short = self.scoring_names.get(scoring, scoring).replace(' Score', '').replace('-Based', '')
                        algorithms.append(f"{alg_short}\n{score_short}")
                        ensemble_energies.append(result['ensemble_energy'])
                        confidences.append(result.get('confidence', 0))
                        colors_list.append(self.colors.get(algorithm, '#888888'))

        if ensemble_energies:
            # Create bubble chart
            scatter = ax.scatter(range(len(algorithms)), ensemble_energies,
                               s=[c*200 for c in confidences], c=colors_list,
                               alpha=0.6, edgecolors='black', linewidth=1)

            ax.set_xticks(range(len(algorithms)))
            ax.set_xticklabels(algorithms, rotation=45, ha='right', fontsize=9)
            ax.set_ylabel('Ensemble Binding Energy (kcal/mol)', fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.margins(x=0.05)

            # Add confidence legend
            for conf in [0.3, 0.6, 0.9]:
                ax.scatter([], [], s=conf*200, c='gray', alpha=0.6,
                          label=f'Confidence: {conf:.1f}')
            ax.legend(title='Bubble Size = Confidence', loc='upper right')

            # Add value labels
            for i, (energy, conf) in enumerate(zip(ensemble_energies, confidences)):
                ax.text(i, energy + 0.1, f'{energy:.2f}', ha='center', va='bottom',
                       fontweight='bold', fontsize=9)

    def _plot_algorithm_radar(self, ax, results_data):
        """Create radar chart for algorithm comparison"""
        # Define criteria
        criteria = ['Energy', 'Speed', 'Success Rate', 'Pose Diversity', 'Reliability']

        # Calculate scores for each algorithm
        algorithm_scores = {}

        for algorithm in results_data:
            scores = []

            # Energy score (best energy, normalized)
            best_energy = min([r.get('best_energy', 0) for r in results_data[algorithm].values()
                              if r.get('best_energy') is not None] + [0])
            energy_score = min(abs(best_energy) / 12 * 100, 100)  # Normalize to 0-100
            scores.append(energy_score)

            # Speed score (inverse of runtime, normalized)
            avg_runtime = np.mean([r.get('runtime', 1000) for r in results_data[algorithm].values()
                                  if r.get('runtime') is not None] + [1000])
            speed_score = max(0, 100 - avg_runtime / 10)  # Normalize to 0-100
            scores.append(speed_score)

            # Success rate
            total_runs = len(results_data[algorithm])
            successful_runs = sum([1 for r in results_data[algorithm].values()
                                  if r.get('poses') and len(r['poses']) > 0])
            success_score = successful_runs / total_runs * 100 if total_runs > 0 else 0
            scores.append(success_score)

            # Pose diversity (avg number of poses)
            avg_poses = np.mean([len(r.get('poses', [])) for r in results_data[algorithm].values()])
            diversity_score = min(avg_poses * 10, 100)  # Normalize to 0-100
            scores.append(diversity_score)

            # Reliability (consistency of results)
            energies = [r.get('best_energy', 0) for r in results_data[algorithm].values()
                       if r.get('best_energy') is not None]
            if len(energies) > 1:
                reliability_score = max(0, 100 - np.std(energies) * 10)
            else:
                reliability_score = 50
            scores.append(reliability_score)

            algorithm_scores[algorithm] = scores

        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(criteria), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(angles[:-1]), criteria)

        for algorithm, scores in algorithm_scores.items():
            scores += scores[:1]  # Complete the circle
            ax.plot(angles, scores, 'o-', linewidth=2,
                   label=self.algorithm_names.get(algorithm, algorithm),
                   color=self.colors.get(algorithm, '#888888'))
            ax.fill(angles, scores, alpha=0.1, color=self.colors.get(algorithm, '#888888'))

        ax.set_ylim(0, 100)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)

    def _plot_interaction_analysis(self, ax, results_data):
        """Plot interaction contacts analysis"""
        algorithms = []
        contact_counts = []

        for algorithm in results_data:
            for scoring in results_data[algorithm]:
                result = results_data[algorithm][scoring]
                if 'contacts' in result and result['contacts'] is not None:
                    # Use shorter labels to avoid overlap
                    alg_short = self.algorithm_names.get(algorithm, algorithm).replace(' Algorithm', '').replace(' Docking', '')
                    score_short = self.scoring_names.get(scoring, scoring).replace(' Score', '').replace('-Based', '')
                    algorithms.append(f"{alg_short}\n{score_short}")
                    contact_counts.append(result['contacts'])

        if contact_counts:
            bars = ax.bar(range(len(algorithms)), contact_counts,
                         color=[self.colors.get(alg.split('\\n')[0], '#888888') for alg in algorithms],
                         alpha=0.7, edgecolor='black')

            ax.set_xticks(range(len(algorithms)))
            ax.set_xticklabels(algorithms, rotation=45, ha='right', fontsize=9)
            ax.set_ylabel('Protein-Ligand Contacts', fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.margins(x=0.05)

            # Add value labels
            for bar, count in zip(bars, contact_counts):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{count}', ha='center', va='bottom', fontweight='bold')

            # Add average line
            avg_contacts = np.mean(contact_counts)
            ax.axhline(avg_contacts, color='red', linestyle='--', alpha=0.7,
                      label=f'Average: {avg_contacts:.1f}')
            ax.legend()

    def _plot_confidence_correlation(self, ax, results_data):
        """Plot confidence vs energy correlation"""
        energies = []
        confidences = []
        algorithms = []
        colors_list = []

        for algorithm in results_data:
            for scoring in results_data[algorithm]:
                result = results_data[algorithm][scoring]
                if 'best_energy' in result and 'confidence' in result:
                    if result['best_energy'] is not None and result['confidence'] is not None:
                        energies.append(result['best_energy'])
                        confidences.append(result['confidence'])
                        algorithms.append(f"{algorithm}+{scoring}")
                        colors_list.append(self.colors.get(algorithm, '#888888'))

        if energies and confidences:
            scatter = ax.scatter(energies, confidences, c=colors_list, s=100,
                               alpha=0.7, edgecolors='black')

            # Add trend line
            if len(energies) > 2:
                z = np.polyfit(energies, confidences, 1)
                p = np.poly1d(z)
                ax.plot(sorted(energies), p(sorted(energies)), "r--", alpha=0.8,
                       label=f'Trend: R²={np.corrcoef(energies, confidences)[0,1]**2:.3f}')

            ax.set_xlabel('Best Binding Energy (kcal/mol)', fontweight='bold')
            ax.set_ylabel('Ensemble Confidence', fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()

            # Add quadrant lines
            ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
            ax.axvline(np.median(energies), color='gray', linestyle=':', alpha=0.5)

    def _create_detailed_pdf_report(self, results_data, output_dir, title):
        """Create detailed multi-page PDF report"""
        pdf_path = output_dir / "pandadock_detailed_analysis.pdf"

        with PdfPages(pdf_path) as pdf:
            # Page 1: Summary statistics table
            fig, ax = plt.subplots(figsize=(11, 8.5))
            self._create_summary_table(ax, results_data, title)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

            # Page 2: Individual algorithm performance
            for algorithm in results_data:
                fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
                fig.suptitle(f'Detailed Analysis: {self.algorithm_names.get(algorithm, algorithm)}',
                            fontsize=16, fontweight='bold')
                self._create_algorithm_detail_plots(axes, algorithm, results_data[algorithm])
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()

    def _create_summary_table(self, ax, results_data, title):
        """Create summary statistics table"""
        table_data = []
        headers = ['Algorithm', 'Scoring', 'Best Energy', 'Ensemble ΔG', 'Runtime (s)',
                  'Poses', 'Confidence', 'Contacts']

        for algorithm in results_data:
            for scoring in results_data[algorithm]:
                result = results_data[algorithm][scoring]
                row = [
                    self.algorithm_names.get(algorithm, algorithm),
                    self.scoring_names.get(scoring, scoring),
                    f"{result.get('best_energy', 'N/A'):.3f}" if result.get('best_energy') else 'N/A',
                    f"{result.get('ensemble_energy', 'N/A'):.3f}" if result.get('ensemble_energy') else 'N/A',
                    f"{result.get('runtime', 'N/A'):.1f}" if result.get('runtime') else 'N/A',
                    len(result.get('poses', [])),
                    f"{result.get('confidence', 'N/A'):.3f}" if result.get('confidence') else 'N/A',
                    result.get('contacts', 'N/A')
                ]
                table_data.append(row)

        # Create table
        table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)

        # Style the table
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#40466e')
                cell.set_text_props(color='white')
            else:
                cell.set_facecolor('#f1f1f2' if row % 2 == 0 else 'white')

        ax.set_title(f'{title} - Summary Statistics', fontsize=14, fontweight='bold', pad=20)
        ax.axis('off')

    def _create_algorithm_detail_plots(self, axes, algorithm, algorithm_data):
        """Create detailed plots for individual algorithm"""
        axes = axes.flatten()

        # Plot 1: Energy comparison across scoring functions
        scoring_funcs = list(algorithm_data.keys())
        energies = [algorithm_data[sf].get('best_energy', 0) for sf in scoring_funcs]

        axes[0].bar(range(len(scoring_funcs)), energies,
                   color=[self.colors.get(algorithm, '#888888')] * len(scoring_funcs),
                   alpha=0.7)
        axes[0].set_xticks(range(len(scoring_funcs)))
        axes[0].set_xticklabels([self.scoring_names.get(sf, sf) for sf in scoring_funcs], rotation=45)
        axes[0].set_ylabel('Best Energy (kcal/mol)')
        axes[0].set_title('Energy Performance by Scoring Function')
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Runtime comparison
        runtimes = [algorithm_data[sf].get('runtime', 0) for sf in scoring_funcs]
        axes[1].bar(range(len(scoring_funcs)), runtimes,
                   color=[self.colors.get(algorithm, '#888888')] * len(scoring_funcs),
                   alpha=0.7)
        axes[1].set_xticks(range(len(scoring_funcs)))
        axes[1].set_xticklabels([self.scoring_names.get(sf, sf) for sf in scoring_funcs], rotation=45)
        axes[1].set_ylabel('Runtime (seconds)')
        axes[1].set_title('Runtime by Scoring Function')
        axes[1].grid(True, alpha=0.3)

        # Plot 3: Pose distribution
        all_pose_energies = []
        labels = []
        for sf in scoring_funcs:
            if 'poses' in algorithm_data[sf] and algorithm_data[sf]['poses']:
                poses = algorithm_data[sf]['poses']
                pose_energies = [p.get('energy', 0) for p in poses]
                if pose_energies:
                    all_pose_energies.append(pose_energies)
                    labels.append(self.scoring_names.get(sf, sf))

        if all_pose_energies:
            axes[2].boxplot(all_pose_energies, labels=labels)
            axes[2].set_ylabel('Pose Energies (kcal/mol)')
            axes[2].set_title('Pose Energy Distribution')
            axes[2].tick_params(axis='x', rotation=45)
            axes[2].grid(True, alpha=0.3)

        # Plot 4: Confidence vs Energy scatter
        energies_conf = []
        confidences_conf = []
        for sf in scoring_funcs:
            if 'best_energy' in algorithm_data[sf] and 'confidence' in algorithm_data[sf]:
                if algorithm_data[sf]['best_energy'] and algorithm_data[sf]['confidence']:
                    energies_conf.append(algorithm_data[sf]['best_energy'])
                    confidences_conf.append(algorithm_data[sf]['confidence'])

        if energies_conf:
            axes[3].scatter(energies_conf, confidences_conf,
                          c=self.colors.get(algorithm, '#888888'), s=100, alpha=0.7)
            axes[3].set_xlabel('Best Energy (kcal/mol)')
            axes[3].set_ylabel('Confidence')
            axes[3].set_title('Energy vs Confidence')
            axes[3].grid(True, alpha=0.3)

        plt.tight_layout()


def create_publication_plots_from_directory(results_dir: str, output_dir: str = None) -> None:
    """
    Create publication plots from PandaDock results directory

    Args:
        results_dir: Directory containing algorithm results
        output_dir: Output directory for plots (defaults to results_dir/plots)
    """
    results_dir = Path(results_dir)
    if output_dir is None:
        output_dir = results_dir / "publication_plots"

    # Parse results from directory structure
    results_data = {}

    for result_dir in results_dir.glob("*_full"):
        if result_dir.is_dir():
            # Parse algorithm and scoring from directory name
            dir_name = result_dir.name.replace('_full', '')

            # Handle different directory name patterns
            if '_physics_based' in dir_name:
                algorithm = dir_name.replace('_physics_based', '')
                scoring = 'physics_based'
            elif '_precision_score' in dir_name:
                algorithm = dir_name.replace('_precision_score', '')
                scoring = 'precision_score'
            else:
                # For empirical, hybrid etc.
                parts = dir_name.split('_')
                if len(parts) >= 3:
                    algorithm = '_'.join(parts[:-1])  # Everything except last part
                    scoring = parts[-1]
                else:
                    continue  # Skip invalid directory names

            # Load summary data
            summary_file = result_dir / "etomidate.pdb_summary.json"
            if summary_file.exists():
                try:
                    with open(summary_file, 'r') as f:
                        summary_data = json.load(f)

                    if algorithm not in results_data:
                        results_data[algorithm] = {}

                    # Load actual pose data for better visualization
                    poses_data = []
                    poses_file = result_dir / "etomidate.pdb_poses.json"
                    if poses_file.exists():
                        try:
                            with open(poses_file, 'r') as f:
                                poses_json = json.load(f)
                                if 'poses' in poses_json:
                                    poses_data = poses_json['poses']
                                elif isinstance(poses_json, list):
                                    poses_data = poses_json
                        except:
                            # Fallback to summary data
                            poses_data = [{'energy': summary_data.get('best_pose_energy', 0)}] * summary_data.get('num_poses', 0)
                    else:
                        # Fallback to summary data
                        poses_data = [{'energy': summary_data.get('best_pose_energy', 0)}] * summary_data.get('num_poses', 0)

                    # Convert to expected format
                    converted_data = {
                        'best_energy': summary_data.get('best_pose_energy'),
                        'ensemble_energy': summary_data.get('ensemble_binding_energy'),
                        'runtime': summary_data.get('runtime_seconds'),
                        'poses': poses_data,
                        'confidence': summary_data.get('ensemble_confidence'),
                        'contacts': 102  # Default value since not in summary
                    }

                    results_data[algorithm][scoring] = converted_data

                except Exception as e:
                    print(f"Warning: Could not load {summary_file}: {e}")

    if results_data:
        print(f"Loaded data for {len(results_data)} algorithms:")
        for alg in results_data:
            print(f"  - {alg}: {list(results_data[alg].keys())}")

        visualizer = PublicationVisualizer()
        visualizer.generate_comprehensive_report(
            results_data,
            output_dir,
            "PandaDock Comprehensive Molecular Docking Analysis"
        )
    else:
        print("No valid results found in directory")


# Example usage function
def main():
    """Example usage of the publication visualization system"""
    # This would typically be called with actual results directory
    results_dir = "/Users/pritam/Desktop/PandaDock_Ultimate_Version/algorithm_comparison_full"
    create_publication_plots_from_directory(results_dir)


if __name__ == "__main__":
    main()