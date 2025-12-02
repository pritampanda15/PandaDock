"""
Professional Affinity Plotting for PandaDock

Creates modern, publication-quality binding affinity plots using seaborn and plotly.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging
from ..core import DockingResult, Pose

# Optional seaborn import
try:
    import seaborn as sns
    HAS_SEABORN = True
    sns.set_palette("husl")
except ImportError:
    HAS_SEABORN = False

# Set modern style
plt.style.use('default')


class AffinityPlotter:
    """Professional binding affinity visualization"""

    def __init__(self, style: str = "modern"):
        self.logger = logging.getLogger("pandadock.visualization.plotter")
        self.style = style
        self._setup_style()

    def _setup_style(self):
        """Setup modern plotting style"""
        # Modern color palette
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'accent': '#F18F01',
            'success': '#C73E1D',
            'gradient': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        }

        # Font settings
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
            'font.size': 12,
            'axes.titlesize': 16,
            'axes.labelsize': 14,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'figure.titlesize': 18
        })

    def create_binding_affinity_plot(self, result: DockingResult, output_file: Path,
                                    show_top_n: int = 20, include_ensemble: bool = True):
        """Alias for plot_binding_affinities for backward compatibility"""
        return self.plot_binding_affinities(result, output_file, show_top_n, include_ensemble)

    def create_interaction_energy_plot(self, interactions: Dict, output_file: Path):
        """Create interaction energy decomposition plot"""
        if not interactions or 'interaction_types' not in interactions:
            self.logger.warning("No interaction data to plot")
            return

        # Extract interaction types and values
        interaction_data = interactions['interaction_types']
        if not interaction_data:
            self.logger.warning("No interaction types found")
            return

        # Create bar plot
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('white')

        types = list(interaction_data.keys())
        values = list(interaction_data.values())

        bars = ax.bar(types, values, color=self.colors['primary'], alpha=0.8, edgecolor='white', linewidth=1)

        # Styling
        ax.set_title('Protein-Ligand Interaction Analysis', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Interaction Type', fontsize=12)
        ax.set_ylabel('Number of Contacts', fontsize=12)
        ax.tick_params(axis='x', rotation=45)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        self.logger.info(f"Interaction energy plot saved to {output_file}")

    def plot_binding_affinities(self, result: DockingResult, output_file: Path,
                              show_top_n: int = 20, include_ensemble: bool = True):
        """
        Create professional binding affinity plot

        Args:
            result: DockingResult with poses and energies
            output_file: Output PNG file path
            show_top_n: Number of top poses to show
            include_ensemble: Whether to show ensemble average
        """
        if not result.poses:
            self.logger.warning("No poses to plot")
            return

        # Get top poses
        top_poses = result.get_top_poses(show_top_n)
        pose_numbers = list(range(1, len(top_poses) + 1))
        energies = [pose.energy for pose in top_poses]

        # Create figure with modern styling
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor('white')

        # Main scatter plot with gradient colors
        scatter = ax.scatter(
            pose_numbers, energies,
            c=pose_numbers, cmap='viridis',
            s=80, alpha=0.8, edgecolors='white', linewidth=1.5
        )

        # Add connecting line
        ax.plot(pose_numbers, energies,
               color=self.colors['primary'], alpha=0.6, linewidth=2, linestyle='--')

        # Highlight best pose
        if energies:
            ax.scatter([1], [energies[0]],
                      color=self.colors['accent'], s=150,
                      edgecolors='white', linewidth=2, zorder=5,
                      marker='*', label='Best Pose')

        # Add ensemble energy line if available
        if include_ensemble and hasattr(result, 'ensemble_binding_energy'):
            if np.isfinite(result.ensemble_binding_energy):
                ax.axhline(
                    result.ensemble_binding_energy,
                    color=self.colors['secondary'],
                    linewidth=3, linestyle='-', alpha=0.8,
                    label=f'Ensemble ΔG: {result.ensemble_binding_energy:.2f} kcal/mol'
                )

        # Styling
        ax.set_xlabel('Pose Rank', fontweight='bold')
        ax.set_ylabel('Binding Energy (kcal/mol)', fontweight='bold')
        ax.set_title(f'Binding Affinities - {result.ligand_name}',
                    fontweight='bold', pad=20)

        # Grid and spines
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('gray')
        ax.spines['bottom'].set_color('gray')

        # Color bar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label('Pose Rank', fontweight='bold')

        # Legend
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)

        # Add statistics box
        self._add_statistics_box(ax, top_poses, result)

        # Tight layout and save
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()

        self.logger.info(f"Binding affinity plot saved to {output_file}")

    def _add_statistics_box(self, ax, poses: List[Pose], result: DockingResult):
        """Add statistics box to the plot"""
        energies = [pose.energy for pose in poses]

        stats_text = [
            f"Poses: {len(poses)}",
            f"Best: {min(energies):.2f} kcal/mol",
            f"Mean: {np.mean(energies):.2f} kcal/mol",
            f"Std: {np.std(energies):.2f} kcal/mol"
        ]

        if hasattr(result, 'ensemble_confidence'):
            stats_text.append(f"Confidence: {result.ensemble_confidence:.3f}")

        stats_str = '\n'.join(stats_text)

        # Add text box
        ax.text(0.02, 0.98, stats_str,
               transform=ax.transAxes,
               verticalalignment='top',
               bbox=dict(boxstyle='round,pad=0.5',
                        facecolor='lightgray', alpha=0.8),
               fontsize=10, fontweight='bold')

    def plot_energy_landscape(self, results: List[DockingResult], output_file: Path):
        """Plot energy landscape comparison across multiple ligands"""
        if not results:
            return

        fig, ax = plt.subplots(figsize=(14, 8))

        if HAS_SEABORN:
            colors = sns.color_palette("husl", len(results))
        else:
            colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

        for i, result in enumerate(results):
            if not result.poses:
                continue

            top_poses = result.get_top_poses(10)
            energies = [pose.energy for pose in top_poses]
            poses = list(range(1, len(energies) + 1))

            ax.plot(poses, energies,
                   color=colors[i], linewidth=2, marker='o',
                   label=result.ligand_name, alpha=0.8)

        ax.set_xlabel('Pose Rank', fontweight='bold')
        ax.set_ylabel('Binding Energy (kcal/mol)', fontweight='bold')
        ax.set_title('Energy Landscape Comparison', fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Energy landscape plot saved to {output_file}")

    def plot_ensemble_convergence(self, result: DockingResult, output_file: Path):
        """Plot Boltzmann ensemble convergence"""
        if not result.poses or not hasattr(result, 'boltzmann_weights'):
            return

        weights = np.array(result.boltzmann_weights)
        cumulative_weights = np.cumsum(weights)
        poses = list(range(1, len(weights) + 1))

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Individual weights
        ax1.bar(poses, weights, color=self.colors['primary'], alpha=0.7)
        ax1.set_xlabel('Pose Rank')
        ax1.set_ylabel('Boltzmann Weight')
        ax1.set_title('Individual Pose Weights')
        ax1.grid(True, alpha=0.3)

        # Cumulative weights
        ax2.plot(poses, cumulative_weights,
                color=self.colors['secondary'], linewidth=3, marker='o')
        ax2.axhline(0.95, color='red', linestyle='--', alpha=0.7,
                   label='95% threshold')
        ax2.set_xlabel('Pose Rank')
        ax2.set_ylabel('Cumulative Weight')
        ax2.set_title('Ensemble Convergence')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Ensemble convergence plot saved to {output_file}")