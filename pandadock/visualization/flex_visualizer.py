"""
Professional visualization tools for flexible docking results
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
from rdkit import Chem

from pandadock.flex_docking.core import FlexibleDockingResult, FlexiblePose
from pandadock.docking.visualization.visualizer import DockingVisualizer


class FlexDockingVisualizer(DockingVisualizer):
    """Advanced visualization for flexible docking results"""

    def __init__(self):
        super().__init__()

    def save_flex_complexes(self, result: FlexibleDockingResult, receptor_file: str,
                           ligand_mol: Chem.Mol, output_dir: Path) -> None:
        """Save flexible docking complex structures with refined receptors"""

        complexes_dir = output_dir / "complexes"
        complexes_dir.mkdir(exist_ok=True)

        # Save top flexible poses with their refined receptors
        top_poses = result.get_top_poses(10)

        for i, pose in enumerate(top_poses, 1):
            # Find the corresponding refined receptor
            refined_receptor = self._find_refined_receptor(pose, result.refined_complexes)

            if refined_receptor:
                # Create complex with refined receptor
                complex_file = complexes_dir / f"flex_complex_{i}.pdb"
                self._create_flex_complex(refined_receptor, pose, ligand_mol, complex_file)

                # Save ligand pose only (no individual refined receptor)
                ligand_file = complexes_dir / f"flex_ligand_{i}.pdb"
                self.save_pose_pdb(pose, ligand_file, ligand_mol)

        # An SDF alongside the PDBs, matching what `pandadock dock` writes. PDB
        # cannot carry bond orders or formal charges, so a ligand read back from
        # one has its bonds inferred from distance, which routinely mis-assigns
        # aromatic rings.
        self._save_flex_poses_sdf(top_poses, ligand_mol, output_dir)

    def _save_flex_poses_sdf(self, poses, ligand_mol: Chem.Mol, output_dir: Path) -> None:
        """Write all flexible-docking poses to one SDF, annotated with IFD terms."""
        if ligand_mol is None or not poses:
            return
        try:
            writer = Chem.SDWriter(str(output_dir / "poses.sdf"))
            try:
                for rank, pose in enumerate(poses, 1):
                    mol = Chem.Mol(ligand_mol)
                    mol.RemoveAllConformers()
                    conf = Chem.Conformer(mol.GetNumAtoms())
                    n = min(mol.GetNumAtoms(), len(pose.coordinates))
                    for i in range(n):
                        conf.SetAtomPosition(i, pose.coordinates[i].tolist())
                    mol.AddConformer(conf, assignId=True)

                    mol.SetProp("_Name", f"flex_pose{rank}")
                    mol.SetProp("rank", str(rank))
                    for prop, value in (
                        ("ifd_score", getattr(pose, "ifd_score", None)),
                        ("binding_energy", getattr(pose, "binding_energy", None)),
                        ("refinement_cost", getattr(pose, "refinement_cost", None)),
                        ("receptor_rmsd", getattr(pose, "receptor_rmsd", None)),
                    ):
                        if value is not None:
                            mol.SetProp(prop, f"{float(value):.3f}")
                    writer.write(mol)
            finally:
                writer.close()
            self.logger.info("Saved flexible docking poses to %s", output_dir / "poses.sdf")
        except Exception as exc:
            self.logger.error("Could not write poses.sdf: %s", exc)

    def _find_refined_receptor(self, pose: FlexiblePose, refined_complexes) -> Optional[str]:
        """Find the refined receptor corresponding to a pose"""
        # Simplified - in production you'd track pose-receptor relationships
        if refined_complexes:
            return refined_complexes[0].receptor  # Use first refined receptor
        return None

    def _create_flex_complex(self, receptor_file: str, pose: FlexiblePose,
                           ligand_mol: Chem.Mol, output_file: Path) -> None:
        """Create complex PDB with refined receptor and ligand pose"""

        # Read refined receptor
        with open(receptor_file, 'r') as f:
            receptor_lines = [line for line in f if line.startswith(('ATOM', 'HETATM'))]

        # Verify pose has coordinates
        if not hasattr(pose, 'coordinates') or pose.coordinates is None or len(pose.coordinates) == 0:
            print(f"Warning: Pose has no coordinates! Using ligand_mol conformer instead.")
            # Fallback: get coordinates from ligand_mol
            if ligand_mol.GetNumConformers() > 0:
                conf = ligand_mol.GetConformer()
                import numpy as np
                pose_coords = np.array([conf.GetAtomPosition(i) for i in range(ligand_mol.GetNumAtoms())])
            else:
                print(f"ERROR: No conformer in ligand_mol either! Cannot create complex.")
                # Write receptor-only complex
                with open(output_file, 'w') as f:
                    f.write("REMARK   WARNING: No ligand coordinates available\n")
                    f.write("REMARK   Receptor-only structure\n")
                    for line in receptor_lines:
                        f.write(line)
                    f.write("END\n")
                return
        else:
            pose_coords = pose.coordinates

        # Create ligand ATOM records
        ligand_lines = []
        atom_serial = len(receptor_lines) + 1

        for i, coord in enumerate(pose_coords):
            if i < ligand_mol.GetNumAtoms():
                rdkit_atom = ligand_mol.GetAtomWithIdx(i)
                element = rdkit_atom.GetSymbol().upper()
                atom_name = f'{element}{i + 1}'[:4]
            else:
                element = 'C'
                atom_name = f'C{i + 1}'[:4]

            # Strict PDB column layout. The previous format string carried an
            # extra space after the serial number, which shifted resName, chainID
            # and every subsequent field one column right. Strict parsers then
            # read the residue name out of the altLoc column and the coordinates
            # out of alignment, so the file could not be reloaded. The ligand is
            # also a HETATM record, not ATOM.
            #
            #  1-6 record | 7-11 serial | 13-16 name | 17 altLoc | 18-20 resName
            #  22 chain   | 23-26 resSeq | 31-38 x | 39-46 y | 47-54 z
            #  55-60 occupancy | 61-66 bfactor | 77-78 element
            name_field = atom_name.ljust(4) if len(atom_name) == 4 else f" {atom_name:<3}"
            atom_line = (
                f"HETATM{atom_serial:5d} {name_field}"
                f" {'LIG':<3} L{1:4d}    "
                f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}"
                f"{1.00:6.2f}{20.00:6.2f}"
                f"{'':10}{element:>2}\n"
            )
            ligand_lines.append(atom_line)
            atom_serial += 1

        # Write combined complex
        with open(output_file, 'w') as f:
            # Write header
            f.write("REMARK   Flexible docking complex with refined receptor\n")
            f.write("REMARK   Generated by PandaDock-Flex\n")
            f.write(f"REMARK   Ligand atoms: {len(ligand_lines)}\n")
            f.write(f"REMARK   Receptor atoms: {len(receptor_lines)}\n")

            # Write receptor atoms
            for line in receptor_lines:
                f.write(line)

            # Write ligand atoms
            for line in ligand_lines:
                f.write(line)

            f.write("END\n")

    def _copy_file(self, src: str, dst: Path) -> None:
        """Copy file from src to dst"""
        try:
            with open(src, 'r') as src_f, open(dst, 'w') as dst_f:
                dst_f.write(src_f.read())
        except Exception as e:
            print(f"Warning: Could not copy {src} to {dst}: {e}")

    def generate_ifd_report(self, result: FlexibleDockingResult, output_dir: Path) -> None:
        """Generate comprehensive IFD analysis report"""

        report_dir = output_dir / "analysis"
        report_dir.mkdir(exist_ok=True)

        # Generate plots
        self._plot_ifd_score_distribution(result, report_dir)
        self._plot_energy_components(result, report_dir)
        self._plot_refinement_costs(result, report_dir)
        self._plot_pose_clustering(result, report_dir)

        # Generate HTML report
        self._generate_html_report(result, report_dir)

    def _plot_ifd_score_distribution(self, result: FlexibleDockingResult, output_dir: Path) -> None:
        """Plot distribution of IFD scores"""

        if not result.final_poses:
            return

        plt.figure(figsize=(10, 6))
        ifd_scores = [pose.ifd_score for pose in result.final_poses]

        # Histogram
        plt.subplot(1, 2, 1)
        plt.hist(ifd_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('IFD Score (kcal/mol)')
        plt.ylabel('Number of Poses')
        plt.title('IFD Score Distribution')
        plt.grid(True, alpha=0.3)

        # Box plot
        plt.subplot(1, 2, 2)
        plt.boxplot(ifd_scores, vert=True)
        plt.ylabel('IFD Score (kcal/mol)')
        plt.title('IFD Score Statistics')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "ifd_score_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_energy_components(self, result: FlexibleDockingResult, output_dir: Path) -> None:
        """Plot energy component breakdown"""

        if not result.final_poses:
            return

        # Prepare data
        binding_energies = [pose.binding_energy for pose in result.final_poses]
        refinement_costs = [pose.refinement_cost for pose in result.final_poses]
        ifd_scores = [pose.ifd_score for pose in result.final_poses]

        plt.figure(figsize=(12, 8))

        # Scatter plot: Binding Energy vs Refinement Cost
        plt.subplot(2, 2, 1)
        plt.scatter(binding_energies, refinement_costs, alpha=0.6, c=ifd_scores,
                   cmap='viridis', s=50)
        plt.colorbar(label='IFD Score (kcal/mol)')
        plt.xlabel('Binding Energy (kcal/mol)')
        plt.ylabel('Refinement Cost (kcal/mol)')
        plt.title('Energy Components')
        plt.grid(True, alpha=0.3)

        # Binding energy distribution
        plt.subplot(2, 2, 2)
        plt.hist(binding_energies, bins=15, alpha=0.7, color='lightcoral')
        plt.xlabel('Binding Energy (kcal/mol)')
        plt.ylabel('Count')
        plt.title('Binding Energy Distribution')
        plt.grid(True, alpha=0.3)

        # Refinement cost distribution
        plt.subplot(2, 2, 3)
        plt.hist(refinement_costs, bins=15, alpha=0.7, color='lightgreen')
        plt.xlabel('Refinement Cost (kcal/mol)')
        plt.ylabel('Count')
        plt.title('Refinement Cost Distribution')
        plt.grid(True, alpha=0.3)

        # Energy correlation
        plt.subplot(2, 2, 4)
        plt.scatter(binding_energies, ifd_scores, alpha=0.6, color='orange')
        plt.xlabel('Binding Energy (kcal/mol)')
        plt.ylabel('IFD Score (kcal/mol)')
        plt.title('Binding Energy vs IFD Score')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "energy_components.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_refinement_costs(self, result: FlexibleDockingResult, output_dir: Path) -> None:
        """Plot refinement costs and receptor changes"""

        if not result.refined_complexes:
            return

        plt.figure(figsize=(10, 6))

        # Refinement costs
        energy_costs = [complex_info.energy_cost for complex_info in result.refined_complexes]
        rmsd_values = [complex_info.rmsd_from_original for complex_info in result.refined_complexes]

        plt.subplot(1, 2, 1)
        plt.bar(range(len(energy_costs)), energy_costs, alpha=0.7, color='salmon')
        plt.xlabel('Refined Complex')
        plt.ylabel('Energy Cost (kcal/mol)')
        plt.title('Receptor Refinement Costs')
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.bar(range(len(rmsd_values)), rmsd_values, alpha=0.7, color='lightblue')
        plt.xlabel('Refined Complex')
        plt.ylabel('RMSD from Original (Å)')
        plt.title('Receptor Conformational Change')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "refinement_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_pose_clustering(self, result: FlexibleDockingResult, output_dir: Path) -> None:
        """Visualize pose clustering in 2D"""

        if len(result.final_poses) < 2:
            return

        # Simple 2D projection using first two coordinates
        x_coords = [pose.center[0] for pose in result.final_poses]
        y_coords = [pose.center[1] for pose in result.final_poses]
        ifd_scores = [pose.ifd_score for pose in result.final_poses]

        plt.figure(figsize=(10, 8))

        # Scatter plot colored by IFD score
        scatter = plt.scatter(x_coords, y_coords, c=ifd_scores, s=100,
                             cmap='RdYlBu_r', alpha=0.7, edgecolors='black')
        plt.colorbar(scatter, label='IFD Score (kcal/mol)')

        # Label top poses
        for i, pose in enumerate(result.final_poses[:5]):
            plt.annotate(f'{i+1}', (pose.center[0], pose.center[1]),
                        xytext=(5, 5), textcoords='offset points',
                        fontweight='bold', fontsize=12)

        plt.xlabel('X Coordinate (Å)')
        plt.ylabel('Y Coordinate (Å)')
        plt.title('Flexible Docking Poses - Spatial Distribution')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "pose_clustering.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _generate_html_report(self, result: FlexibleDockingResult, output_dir: Path) -> None:
        """Generate comprehensive HTML analysis report"""

        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>PandaDock-Flex Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background-color: #f0f8ff; padding: 20px; border-radius: 10px; }}
        .section {{ margin: 20px 0; }}
        .stats-table {{ border-collapse: collapse; width: 100%; }}
        .stats-table th, .stats-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        .stats-table th {{ background-color: #4CAF50; color: white; }}
        .pose-table {{ border-collapse: collapse; width: 100%; }}
        .pose-table th, .pose-table td {{ border: 1px solid #ddd; padding: 6px; text-align: center; }}
        .pose-table th {{ background-color: #2196F3; color: white; }}
        .image-container {{ text-align: center; margin: 20px 0; }}
        .best-pose {{ background-color: #e8f5e8; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🧬 PandaDock-Flex Analysis Report</h1>
        <p>Professional Induced-Fit Docking Results</p>
        <p><strong>Generated:</strong> {result.parameters.get('timestamp', 'N/A')}</p>
    </div>

    <div class="section">
        <h2>📊 Summary Statistics</h2>
        <table class="stats-table">
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Initial Poses Generated</td><td>{result.statistics['initial_poses_count']}</td></tr>
            <tr><td>Refined Receptor Conformations</td><td>{result.statistics['refined_complexes_count']}</td></tr>
            <tr><td>Final IFD Poses</td><td>{result.statistics['final_poses_count']}</td></tr>
            <tr><td>Flexible Residues</td><td>{result.statistics['flexible_residues_count']}</td></tr>
            <tr><td>Total Runtime</td><td>{result.statistics['total_runtime_seconds']:.1f} seconds</td></tr>
            <tr><td>Best IFD Score</td><td>{result.statistics.get('best_ifd_score', 'N/A'):.3f} kcal/mol</td></tr>
            <tr><td>Mean Binding Energy</td><td>{result.statistics.get('mean_binding_energy', 'N/A'):.3f} kcal/mol</td></tr>
            <tr><td>Mean Refinement Cost</td><td>{result.statistics.get('mean_refinement_cost', 'N/A'):.3f} kcal/mol</td></tr>
        </table>
    </div>

    <div class="section">
        <h2>🏆 Top Flexible Docking Poses</h2>
        <table class="pose-table">
            <tr>
                <th>Rank</th><th>IFD Score</th><th>Binding Energy</th><th>Refinement Cost</th>
                <th>Ligand Strain</th><th>H-Bonds</th><th>Contacts</th>
            </tr>
        """

        # Add top poses
        for i, pose in enumerate(result.get_top_poses(10), 1):
            row_class = "best-pose" if i == 1 else ""
            h_bonds = pose.interaction_fingerprint.get('hydrogen_bonds', 0)
            contacts = len(pose.flexible_residue_contacts)

            html_content += f"""
            <tr class="{row_class}">
                <td>{i}</td>
                <td>{pose.ifd_score:.3f}</td>
                <td>{pose.binding_energy:.3f}</td>
                <td>{pose.refinement_cost:.3f}</td>
                <td>{pose.ligand_strain_energy:.3f}</td>
                <td>{h_bonds:.1f}</td>
                <td>{contacts}</td>
            </tr>
            """

        html_content += f"""
        </table>
    </div>

    <div class="section">
        <h2>📈 Analysis Plots</h2>
        <div class="image-container">
            <img src="ifd_score_distribution.png" alt="IFD Score Distribution" width="800">
        </div>
        <div class="image-container">
            <img src="energy_components.png" alt="Energy Components" width="900">
        </div>
        <div class="image-container">
            <img src="refinement_analysis.png" alt="Refinement Analysis" width="800">
        </div>
        <div class="image-container">
            <img src="pose_clustering.png" alt="Pose Clustering" width="700">
        </div>
    </div>

    <div class="section">
        <h2>🔧 Flexible Residues</h2>
        <p><strong>Residues refined:</strong> {', '.join(result.flexible_residues)}</p>
        <p>These residues were allowed conformational flexibility during the induced-fit process.</p>
    </div>

    <div class="section">
        <h2>⚙️ Parameters Used</h2>
        <table class="stats-table">
        """

        for key, value in result.parameters.items():
            html_content += f"<tr><td>{key}</td><td>{value}</td></tr>"

        html_content += """
        </table>
    </div>

    <div class="section">
        <h2>📁 Output Files</h2>
        <ul>
            <li><strong>complexes/</strong> - Refined receptor-ligand complex structures</li>
            <li><strong>flex_docking_summary.json</strong> - Detailed numerical results</li>
            <li><strong>flex_poses_detailed.json</strong> - Complete pose information</li>
            <li><strong>refined_complexes.json</strong> - Receptor refinement details</li>
        </ul>
    </div>

    <footer style="margin-top: 40px; text-align: center; color: #666;">
        <p>Generated by PandaDock-Flex • Professional Induced-Fit Docking</p>
    </footer>
</body>
</html>
        """

        # Save HTML report
        with open(output_dir / "flex_docking_report.html", 'w') as f:
            f.write(html_content)

        print(f"✓ Comprehensive IFD report generated: {output_dir}/flex_docking_report.html")