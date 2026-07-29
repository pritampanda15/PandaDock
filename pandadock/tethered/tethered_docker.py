#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tethered Docker for Constrained Docking

Performs molecular docking with distance constraints to validate scoring
functions and reproduce crystallographic binding modes.

Author: Pritam Kumar Panda @ Stanford University
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from Bio.PDB import PDBParser
import subprocess
import tempfile


class TetheredDocker:
    """Perform tethered/constrained molecular docking"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def run_tethered_docking(
        self,
        receptor_file: str,
        ligand_file: str,
        reference_pose: Optional[str] = None,
        tether_radius: float = 2.0,
        scoring_function: str = 'physics_based',
        algorithm: str = 'genetic',
        output_dir: str = '.',
        num_poses: int = 10
    ) -> Dict[str, Any]:
        """
        Run tethered docking with distance constraints

        Args:
            receptor_file: Receptor PDB file
            ligand_file: Ligand structure file
            reference_pose: Reference pose PDB for constraints
            tether_radius: Maximum distance from reference (Å)
            scoring_function: Scoring function to use
            algorithm: Docking algorithm
            output_dir: Output directory
            num_poses: Number of poses to generate

        Returns:
            Dictionary with docking results and statistics
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            self.logger.info(f"Starting tethered docking")
            self.logger.info(f"Receptor: {receptor_file}")
            self.logger.info(f"Ligand: {ligand_file}")
            self.logger.info(f"Reference: {reference_pose}")
            self.logger.info(f"Tether radius: {tether_radius} Å")

            # Parse reference pose if provided
            reference_center = None
            if reference_pose:
                reference_center = self._get_ligand_center(reference_pose)
                self.logger.info(f"Reference center: {reference_center}")

            # Setup docking configuration
            docking_config = self._create_docking_config(
                receptor_file=receptor_file,
                ligand_file=ligand_file,
                reference_center=reference_center,
                tether_radius=tether_radius,
                scoring_function=scoring_function,
                algorithm=algorithm,
                num_poses=num_poses,
                output_dir=output_path,
                reference_pose=reference_pose
            )

            # Run docking
            docking_results = self._execute_tethered_docking(docking_config)

            # Analyze results
            analysis_results = self._analyze_tethered_results(
                docking_results, reference_pose, tether_radius, output_path
            )

            # Generate comprehensive report
            self._generate_tethered_report(docking_results, analysis_results, output_path)

            return {
                'best_score': docking_results.get('best_score'),
                'num_poses': len(docking_results.get('poses', [])),
                'avg_rmsd': analysis_results.get('avg_rmsd'),
                'poses_within_constraint': analysis_results.get('poses_within_constraint'),
                'output_files': analysis_results.get('output_files', {}),
                'summary_report': str(output_path / 'tethered_docking_report.json')
            }

        except Exception as e:
            self.logger.error(f"Tethered docking failed: {e}")
            raise

    def _get_ligand_center(self, pdb_file: str) -> np.ndarray:
        """
        Centroid of a reference ligand.

        Accepts SDF and MOL2 as well as PDB. This used to parse every reference
        with BioPython's PDB parser, which returns no atoms for an SDF and so
        failed with "No atoms found in reference pose" -- for the format users
        are most likely to have, since it is what the ligand option itself takes.
        Hydrogens are excluded so the centroid does not shift with protonation.
        """
        try:
            suffix = Path(pdb_file).suffix.lower()
            coordinates = []

            if suffix in ('.sdf', '.mol', '.mol2'):
                from rdkit import Chem

                if suffix == '.mol2':
                    mol = Chem.MolFromMol2File(pdb_file, removeHs=False, sanitize=False)
                elif suffix == '.mol':
                    mol = Chem.MolFromMolFile(pdb_file, removeHs=False, sanitize=False)
                else:
                    mol = next(
                        (m for m in Chem.SDMolSupplier(pdb_file, removeHs=False,
                                                       sanitize=False) if m is not None),
                        None,
                    )
                if mol is not None and mol.GetNumConformers() > 0:
                    positions = mol.GetConformer().GetPositions()
                    coordinates = [
                        positions[a.GetIdx()]
                        for a in mol.GetAtoms()
                        if a.GetAtomicNum() > 1
                    ] or list(positions)
            else:
                parser = PDBParser(QUIET=True)
                structure = parser.get_structure('ligand', pdb_file)
                for model in structure:
                    for chain in model:
                        for residue in chain:
                            for atom in residue:
                                if (atom.element or '').strip().upper() == 'H':
                                    continue
                                coordinates.append(atom.get_coord())

            if not coordinates:
                raise ValueError(
                    f"No atoms found in reference pose '{pdb_file}'. "
                    f"Supported formats: PDB, SDF, MOL, MOL2."
                )

            return np.mean(np.asarray(coordinates, dtype=float), axis=0)

        except Exception as e:
            self.logger.error(f"Error calculating ligand center: {e}")
            raise

    def _create_docking_config(
        self,
        receptor_file: str,
        ligand_file: str,
        reference_center: Optional[np.ndarray],
        tether_radius: float,
        scoring_function: str,
        algorithm: str,
        num_poses: int,
        output_dir: Path,
        reference_pose: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create docking configuration for tethered docking"""

        config = {
            'receptor': str(receptor_file),
            'ligand': str(ligand_file),
            'scoring_function': scoring_function,
            'algorithm': algorithm,
            'num_poses': num_poses,
            'output_dir': str(output_dir),
            'reference_pose_file': str(reference_pose) if reference_pose else None
        }

        # Add search space configuration
        if reference_center is not None:
            # Constrained search space centered on reference
            config['search_space'] = {
                'center': reference_center.tolist(),
                'size': [tether_radius * 2] * 3,  # Search box size
                'constraint_type': 'spherical',
                'constraint_radius': tether_radius
            }
        else:
            # Default search space (unconstrained)
            config['search_space'] = {
                'center': [0, 0, 0],  # Will be auto-calculated
                'size': [20, 20, 20],
                'constraint_type': 'none'
            }

        # Algorithm-specific parameters
        if algorithm == 'genetic':
            config['algorithm_params'] = {
                'population_size': 50,
                'generations': 20,
                'mutation_rate': 0.2,
                'crossover_rate': 0.8,
                'elite_fraction': 0.1
            }
        elif algorithm == 'hierarchical':
            config['algorithm_params'] = {
                'coarse_samples': 100,
                'medium_samples': 50,
                'fine_samples': 20
            }
        elif algorithm == 'monte_carlo':
            config['algorithm_params'] = {
                'num_steps': 10000,
                'temperature': 300.0,
                'cooling_rate': 0.99
            }

        return config

    def _execute_tethered_docking(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a real docking search with a centroid restraint toward the reference.

        The previous implementation was a simulation, as its own comment stated:
        pose centres were drawn from a Gaussian around the reference, scores were
        `-8.0 - numpy.random.exponential(2.0)`, and coordinates were either the
        reference jittered by 0.05 A or a synthetic benzene ring. Neither the
        ligand nor the receptor was read, so every reported score and RMSD was a
        random number.

        The restraint is applied inside the objective as a flat-bottom penalty:
        free movement within the tether radius, harmonic cost beyond it. That
        steers the optimizer into the allowed region rather than filtering
        out-of-range results afterwards.
        """
        from ..docking.algorithms.pandacore import PandaCoreDocker
        from ..docking.scoring.vina_scoring import VinaScoring

        self.logger.info("Executing tethered docking...")

        reference_center = np.array(config['search_space']['center'], dtype=float)
        constraint_radius = float(config['search_space'].get('constraint_radius', 10.0))

        ligand_mol = self._load_ligand(config['ligand'])
        if ligand_mol is None:
            raise ValueError(f"Could not read ligand from {config['ligand']}")

        # Box must accommodate the tether sphere plus the ligand's own extent.
        positions = np.asarray(ligand_mol.GetConformer().GetPositions())
        extent = float(np.max(positions.max(axis=0) - positions.min(axis=0)))
        box = np.full(3, max(2.0 * constraint_radius + extent + 4.0, 16.0))

        docker = PandaCoreDocker()
        docker.set_scoring_function(VinaScoring())
        result = docker.dock(
            receptor_file=config['receptor'],
            ligand_mol=ligand_mol,
            grid_center=reference_center,
            grid_dimensions=box,
            num_poses=config['num_poses'],
            tether_center=reference_center,
            tether_radius=constraint_radius,
            seed=config.get('seed'),
        )

        heavy = [a.GetIdx() for a in ligand_mol.GetAtoms() if a.GetAtomicNum() > 1]

        poses = []
        for rank, pose in enumerate(result.poses, 1):
            centroid = pose.coordinates[heavy].mean(axis=0)
            distance = float(np.linalg.norm(centroid - reference_center))
            poses.append({
                'rank': rank,
                'score': float(pose.energy),
                'center': centroid.tolist(),
                'rmsd': distance,
                'within_constraint': bool(distance <= constraint_radius),
                'coordinates': pose.coordinates.tolist(),
            })

        return {
            'poses': poses,
            'best_score': poses[0]['score'] if poses else None,
            'num_poses': len(poses),
            'reference_center': reference_center.tolist(),
            'constraint_radius': constraint_radius,
            'runtime_seconds': result.runtime_seconds,
        }

    def _load_ligand(self, ligand_file: str):
        """Read a ligand from SDF, MOL, MOL2 or PDB."""
        from rdkit import Chem

        suffix = Path(ligand_file).suffix.lower()
        if suffix == '.mol2':
            mol = Chem.MolFromMol2File(ligand_file, removeHs=False)
        elif suffix == '.mol':
            mol = Chem.MolFromMolFile(ligand_file, removeHs=False)
        elif suffix == '.pdb':
            mol = Chem.MolFromPDBFile(ligand_file, removeHs=False)
        else:
            mol = next(
                (m for m in Chem.SDMolSupplier(ligand_file, removeHs=False) if m is not None),
                None,
            )
        return mol

    def _generate_realistic_coordinates(self, pose_center: np.ndarray,
                                        reference_center: np.ndarray) -> List[List[float]]:
        """
        Deprecated. Retained only so older callers fail loudly.

        This used to produce "poses" by taking the reference coordinates, applying
        a random rotation and 0.05 A of noise, and returning the result -- so a
        redocking run reported a jittered copy of the answer rather than a
        prediction. When the reference could not be read it fell through to
        `_generate_simple_molecule`, emitting a synthetic benzene ring in place of
        the user's ligand: neither the ligand nor the receptor entered the
        calculation at all.

        Tethered docking now runs the real search with a flat-bottom centroid
        restraint; see `TetheredDocker._run_tethered_search`.
        """
        raise NotImplementedError(
            "Synthetic pose generation has been removed. Tethered docking now "
            "performs a real search with a centroid restraint."
        )

    def _generate_random_rotation(self) -> np.ndarray:
        """Generate a random 3D rotation matrix"""
        # Generate random rotation using Rodrigues' rotation formula
        axis = np.random.normal(0, 1, 3)
        axis = axis / np.linalg.norm(axis)

        # Random rotation angle (up to 15 degrees for better geometry preservation)
        angle = np.random.uniform(0, np.pi/12)

        # Rodrigues' rotation formula
        K = np.array([[0, -axis[2], axis[1]],
                     [axis[2], 0, -axis[0]],
                     [-axis[1], axis[0], 0]])

        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
        return R

    def _generate_simple_molecule(self, center: np.ndarray) -> List[List[float]]:
        """Generate a simple but realistic small molecule structure"""
        coordinates = []

        # Create a benzene-like ring structure
        ring_radius = 1.4  # Typical aromatic C-C distance

        # Central ring (6 atoms)
        for i in range(6):
            angle = 2 * np.pi * i / 6
            x = center[0] + ring_radius * np.cos(angle)
            y = center[1] + ring_radius * np.sin(angle)
            z = center[2] + np.random.normal(0, 0.1)  # Small z variation
            coordinates.append([x, y, z])

        # Add some substituents
        for i in range(4):
            # Pick random ring atom as attachment point
            ring_idx = np.random.randint(0, 6)
            ring_pos = np.array(coordinates[ring_idx])

            # Direction away from center
            direction = ring_pos - center
            direction = direction / np.linalg.norm(direction)

            # Add substituent 1.5 Å away
            substituent_pos = ring_pos + direction * 1.5
            substituent_pos += np.random.normal(0, 0.2, 3)  # Add some randomness
            coordinates.append(substituent_pos.tolist())

        return coordinates

    def _analyze_tethered_results(
        self,
        docking_results: Dict[str, Any],
        reference_pose: Optional[str],
        tether_radius: float,
        output_dir: Path
    ) -> Dict[str, Any]:
        """Analyze tethered docking results"""

        poses = docking_results.get('poses', [])
        if not poses:
            return {'avg_rmsd': 0, 'poses_within_constraint': 0, 'output_files': {}}

        # Calculate statistics
        rmsd_values = [pose.get('rmsd', 0) for pose in poses]
        avg_rmsd = np.mean(rmsd_values) if rmsd_values else 0

        poses_within_constraint = sum(1 for pose in poses if pose.get('within_constraint', False))

        # Save detailed results
        output_files = {}

        # 1. Save poses as PDB files
        poses_dir = output_dir / 'poses'
        poses_dir.mkdir(exist_ok=True)

        for i, pose in enumerate(poses[:10]):  # Save top 10 poses
            pose_file = poses_dir / f'tethered_pose_{i+1}.pdb'
            self._save_pose_as_pdb(pose, pose_file)
            if i == 0:  # Record best pose
                output_files['best_pose'] = str(pose_file)

        output_files['poses_directory'] = str(poses_dir)

        # 2. Save results summary
        summary_file = output_dir / 'tethered_results.json'
        with open(summary_file, 'w') as f:
            json.dump({
                'docking_config': docking_results.get('config'),
                'statistics': {
                    'num_poses': len(poses),
                    'best_score': docking_results.get('best_score'),
                    'avg_rmsd': avg_rmsd,
                    'poses_within_constraint': poses_within_constraint,
                    'constraint_success_rate': poses_within_constraint / len(poses) if poses else 0
                },
                'poses': poses
            }, f, indent=2)

        output_files['results_summary'] = str(summary_file)

        # 3. Generate analysis plots (mock)
        plots_dir = output_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)

        rmsd_plot = plots_dir / 'rmsd_vs_score.png'
        self._generate_rmsd_plot(poses, rmsd_plot)
        output_files['rmsd_plot'] = str(rmsd_plot)

        constraint_plot = plots_dir / 'constraint_analysis.png'
        self._generate_constraint_plot(poses, tether_radius, constraint_plot)
        output_files['constraint_plot'] = str(constraint_plot)

        return {
            'avg_rmsd': avg_rmsd,
            'poses_within_constraint': poses_within_constraint,
            'constraint_success_rate': poses_within_constraint / len(poses) if poses else 0,
            'output_files': output_files
        }

    def _save_pose_as_pdb(self, pose: Dict[str, Any], output_file: Path):
        """Save pose coordinates as PDB file"""
        coordinates = pose.get('coordinates', [])
        center = pose.get('center', [0, 0, 0])

        # Get reference atom information if available
        ref_atoms = self._get_reference_atom_info()

        with open(output_file, 'w') as f:
            f.write(f"REMARK Generated by PandaDock Tethered Docking\n")
            f.write(f"REMARK Pose rank: {pose.get('rank', 0)}\n")
            f.write(f"REMARK Score: {pose.get('score', 0):.3f} kcal/mol\n")
            f.write(f"REMARK RMSD: {pose.get('rmsd', 0):.3f} Å\n")
            f.write(f"REMARK Center: {center[0]:.3f} {center[1]:.3f} {center[2]:.3f}\n")

            for i, coord in enumerate(coordinates):
                if ref_atoms and i < len(ref_atoms):
                    # Use reference atom information
                    atom_name = ref_atoms[i]['name']
                    element = ref_atoms[i]['element']
                else:
                    # Fallback to generic naming
                    atom_name = f"C{i+1}" if i < 30 else f"C{i+1}"
                    element = "C"

                # Proper PDB formatting
                f.write(f"HETATM{i+1:5d} {atom_name:>4s} LIG A   1    {coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}  1.00 20.00           {element:<2s}\n")

            # Add CONECT records for better bond visualization
            self._write_conect_records(f, ref_atoms, coordinates)

            f.write("END\n")

    def _get_reference_atom_info(self) -> Optional[List[Dict[str, str]]]:
        """Get atom names and elements from reference structure"""
        try:
            if not hasattr(self, 'reference_pose_file') or not self.reference_pose_file:
                return None

            from Bio.PDB import PDBParser

            parser = PDBParser(QUIET=True)
            structure = parser.get_structure('reference', self.reference_pose_file)

            atoms = []
            for model in structure:
                for chain in model:
                    for residue in chain:
                        for atom in residue:
                            element = atom.element.strip() if hasattr(atom, 'element') and atom.element.strip() else atom.get_name()[0]
                            atoms.append({
                                'name': atom.get_name(),
                                'element': element
                            })

            return atoms if atoms else None

        except Exception as e:
            self.logger.warning(f"Could not load reference atom info: {e}")
            return None

    def _write_conect_records(self, f, ref_atoms: Optional[List[Dict]], coordinates: List[List[float]]):
        """Write CONECT records to define bonds explicitly"""
        try:
            if not ref_atoms or not coordinates:
                return

            # Simple bond detection based on distance
            max_bond_distance = {
                ('C', 'C'): 1.8, ('C', 'N'): 1.7, ('C', 'O'): 1.6, ('C', 'H'): 1.3,
                ('N', 'N'): 1.6, ('N', 'O'): 1.6, ('N', 'H'): 1.2,
                ('O', 'O'): 1.6, ('O', 'H'): 1.2,
                ('H', 'H'): 0.8  # Very rare
            }

            bonds = []
            for i in range(len(coordinates)):
                for j in range(i + 1, len(coordinates)):
                    if i >= len(ref_atoms) or j >= len(ref_atoms):
                        continue

                    elem1 = ref_atoms[i]['element']
                    elem2 = ref_atoms[j]['element']

                    # Get maximum bond distance for this pair
                    key = tuple(sorted([elem1, elem2]))
                    max_dist = max_bond_distance.get(key, 1.8)

                    # Calculate distance
                    coord1 = np.array(coordinates[i])
                    coord2 = np.array(coordinates[j])
                    distance = np.linalg.norm(coord1 - coord2)

                    if distance <= max_dist:
                        bonds.append((i + 1, j + 1))  # PDB uses 1-based indexing

            # Write CONECT records (max 4 bonds per line in PDB format)
            bond_dict = {}
            for atom1, atom2 in bonds:
                if atom1 not in bond_dict:
                    bond_dict[atom1] = []
                if atom2 not in bond_dict:
                    bond_dict[atom2] = []
                bond_dict[atom1].append(atom2)
                bond_dict[atom2].append(atom1)

            for atom, bonded_atoms in sorted(bond_dict.items()):
                # Write CONECT records in chunks of 4
                for i in range(0, len(bonded_atoms), 4):
                    chunk = bonded_atoms[i:i+4]
                    conect_line = f"CONECT{atom:5d}"
                    for bonded in chunk:
                        conect_line += f"{bonded:5d}"
                    f.write(conect_line + "\n")

        except Exception as e:
            self.logger.warning(f"Could not write CONECT records: {e}")

    def _generate_rmsd_plot(self, poses: List[Dict], output_file: Path):
        """Generate RMSD vs Score plot"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            # Extract data
            scores = [pose.get('score', 0) for pose in poses]
            rmsds = [pose.get('rmsd', 0) for pose in poses]
            ranks = [pose.get('rank', i+1) for i, pose in enumerate(poses)]

            # Create the plot
            plt.figure(figsize=(10, 6))

            # Scatter plot
            scatter = plt.scatter(rmsds, scores, c=ranks, cmap='viridis', s=100, alpha=0.7)

            # Add colorbar
            cbar = plt.colorbar(scatter)
            cbar.set_label('Pose Rank', rotation=270, labelpad=15)

            # Labels and title
            plt.xlabel('RMSD from Reference (Å)', fontsize=12)
            plt.ylabel('Docking Score (kcal/mol)', fontsize=12)
            plt.title('Tethered Docking: RMSD vs Docking Score', fontsize=14, fontweight='bold')

            # Add grid
            plt.grid(True, alpha=0.3)

            # Annotate best pose
            best_pose_idx = np.argmin(scores)
            plt.annotate(f'Best Pose\n(Rank {ranks[best_pose_idx]})',
                        xy=(rmsds[best_pose_idx], scores[best_pose_idx]),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

            # Save plot
            plt.tight_layout()
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()

            self.logger.info(f"RMSD plot saved: {output_file}")

        except ImportError:
            # Fallback to text if matplotlib not available
            with open(output_file.with_suffix('.txt'), 'w') as f:
                f.write("RMSD vs Score Analysis\n")
                f.write("=====================\n")
                for pose in poses:
                    f.write(f"Pose {pose.get('rank', 0):2d}: Score = {pose.get('score', 0):7.3f}, RMSD = {pose.get('rmsd', 0):6.3f}\n")

            self.logger.info(f"RMSD analysis saved as text: {output_file.with_suffix('.txt')}")

    def _generate_constraint_plot(self, poses: List[Dict], tether_radius: float, output_file: Path):
        """Generate constraint satisfaction analysis"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            # Extract data
            rmsds = [pose.get('rmsd', 0) for pose in poses]
            within_constraint = [pose.get('within_constraint', False) for pose in poses]
            scores = [pose.get('score', 0) for pose in poses]

            # Create the plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Plot 1: RMSD distribution with constraint boundary
            colors = ['green' if w else 'red' for w in within_constraint]
            ax1.scatter(range(len(rmsds)), rmsds, c=colors, s=100, alpha=0.7)
            ax1.axhline(y=tether_radius, color='blue', linestyle='--', linewidth=2,
                       label=f'Tether Radius ({tether_radius:.1f} Å)')
            ax1.set_xlabel('Pose Index', fontsize=12)
            ax1.set_ylabel('RMSD from Reference (Å)', fontsize=12)
            ax1.set_title('Constraint Satisfaction by Pose', fontsize=14, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Plot 2: Constraint satisfaction pie chart
            within_count = sum(within_constraint)
            outside_count = len(poses) - within_count
            labels = [f'Within Constraint\n({within_count} poses)', f'Outside Constraint\n({outside_count} poses)']
            sizes = [within_count, outside_count]
            colors_pie = ['green', 'red']

            ax2.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=90)
            ax2.set_title('Overall Constraint Satisfaction', fontsize=14, fontweight='bold')

            # Save plot
            plt.tight_layout()
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()

            self.logger.info(f"Constraint plot saved: {output_file}")

        except ImportError:
            # Fallback to text if matplotlib not available
            with open(output_file.with_suffix('.txt'), 'w') as f:
                f.write("Constraint Satisfaction Analysis\n")
                f.write("===============================\n")
                f.write(f"Tether radius: {tether_radius:.1f} Å\n\n")

                within_count = sum(1 for pose in poses if pose.get('within_constraint', False))
                f.write(f"Poses within constraint: {within_count}/{len(poses)} ({within_count/len(poses)*100:.1f}%)\n")

            self.logger.info(f"Constraint analysis saved as text: {output_file.with_suffix('.txt')}")

    def _generate_tethered_report(
        self,
        docking_results: Dict[str, Any],
        analysis_results: Dict[str, Any],
        output_dir: Path
    ):
        """Generate comprehensive tethered docking report"""

        report_file = output_dir / 'tethered_docking_report.json'

        report = {
            'tethered_docking_report': {
                'summary': {
                    'total_poses': len(docking_results.get('poses', [])),
                    'best_score': docking_results.get('best_score'),
                    'average_rmsd': analysis_results.get('avg_rmsd'),
                    'poses_within_constraint': analysis_results.get('poses_within_constraint'),
                    'constraint_success_rate': analysis_results.get('constraint_success_rate')
                },
                'configuration': docking_results.get('config'),
                'detailed_results': analysis_results,
                'output_files': analysis_results.get('output_files', {})
            }
        }

        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        self.logger.info(f"Tethered docking report saved: {report_file}")