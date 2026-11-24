"""
Four-phase induced-fit docking implementation
"""

import numpy as np
import os
import shutil
import tempfile
import logging
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
from Bio.PDB import PDBParser, PDBIO, Select
from Bio.PDB.Structure import Structure

from .core import FlexiblePose, RefinedComplex, FlexibleResidue
from ..docking.core import Pose, DockingResult
from ..docking.algorithms.monte_carlo_cpu import MonteCarloDocker
from ..docking.scoring.physics_based import PhysicsBasedScoring
from ..docking.scoring.vina_scoring import VinaScoring


class SoftRigidDocker:
    """Phase 1: Initial soft rigid docking with reduced van der Waals penalties"""

    def __init__(self, use_gpu: bool = False, gpu_id: int = 0, cpu_workers: Optional[int] = None,
                 max_memory_gb: float = 8.0):
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        self.cpu_workers = cpu_workers
        self.max_memory_gb = max_memory_gb
        self.soft_scale = 0.7
        self.logger = logging.getLogger("pandadock.flex.phase1")

        # Initialize underlying rigid docking engine
        self.rigid_docker = MonteCarloDocker()
        self.soft_scorer = SoftPotentialScoring()

    def configure_soft_potential(self, soft_scale: float) -> None:
        """Configure the soft potential scaling factor"""
        self.soft_scale = soft_scale
        self.soft_scorer.set_vdw_scaling(soft_scale)
        self.logger.info(f"Configured soft potential with scaling factor: {soft_scale}")

    def dock(self, receptor_file: str, ligand_mol: Chem.Mol, binding_site_center: np.ndarray,
             binding_site_radius: float, max_poses: int = 20, energy_cutoff: float = 50.0) -> DockingResult:
        """
        Perform initial soft rigid docking

        Args:
            receptor_file: Path to receptor PDB
            ligand_mol: RDKit ligand molecule
            binding_site_center: Center coordinates of binding site
            binding_site_radius: Radius for docking grid
            max_poses: Maximum poses to retain
            energy_cutoff: Energy cutoff for pose filtering

        Returns:
            DockingResult with soft-potential poses
        """
        self.logger.info(f"Starting soft rigid docking with {self.soft_scale:.2f} VDW scaling")

        # Set up grid dimensions from radius
        grid_dimensions = np.array([binding_site_radius * 2] * 3)

        # Configure the rigid docker with soft scoring
        self.rigid_docker._scoring_function = self.soft_scorer

        # Perform docking with enhanced sampling for soft potentials
        result = self.rigid_docker.dock(
            receptor_file=receptor_file,
            ligand_mol=ligand_mol,
            grid_center=binding_site_center,
            grid_dimensions=grid_dimensions,
            num_poses=max_poses * 2,  # Generate more poses for soft filtering
            fast=False  # Use full sampling for better coverage
        )

        # Filter poses by energy cutoff
        filtered_poses = [p for p in result.poses if p.energy <= energy_cutoff]
        result.poses = filtered_poses[:max_poses]

        self.logger.info(f"Generated {len(result.poses)} soft rigid poses")
        return result


class ReceptorRefiner:
    """Phase 2: Receptor conformational refinement around ligand poses"""

    def __init__(self, use_gpu: bool = False, gpu_id: int = 0, cpu_workers: Optional[int] = None,
                 max_memory_gb: float = 8.0):
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        self.cpu_workers = cpu_workers
        self.max_memory_gb = max_memory_gb
        self.logger = logging.getLogger("pandadock.flex.phase2")

        # Refined receptors are written to a private temporary directory rather
        # than the working directory. Writing them to the CWD left multi-megabyte
        # files behind whenever a run was interrupted, and made two concurrent
        # runs in one directory delete each other's files during cleanup.
        self._temp_dir: Optional[str] = None
        self._refined_counter = 0

        # Refinement configuration
        self.refinement_method = 'minimization'
        self.max_cycles = 3
        self.refine_loops = False
        self.refine_ligand = True

    def configure_refinement(self, method: str = 'minimization', max_cycles: int = 3,
                           refine_loops: bool = False, refine_ligand: bool = True) -> None:
        """Configure refinement parameters"""
        self.refinement_method = method
        self.max_cycles = max_cycles
        self.refine_loops = refine_loops
        self.refine_ligand = refine_ligand

        self.logger.info(f"Configured refinement: {method}, {max_cycles} cycles, "
                        f"loops={'Yes' if refine_loops else 'No'}")

    def auto_detect_flexible_residues(self, receptor_file: str, poses: List[Pose],
                                    distance_threshold: float = 6.0) -> List[str]:
        """
        Automatically detect residues that should be flexible based on ligand pose proximity

        Args:
            receptor_file: Path to receptor PDB
            poses: List of ligand poses
            distance_threshold: Distance cutoff for flexible residue detection

        Returns:
            List of residue identifiers (e.g., ['A:123', 'A:145'])
        """
        self.logger.info(f"Auto-detecting flexible residues within {distance_threshold:.1f} Å")

        # Load receptor structure
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('receptor', receptor_file)

        flexible_residues = set()

        for pose in poses:
            # Find residues within distance of any ligand atom
            for ligand_atom in pose.coordinates:
                for model in structure:
                    for chain in model:
                        for residue in chain:
                            # Skip water and heterogens initially
                            if residue.id[0] != ' ':
                                continue

                            # Check CA atom distance (representing residue position)
                            if 'CA' in residue:
                                ca_coord = residue['CA'].get_coord()
                                distance = np.linalg.norm(ligand_atom - ca_coord)

                                if distance <= distance_threshold:
                                    res_id = f"{chain.id}:{residue.id[1]}"
                                    flexible_residues.add(res_id)

        flexible_residues = sorted(list(flexible_residues))
        self.logger.info(f"Detected {len(flexible_residues)} flexible residues: {flexible_residues}")
        return flexible_residues

    def refine_receptor_conformations(self, receptor_file: str, initial_poses: List[Pose],
                                    flexible_residues: List[str],
                                    binding_site_center: np.ndarray) -> List[RefinedComplex]:
        """
        Refine receptor conformations for each promising ligand pose

        Args:
            receptor_file: Original receptor PDB
            initial_poses: Poses from soft rigid docking
            flexible_residues: List of residues to make flexible
            binding_site_center: Center of binding site

        Returns:
            List of refined receptor-ligand complexes
        """
        self.logger.info(f"Refining receptor for {len(initial_poses)} poses with "
                        f"{len(flexible_residues)} flexible residues")

        refined_complexes = []

        for i, pose in enumerate(initial_poses):
            self.logger.debug(f"Refining receptor for pose {i+1}/{len(initial_poses)}")

            try:
                # Perform side-chain refinement
                refined_complex = self._refine_single_complex(
                    receptor_file, pose, flexible_residues, binding_site_center
                )

                if refined_complex:
                    refined_complexes.append(refined_complex)

            except Exception as e:
                self.logger.warning(f"Failed to refine complex for pose {i+1}: {e}")
                continue

        self.logger.info(f"Successfully refined {len(refined_complexes)} receptor conformations")
        return refined_complexes

    def _refine_single_complex(self, receptor_file: str, pose: Pose,
                             flexible_residues: List[str],
                             binding_site_center: np.ndarray) -> Optional[RefinedComplex]:
        """Refine a single receptor-ligand complex"""

        # Load original receptor
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('receptor', receptor_file)

        # Prepare flexible residue objects
        flex_res_objects = []
        for res_id in flexible_residues:
            chain_id, res_num = res_id.split(':')
            res_num = int(res_num)

            # Find the residue in structure
            try:
                residue = structure[0][chain_id][(' ', res_num, ' ')]
                atoms = list(residue.get_atoms())
                original_coords = np.array([atom.get_coord() for atom in atoms], dtype=np.float64)

                flex_residue = FlexibleResidue(
                    chain_id=chain_id,
                    residue_number=res_num,
                    residue_name=residue.get_resname(),
                    original_conformation=original_coords,
                    refined_conformations=[],
                    energy_change=0.0
                )
                flex_residue.atom_names = [atom.get_name().strip().upper() for atom in atoms]
                flex_res_objects.append(flex_residue)

            except KeyError:
                self.logger.warning(f"Residue {res_id} not found in structure")
                continue

        # Perform refinement based on method
        if self.refinement_method == 'minimization':
            refined_structure, energy_cost = self._minimize_side_chains(
                structure, pose, flex_res_objects
            )
        elif self.refinement_method == 'monte_carlo':
            refined_structure, energy_cost = self._monte_carlo_side_chains(
                structure, pose, flex_res_objects
            )
        else:
            self.logger.error(f"Unknown refinement method: {self.refinement_method}")
            return None

        # Save refined receptor to a temporary file.
        #
        # The name is a per-instance counter, not id(pose). CPython reuses id()
        # values once an object is collected, so two poses could receive the same
        # filename and the second would overwrite the first -- meaning the final
        # redocking phase could dock into the wrong refined receptor.
        refined_pdb_path = os.path.join(
            self._ensure_temp_dir(), f"refined_receptor_{self._refined_counter:04d}.pdb"
        )
        self._refined_counter += 1
        io = PDBIO()
        io.set_structure(refined_structure)
        io.save(refined_pdb_path)

        # Calculate RMSD from original
        rmsd = self._calculate_receptor_rmsd(receptor_file, refined_pdb_path, flexible_residues)

        return RefinedComplex(
            receptor=refined_pdb_path,
            ligand_pose=pose,
            flexible_residues=flex_res_objects,
            energy_cost=energy_cost,
            rmsd_from_original=rmsd
        )

    def _minimize_side_chains(self, structure: Structure, pose: Pose,
                            flexible_residues: List[FlexibleResidue]) -> Tuple[Structure, float]:
        """
        Minimize side-chain conformations using simple energy minimization

        This is a simplified implementation. In production, you'd use:
        - OpenMM for molecular mechanics
        - RDKit's conformer optimization
        - Custom MM force fields
        """
        self.logger.debug("Performing side-chain minimization")

        # For now, use a simplified approach with rotamer sampling
        total_energy_cost = 0.0

        for flex_res in flexible_residues:
            # Sample rotamer conformations (simplified)
            rotamer_conformations = self._sample_rotamers(flex_res, pose)

            # Select best rotamer based on simple scoring
            best_rotamer, best_energy = self._select_best_rotamer(
                rotamer_conformations, pose, structure
            )

            if best_rotamer is not None:
                flex_res.refined_conformations = [best_rotamer]
                flex_res.energy_change = best_energy
                total_energy_cost += best_energy

                # Update structure with best rotamer (simplified)
                self._update_residue_conformation(structure, flex_res, best_rotamer)

        return structure, total_energy_cost

    # Backbone and anchor atom names that must NEVER move during chi1 rotation
    _BACKBONE_AND_CB = frozenset({'N', 'CA', 'C', 'O', 'OXT', 'CB', 'H', 'HA', 'HA2', 'HA3', 'HB', 'HB2', 'HB3'})
    # Atoms fixed during chi2 rotation (backbone + CB + CG/CG1)
    _FIXED_FOR_CHI2 = _BACKBONE_AND_CB | frozenset({'CG', 'CG1', 'HG', 'HG2', 'HG3'})

    def _sample_rotamers(self, flexible_residue: FlexibleResidue,
                        pose: Pose) -> List[np.ndarray]:
        """Sample side-chain rotamers by rotating around actual chi1/chi2 bonds.

        Uses canonical Dunbrack-like staggered torsion increments (0, +/-60,
        +/-120, 180 deg) around the CA->CB bond (chi1) and, for longer side
        chains, CB->CG/CG1 (chi2). Backbone atoms and CA/CB are kept fixed so
        peptide geometry is strictly preserved.
        """
        original = np.asarray(flexible_residue.original_conformation, dtype=np.float64)
        resname = flexible_residue.residue_name.strip().upper()
        atom_names = getattr(flexible_residue, 'atom_names', None)

        # GLY, ALA, and PRO have no free chi1 side-chain torsion
        if resname in ('GLY', 'ALA', 'PRO') or not atom_names or len(atom_names) != len(original):
            return [original.copy()]

        name_to_idx = {name: i for i, name in enumerate(atom_names)}
        if 'CA' not in name_to_idx or 'CB' not in name_to_idx:
            return [original.copy()]

        ca_idx = name_to_idx['CA']
        cb_idx = name_to_idx['CB']
        chi1_moving = np.array(
            [name not in self._BACKBONE_AND_CB for name in atom_names], dtype=bool
        )
        if not np.any(chi1_moving):
            return [original.copy()]

        chi1_angles = [0.0, 60.0, -60.0, 120.0, -120.0, 180.0]
        cg_name = 'CG' if 'CG' in name_to_idx else ('CG1' if 'CG1' in name_to_idx else None)
        chi2_moving = None
        if cg_name is not None:
            mask2 = np.array(
                [name not in self._FIXED_FOR_CHI2 for name in atom_names], dtype=bool
            )
            if np.any(mask2):
                chi2_moving = mask2

        rotamers: List[np.ndarray] = []
        for a1 in chi1_angles:
            c1 = self._rotate_side_chain(
                original, a1,
                axis_origin=original[ca_idx],
                axis_end=original[cb_idx],
                moving_mask=chi1_moving,
            )
            rotamers.append(c1)
            if chi2_moving is not None and cg_name is not None:
                cg_idx = name_to_idx[cg_name]
                for a2 in (-60.0, 60.0, 180.0):
                    c2 = self._rotate_side_chain(
                        c1, a2,
                        axis_origin=c1[cb_idx],
                        axis_end=c1[cg_idx],
                        moving_mask=chi2_moving,
                    )
                    rotamers.append(c2)

        return rotamers

    def _rotate_side_chain(
        self,
        coords: np.ndarray,
        angle_deg: float,
        axis_origin: Optional[np.ndarray] = None,
        axis_end: Optional[np.ndarray] = None,
        moving_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Rotate distal side-chain atoms around a covalent bond axis (Rodrigues formula)."""
        out = np.asarray(coords, dtype=np.float64).copy()
        if abs(angle_deg) < 1e-12:
            return out

        angle_rad = np.deg2rad(angle_deg)
        if axis_origin is None or axis_end is None:
            if len(out) < 2:
                return out
            axis_origin = out[0]
            axis_end = out[1]

        axis = np.asarray(axis_end, dtype=np.float64) - np.asarray(axis_origin, dtype=np.float64)
        norm = float(np.linalg.norm(axis))
        if norm < 1e-8:
            return out
        k = axis / norm

        if moving_mask is None:
            moving_mask = np.ones(len(out), dtype=bool)

        pts = out[moving_mask] - axis_end
        cos_a = float(np.cos(angle_rad))
        sin_a = float(np.sin(angle_rad))
        k_dot_v = pts @ k
        k_cross_v = np.cross(k, pts)
        rotated = pts * cos_a + k_cross_v * sin_a + np.outer(k_dot_v, k) * (1.0 - cos_a)
        out[moving_mask] = rotated + axis_end
        return out

    def _select_best_rotamer(self, rotamers: List[np.ndarray], pose: Pose,
                           structure: Structure) -> Tuple[Optional[np.ndarray], float]:
        """
        Choose the best rotamer, and report the strain it leaves behind.

        Selection uses the full energy (clash penalty plus contact reward), but
        the value returned as the refinement cost is the residual clash strain
        alone, which is non-negative.

        Returning the full energy conflated the two. With no clashes it is purely
        a negative contact reward, and the IFD score adds
        `refinement_cost * refinement_penalty_weight` -- so a negative cost
        improved the score. A pose demanding more side-chain rearrangement scored
        better than one needing none, inverting the purpose of an induced-fit
        penalty.
        """
        best_rotamer = None
        best_energy = float('inf')
        best_strain = 0.0

        for rotamer in rotamers:
            energy, strain = self._calculate_rotamer_energy(
                rotamer, pose.coordinates, return_strain=True
            )

            if energy < best_energy:
                best_energy = energy
                best_strain = strain
                best_rotamer = rotamer

        return best_rotamer, best_strain

    def _calculate_rotamer_energy(self, rotamer_coords: np.ndarray,
                                ligand_coords: np.ndarray,
                                return_strain: bool = False):
        """
        Simplified energy for a rotamer conformation.

        Returns the combined energy used to rank rotamers. With
        `return_strain=True` it also returns the clash component on its own,
        which is the non-negative strain the receptor carries in that
        conformation and the quantity an induced-fit penalty should use.
        """

        # Simple distance-based scoring
        min_distances = []
        for rot_atom in rotamer_coords:
            distances = np.linalg.norm(ligand_coords - rot_atom, axis=1)
            min_distances.append(np.min(distances))

        # Penalty for clashes (< 2.5 Å)
        clash_penalty = sum(max(0, 2.5 - d) * 10 for d in min_distances)

        # Reward for favorable contacts (2.5-4.0 Å)
        favorable_contacts = sum(1 for d in min_distances if 2.5 <= d <= 4.0)
        contact_reward = favorable_contacts * -0.5

        if return_strain:
            return clash_penalty + contact_reward, clash_penalty
        return clash_penalty + contact_reward

    def _update_residue_conformation(self, structure: Structure,
                                   flexible_residue: FlexibleResidue,
                                   new_coords: np.ndarray) -> None:
        """Update residue conformation in structure (simplified)"""

        # Find and update the residue atoms
        try:
            chain = structure[0][flexible_residue.chain_id]
            residue = chain[(' ', flexible_residue.residue_number, ' ')]

            # Update coordinates (simplified - should match atom ordering)
            atoms = list(residue.get_atoms())
            for i, atom in enumerate(atoms):
                if i < len(new_coords):
                    atom.set_coord(new_coords[i])
        except (KeyError, IndexError) as e:
            self.logger.warning(f"Failed to update residue conformation: {e}")

    def _monte_carlo_side_chains(self, structure: Structure, pose: Pose,
                               flexible_residues: List[FlexibleResidue]) -> Tuple[Structure, float]:
        """Metropolis Monte Carlo sampling over side-chain rotamer states."""
        self.logger.debug("Performing Monte Carlo side-chain sampling")

        if not flexible_residues:
            return structure, 0.0

        rng = np.random.default_rng(42)
        rotamer_pools = [self._sample_rotamers(fr, pose) for fr in flexible_residues]
        current_idx = [0 for _ in flexible_residues]
        current_energies = [
            self._calculate_rotamer_energy(pool[0], pose.coordinates, return_strain=True)
            for pool in rotamer_pools
        ]

        best_idx = list(current_idx)
        best_total_e = sum(e for e, _ in current_energies)
        best_strains = [s for _, s in current_energies]

        n_steps = max(10, int(self.refinement_cycles) * max(1, len(flexible_residues)) * 5)
        temperature = 1.2

        for _ in range(n_steps):
            r_i = int(rng.integers(0, len(flexible_residues)))
            pool = rotamer_pools[r_i]
            if len(pool) <= 1:
                continue
            cand = int(rng.integers(0, len(pool)))
            cand_e, cand_s = self._calculate_rotamer_energy(
                pool[cand], pose.coordinates, return_strain=True
            )
            cur_e, _ = current_energies[r_i]
            delta = cand_e - cur_e
            if delta <= 0.0 or rng.random() < np.exp(-delta / temperature):
                current_idx[r_i] = cand
                current_energies[r_i] = (cand_e, cand_s)
                total_e = sum(e for e, _ in current_energies)
                if total_e < best_total_e:
                    best_total_e = total_e
                    best_idx = list(current_idx)
                    best_strains = [s for _, s in current_energies]

        total_strain = 0.0
        for fr, pool, idx, strain in zip(flexible_residues, rotamer_pools, best_idx, best_strains):
            chosen = pool[idx]
            fr.refined_conformations = [chosen]
            fr.energy_change = strain
            total_strain += strain
            self._update_residue_conformation(structure, fr, chosen)

        return structure, float(total_strain)

    def _calculate_receptor_rmsd(self, original_pdb: str, refined_pdb: str,
                               flexible_residues: List[str]) -> float:
        """Calculate heavy-atom RMSD of flexible residues between original and refined PDBs."""
        parser = PDBParser(QUIET=True)
        orig_struct = parser.get_structure('orig', original_pdb)
        ref_struct = parser.get_structure('ref', refined_pdb)

        orig_coords = []
        ref_coords = []
        for res_id in flexible_residues:
            try:
                chain_id, res_num_str = res_id.split(':')
                key = (' ', int(res_num_str), ' ')
                r_orig = orig_struct[0][chain_id][key]
                r_ref = ref_struct[0][chain_id][key]
            except (ValueError, KeyError):
                continue

            for atom in r_orig.get_atoms():
                if atom.element.strip().upper() == 'H':
                    continue
                name = atom.get_name()
                if name in r_ref:
                    orig_coords.append(atom.get_coord())
                    ref_coords.append(r_ref[name].get_coord())

        if not orig_coords:
            return 0.0

        diff = np.asarray(ref_coords, dtype=np.float64) - np.asarray(orig_coords, dtype=np.float64)
        return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


    def _ensure_temp_dir(self) -> str:
        """Create this refiner's temporary directory on first use."""
        if self._temp_dir is None:
            self._temp_dir = tempfile.mkdtemp(prefix="pandadock_flex_")
            self.logger.debug("Refined receptors will be written to %s", self._temp_dir)
        return self._temp_dir

    def cleanup(self) -> None:
        """
        Remove the temporary refined receptors.

        Safe to call more than once, and safe to call after a failure. Cleanup
        used to run only on the success path of the CLI, so an interrupted or
        failed run left every refined receptor behind -- roughly 375 KB each.
        """
        if self._temp_dir and os.path.isdir(self._temp_dir):
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self.logger.debug("Removed temporary directory %s", self._temp_dir)
        self._temp_dir = None

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass


class FinalRedocker:
    """Phase 3: Final redocking into refined receptor conformations"""

    def __init__(self, use_gpu: bool = False, gpu_id: int = 0, cpu_workers: Optional[int] = None,
                 max_memory_gb: float = 8.0):
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        self.cpu_workers = cpu_workers
        self.max_memory_gb = max_memory_gb
        self.final_scale = 1.0
        self.logger = logging.getLogger("pandadock.flex.phase3")

        # Initialize precise docking engine on the same Vina scale as `pandadock dock`
        self.precise_docker = MonteCarloDocker()
        self.precise_scorer = VinaScoring()

    def configure_final_potential(self, final_scale: float = 1.0) -> None:
        """Configure final docking potential (typically 1.0 for full precision)"""
        self.final_scale = final_scale
        self.logger.info(f"Configured final potential with scaling factor: {final_scale}")

    def redock(self, refined_receptor: str, ligand_mol: Chem.Mol,
               binding_site_center: np.ndarray, binding_site_radius: float,
               max_poses: int = 10) -> DockingResult:
        """
        Redock ligand into refined receptor conformation

        Args:
            refined_receptor: Path to refined receptor PDB
            ligand_mol: RDKit ligand molecule
            binding_site_center: Center of binding site
            binding_site_radius: Radius for docking
            max_poses: Maximum poses to retain

        Returns:
            DockingResult with precisely docked poses
        """
        self.logger.debug(f"Redocking into refined receptor: {refined_receptor}")

        # Set up precise scoring
        self.precise_docker._scoring_function = self.precise_scorer

        # Perform high-precision docking
        grid_dimensions = np.array([binding_site_radius * 1.5] * 3)  # Smaller grid for precision

        result = self.precise_docker.dock(
            receptor_file=refined_receptor,
            ligand_mol=ligand_mol,
            grid_center=binding_site_center,
            grid_dimensions=grid_dimensions,
            num_poses=max_poses * 2,  # Generate extra for selection
            fast=False  # Use full sampling
        )

        # Convert to FlexiblePose objects
        flex_poses = []
        for pose in result.poses[:max_poses]:
            flex_pose = FlexiblePose(
                coordinates=pose.coordinates,
                center=pose.center,
                rotation=pose.rotation,
                conformer_id=pose.conformer_id,
                energy=pose.energy,
                confidence=pose.confidence
            )
            flex_pose.binding_energy = pose.energy
            flex_poses.append(flex_pose)

        result.poses = flex_poses
        return result


class IFDScorer:
    """Phase 4: Induced-Fit Docking scoring and ranking"""

    def __init__(self, scoring_function: str = 'ifd_composite'):
        self.scoring_function = scoring_function
        self.logger = logging.getLogger("pandadock.flex.phase4")

        # Scoring weights aligned with Schrödinger IFDScore convention:
        # IFDScore = Binding_Energy + 0.05 * Receptor_Strain + 0.05 * Rel_Ligand_Strain + Interaction_Bonus
        self.binding_weight = 1.0
        self.refinement_penalty_weight = 0.05
        self.ligand_strain_weight = 0.05

    def score_ifd_poses(self, poses: List[FlexiblePose], receptor_file: str,
                       ligand_mol: Chem.Mol) -> List[FlexiblePose]:
        """
        Calculate comprehensive IFD scores on the same kcal/mol scale as `pandadock dock`.

        IFD Score = Binding Energy
                    + 0.05 * Receptor_Refinement_Strain (capped at 2.0 kcal/mol)
                    + 0.05 * Relative_Ligand_Strain (capped at 1.5 kcal/mol)
                    + Interaction_Bonus (actual H-bonds & pi-stacking)
        """
        self.logger.info(f"Calculating IFD scores for {len(poses)} poses")

        for pose in poses:
            rec_penalty = min(max(0.0, float(pose.refinement_cost)) * self.refinement_penalty_weight, 2.0)

            pose.ligand_strain_energy = self._calculate_ligand_strain(pose, ligand_mol)
            pose.interaction_fingerprint = self._calculate_interactions(
                pose, receptor_file, ligand_mol
            )

            strain_penalty = min(max(0.0, pose.ligand_strain_energy) * self.ligand_strain_weight, 1.5)
            hb = float(pose.interaction_fingerprint.get('hydrogen_bonds', 0.0))
            pi = float(pose.interaction_fingerprint.get('aromatic_stacking', 0.0))
            interaction_bonus = -0.15 * min(hb, 4.0) - 0.10 * min(pi, 3.0)

            pose.ifd_score = (
                self.binding_weight * float(pose.binding_energy)
                + rec_penalty
                + strain_penalty
                + interaction_bonus
            )

        return poses

    def _calculate_ligand_strain(self, pose: FlexiblePose, ligand_mol: Chem.Mol) -> float:
        """Calculate relative ligand conformational strain (kcal/mol) via RDKit MMFF94/UFF.

        Computes `max(0.0, E_FF(pose) - E_FF(reference_conformer))` so differences
        between external SDF bond lengths and MMFF94 equilibrium values cancel out,
        leaving genuine torsional/conformational strain introduced during docking.
        """
        from rdkit.Chem import AllChem

        def _ff_energy(mol_obj: Chem.Mol) -> Optional[float]:
            props = AllChem.MMFFGetMoleculeProperties(mol_obj)
            ff = (
                AllChem.MMFFGetMoleculeForceField(mol_obj, props)
                if props is not None
                else AllChem.UFFGetMoleculeForceField(mol_obj)
            )
            return float(ff.CalcEnergy()) if ff is not None else None

        try:
            coords = np.asarray(pose.coordinates, dtype=np.float64)
            if ligand_mol.GetNumAtoms() != len(coords):
                return 0.0

            pose_mol = Chem.Mol(ligand_mol)
            pose_mol.RemoveAllConformers()
            conf = Chem.Conformer(pose_mol.GetNumAtoms())
            for i in range(pose_mol.GetNumAtoms()):
                conf.SetAtomPosition(i, coords[i].tolist())
            pose_mol.AddConformer(conf, assignId=True)

            e_pose = _ff_energy(pose_mol)
            if e_pose is None:
                return 0.0

            if ligand_mol.GetNumConformers() > 0:
                e_ref = _ff_energy(Chem.Mol(ligand_mol))
                if e_ref is not None:
                    return float(max(0.0, e_pose - e_ref))

            # Fallback if input had no conformer: minimize pose locally
            props = AllChem.MMFFGetMoleculeProperties(pose_mol)
            ff = (
                AllChem.MMFFGetMoleculeForceField(pose_mol, props)
                if props is not None
                else AllChem.UFFGetMoleculeForceField(pose_mol)
            )
            if ff is None:
                return 0.0
            ff.Minimize(maxIts=200)
            return float(max(0.0, e_pose - float(ff.CalcEnergy())))
        except Exception as exc:
            self.logger.debug("Ligand strain calculation fell back to 0.0: %s", exc)
            return 0.0

    def _calculate_interactions(self, pose: FlexiblePose, receptor_file: str,
                              ligand_mol: Chem.Mol) -> Dict[str, float]:
        """Calculate real protein-ligand interaction counts from structure geometry."""
        try:
            from ..docking.analysis.interaction_analysis import InteractionAnalyzer

            parser = PDBParser(QUIET=True)
            receptor = parser.get_structure('receptor', receptor_file)
            analyzer = InteractionAnalyzer()
            details = analyzer.analyze_pose_interactions(pose.coordinates, receptor, ligand_mol)
            counts = details.get('interaction_types', {})
            return {
                'hydrogen_bonds': float(counts.get('hydrogen_bonds', 0)),
                'hydrophobic_contacts': float(counts.get('hydrophobic', 0)),
                'electrostatic': float(counts.get('electrostatic', 0)),
                'aromatic_stacking': float(counts.get('pi_stacking', 0)),
            }
        except Exception as exc:
            self.logger.debug("IFD interaction analysis failed: %s", exc)
            return {
                'hydrogen_bonds': 0.0,
                'hydrophobic_contacts': 0.0,
                'electrostatic': 0.0,
                'aromatic_stacking': 0.0,
            }

    def cluster_poses(self, poses: List[FlexiblePose],
                     rmsd_threshold: float = 2.0) -> List[FlexiblePose]:
        """Cluster poses by RMSD and select representatives"""
        self.logger.info(f"Clustering {len(poses)} poses with RMSD threshold {rmsd_threshold:.1f} Å")

        if not poses:
            return poses

        # Sort by IFD score first
        sorted_poses = sorted(poses, key=lambda p: p.ifd_score)

        # Simple clustering - select diverse poses
        clustered_poses = [sorted_poses[0]]  # Always keep best

        for pose in sorted_poses[1:]:
            is_diverse = True
            for clustered_pose in clustered_poses:
                rmsd = self._calculate_pose_rmsd(pose, clustered_pose)
                if rmsd < rmsd_threshold:
                    is_diverse = False
                    break

            if is_diverse:
                clustered_poses.append(pose)

        self.logger.info(f"Selected {len(clustered_poses)} diverse poses after clustering")
        return clustered_poses

    def _calculate_pose_rmsd(self, pose1: FlexiblePose, pose2: FlexiblePose) -> float:
        """Calculate RMSD between two poses"""
        if len(pose1.coordinates) != len(pose2.coordinates):
            return float('inf')

        diff = pose1.coordinates - pose2.coordinates
        return np.sqrt(np.mean(np.sum(diff**2, axis=1)))


class SoftPotentialScoring:
    """Soft potential scoring for Phase 1 docking"""

    def __init__(self):
        self.base_scorer = PhysicsBasedScoring()
        self.vdw_scaling = 0.7

    def set_vdw_scaling(self, scaling: float) -> None:
        """Set van der Waals scaling factor"""
        self.vdw_scaling = scaling

    def calculate_binding_energy(self, ligand_coords: np.ndarray,
                                receptor_structure, ligand_mol: Chem.Mol = None) -> float:
        """Calculate binding energy with soft VDW potentials"""

        # Get base energy
        base_energy = self.base_scorer.calculate_binding_energy(
            ligand_coords, receptor_structure, ligand_mol
        )

        # Apply soft scaling to reduce VDW penalties
        # This is simplified - in production you'd modify the VDW terms specifically
        if base_energy > 0:
            # Reduce positive (unfavorable) energies more aggressively
            soft_energy = base_energy * self.vdw_scaling
        else:
            # Keep favorable energies mostly unchanged
            soft_energy = base_energy * (0.8 + 0.2 * self.vdw_scaling)

        return soft_energy