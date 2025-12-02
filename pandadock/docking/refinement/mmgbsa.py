"""
MM-GBSA Rescoring and Refinement

Implements molecular mechanics with generalized Born surface area (MM-GBSA) rescoring:
1. Molecular mechanics energy (bonded + non-bonded)
2. Generalized Born solvation energy
3. Surface area solvation correction
4. Entropy estimation (optional)
5. Binding free energy decomposition

This provides more accurate binding free energies than docking scores alone
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import logging
from scipy.optimize import minimize
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from Bio.PDB import Structure
import math

from ..core import Pose


class MMGBSARescoring:
    """MM-GBSA rescoring for improved binding affinity prediction"""

    def __init__(self, **params):
        self.logger = logging.getLogger("pandadock.refinement.mmgbsa")

        # MM-GBSA parameters
        self.gb_model = params.get('gb_model', 'GB-OBC2')  # GB model variant
        self.surface_tension = params.get('surface_tension', 0.0072)  # kcal/(mol·Å²)
        self.dielectric_interior = params.get('dielectric_interior', 1.0)
        self.dielectric_exterior = params.get('dielectric_exterior', 80.0)
        self.salt_concentration = params.get('salt_concentration', 0.15)  # M

        # Force field parameters (simplified AMBER/CHARMM-like)
        self.bond_force_constants = {'CC': 317, 'CN': 337, 'CO': 320, 'NN': 410, 'NO': 410}
        self.angle_force_constants = {'CCC': 63, 'CCN': 70, 'CNN': 70, 'COO': 80}
        self.torsion_barriers = {'CCCC': 1.4, 'NCCN': 2.0, 'OCCN': 1.8}

        # Atomic radii for GB calculations (Å)
        self.gb_radii = {
            'C': 1.70, 'N': 1.55, 'O': 1.50, 'S': 1.80, 'P': 1.85,
            'H': 1.20, 'F': 1.47, 'Cl': 1.77, 'Br': 1.90, 'I': 2.09
        }

        # van der Waals parameters
        self.vdw_radii = {
            'C': 1.70, 'N': 1.55, 'O': 1.52, 'S': 1.80, 'P': 1.80,
            'H': 1.20, 'F': 1.47, 'Cl': 1.75, 'Br': 1.85, 'I': 1.98
        }

        self.logger.info("MM-GBSA rescoring initialized with GB model: " + self.gb_model)

    def rescore_poses(self, poses: List[Pose], ligand_mol: Chem.Mol,
                     receptor_structure: Structure) -> List[Pose]:
        """Rescore poses using MM-GBSA"""
        rescored_poses = []

        for i, pose in enumerate(poses):
            try:
                self.logger.debug(f"MM-GBSA rescoring pose {i+1}/{len(poses)}")

                # Calculate MM-GBSA binding free energy
                binding_free_energy = self.calculate_binding_free_energy(
                    pose, ligand_mol, receptor_structure
                )

                # Create new pose with MM-GBSA energy
                rescored_pose = Pose(
                    coordinates=pose.coordinates,
                    center=pose.center,
                    rotation=pose.rotation,
                    energy=binding_free_energy,
                    conformer_id=pose.conformer_id
                )

                # Store energy decomposition
                if hasattr(self, '_last_energy_components'):
                    rescored_pose.mmgbsa_components = self._last_energy_components

                rescored_poses.append(rescored_pose)

            except Exception as e:
                self.logger.warning(f"MM-GBSA rescoring failed for pose {i+1}: {e}")
                rescored_poses.append(pose)  # Keep original if rescoring fails

        return rescored_poses

    def calculate_binding_free_energy(self, pose: Pose, ligand_mol: Chem.Mol,
                                    receptor_structure: Structure) -> float:
        """
        Calculate MM-GBSA binding free energy

        ΔG_bind = ΔE_MM + ΔG_solv - TΔS
        where:
        ΔE_MM = molecular mechanics energy difference
        ΔG_solv = solvation free energy difference (GB + SA)
        TΔS = entropy contribution
        """
        # Get system components
        ligand_coords = pose.coordinates
        receptor_atoms = list(receptor_structure.get_atoms())
        receptor_coords = np.array([atom.get_coord() for atom in receptor_atoms])

        # Calculate energy components
        energy_components = {}

        # 1. Molecular mechanics energy
        mm_energy = self._calculate_mm_energy(
            ligand_coords, receptor_coords, ligand_mol, receptor_atoms
        )
        energy_components['MM_energy'] = mm_energy

        # 2. Generalized Born solvation energy
        gb_energy = self._calculate_gb_solvation_energy(
            ligand_coords, receptor_coords, ligand_mol, receptor_atoms
        )
        energy_components['GB_solvation'] = gb_energy

        # 3. Surface area solvation correction
        sa_energy = self._calculate_surface_area_energy(
            ligand_coords, receptor_coords, ligand_mol, receptor_atoms
        )
        energy_components['SA_solvation'] = sa_energy

        # 4. Entropy contribution (simplified)
        entropy_energy = self._calculate_entropy_contribution(ligand_mol)
        energy_components['entropy'] = entropy_energy

        # Total binding free energy
        total_energy = mm_energy + gb_energy + sa_energy + entropy_energy

        # Store components for analysis
        self._last_energy_components = energy_components

        self.logger.debug(f"MM-GBSA components: {energy_components}")
        self.logger.debug(f"Total MM-GBSA energy: {total_energy:.3f} kcal/mol")

        return total_energy

    def _calculate_mm_energy(self, ligand_coords: np.ndarray, receptor_coords: np.ndarray,
                           ligand_mol: Chem.Mol, receptor_atoms: List) -> float:
        """Calculate molecular mechanics energy"""
        mm_energy = 0.0

        # Bond energy (ligand internal)
        mm_energy += self._calculate_bond_energy(ligand_mol)

        # Angle energy (ligand internal)
        mm_energy += self._calculate_angle_energy(ligand_mol)

        # Torsion energy (ligand internal)
        mm_energy += self._calculate_torsion_energy(ligand_mol)

        # Non-bonded interactions (van der Waals + electrostatic)
        mm_energy += self._calculate_nonbonded_energy(
            ligand_coords, receptor_coords, ligand_mol, receptor_atoms
        )

        return mm_energy

    def _calculate_bond_energy(self, ligand_mol: Chem.Mol) -> float:
        """Calculate bond stretching energy"""
        bond_energy = 0.0

        for bond in ligand_mol.GetBonds():
            atom1 = bond.GetBeginAtom()
            atom2 = bond.GetEndAtom()

            # Get bond type
            bond_type = f"{atom1.GetSymbol()}{atom2.GetSymbol()}"
            if bond_type not in self.bond_force_constants:
                bond_type = bond_type[::-1]  # Try reverse

            if bond_type in self.bond_force_constants:
                # Get ideal bond length (simplified)
                ideal_length = self._get_ideal_bond_length(atom1.GetSymbol(), atom2.GetSymbol())

                # Current bond length would need 3D coordinates
                # For now, assume bonds are at ideal length
                current_length = ideal_length

                # Harmonic potential: E = k(r - r0)²
                force_constant = self.bond_force_constants[bond_type]
                bond_energy += force_constant * (current_length - ideal_length) ** 2

        return bond_energy

    def _calculate_angle_energy(self, ligand_mol: Chem.Mol) -> float:
        """Calculate angle bending energy"""
        angle_energy = 0.0

        # Find all angles in molecule
        for atom in ligand_mol.GetAtoms():
            neighbors = [neighbor.GetSymbol() for neighbor in atom.GetNeighbors()]
            if len(neighbors) >= 2:
                # For each pair of neighbors, form an angle
                for i in range(len(neighbors)):
                    for j in range(i+1, len(neighbors)):
                        angle_type = f"{neighbors[i]}{atom.GetSymbol()}{neighbors[j]}"

                        if angle_type in self.angle_force_constants:
                            # Assume angles are at ideal values for simplicity
                            ideal_angle = 109.5  # degrees (tetrahedral)
                            current_angle = ideal_angle

                            force_constant = self.angle_force_constants[angle_type]
                            angle_deviation = math.radians(current_angle - ideal_angle)
                            angle_energy += force_constant * angle_deviation ** 2

        return angle_energy

    def _calculate_torsion_energy(self, ligand_mol: Chem.Mol) -> float:
        """Calculate torsional energy"""
        torsion_energy = 0.0

        # Find rotatable bonds and calculate torsions
        for bond in ligand_mol.GetBonds():
            if not bond.IsInRing():
                atom1 = bond.GetBeginAtom()
                atom2 = bond.GetEndAtom()

                # Get atoms for torsion angle
                neighbors1 = [n for n in atom1.GetNeighbors() if n.GetIdx() != atom2.GetIdx()]
                neighbors2 = [n for n in atom2.GetNeighbors() if n.GetIdx() != atom1.GetIdx()]

                if neighbors1 and neighbors2:
                    # Form torsion type
                    torsion_type = f"{neighbors1[0].GetSymbol()}{atom1.GetSymbol()}{atom2.GetSymbol()}{neighbors2[0].GetSymbol()}"

                    if torsion_type in self.torsion_barriers:
                        # Assume torsion is at minimum for simplicity
                        barrier = self.torsion_barriers[torsion_type]
                        torsion_angle = 0.0  # radians from minimum

                        # Cosine potential: E = V(1 + cos(nφ - γ))
                        torsion_energy += barrier * (1 + math.cos(3 * torsion_angle))

        return torsion_energy

    def _calculate_nonbonded_energy(self, ligand_coords: np.ndarray, receptor_coords: np.ndarray,
                                  ligand_mol: Chem.Mol, receptor_atoms: List) -> float:
        """Calculate non-bonded (vdW + electrostatic) energy"""
        nonbonded_energy = 0.0

        # Get partial charges
        try:
            AllChem.ComputeGasteigerCharges(ligand_mol)
            ligand_charges = [float(atom.GetProp('_GasteigerCharge')) for atom in ligand_mol.GetAtoms()]
        except:
            ligand_charges = [0.0] * ligand_mol.GetNumAtoms()

        receptor_charges = self._get_receptor_charges(receptor_atoms)

        # Calculate pairwise interactions
        for i, lig_coord in enumerate(ligand_coords):
            lig_element = ligand_mol.GetAtomWithIdx(int(i)).GetSymbol()
            lig_charge = ligand_charges[i]

            for j, rec_coord in enumerate(receptor_coords):
                rec_element = receptor_atoms[j].element
                rec_charge = receptor_charges[j]

                dist = np.linalg.norm(lig_coord - rec_coord)

                if dist < 12.0:  # Cutoff for non-bonded interactions
                    # van der Waals energy (Lennard-Jones)
                    vdw_energy = self._calculate_vdw_pair_energy(
                        lig_element, rec_element, dist
                    )

                    # Electrostatic energy
                    if dist > 1.0:  # Avoid singularities
                        coulomb_constant = 332.0637  # kcal·Å/(mol·e²)
                        dielectric = self.dielectric_interior
                        electrostatic_energy = (coulomb_constant * lig_charge * rec_charge) / (dielectric * dist)
                    else:
                        electrostatic_energy = 0.0

                    nonbonded_energy += vdw_energy + electrostatic_energy

        return nonbonded_energy

    def _calculate_vdw_pair_energy(self, element1: str, element2: str, distance: float) -> float:
        """Calculate van der Waals energy between two atoms"""
        # Get van der Waals parameters
        radius1 = self.vdw_radii.get(element1, 1.7)
        radius2 = self.vdw_radii.get(element2, 1.7)

        sigma = radius1 + radius2
        epsilon = 0.1  # Simplified well depth

        if distance > sigma * 0.8:  # Avoid numerical issues
            # Lennard-Jones potential
            r6 = (sigma / distance) ** 6
            r12 = r6 ** 2
            vdw_energy = 4 * epsilon * (r12 - r6)
        else:
            vdw_energy = 1000.0  # Large penalty for severe clashes

        return vdw_energy

    def _calculate_gb_solvation_energy(self, ligand_coords: np.ndarray, receptor_coords: np.ndarray,
                                     ligand_mol: Chem.Mol, receptor_atoms: List) -> float:
        """Calculate Generalized Born solvation energy"""
        # Get partial charges
        try:
            AllChem.ComputeGasteigerCharges(ligand_mol)
            ligand_charges = [float(atom.GetProp('_GasteigerCharge')) for atom in ligand_mol.GetAtoms()]
        except:
            ligand_charges = [0.0] * ligand_mol.GetNumAtoms()

        receptor_charges = self._get_receptor_charges(receptor_atoms)

        # Combine all atoms
        all_coords = np.vstack([ligand_coords, receptor_coords])
        all_charges = ligand_charges + receptor_charges
        all_elements = [ligand_mol.GetAtomWithIdx(int(i)).GetSymbol() for i in range(len(ligand_coords))] + \
                      [atom.element for atom in receptor_atoms]

        # Calculate effective Born radii
        born_radii = self._calculate_born_radii(all_coords, all_elements)

        # Calculate GB energy
        gb_energy = 0.0
        coulomb_constant = 332.0637  # kcal·Å/(mol·e²)

        # Self energy
        for i, (charge, born_radius) in enumerate(zip(all_charges, born_radii)):
            gb_energy += -coulomb_constant * charge**2 * (1/self.dielectric_interior - 1/self.dielectric_exterior) / (2 * born_radius)

        # Pairwise interactions
        for i in range(len(all_coords)):
            for j in range(i+1, len(all_coords)):
                qi, qj = all_charges[i], all_charges[j]
                ri, rj = born_radii[i], born_radii[j]

                dist = np.linalg.norm(all_coords[i] - all_coords[j])
                if dist > 0:
                    # Still-Tempczyk formula
                    fgb = math.sqrt(dist**2 + ri * rj * math.exp(-dist**2 / (4 * ri * rj)))
                    gb_energy += -coulomb_constant * qi * qj * (1/self.dielectric_interior - 1/self.dielectric_exterior) / fgb

        return gb_energy

    def _calculate_born_radii(self, coords: np.ndarray, elements: List[str]) -> List[float]:
        """Calculate effective Born radii using Onufriev-Bashford-Case model"""
        born_radii = []

        for i, element in enumerate(elements):
            # Get intrinsic radius
            intrinsic_radius = self.gb_radii.get(element, 1.7)

            # Calculate volume integral (simplified)
            integral = 0.0
            coord_i = coords[i]

            for j, coord_j in enumerate(coords):
                if i != j:
                    dist = np.linalg.norm(coord_i - coord_j)
                    radius_j = self.gb_radii.get(elements[j], 1.7)

                    if dist < intrinsic_radius + radius_j:
                        # Simplified integral calculation
                        integral += radius_j**3 / max(dist, 0.1)**3

            # Effective Born radius (OBC2 model)
            psi = integral * intrinsic_radius / 16.0
            born_radius = 1.0 / (1.0/intrinsic_radius - math.tanh(0.8*psi + 2.9*psi**2 + 1.9*psi**3)/intrinsic_radius)

            born_radii.append(max(born_radius, intrinsic_radius * 0.5))

        return born_radii

    def _calculate_surface_area_energy(self, ligand_coords: np.ndarray, receptor_coords: np.ndarray,
                                     ligand_mol: Chem.Mol, receptor_atoms: List) -> float:
        """Calculate surface area solvation energy"""
        # Calculate solvent accessible surface area (SASA)
        all_coords = np.vstack([ligand_coords, receptor_coords])
        all_elements = [ligand_mol.GetAtomWithIdx(int(i)).GetSymbol() for i in range(len(ligand_coords))] + \
                      [atom.element for atom in receptor_atoms]

        # Simplified SASA calculation
        total_sasa = 0.0
        probe_radius = 1.4  # Water probe radius in Å

        for i, element in enumerate(all_elements):
            atom_radius = self.vdw_radii.get(element, 1.7)
            effective_radius = atom_radius + probe_radius

            # Count neighboring atoms within interaction distance
            neighbors = 0
            coord_i = all_coords[i]

            for j, coord_j in enumerate(all_coords):
                if i != j:
                    dist = np.linalg.norm(coord_i - coord_j)
                    neighbor_radius = self.vdw_radii.get(all_elements[j], 1.7)

                    if dist < effective_radius + neighbor_radius + probe_radius:
                        neighbors += 1

            # Approximate SASA based on neighbor count
            max_surface_area = 4 * math.pi * effective_radius**2
            burial_factor = min(neighbors / 12.0, 1.0)  # Normalized by coordination number
            sasa = max_surface_area * (1.0 - burial_factor)

            total_sasa += sasa

        # Surface area energy
        sa_energy = self.surface_tension * total_sasa

        return sa_energy

    def _calculate_entropy_contribution(self, ligand_mol: Chem.Mol) -> float:
        """Calculate entropy contribution (simplified)"""
        # Rotational and translational entropy loss upon binding
        # This is typically calculated from MD simulations or normal mode analysis

        # Simplified model based on molecular properties
        molecular_weight = Descriptors.MolWt(ligand_mol)
        num_rotatable_bonds = Descriptors.NumRotatableBonds(ligand_mol)

        # Translational and rotational entropy (standard state)
        # Values in kcal/mol at 298K
        translational_entropy = 0.0  # Often ignored in relative calculations
        rotational_entropy = 0.0     # Often ignored in relative calculations

        # Conformational entropy loss (major contributor)
        conformational_entropy = num_rotatable_bonds * 1.0  # ~1 kcal/mol per rotatable bond

        total_entropy = translational_entropy + rotational_entropy + conformational_entropy

        return total_entropy

    def _get_ideal_bond_length(self, element1: str, element2: str) -> float:
        """Get ideal bond length between two elements"""
        bond_lengths = {
            ('C', 'C'): 1.54, ('C', 'N'): 1.47, ('C', 'O'): 1.43,
            ('C', 'S'): 1.81, ('N', 'N'): 1.45, ('N', 'O'): 1.40,
            ('O', 'O'): 1.48, ('C', 'H'): 1.09, ('N', 'H'): 1.01,
            ('O', 'H'): 0.96, ('S', 'H'): 1.34
        }

        bond_pair = tuple(sorted([element1, element2]))
        return bond_lengths.get(bond_pair, 1.5)  # Default bond length

    def _get_receptor_charges(self, receptor_atoms: List) -> List[float]:
        """Get receptor partial charges (simplified)"""
        charges = []

        for atom in receptor_atoms:
            residue_name = atom.get_parent().get_resname()
            atom_name = atom.get_name()
            element = atom.element

            charge = 0.0

            # Simplified charge assignment
            if residue_name == 'ARG':
                if atom_name in ['NH1', 'NH2']:
                    charge = 0.8
                elif atom_name == 'NE':
                    charge = 0.4
            elif residue_name == 'LYS':
                if atom_name == 'NZ':
                    charge = 1.0
            elif residue_name == 'ASP':
                if atom_name in ['OD1', 'OD2']:
                    charge = -0.8
            elif residue_name == 'GLU':
                if atom_name in ['OE1', 'OE2']:
                    charge = -0.8
            elif residue_name == 'HIS':
                if atom_name in ['ND1', 'NE2']:
                    charge = 0.4
            # Add more residue types as needed

            charges.append(charge)

        return charges